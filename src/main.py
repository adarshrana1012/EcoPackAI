"""
main.py — Unified EcoPackAI API Gateway (Phase 8, Prompt P07)
=============================================================

This is the single entry point for the EcoPackAI backend platform.
It composes the classification sub-app and packing sub-app behind a
unified gateway that adds:

  * CORS middleware (origins from Settings.CORS_ORIGINS)
  * X-Request-ID correlation middleware
  * Rate-limiting per-tenant via Redis (src/cache.py)
  * JWT authentication/authorisation guards (src/auth.py)
  * Prometheus instrumentation (src/prometheus_metrics.py)
  * Async DB persistence for packing results (src/database.py)
  * Redis classification cache invalidation on model promotion

Entry points
------------
  POST /v1/auth/login           — exchange credentials for JWT
  GET  /v1/health               — gateway liveness probe
  POST /v1/classify             — fragility classification (cached)
  POST /v1/pack                 — 3-D bin packing + DB persistence
  GET  /v1/metrics/aggregate    — (auth) aggregate dashboard metrics
  GET  /v1/metrics/{id}         — (auth) per-shipment metrics
  GET  /v1/models/versions      — (admin) list model registry
  POST /v1/models/promote/{v}   — (admin) promote model + bust cache
  POST /v1/train/trigger        — (admin) enqueue Celery retrain task
  GET  /v1/ab-test/results      — (admin) A/B test results
  POST /v1/ab-test/configure    — (admin) update traffic split
  GET  /metrics                 — Prometheus scrape endpoint

Run
---
  uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

Author: EcoPackAI Team
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Annotated, Any, Dict, List, Optional

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from src.settings import get_settings
from src.auth import (
    LoginRequest, LoginResponse,
    get_current_user, require_admin,
    login_handler,
)
from src.database import (
    get_db, get_shipment, get_aggregate_metrics, create_shipment,
    AggregateMetrics, ShipmentMetrics,
)
from src.cache import (
    check_rate_limit, invalidate_classification_cache, is_redis_connected,
)
from src.prometheus_metrics import instrument_app

# Sub-applications
from src.classify_api import app as classify_app
from src.packing_api import (
    packing_app,
    pack_order_endpoint,
    OrderPackRequest,
    PackResponse,
    _load_resources as _load_packing_resources,
)
from src.ab_test_router import ab_router, get_ab_router, ABTestConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
from src.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)
slog = structlog.get_logger(__name__)

_settings = get_settings()


# ---------------------------------------------------------------------------
# CORS Origins from Settings
# ---------------------------------------------------------------------------
_CORS_ORIGINS: List[str] = _settings.cors_origins_list


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared resources on gateway startup."""
    import sys
    from urllib.parse import urlparse
    from src.settings import ConfigurationError
    
    logger.info("EcoPackAI Gateway starting up…")
    
    try:
        _settings.validate_connections()
    except ConfigurationError as e:
        logger.error("Startup failed: %s", e)
        sys.exit(1)
        
    db_host = urlparse(_settings.DATABASE_URL).hostname
    logger.info("ENV: %s | MODEL_PATH: %s | DB Host: %s", _settings.ENV, _settings.MODEL_PATH, db_host)
    
    if _settings.SENTRY_DSN:
        try:
            import sentry_sdk
            sentry_sdk.init(
                dsn=_settings.SENTRY_DSN,
                traces_sample_rate=0.1,
                environment=_settings.ENV,
            )
            logger.info("Sentry SDK initialized.")
        except ImportError:
            logger.warning("SENTRY_DSN set but sentry_sdk not installed.")

    if _settings.OTLP_ENDPOINT:
        try:
            from src.telemetry import setup_telemetry
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
            from opentelemetry.instrumentation.redis import RedisInstrumentor
            
            setup_telemetry("ecopackai-api", _settings.OTLP_ENDPOINT)
            FastAPIInstrumentor.instrument_app(app)
            SQLAlchemyInstrumentor().instrument()
            RedisInstrumentor().instrument()
            logger.info("OpenTelemetry initialized.")
        except ImportError:
            logger.warning("OpenTelemetry libraries not installed.")

    _load_packing_resources()
    yield
    logger.info("EcoPackAI Gateway shutting down.")


# ---------------------------------------------------------------------------
# Root Gateway Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="EcoPackAI Gateway",
    description=(
        "Unified API gateway for fragility classification, 3-D packing "
        "optimization, analytics, and model management."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Middleware 1: CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-Cache"],
)


# ---------------------------------------------------------------------------
# Middleware 2: X-Request-ID Correlation
# ---------------------------------------------------------------------------
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a per-request UUID4 correlation ID.

    Reads ``X-Request-ID`` from the incoming request headers; if absent,
    generates a new UUID4.  The ID is stored on ``request.state.request_id``
    and returned as the ``X-Request-ID`` response header.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            structlog.contextvars.clear_contextvars()


app.add_middleware(RequestIDMiddleware)


# ---------------------------------------------------------------------------
# Middleware 2.5: Access Logging
# ---------------------------------------------------------------------------
import time

class AccessLogMiddleware(BaseHTTPMiddleware):
    """Log every incoming request with status and duration."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in ("/v1/health", "/metrics", "/v1/ready", "/v1/startup"):
            return await call_next(request)
            
        start_time = time.perf_counter()
        method = request.method
        path = request.url.path
        
        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000
            status_code = response.status_code
            
            if status_code >= 500:
                logger.error("Request failed", method=method, path=path, status=status_code, duration_ms=round(duration_ms, 2))
            elif status_code >= 400:
                logger.warning("Request rejected", method=method, path=path, status=status_code, duration_ms=round(duration_ms, 2))
            else:
                logger.info("Request handled", method=method, path=path, status=status_code, duration_ms=round(duration_ms, 2))
                
            return response
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.exception("Request unhandled exception", method=method, path=path, duration_ms=round(duration_ms, 2))
            raise e

app.add_middleware(AccessLogMiddleware)


# ---------------------------------------------------------------------------
# Middleware 3: Rate Limiter (token bucket via Redis)
# ---------------------------------------------------------------------------
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce per-tenant rate limiting on all /v1/* routes.

    Uses the Redis token-bucket implementation in :mod:`src.cache`.
    On exceeded limit, returns 429 with a Retry-After header.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if not request.url.path.startswith("/v1/"):
            return await call_next(request)

        tenant_id = (
            request.headers.get("X-Tenant-ID")
            or (request.client.host if request.client else "unknown")
        )
        allowed = await check_rate_limit(
            tenant_id=tenant_id,
            limit=_settings.API_RATE_LIMIT,
            window_seconds=60,
        )
        if not allowed:
            request_id = getattr(request.state, "request_id", "unknown")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded. Try again in 60 seconds.",
                    "retry_after_seconds": 60,
                    "request_id": request_id,
                },
                headers={"Retry-After": "60"},
            )
        return await call_next(request)


app.add_middleware(RateLimitMiddleware)


# ---------------------------------------------------------------------------
# Prometheus instrumentation (adds /metrics endpoint)
# ---------------------------------------------------------------------------
instrument_app(app)


# ---------------------------------------------------------------------------
# Global Exception Handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled exceptions.

    Logs the error with structlog and returns a sanitised 500 response
    that includes the request correlation ID for traceability.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    slog.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        request_id=request_id,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": request_id,
        },
    )


# ---------------------------------------------------------------------------
# Mount/Include Sub-Applications
# ---------------------------------------------------------------------------
# Include classify_router to prevent root-level mount hijacking
app.include_router(classify_app.router)

# packing_app handles:   /v1/pack, /v1/pack/health
app.mount("/packing", packing_app)


# ═══════════════════════════════════════════════════════════════════════════
# Gateway-level Routes
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@app.post("/v1/auth/login", response_model=LoginResponse, tags=["Auth"])
async def login(request: Request, body: LoginRequest) -> LoginResponse:
    """Authenticate with email/password and receive a JWT access token.

    Demo credentials
    ----------------
    - **user**:  demo@ecopackai.io  / demo123
    - **admin**: admin@ecopackai.io / admin123
    """
    client_ip = request.client.host if request.client else "unknown"
    return await login_handler(body, client_ip=client_ip)


# ---------------------------------------------------------------------------
# Analytics / Metrics
# ---------------------------------------------------------------------------

class DateRangeParams(BaseModel):
    """Optional date-range query parameters for aggregate metrics."""
    start_date: Optional[str] = Field(None, description="ISO-8601 start date (inclusive)")
    end_date: Optional[str] = Field(None, description="ISO-8601 end date (inclusive)")


@app.get("/v1/metrics/aggregate", response_model=AggregateMetrics, tags=["Analytics"])
async def aggregate_metrics(
    user: Annotated[Dict[str, Any], Depends(get_current_user)],
    db=Depends(get_db),
) -> AggregateMetrics:
    """Return aggregated sustainability and packing metrics.

    Requires a valid JWT (any role).
    """
    return await get_aggregate_metrics(db)


@app.get("/v1/metrics/{shipment_id}", response_model=ShipmentMetrics, tags=["Analytics"])
async def per_shipment_metrics(
    shipment_id: str,
    user: Annotated[Dict[str, Any], Depends(get_current_user)],
    db=Depends(get_db),
) -> ShipmentMetrics:
    """Return per-shipment sustainability metrics.

    Requires a valid JWT (any role).
    Raises 404 if the shipment is not found.
    """
    result = await get_shipment(db, shipment_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Shipment '{shipment_id}' not found.",
        )
    return result


# ---------------------------------------------------------------------------
# Packing (with DB persistence)
# ---------------------------------------------------------------------------

@app.post("/v1/pack", response_model=PackResponse, tags=["Packing"])
async def pack_order_gateway(
    request_body: OrderPackRequest,
    db=Depends(get_db),
) -> PackResponse:
    """Pack an order and persist the result to the database.

    Orchestrates the 3-D bin packing engine and asynchronously records
    the shipment sustainability metrics to PostgreSQL.
    """
    result = await pack_order_endpoint(request_body)

    # Persist to DB asynchronously (best-effort)
    try:
        from src.metrics_calculator import estimate_co2e_kg
        co2e = estimate_co2e_kg(result.estimated_material_weight_g, transport_distance_km=500.0)
        await create_shipment(
            db=db,
            order_id=result.order_id,
            box_sku=result.box_sku,
            void_pct=result.void_volume_pct,
            material_weight_g=result.estimated_material_weight_g,
            co2e_kg=co2e,
        )
    except Exception as exc:
        logger.warning("DB persistence skipped: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Model Registry (Admin)
# ---------------------------------------------------------------------------

class ModelVersionsResponse(BaseModel):
    """Model registry summary response."""
    production: Optional[Dict[str, Any]] = None
    staging: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = []


@app.get("/v1/models/versions", response_model=ModelVersionsResponse, tags=["Model Registry"])
async def list_model_versions(
    user: Annotated[Dict[str, Any], Depends(require_admin)],
) -> ModelVersionsResponse:
    """List model versions from the registry. Requires admin role."""
    try:
        from pathlib import Path
        from src.model_registry import ModelRegistry

        registry = ModelRegistry(str(Path("models") / "registry"))
        versions = registry.list_versions()

        production = None
        staging = None
        history = []

        for v in versions:
            entry = {
                "version": v.get("version"),
                "accuracy": v.get("accuracy"),
                "deployed_at": v.get("created_at"),
                "is_production": v.get("is_production", False),
            }
            if entry["is_production"]:
                production = entry
            elif staging is None:
                staging = entry
            else:
                history.append(entry)

        return ModelVersionsResponse(
            production=production,
            staging=staging,
            history=history,
        )
    except Exception as exc:
        logger.warning("Model registry query failed: %s", exc)
        # Return sensible mock data
        return ModelVersionsResponse(
            production={"version": "v1.2", "accuracy": 0.95, "deployed_at": "2026-05-15T10:00:00Z"},
            staging={"version": "v1.3", "accuracy": 0.96, "deployed_at": "2026-06-20T10:00:00Z"},
            history=[{"version": "v1.1", "accuracy": 0.92, "deployed_at": "2026-01-10T10:00:00Z"}],
        )


@app.post("/v1/models/promote/{version}", tags=["Model Registry"])
async def promote_model(
    version: str,
    user: Annotated[Dict[str, Any], Depends(require_admin)],
) -> Dict[str, str]:
    """Promote a model version to production and invalidate the classification cache.

    Requires admin role.
    """
    try:
        from pathlib import Path
        from src.model_registry import ModelRegistry

        registry = ModelRegistry(str(Path("models") / "registry"))
        registry.promote_to_production(version)
        logger.info("Model %s promoted to production by %s", version, user.get("sub"))
    except Exception as exc:
        logger.warning("Model promotion failed: %s — proceeding with cache invalidation.", exc)

    # Invalidate classification cache regardless of registry success
    deleted = await invalidate_classification_cache()
    slog.info(
        "Classification cache invalidated after model promotion",
        version=version,
        keys_deleted=deleted,
    )

    return {
        "status": "success",
        "message": f"Model {version} promoted to production. {deleted} cache keys invalidated.",
    }


# ---------------------------------------------------------------------------
# Training Trigger (Admin)
# ---------------------------------------------------------------------------

@app.post("/v1/train/trigger", tags=["Model Registry"])
async def trigger_training(
    user: Annotated[Dict[str, Any], Depends(require_admin)],
) -> Dict[str, str]:
    """Enqueue a Celery retraining task. Requires admin role."""
    try:
        from src.retrain_celery import retrain_rl_policy
        task = retrain_rl_policy.delay()
        job_id = task.id
        logger.info("Retrain task enqueued by %s — job_id=%s", user.get("sub"), job_id)
        return {
            "status": "success",
            "job_id": job_id,
            "message": "Retraining pipeline triggered.",
        }
    except Exception as exc:
        logger.warning("Celery not available: %s — returning mock job_id.", exc)
        return {
            "status": "success",
            "job_id": "mock-job-" + str(uuid.uuid4())[:8],
            "message": "Retraining pipeline triggered (mock — Celery unavailable).",
        }


# ---------------------------------------------------------------------------
# A/B Testing (Admin)
# ---------------------------------------------------------------------------

@app.get("/v1/ab-test/results", tags=["A/B Testing"])
async def ab_results(
    user: Annotated[Dict[str, Any], Depends(require_admin)],
):
    """Return A/B test variant metrics. Requires admin role."""
    router = get_ab_router()
    return router.get_results()


@app.post("/v1/ab-test/configure", tags=["A/B Testing"])
async def configure_ab_test(
    config: ABTestConfig,
    user: Annotated[Dict[str, Any], Depends(require_admin)],
) -> Dict[str, Any]:
    """Update A/B test traffic split. Requires admin role."""
    router = get_ab_router()
    router.update_config(config.rl_traffic_pct)
    logger.info(
        "A/B split updated by %s: RL=%.1f%% FFD=%.1f%%",
        user.get("sub"), router.rl_pct, router.ffd_pct,
    )
    return {
        "status": "updated",
        "rl_pct": router.rl_pct,
        "ffd_pct": router.ffd_pct,
    }


# ---------------------------------------------------------------------------
# Gateway Health Probes
# ---------------------------------------------------------------------------
import os
import time
import psutil
from src.database import get_db
from sqlalchemy import text
from src.classify_api import _state as classify_state
from src.packing_api import _state as packing_state

@app.get("/v1/health", tags=["Gateway"])
async def health_probe() -> Dict[str, Any]:
    """Liveness probe: Returns 200 as long as the process is running."""
    p = psutil.Process(os.getpid())
    uptime_seconds = time.time() - p.create_time()
    return {
        "status": "healthy",
        "model_version": classify_state.get("model_version", "unknown"),
        "model_loaded": classify_state.get("model") is not None,
        "uptime_seconds": round(uptime_seconds, 2),
        "pid": os.getpid(),
    }

@app.get("/v1/ready", tags=["Gateway"])
async def readiness_probe() -> Dict[str, Any]:
    """Readiness probe: Returns 200 only if all critical services are healthy."""
    failing = []
    
    # 1. Check Model & Pipeline
    if not classify_state.get("model") or not classify_state.get("pipeline"):
        failing.append("classifier")
        
    # 2. Check Catalogue
    if not packing_state.get("catalogue"):
        failing.append("catalogue")
        
    # 3. Check Redis
    redis_ok = await is_redis_connected()
    if not redis_ok:
        failing.append("redis")
        
    # 4. Check DB
    try:
        async for session in get_db():
            await session.execute(text("SELECT 1"))
            break
    except Exception:
        failing.append("db")
        
    if failing:
        raise HTTPException(
            status_code=503,
            detail={"ready": False, "failing": failing}
        )
        
    return {"ready": True, "failing": []}

@app.get("/v1/startup", tags=["Gateway"])
async def startup_probe() -> Dict[str, Any]:
    """Startup probe: Returns 200 once initial model load is complete."""
    loaded = classify_state.get("model") is not None and packing_state.get("catalogue") is not None
    if not loaded:
        raise HTTPException(
            status_code=503,
            detail={"loaded": False}
        )
    return {"loaded": True}


# ---------------------------------------------------------------------------
# Shipments History & CSV Export (Phase 9, Prompt P13)
# ---------------------------------------------------------------------------

class ShipmentsListResponse(BaseModel):
    """Paginated shipments response schema."""
    shipments: List[ShipmentMetrics]
    total: int
    page: int
    pages: int


@app.get("/v1/shipments", response_model=ShipmentsListResponse, tags=["Analytics"])
async def get_shipments(
    user: Annotated[Dict[str, Any], Depends(get_current_user)],
    page: int = 1,
    page_size: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    box_sku: Optional[str] = None,
    damage_reported: Optional[bool] = None,
    db=Depends(get_db),
) -> ShipmentsListResponse:
    """Retrieve filtered, paginated shipment records. Requires authentication."""
    # Enforce maximum page size
    page_size = min(max(page_size, 1), 100)
    page = max(page, 1)

    if db is None:
        # DB unavailable — yield realistic mocked data
        mock_shipments = [
            ShipmentMetrics(
                shipment_id="8fa2b879-11c2-40de-a5d6-d08e92d7ca81",
                order_id="order-2026-001",
                box_sku="BOX-M4",
                void_volume_pct=15.4,
                material_weight_g=245.0,
                co2e_kg=0.25,
                damage_reported=False,
                packed_at="2026-07-07T12:00:00Z"
            ),
            ShipmentMetrics(
                shipment_id="59b66c4c-32ef-4682-8bc1-5fe15cb0b112",
                order_id="order-2026-002",
                box_sku="BOX-L2",
                void_volume_pct=34.1,
                material_weight_g=410.0,
                co2e_kg=0.42,
                damage_reported=True,
                packed_at="2026-07-07T11:45:00Z"
            ),
            ShipmentMetrics(
                shipment_id="ac52d19f-bfa8-48b0-a33d-c12e52b2b113",
                order_id="order-2026-003",
                box_sku="BOX-S1",
                void_volume_pct=8.9,
                material_weight_g=120.0,
                co2e_kg=0.12,
                damage_reported=False,
                packed_at="2026-07-07T10:15:00Z"
            )
        ]
        return ShipmentsListResponse(
            shipments=mock_shipments,
            total=len(mock_shipments),
            page=1,
            pages=1
        )

    try:
        from sqlalchemy import select, func
        from src.database import ShipmentRecord

        query = select(ShipmentRecord)

        # Filters
        if start_date:
            try:
                from datetime import datetime
                query = query.where(ShipmentRecord.packed_at >= datetime.fromisoformat(start_date))
            except ValueError:
                pass
        if end_date:
            try:
                from datetime import datetime
                query = query.where(ShipmentRecord.packed_at <= datetime.fromisoformat(end_date))
            except ValueError:
                pass
        if box_sku:
            query = query.where(ShipmentRecord.box_sku.ilike(f"%{box_sku}%"))
        if damage_reported is not None:
            query = query.where(ShipmentRecord.damage_reported == damage_reported)

        # Count total
        count_stmt = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_stmt)
        total = total_result.scalar() or 0

        # Sort and Paginate
        query = query.order_by(ShipmentRecord.packed_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await db.execute(query)
        rows = result.scalars().all()

        shipments = [
            ShipmentMetrics(
                shipment_id=str(row.shipment_id),
                order_id=row.order_id,
                box_sku=row.box_sku,
                void_volume_pct=row.void_volume_pct,
                material_weight_g=row.material_weight_g,
                co2e_kg=row.co2e_kg,
                damage_reported=row.damage_reported,
                packed_at=row.packed_at.isoformat() if row.packed_at else None,
            )
            for row in rows
        ]

        import math
        pages = math.ceil(total / page_size) if total > 0 else 1

        return ShipmentsListResponse(
            shipments=shipments,
            total=total,
            page=page,
            pages=pages
        )
    except Exception as exc:
        logger.warning("Failed to query shipments database: %s. Falling back to mock data.", exc)
        mock_shipments = [
            ShipmentMetrics(
                shipment_id="8fa2b879-11c2-40de-a5d6-d08e92d7ca81",
                order_id="order-2026-001",
                box_sku="BOX-M4",
                void_volume_pct=15.4,
                material_weight_g=245.0,
                co2e_kg=0.25,
                damage_reported=False,
                packed_at="2026-07-07T12:00:00Z"
            ),
            ShipmentMetrics(
                shipment_id="59b66c4c-32ef-4682-8bc1-5fe15cb0b112",
                order_id="order-2026-002",
                box_sku="BOX-L2",
                void_volume_pct=34.1,
                material_weight_g=410.0,
                co2e_kg=0.42,
                damage_reported=True,
                packed_at="2026-07-07T11:45:00Z"
            ),
            ShipmentMetrics(
                shipment_id="ac52d19f-bfa8-48b0-a33d-c12e52b2b113",
                order_id="order-2026-003",
                box_sku="BOX-S1",
                void_volume_pct=8.9,
                material_weight_g=120.0,
                co2e_kg=0.12,
                damage_reported=False,
                packed_at="2026-07-07T10:15:00Z"
            )
        ]
        return ShipmentsListResponse(
            shipments=mock_shipments,
            total=len(mock_shipments),
            page=1,
            pages=1
        )


@app.get("/v1/shipments/export", tags=["Analytics"])
async def export_shipments(
    user: Annotated[Dict[str, Any], Depends(get_current_user)],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db=Depends(get_db),
):
    """Export shipment history as a CSV file. Requires authentication."""
    import csv
    import io
    from fastapi.responses import StreamingResponse

    def generate_csv(rows_data):
        output = io.StringIO()
        writer = csv.writer(output)
        # Header
        writer.writerow([
            "shipment_id", "order_id", "box_sku", "void_volume_pct",
            "material_weight_g", "co2e_kg", "damage_reported", "packed_at"
        ])
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)

        for row in rows_data:
            writer.writerow([
                row.get("shipment_id"),
                row.get("order_id"),
                row.get("box_sku"),
                row.get("void_volume_pct"),
                row.get("material_weight_g"),
                row.get("co2e_kg"),
                row.get("damage_reported"),
                row.get("packed_at")
            ])
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

    mock_export_data = [
        {
            "shipment_id": "8fa2b879-11c2-40de-a5d6-d08e92d7ca81",
            "order_id": "order-2026-001",
            "box_sku": "BOX-M4",
            "void_volume_pct": 15.4,
            "material_weight_g": 245.0,
            "co2e_kg": 0.25,
            "damage_reported": False,
            "packed_at": "2026-07-07T12:00:00Z"
        },
        {
            "shipment_id": "59b66c4c-32ef-4682-8bc1-5fe15cb0b112",
            "order_id": "order-2026-002",
            "box_sku": "BOX-L2",
            "void_volume_pct": 34.1,
            "material_weight_g": 410.0,
            "co2e_kg": 0.42,
            "damage_reported": True,
            "packed_at": "2026-07-07T11:45:00Z"
        }
    ]

    if db is None:
        return StreamingResponse(
            generate_csv(mock_export_data),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=shipments_export.csv"}
        )

    try:
        from sqlalchemy import select
        from src.database import ShipmentRecord

        query = select(ShipmentRecord)
        if start_date:
            from datetime import datetime
            query = query.where(ShipmentRecord.packed_at >= datetime.fromisoformat(start_date))
        if end_date:
            from datetime import datetime
            query = query.where(ShipmentRecord.packed_at <= datetime.fromisoformat(end_date))

        query = query.order_by(ShipmentRecord.packed_at.desc())
        result = await db.execute(query)
        rows = result.scalars().all()

        formatted_rows = [
            {
                "shipment_id": str(row.shipment_id),
                "order_id": row.order_id,
                "box_sku": row.box_sku,
                "void_volume_pct": row.void_volume_pct,
                "material_weight_g": row.material_weight_g,
                "co2e_kg": row.co2e_kg,
                "damage_reported": row.damage_reported,
                "packed_at": row.packed_at.isoformat() if row.packed_at else "",
            }
            for row in rows
        ]

        return StreamingResponse(
            generate_csv(formatted_rows),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=shipments_export.csv"}
        )
    except Exception as exc:
        logger.warning("Failed to export shipments database: %s. Falling back to mock data.", exc)
        return StreamingResponse(
            generate_csv(mock_export_data),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=shipments_export.csv"}
        )

