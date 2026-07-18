"""
prometheus_metrics.py — Prometheus Instrumentation for EcoPackAI (Prompt 31)
============================================================================

Instruments the FastAPI application with Prometheus metrics and exposes
a ``/metrics`` endpoint.  Includes a Grafana dashboard JSON generator.

Custom Metrics
--------------
* ``ecopackai_void_pct_histogram`` — Histogram of void volume percentages
* ``ecopackai_safety_violations_total`` — Counter of safety violations
* ``ecopackai_co2e_saved_kg_total`` — Counter of CO₂e saved (kg)

Author: EcoPackAI Team
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing Prometheus client
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info,
        generate_latest, CONTENT_TYPE_LATEST,
    )
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.middleware.base import BaseHTTPMiddleware
    import time

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False
    logger.warning("prometheus_client not installed. Metrics disabled.")


# ═══════════════════════════════════════════════════════════════════════════
# Custom Metrics
# ═══════════════════════════════════════════════════════════════════════════

if _HAS_PROMETHEUS:
    # Void volume percentage histogram
    ecopackai_void_pct_histogram = Histogram(
        "ecopackai_void_pct",
        "Distribution of void volume percentages across packed orders",
        buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    )

    # Safety violations counter
    ecopackai_safety_violations_total = Counter(
        "ecopackai_safety_violations_total",
        "Total number of fragility constraint violations",
    )

    # CO2e saved counter
    ecopackai_co2e_saved_kg_total = Counter(
        "ecopackai_co2e_saved_kg_total",
        "Total CO2-equivalent emissions saved in kilograms",
    )

    # Request latency histogram
    ecopackai_request_duration = Histogram(
        "ecopackai_request_duration_seconds",
        "HTTP request duration in seconds",
        ["method", "endpoint", "status_code"],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    )

    # Active requests gauge
    ecopackai_requests_in_progress = Gauge(
        "ecopackai_requests_in_progress",
        "Number of HTTP requests currently being processed",
    )

    # App info
    ecopackai_app_info = Info(
        "ecopackai_app",
        "Application information",
    )
    ecopackai_app_info.info({
        "version": "1.0.0",
        "component": "packing_service",
    })
else:
    # Stubs when prometheus_client not installed
    class _Stub:
        def observe(self, *a, **kw): pass
        def inc(self, *a, **kw): pass
        def labels(self, *a, **kw): return self

    ecopackai_void_pct_histogram = _Stub()
    ecopackai_safety_violations_total = _Stub()
    ecopackai_co2e_saved_kg_total = _Stub()
    ecopackai_request_duration = _Stub()
    ecopackai_requests_in_progress = _Stub()


# ═══════════════════════════════════════════════════════════════════════════
# Recording helpers
# ═══════════════════════════════════════════════════════════════════════════

def record_packing_metrics(
    void_pct: float,
    violations: int = 0,
    co2e_saved_kg: float = 0.0,
) -> None:
    """Record packing metrics for a single order.

    Parameters
    ----------
    void_pct : float
        Void volume percentage of the packed order.
    violations : int
        Number of constraint violations.
    co2e_saved_kg : float
        CO₂e saved compared to baseline (kg).
    """
    ecopackai_void_pct_histogram.observe(void_pct)
    if violations > 0:
        ecopackai_safety_violations_total.inc(violations)
    if co2e_saved_kg > 0:
        ecopackai_co2e_saved_kg_total.inc(co2e_saved_kg)


# ═══════════════════════════════════════════════════════════════════════════
# Middleware & /metrics endpoint
# ═══════════════════════════════════════════════════════════════════════════

def instrument_app(app: FastAPI) -> None:
    """Add Prometheus instrumentation to a FastAPI application.

    Adds request duration tracking middleware and a ``/metrics`` endpoint.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application to instrument.
    """
    if not _HAS_PROMETHEUS:
        logger.warning("Prometheus client not installed. Skipping instrumentation.")
        return

    class PrometheusMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            ecopackai_requests_in_progress.inc()
            start = time.perf_counter()

            try:
                response = await call_next(request)
            except Exception:
                ecopackai_requests_in_progress.dec()
                raise

            duration = time.perf_counter() - start
            ecopackai_request_duration.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code,
            ).observe(duration)
            ecopackai_requests_in_progress.dec()

            return response

    app.add_middleware(PrometheusMiddleware)

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    logger.info("Prometheus instrumentation added to FastAPI app.")


# ═══════════════════════════════════════════════════════════════════════════
# Grafana Dashboard JSON
# ═══════════════════════════════════════════════════════════════════════════

def generate_grafana_dashboard(output_path: str = "grafana_dashboard.json") -> Path:
    """Generate a Grafana dashboard JSON with 3 panels.

    Panels
    ------
    1. **Void Volume Distribution** — Histogram of void_pct
    2. **Safety Violations** — Counter time series
    3. **CO₂e Saved** — Cumulative counter

    Parameters
    ----------
    output_path : str
        Where to save the JSON file.

    Returns
    -------
    Path
        Path to the saved file.
    """
    dashboard = {
        "dashboard": {
            "id": None,
            "uid": "ecopackai-sustainability",
            "title": "EcoPackAI Sustainability Dashboard",
            "tags": ["ecopackai", "sustainability", "packing"],
            "timezone": "utc",
            "refresh": "30s",
            "time": {"from": "now-24h", "to": "now"},
            "panels": [
                {
                    "id": 1,
                    "title": "📦 Void Volume Distribution",
                    "type": "histogram",
                    "gridPos": {"h": 8, "w": 8, "x": 0, "y": 0},
                    "targets": [{
                        "expr": "histogram_quantile(0.5, rate(ecopackai_void_pct_bucket[5m]))",
                        "legendFormat": "p50",
                    }, {
                        "expr": "histogram_quantile(0.95, rate(ecopackai_void_pct_bucket[5m]))",
                        "legendFormat": "p95",
                    }],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent",
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": 50},
                                    {"color": "red", "value": 80},
                                ],
                            },
                        },
                    },
                },
                {
                    "id": 2,
                    "title": "⚠️ Safety Violations (Rate)",
                    "type": "timeseries",
                    "gridPos": {"h": 8, "w": 8, "x": 8, "y": 0},
                    "targets": [{
                        "expr": "rate(ecopackai_safety_violations_total[5m])",
                        "legendFormat": "violations/s",
                    }],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "ops",
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "red", "value": 0.1},
                                ],
                            },
                        },
                    },
                },
                {
                    "id": 3,
                    "title": "🌱 CO₂e Saved (Cumulative)",
                    "type": "stat",
                    "gridPos": {"h": 8, "w": 8, "x": 16, "y": 0},
                    "targets": [{
                        "expr": "ecopackai_co2e_saved_kg_total",
                        "legendFormat": "Total CO₂e saved",
                    }],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "masskg",
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                ],
                            },
                        },
                    },
                },
            ],
            "schemaVersion": 39,
            "version": 1,
        },
        "overwrite": True,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(dashboard, f, indent=2)

    logger.info("Grafana dashboard saved to %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = generate_grafana_dashboard()
    print(f"Dashboard: {p}")
