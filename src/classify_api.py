"""
classify_api.py — FastAPI Classification Service for EcoPackAI
==============================================================

Exposes ``POST /v1/classify`` for real-time fragility classification and
``GET /v1/health`` for liveness probes.

Usage
-----
    uvicorn src.classify_api:app --host 0.0.0.0 --port 8000

Author: EcoPackAI Team
"""

from __future__ import annotations

import structlog
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.packing_api import (
    pack_order_endpoint,
    packing_health,
    OrderPackRequest,
    PackResponse,
    _load_resources as _load_packing_resources,
)
from src.ab_test_router import ab_router
from src.cache import (
    build_cache_key,
    get_cached_classification,
    set_cached_classification,
    is_redis_connected,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# (Logging configured in main.py)
logger = structlog.get_logger(__name__)

from opentelemetry import trace
tracer = trace.get_tracer(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIER_LABELS: Dict[int, str] = {0: "None", 1: "Low", 2: "Medium", 3: "Critical"}
VALID_MATERIALS = ["glass", "electronics", "apparel", "fragile_liquid", "standard"]

FEATURE_COLUMNS = [
    "length_cm", "width_cm", "height_cm", "weight_g",
    "material_type", "volume_cm3", "aspect_ratio",
    "historical_material_volume_cm3",
]

MODEL_DIR = Path("models")
REGISTRY_DIR = MODEL_DIR / "registry"


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class ProductFeatures(BaseModel):
    """Input schema for product classification."""
    length_cm: float = Field(..., gt=0, description="Product length in cm")
    width_cm: float = Field(..., gt=0, description="Product width in cm")
    height_cm: float = Field(..., gt=0, description="Product height in cm")
    weight_g: float = Field(..., gt=0, description="Product weight in grams")
    material_type: str = Field(..., description="Material category")

    @field_validator("material_type")
    @classmethod
    def validate_material(cls, v: str) -> str:
        if v not in VALID_MATERIALS:
            raise ValueError(f"material_type must be one of {VALID_MATERIALS}")
        return v

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "length_cm": 15.0,
                "width_cm": 10.0,
                "height_cm": 8.0,
                "weight_g": 500.0,
                "material_type": "glass",
            }
        ]
    }}


class FragilityResponse(BaseModel):
    """Output schema for fragility classification."""
    tier: int = Field(..., ge=0, le=3, description="Fragility tier (0-3)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    tier_label: str = Field(..., description="Human-readable tier label")
    probabilities: Dict[str, float] = Field(..., description="Per-class probabilities")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_version: str
    model_loaded: bool
    redis_connected: bool = False


# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {
    "model": None,
    "pipeline": None,
    "model_version": "unknown",
}


def _load_model() -> None:
    """Attempt to load model and feature pipeline from disk."""
    # Try ModelRegistry first
    try:
        from src.model_registry import ModelRegistry
        registry = ModelRegistry(str(REGISTRY_DIR))
        model, meta = registry.get_production_model()
        _state["model"] = model
        _state["model_version"] = meta.get("version", "registry")
        logger.info("Model loaded from registry (version=%s)",
                    _state["model_version"])
    except Exception as e:
        logger.info("Registry load failed (%s), trying direct path...", e)
        model_path = MODEL_DIR / "best_model.joblib"
        if model_path.exists():
            _state["model"] = joblib.load(model_path)
            _state["model_version"] = "direct-load"
            logger.info("Model loaded from %s", model_path)
        else:
            logger.warning("No model found. /v1/classify will return 503.")

    # Load feature pipeline
    pipeline_path = MODEL_DIR / "feature_pipeline.joblib"
    if pipeline_path.exists():
        _state["pipeline"] = joblib.load(pipeline_path)
        logger.info("Feature pipeline loaded from %s", pipeline_path)
    else:
        logger.warning("Feature pipeline not found at %s", pipeline_path)


# ---------------------------------------------------------------------------
# App Lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    _load_model()
    _load_packing_resources()
    yield
    logger.info("Shutting down classification service.")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="EcoPackAI Classification Service",
    description="Fragility classification for e-commerce products",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ab_router)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/classify", response_model=FragilityResponse)
async def classify_product(
    product: ProductFeatures,
    response: Response,
) -> FragilityResponse:
    """Classify a product's fragility tier.

    Accepts product dimensions and material type, runs the preprocessing
    pipeline and Random Forest classifier, and returns the predicted
    fragility tier with confidence scores.

    Responses are cached in Redis by SHA-256 keyed on the feature vector.
    Cache hits are indicated by the ``X-Cache: HIT`` response header.
    """
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first using train_classifier.py.",
        )

    if _state["pipeline"] is None:
        raise HTTPException(
            status_code=503,
            detail="Feature pipeline not loaded.",
        )

    # -----------------------------------------------------------------------
    # Cache lookup
    # -----------------------------------------------------------------------
    feature_dict = {
        "length_cm": product.length_cm,
        "width_cm": product.width_cm,
        "height_cm": product.height_cm,
        "weight_g": product.weight_g,
        "material_type": product.material_type,
    }
    cache_key = build_cache_key(feature_dict)
    cached = await get_cached_classification(cache_key)
    if cached is not None:
        response.headers["X-Cache"] = "HIT"
        return FragilityResponse(**cached)

    try:
        # Build single-row DataFrame matching training schema
        input_df = pd.DataFrame([{
            "product_id": "inference",
            "length_cm": product.length_cm,
            "width_cm": product.width_cm,
            "height_cm": product.height_cm,
            "weight_g": product.weight_g,
            "material_type": product.material_type,
            "fragility_label": 0,  # placeholder, not used for prediction
            "historical_material_volume_cm3": (
                product.length_cm * product.width_cm * product.height_cm
            ),
        }])

        # Apply fitted feature pipeline
        transformed = _state["pipeline"].transform(input_df)

        # Select feature columns
        X = transformed[FEATURE_COLUMNS]

        # Predict
        model = _state["model"]

        with tracer.start_as_current_span("ml.inference") as span:
            span.set_attribute("ml.model.version", _state["model_version"])
            prediction = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0]
            confidence = float(np.max(proba))
            
            span.set_attribute("ml.prediction.tier", prediction)
            span.set_attribute("ml.prediction.confidence", confidence)

        # Build per-class probability map
        probabilities = {
            TIER_LABELS[i]: round(float(p), 4)
            for i, p in enumerate(proba)
        }

        result = FragilityResponse(
            tier=prediction,
            confidence=round(confidence, 4),
            tier_label=TIER_LABELS[prediction],
            probabilities=probabilities,
        )

        # Store in cache (fire-and-forget — don't fail on cache errors)
        await set_cached_classification(cache_key, result.model_dump())
        response.headers["X-Cache"] = "MISS"
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Classification failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness and readiness probe."""
    redis_ok = await is_redis_connected()
    return HealthResponse(
        status="healthy" if _state["model"] is not None else "degraded",
        model_version=_state["model_version"],
        model_loaded=_state["model"] is not None,
        redis_connected=redis_ok,
    )


# ---------------------------------------------------------------------------
# Mounted / Reused Endpoints (re-exposed at gateway root)
# ---------------------------------------------------------------------------

