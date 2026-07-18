"""
packing_api.py — FastAPI Packing Endpoint for EcoPackAI
=======================================================

``POST /v1/pack`` accepts an order of items, classifies each item's
fragility using the trained RF model, runs the 3D packing engine,
and returns the optimal box assignment with placements.

Author: EcoPackAI Team
"""

from __future__ import annotations

import structlog

from opentelemetry import trace
tracer = trace.get_tracer(__name__)

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

from src.box_catalogue import BoxCatalogue
from src.packing_engine import Box, Item, pack_order

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = Path("models")
VALID_MATERIALS = ["glass", "electronics", "apparel", "fragile_liquid", "standard"]
TIER_LABELS = {0: "None", 1: "Low", 2: "Medium", 3: "Critical"}

FEATURE_COLUMNS = [
    "length_cm", "width_cm", "height_cm", "weight_g",
    "material_type", "volume_cm3", "aspect_ratio",
    "historical_material_volume_cm3",
]

# Approximate material weight per cm3 of packaging
MATERIAL_WEIGHT_FACTOR = 0.015  # grams per cm3 of box volume


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class PackItemRequest(BaseModel):
    """Single item in a packing order."""
    item_id: Optional[str] = Field(default=None, description="Item identifier")
    length_cm: float = Field(..., gt=0)
    width_cm: float = Field(..., gt=0)
    height_cm: float = Field(..., gt=0)
    weight_g: float = Field(..., gt=0)
    material_type: str = Field(default="standard")

    @model_validator(mode="before")
    @classmethod
    def resolve_dimensions_and_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Map length/width/height to length_cm/width_cm/height_cm
            if "length" in data and "length_cm" not in data:
                data["length_cm"] = data["length"]
            if "width" in data and "width_cm" not in data:
                data["width_cm"] = data["width"]
            if "height" in data and "height_cm" not in data:
                data["height_cm"] = data["height"]
            if "material_type" not in data or data["material_type"] is None:
                data["material_type"] = "standard"
        return data

    @field_validator("material_type")
    @classmethod
    def validate_material(cls, v: str) -> str:
        if v not in VALID_MATERIALS:
            raise ValueError(f"material_type must be one of {VALID_MATERIALS}")
        return v


class OrderPackRequest(BaseModel):
    """Packing request for an entire order."""
    order_id: Optional[str] = Field(default=None, description="Order ID")
    items: List[PackItemRequest] = Field(..., min_length=1)
    allow_rotation: bool = Field(default=False, description="Allow item rotation")


class PlacementResponse(BaseModel):
    """Placement coordinates for a single item."""
    item_id: str
    x: float
    y: float
    z: float
    placed_length: float
    placed_width: float
    placed_height: float
    fragility_tier: int
    fragility_label: str


class PackResponse(BaseModel):
    """Response from the packing endpoint."""
    order_id: str
    box_sku: str
    box_dimensions: str
    void_volume_pct: float
    placements: List[PlacementResponse]
    constraint_violations: int
    requires_split: bool
    separate_box_items: List[str]
    estimated_material_weight_g: float
    item_count: int


# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {
    "model": None,
    "pipeline": None,
    "catalogue": None,
}


def _load_resources() -> None:
    """Load model, pipeline, and box catalogue."""
    # Model
    model_path = MODEL_DIR / "best_model.joblib"
    if model_path.exists():
        _state["model"] = joblib.load(model_path)
        logger.info("Packing API: classifier loaded from %s", model_path)
    else:
        logger.warning("Packing API: no classifier found at %s", model_path)

    # Feature pipeline
    pipeline_path = MODEL_DIR / "feature_pipeline.joblib"
    if pipeline_path.exists():
        _state["pipeline"] = joblib.load(pipeline_path)
        logger.info("Packing API: feature pipeline loaded.")
    else:
        logger.warning("Packing API: feature pipeline not found.")

    # Box catalogue
    try:
        cat_path = Path("data/box_catalogue.json")
        if cat_path.exists():
            _state["catalogue"] = BoxCatalogue(cat_path)
        else:
            _state["catalogue"] = BoxCatalogue()  # default
        logger.info("Packing API: box catalogue loaded.")
    except Exception as e:
        logger.error("Failed to load box catalogue: %s", e)
        _state["catalogue"] = BoxCatalogue()


def _classify_item(item: PackItemRequest) -> int:
    """Classify a single item's fragility using the trained model."""
    model = _state.get("model")
    pipeline = _state.get("pipeline")

    if model is None or pipeline is None:
        # Fallback: heuristic based on material
        heuristic = {
            "glass": 3, "fragile_liquid": 3,
            "electronics": 2, "standard": 1, "apparel": 0,
        }
        return heuristic.get(item.material_type, 1)

    # Build DataFrame matching training schema
    input_df = pd.DataFrame([{
        "product_id": item.item_id or "inference",
        "length_cm": item.length_cm,
        "width_cm": item.width_cm,
        "height_cm": item.height_cm,
        "weight_g": item.weight_g,
        "material_type": item.material_type,
        "fragility_label": 0,
        "historical_material_volume_cm3": (
            item.length_cm * item.width_cm * item.height_cm
        ),
    }])

    transformed = pipeline.transform(input_df)
    X = transformed[FEATURE_COLUMNS]
    prediction = int(model.predict(X)[0])
    return prediction


# ---------------------------------------------------------------------------
# App Lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_resources()
    yield
    logger.info("Packing API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
packing_app = FastAPI(
    title="EcoPackAI Packing Service",
    description="3D bin packing optimization for e-commerce orders",
    version="1.0.0",
    lifespan=lifespan,
)

packing_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@packing_app.post("/v1/pack", response_model=PackResponse)
async def pack_order_endpoint(request: OrderPackRequest) -> PackResponse:
    """Pack an order of items into the optimal box.

    For each item, the fragility classifier runs to assign a tier,
    then the 3D FFD packing engine finds the smallest suitable box.
    """
    catalogue: Optional[BoxCatalogue] = _state.get("catalogue")
    if catalogue is None:
        raise HTTPException(503, "Box catalogue not loaded.")

    order_id = request.order_id or str(uuid.uuid4())

    # Classify each item and build Item objects
    packing_items: List[Item] = []
    for i, req_item in enumerate(request.items):
        fragility = _classify_item(req_item)
        item = Item(
            item_id=req_item.item_id or f"item-{i}",
            length=req_item.length_cm,
            width=req_item.width_cm,
            height=req_item.height_cm,
            weight_g=req_item.weight_g,
            fragility_label=fragility,
        )
        packing_items.append(item)

    # Run packing
    with tracer.start_as_current_span("bin_packing.ffd") as span:
        result = catalogue.select_optimal_box(
            packing_items,
            allow_rotation=request.allow_rotation,
        )
        span.set_attribute("packing.item_count", len(packing_items))
        span.set_attribute("packing.void_pct", result.void_volume_pct)
        span.set_attribute("packing.box_sku", result.box.sku)

    # Build placements response
    placements = [
        PlacementResponse(
            item_id=p.item.item_id,
            x=round(p.x, 2),
            y=round(p.y, 2),
            z=round(p.z, 2),
            placed_length=round(p.placed_length, 2),
            placed_width=round(p.placed_width, 2),
            placed_height=round(p.placed_height, 2),
            fragility_tier=p.item.fragility_label,
            fragility_label=TIER_LABELS.get(p.item.fragility_label, "Unknown"),
        )
        for p in result.placements
    ]

    # Estimate material weight (proportional to box surface area)
    box = result.box
    surface_area = 2 * (box.length * box.width + box.length * box.height + box.width * box.height)
    est_material_weight = round(surface_area * MATERIAL_WEIGHT_FACTOR, 2)

    return PackResponse(
        order_id=order_id,
        box_sku=result.box.sku,
        box_dimensions=f"{box.length}x{box.width}x{box.height}",
        void_volume_pct=result.void_volume_pct,
        placements=placements,
        constraint_violations=result.constraint_violations,
        requires_split=result.requires_split,
        separate_box_items=[item.item_id for item in result.requires_separate_box],
        estimated_material_weight_g=est_material_weight,
        item_count=len(result.placements),
    )


@packing_app.get("/v1/pack/health")
async def packing_health():
    """Health check for the packing service."""
    return {
        "status": "healthy" if _state.get("catalogue") else "degraded",
        "classifier_loaded": _state.get("model") is not None,
        "catalogue_loaded": _state.get("catalogue") is not None,
    }
