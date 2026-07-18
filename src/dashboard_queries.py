"""
dashboard_queries.py — Aggregate Dashboard Queries (Prompt 30)
==============================================================

SQLAlchemy queries to power the EcoPackAI analytics dashboard, with
Pydantic response models.

Author: EcoPackAI Team
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing SQLAlchemy; fall back gracefully
# ---------------------------------------------------------------------------
try:
    from sqlalchemy import (
        Column, Float, Integer, String, DateTime, Boolean,
        create_engine, func, text, case, desc,
    )
    from sqlalchemy.orm import Session, DeclarativeBase, sessionmaker

    class Base(DeclarativeBase):
        pass

    class ShipmentRecord(Base):
        """ORM model for the shipments analytics table."""
        __tablename__ = "shipments"

        id = Column(Integer, primary_key=True, autoincrement=True)
        shipment_id = Column(String(64), unique=True, nullable=False)
        product_id = Column(String(64), nullable=False)
        packed_at = Column(DateTime, nullable=False)
        box_sku = Column(String(32), nullable=False)
        box_length = Column(Float, nullable=False)
        box_width = Column(Float, nullable=False)
        box_height = Column(Float, nullable=False)
        void_volume_pct = Column(Float, nullable=False)
        material_weight_g = Column(Float, nullable=False)
        co2e_kg = Column(Float, nullable=False)
        fragility_tier = Column(Integer, nullable=False, default=0)
        constraint_violations = Column(Integer, nullable=False, default=0)
        transport_distance_km = Column(Float, default=0.0)
        damaged = Column(Boolean, default=False)

    _HAS_SQLALCHEMY = True

except ImportError:
    _HAS_SQLALCHEMY = False
    logger.warning("SQLAlchemy not installed. Dashboard queries unavailable.")


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic Response Models
# ═══════════════════════════════════════════════════════════════════════════

class WeeklyMaterialUsage(BaseModel):
    """Weekly material usage aggregate."""
    week_start: str
    week_number: int
    total_material_kg: float
    shipment_count: int


class WeeklyMaterialResponse(BaseModel):
    """Response for weekly material usage over last 12 weeks."""
    weeks: List[WeeklyMaterialUsage]
    total_material_kg: float
    avg_weekly_kg: float


class FragilityDistribution(BaseModel):
    """Fragility tier distribution."""
    tier: int
    tier_label: str
    count: int
    percentage: float


class FragilityDistributionResponse(BaseModel):
    """Response for fragility tier distribution this month."""
    month: str
    total_shipments: int
    distribution: List[FragilityDistribution]


class ProductVoidEntry(BaseModel):
    """Single product void volume entry."""
    product_id: str
    avg_void_pct: float
    shipment_count: int


class TopVoidProductsResponse(BaseModel):
    """Response for top 10 products by void volume."""
    products: List[ProductVoidEntry]


class CO2eSavingsResponse(BaseModel):
    """Response for total CO2e savings vs baseline."""
    total_actual_co2e_kg: float
    total_baseline_co2e_kg: float
    co2e_saved_kg: float
    reduction_pct: float
    baseline_void_pct_assumption: float


# ═══════════════════════════════════════════════════════════════════════════
# Tier Labels
# ═══════════════════════════════════════════════════════════════════════════
TIER_LABELS = {0: "None", 1: "Low", 2: "Medium", 3: "Critical"}


# ═══════════════════════════════════════════════════════════════════════════
# Query Functions
# ═══════════════════════════════════════════════════════════════════════════

def query_weekly_material_usage(
    session: Any,
    weeks: int = 12,
) -> WeeklyMaterialResponse:
    """Query weekly material usage (kg) over the last N weeks.

    Parameters
    ----------
    session : SQLAlchemy Session
        Database session.
    weeks : int
        Number of weeks to look back.

    Returns
    -------
    WeeklyMaterialResponse
    """
    cutoff = datetime.now(timezone.utc) - timedelta(weeks=weeks)

    results = (
        session.query(
            func.date_trunc("week", ShipmentRecord.packed_at).label("week_start"),
            func.sum(ShipmentRecord.material_weight_g / 1000.0).label("total_kg"),
            func.count(ShipmentRecord.id).label("shipment_count"),
        )
        .filter(ShipmentRecord.packed_at >= cutoff)
        .group_by(func.date_trunc("week", ShipmentRecord.packed_at))
        .order_by(func.date_trunc("week", ShipmentRecord.packed_at))
        .all()
    )

    week_data = []
    for i, row in enumerate(results):
        week_data.append(WeeklyMaterialUsage(
            week_start=str(row.week_start),
            week_number=i + 1,
            total_material_kg=round(float(row.total_kg or 0), 2),
            shipment_count=int(row.shipment_count or 0),
        ))

    total = sum(w.total_material_kg for w in week_data)
    avg = total / len(week_data) if week_data else 0

    return WeeklyMaterialResponse(
        weeks=week_data,
        total_material_kg=round(total, 2),
        avg_weekly_kg=round(avg, 2),
    )


def query_fragility_distribution(
    session: Any,
) -> FragilityDistributionResponse:
    """Query fragility tier distribution of shipments this month.

    Parameters
    ----------
    session : SQLAlchemy Session
        Database session.

    Returns
    -------
    FragilityDistributionResponse
    """
    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    results = (
        session.query(
            ShipmentRecord.fragility_tier,
            func.count(ShipmentRecord.id).label("count"),
        )
        .filter(ShipmentRecord.packed_at >= month_start)
        .group_by(ShipmentRecord.fragility_tier)
        .order_by(ShipmentRecord.fragility_tier)
        .all()
    )

    total = sum(r.count for r in results)
    distribution = [
        FragilityDistribution(
            tier=int(r.fragility_tier),
            tier_label=TIER_LABELS.get(int(r.fragility_tier), "Unknown"),
            count=int(r.count),
            percentage=round(r.count / total * 100, 1) if total > 0 else 0,
        )
        for r in results
    ]

    return FragilityDistributionResponse(
        month=month_start.strftime("%Y-%m"),
        total_shipments=total,
        distribution=distribution,
    )


def query_top_void_products(
    session: Any,
    limit: int = 10,
) -> TopVoidProductsResponse:
    """Query top N products by average void_volume_pct.

    Parameters
    ----------
    session : SQLAlchemy Session
        Database session.
    limit : int
        Number of top products to return.

    Returns
    -------
    TopVoidProductsResponse
    """
    results = (
        session.query(
            ShipmentRecord.product_id,
            func.avg(ShipmentRecord.void_volume_pct).label("avg_void"),
            func.count(ShipmentRecord.id).label("count"),
        )
        .group_by(ShipmentRecord.product_id)
        .order_by(desc(func.avg(ShipmentRecord.void_volume_pct)))
        .limit(limit)
        .all()
    )

    products = [
        ProductVoidEntry(
            product_id=str(r.product_id),
            avg_void_pct=round(float(r.avg_void), 2),
            shipment_count=int(r.count),
        )
        for r in results
    ]

    return TopVoidProductsResponse(products=products)


def query_co2e_savings(
    session: Any,
    baseline_void_pct_increase: float = 30.0,
) -> CO2eSavingsResponse:
    """Compute total CO₂e saved vs a manual-packing baseline.

    Baseline assumes void_volume_pct was 30 percentage points higher
    before EcoPackAI deployment.

    Parameters
    ----------
    session : SQLAlchemy Session
        Database session.
    baseline_void_pct_increase : float
        Additional void % assumed under manual packing.

    Returns
    -------
    CO2eSavingsResponse
    """
    from src.metrics_calculator import (
        estimate_co2e_kg,
        estimate_material_weight_g,
    )

    results = (
        session.query(
            func.sum(ShipmentRecord.co2e_kg).label("total_co2e"),
            func.sum(ShipmentRecord.material_weight_g).label("total_material_g"),
            func.avg(ShipmentRecord.void_volume_pct).label("avg_void"),
            func.avg(ShipmentRecord.transport_distance_km).label("avg_distance"),
            func.sum(ShipmentRecord.material_weight_g / 1000.0).label("total_material_kg"),
        )
        .first()
    )

    actual_co2e = float(results.total_co2e or 0)
    actual_material_g = float(results.total_material_g or 0)
    avg_void = float(results.avg_void or 0)
    avg_distance = float(results.avg_distance or 500)

    # Baseline: higher void means more material was used
    void_ratio = (100.0 - avg_void) / max(1.0, 100.0 - avg_void - baseline_void_pct_increase)
    baseline_material_g = actual_material_g * void_ratio
    baseline_co2e = estimate_co2e_kg(baseline_material_g, avg_distance)

    saved = baseline_co2e - actual_co2e
    reduction_pct = (saved / baseline_co2e * 100) if baseline_co2e > 0 else 0

    return CO2eSavingsResponse(
        total_actual_co2e_kg=round(actual_co2e, 2),
        total_baseline_co2e_kg=round(baseline_co2e, 2),
        co2e_saved_kg=round(saved, 2),
        reduction_pct=round(reduction_pct, 1),
        baseline_void_pct_assumption=baseline_void_pct_increase,
    )
