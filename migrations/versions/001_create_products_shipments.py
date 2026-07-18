"""Create products, shipments, and packing_policies tables.

Revision ID: 001
Revises: None (initial migration)
Create Date: 2026-01-01 00:00:00.000000

This migration establishes the core EcoPackAI schema:

* **products** – Physical attributes of packable items.
* **shipments** – Records of packed shipments with sustainability metrics.
* **packing_policies** – Versioned RL model metadata for packing strategies.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# ---------------------------------------------------------------------------
# Alembic revision identifiers
# ---------------------------------------------------------------------------
revision: str = "001"
down_revision: str | None = None
branch_labels: tuple[str, ...] | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Create the products, shipments, and packing_policies tables."""

    # ------------------------------------------------------------------
    # products
    # ------------------------------------------------------------------
    op.create_table(
        "products",
        sa.Column(
            "product_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            comment="Unique product identifier (UUIDv4).",
        ),
        sa.Column(
            "length_cm",
            sa.Float,
            nullable=False,
            comment="Product length in centimetres.",
        ),
        sa.Column(
            "width_cm",
            sa.Float,
            nullable=False,
            comment="Product width in centimetres.",
        ),
        sa.Column(
            "height_cm",
            sa.Float,
            nullable=False,
            comment="Product height in centimetres.",
        ),
        sa.Column(
            "weight_g",
            sa.Float,
            nullable=False,
            comment="Product weight in grams.",
        ),
        sa.Column(
            "material_type",
            sa.VARCHAR(64),
            nullable=False,
            comment="Primary packaging material category.",
        ),
        sa.Column(
            "fragility_label",
            sa.Integer,
            nullable=False,
            comment="Fragility rating (0 = robust … 4 = extremely fragile).",
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="Row creation timestamp (DB clock).",
        ),
        # -- Check constraints --
        sa.CheckConstraint("length_cm > 0", name="ck_products_length_positive"),
        sa.CheckConstraint("width_cm > 0", name="ck_products_width_positive"),
        sa.CheckConstraint("height_cm > 0", name="ck_products_height_positive"),
        sa.CheckConstraint("weight_g > 0", name="ck_products_weight_positive"),
        comment="Physical product attributes used by the packing optimiser.",
    )

    # Index on fragility_label for fast look-ups during packing decisions
    op.create_index(
        "ix_products_fragility_label",
        "products",
        ["fragility_label"],
    )

    # ------------------------------------------------------------------
    # shipments
    # ------------------------------------------------------------------
    op.create_table(
        "shipments",
        sa.Column(
            "shipment_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            comment="Unique shipment identifier (UUIDv4).",
        ),
        sa.Column(
            "order_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
            comment="Reference to the originating order.",
        ),
        sa.Column(
            "box_sku",
            sa.VARCHAR(64),
            nullable=True,
            comment="SKU of the box type selected for this shipment.",
        ),
        sa.Column(
            "packing_policy_version",
            sa.Integer,
            nullable=True,
            comment="Version of the RL packing policy used.",
        ),
        sa.Column(
            "void_volume_pct",
            sa.Float,
            nullable=True,
            comment="Percentage of box volume that is void fill.",
        ),
        sa.Column(
            "material_weight_g",
            sa.Float,
            nullable=True,
            comment="Total packaging material weight in grams.",
        ),
        sa.Column(
            "co2e_kg",
            sa.Float,
            nullable=True,
            comment="Estimated CO₂-equivalent emissions in kilograms.",
        ),
        sa.Column(
            "damage_reported",
            sa.Boolean,
            server_default=sa.text("false"),
            nullable=False,
            comment="Whether damage was reported for this shipment.",
        ),
        sa.Column(
            "packed_at",
            sa.TIMESTAMP(timezone=True),
            nullable=True,
            comment="Timestamp when the shipment was packed.",
        ),
        comment="Packed-shipment records with sustainability metrics.",
    )

    # Index on packed_at for time-range analytics queries
    op.create_index(
        "ix_shipments_packed_at",
        "shipments",
        ["packed_at"],
    )

    # ------------------------------------------------------------------
    # packing_policies
    # ------------------------------------------------------------------
    op.create_table(
        "packing_policies",
        sa.Column(
            "policy_id",
            sa.Integer,
            primary_key=True,
            autoincrement=True,
            nullable=False,
            comment="Auto-incrementing policy version identifier.",
        ),
        sa.Column(
            "model_path",
            sa.Text,
            nullable=True,
            comment="Filesystem or object-store path to the serialised model.",
        ),
        sa.Column(
            "training_date",
            sa.Date,
            nullable=True,
            comment="Date the model finished training.",
        ),
        sa.Column(
            "avg_reward",
            sa.Float,
            nullable=True,
            comment="Average reward achieved during evaluation.",
        ),
        sa.Column(
            "is_active",
            sa.Boolean,
            server_default=sa.text("false"),
            nullable=False,
            comment="Whether this policy is currently serving production traffic.",
        ),
        comment="Versioned RL packing-policy metadata.",
    )


def downgrade() -> None:
    """Drop the packing_policies, shipments, and products tables.

    Tables are dropped in reverse dependency order.  Indexes and constraints
    are removed automatically when their parent table is dropped.
    """
    op.drop_table("packing_policies")
    op.drop_table("shipments")
    op.drop_table("products")
