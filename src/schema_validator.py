"""
schema_validator.py — Pandera schema validation for EcoPackAI shipping data.

This module enforces column-level constraints on every DataFrame that enters the
feature-engineering pipeline.  It is the first gate in the data-quality chain:

    raw CSV  ──►  validate_dataset()  ──►  feature_pipeline  ──►  model

Responsibilities
----------------
* Column-type enforcement (string / float / int).
* Range checks (strictly positive dimensions, valid enum sets).
* Nullable-column prohibition.
* Human-readable error reporting via ``DataValidationError``.

Usage
-----
>>> from src.schema_validator import validate_dataset
>>> validated_df, report = validate_dataset(raw_df)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema
from pandera.errors import SchemaErrors

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------

class DataValidationError(Exception):
    """Raised when the shipping DataFrame fails schema validation.

    Attributes
    ----------
    error_summary : str
        A human-readable summary of every failing check.
    field_errors : list[dict[str, Any]]
        One entry per failing check, each containing *field*, *check*,
        and *message* keys.
    """

    def __init__(
        self,
        error_summary: str,
        field_errors: List[Dict[str, Any]],
    ) -> None:
        self.error_summary = error_summary
        self.field_errors = field_errors
        super().__init__(error_summary)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DataValidationError(n_field_errors={len(self.field_errors)}, "
            f"summary={self.error_summary!r})"
        )


# ---------------------------------------------------------------------------
# Allowed Values
# ---------------------------------------------------------------------------

VALID_MATERIAL_TYPES: List[str] = [
    "glass",
    "electronics",
    "apparel",
    "fragile_liquid",
    "standard",
]

VALID_FRAGILITY_LABELS: List[int] = [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Schema Definition
# ---------------------------------------------------------------------------

SHIPPING_SCHEMA: DataFrameSchema = DataFrameSchema(
    columns={
        "product_id": Column(
            dtype="object",
            nullable=False,
            checks=[
                Check(lambda s: s.str.len() > 0, error="product_id must be non-empty"),
            ],
            description="Unique product identifier.",
        ),
        "length_cm": Column(
            dtype="float64",
            nullable=False,
            checks=[
                Check.gt(0, error="length_cm must be > 0"),
            ],
            description="Product length in centimetres.",
        ),
        "width_cm": Column(
            dtype="float64",
            nullable=False,
            checks=[
                Check.gt(0, error="width_cm must be > 0"),
            ],
            description="Product width in centimetres.",
        ),
        "height_cm": Column(
            dtype="float64",
            nullable=False,
            checks=[
                Check.gt(0, error="height_cm must be > 0"),
            ],
            description="Product height in centimetres.",
        ),
        "weight_g": Column(
            dtype="float64",
            nullable=False,
            checks=[
                Check.gt(0, error="weight_g must be > 0"),
            ],
            description="Product weight in grams.",
        ),
        "material_type": Column(
            dtype="object",
            nullable=False,
            checks=[
                Check.isin(
                    VALID_MATERIAL_TYPES,
                    error=(
                        f"material_type must be one of {VALID_MATERIAL_TYPES}"
                    ),
                ),
            ],
            description="Category of packaging material.",
        ),
        "fragility_label": Column(
            dtype="int64",
            nullable=False,
            checks=[
                Check.isin(
                    VALID_FRAGILITY_LABELS,
                    error=(
                        f"fragility_label must be one of {VALID_FRAGILITY_LABELS}"
                    ),
                ),
            ],
            description="Fragility class (0 = robust … 3 = very fragile).",
        ),
        "historical_material_volume_cm3": Column(
            dtype="float64",
            nullable=False,
            checks=[
                Check.gt(0, error="historical_material_volume_cm3 must be > 0"),
            ],
            description="Historical average packaging-material volume.",
        ),
    },
    strict=False,   # allow extra columns to pass through
    coerce=True,    # attempt safe type coercion before checking
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate *df* against the shipping-data schema.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or lightly cleaned shipping data.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        A two-element tuple:

        * **validated_df** — the DataFrame after pandera coercion.
        * **validation_report** — a dict containing:
          - ``row_count`` (int)
          - ``column_count`` (int)
          - ``checks_passed`` (bool, always ``True`` on success)

    Raises
    ------
    DataValidationError
        If any column violates its schema constraint.  The exception
        carries a ``field_errors`` list with per-check detail.
    """
    logger.info(
        "Starting schema validation — %d rows × %d columns.",
        len(df),
        len(df.columns),
    )

    try:
        validated_df: pd.DataFrame = SHIPPING_SCHEMA.validate(
            df, lazy=True,
        )
    except SchemaErrors as exc:
        field_errors = _extract_field_errors(exc)
        summary = _build_error_summary(field_errors)
        logger.error("Schema validation FAILED.\n%s", summary)
        raise DataValidationError(
            error_summary=summary,
            field_errors=field_errors,
        ) from exc

    report: Dict[str, Any] = {
        "row_count": len(validated_df),
        "column_count": len(validated_df.columns),
        "checks_passed": True,
    }
    logger.info(
        "Schema validation PASSED — %d rows, %d columns.",
        report["row_count"],
        report["column_count"],
    )
    return validated_df, report


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _extract_field_errors(
    schema_errors: SchemaErrors,
) -> List[Dict[str, Any]]:
    """Turn a ``SchemaErrors`` exception into a flat list of dicts.

    Each dict has three keys:

    * ``field``   – column name (or ``"<dataframe>"``).
    * ``check``   – the check expression that failed.
    * ``message`` – pandera's human-readable error string.
    """
    field_errors: List[Dict[str, Any]] = []
    failure_cases = schema_errors.failure_cases

    if failure_cases is not None and not failure_cases.empty:
        for _, row in failure_cases.iterrows():
            field_errors.append(
                {
                    "field": str(row.get("column", "<dataframe>")),
                    "check": str(row.get("check", "unknown")),
                    "message": str(row.get("failure_case", "")),
                }
            )
    else:
        # Fallback: iterate over the message attribute
        for err in schema_errors.schema_errors:
            reason = err.get("error", str(err))
            field_errors.append(
                {
                    "field": str(err.get("column", "<dataframe>")),
                    "check": str(err.get("check", "unknown")),
                    "message": str(reason),
                }
            )

    return field_errors


def _build_error_summary(field_errors: List[Dict[str, Any]]) -> str:
    """Create a multi-line, human-readable error summary."""
    lines = [f"Schema validation failed with {len(field_errors)} error(s):"]
    for idx, err in enumerate(field_errors, start=1):
        lines.append(
            f"  [{idx}] field={err['field']!r}  check={err['check']!r}  "
            f"value={err['message']!r}"
        )
    return "\n".join(lines)
