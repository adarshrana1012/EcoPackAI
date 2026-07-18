"""
test_schema_validator.py — Tests for the EcoPackAI shipping-data schema validator.

Each test creates a small DataFrame (via pytest fixture), optionally mutates
one column, and asserts that ``validate_dataset`` either succeeds or raises
``DataValidationError`` with the expected field detail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.schema_validator import DataValidationError, validate_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def valid_shipping_df() -> pd.DataFrame:
    """Return a minimal, schema-compliant shipping DataFrame (5 rows)."""
    return pd.DataFrame(
        {
            "product_id": ["P001", "P002", "P003", "P004", "P005"],
            "length_cm": [10.0, 20.0, 15.0, 30.0, 25.0],
            "width_cm": [5.0, 10.0, 8.0, 12.0, 6.0],
            "height_cm": [3.0, 7.0, 4.0, 9.0, 5.0],
            "weight_g": [150.0, 500.0, 250.0, 1200.0, 300.0],
            "material_type": [
                "glass",
                "electronics",
                "apparel",
                "fragile_liquid",
                "standard",
            ],
            "fragility_label": [3, 2, 0, 3, 1],
            "historical_material_volume_cm3": [
                200.0, 1500.0, 600.0, 3500.0, 900.0,
            ],
        }
    )


# ---------------------------------------------------------------------------
# Happy-path
# ---------------------------------------------------------------------------

class TestValidDataset:
    """Tests that valid data passes without error."""

    def test_valid_dataset_passes(
        self, valid_shipping_df: pd.DataFrame,
    ) -> None:
        """A fully compliant DataFrame should return successfully."""
        validated_df, report = validate_dataset(valid_shipping_df)

        assert isinstance(validated_df, pd.DataFrame)
        assert report["checks_passed"] is True
        assert report["row_count"] == len(valid_shipping_df)
        assert report["column_count"] >= len(valid_shipping_df.columns)


# ---------------------------------------------------------------------------
# Negative / boundary dimension checks
# ---------------------------------------------------------------------------

class TestDimensionConstraints:
    """Negative and zero dimensions must be rejected."""

    def test_negative_dimensions_fail(
        self, valid_shipping_df: pd.DataFrame,
    ) -> None:
        """A negative ``length_cm`` should trigger DataValidationError."""
        df = valid_shipping_df.copy()
        df.loc[0, "length_cm"] = -1.0

        with pytest.raises(DataValidationError) as exc_info:
            validate_dataset(df)

        err = exc_info.value
        assert len(err.field_errors) > 0
        failing_fields = {e["field"] for e in err.field_errors}
        assert "length_cm" in failing_fields

    def test_zero_weight_fails(
        self, valid_shipping_df: pd.DataFrame,
    ) -> None:
        """``weight_g = 0`` violates the gt(0) check."""
        df = valid_shipping_df.copy()
        df.loc[2, "weight_g"] = 0.0

        with pytest.raises(DataValidationError) as exc_info:
            validate_dataset(df)

        err = exc_info.value
        failing_fields = {e["field"] for e in err.field_errors}
        assert "weight_g" in failing_fields


# ---------------------------------------------------------------------------
# Categorical / enum checks
# ---------------------------------------------------------------------------

class TestCategoricalConstraints:
    """Invalid enum values must be rejected."""

    def test_invalid_fragility_label(
        self, valid_shipping_df: pd.DataFrame,
    ) -> None:
        """``fragility_label = 5`` is outside the allowed set {0,1,2,3}."""
        df = valid_shipping_df.copy()
        df.loc[1, "fragility_label"] = 5

        with pytest.raises(DataValidationError) as exc_info:
            validate_dataset(df)

        err = exc_info.value
        failing_fields = {e["field"] for e in err.field_errors}
        assert "fragility_label" in failing_fields

    def test_invalid_material_type(
        self, valid_shipping_df: pd.DataFrame,
    ) -> None:
        """``material_type = 'plastic'`` is not in the allowed list."""
        df = valid_shipping_df.copy()
        df.loc[3, "material_type"] = "plastic"

        with pytest.raises(DataValidationError) as exc_info:
            validate_dataset(df)

        err = exc_info.value
        failing_fields = {e["field"] for e in err.field_errors}
        assert "material_type" in failing_fields


# ---------------------------------------------------------------------------
# Null / NaN checks
# ---------------------------------------------------------------------------

class TestNullability:
    """Schema-required columns must not contain NaN."""

    def test_null_values_fail(
        self, valid_shipping_df: pd.DataFrame,
    ) -> None:
        """Injecting NaN into ``weight_g`` and ``length_cm`` must fail."""
        df = valid_shipping_df.copy()
        df.loc[0, "weight_g"] = np.nan
        df.loc[1, "length_cm"] = np.nan

        with pytest.raises(DataValidationError) as exc_info:
            validate_dataset(df)

        err = exc_info.value
        # At least one of the null fields should appear in errors
        failing_fields = {e["field"] for e in err.field_errors}
        assert failing_fields & {"weight_g", "length_cm"}
