"""
test_feature_pipeline.py — Tests for the EcoPackAI feature-engineering pipeline.

Tests cover every transformer's correctness, the full pipeline composition,
and serialisation round-trip behaviour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_pipeline import (
    CONTINUOUS_FEATURES,
    MATERIAL_RISK_ORDER,
    build_feature_pipeline,
    load_pipeline,
    save_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Return a small, valid shipping DataFrame for pipeline tests."""
    return pd.DataFrame(
        {
            "product_id": ["A1", "A2", "A3", "A4", "A5"],
            "length_cm": [10.0, 20.0, 30.0, 40.0, 50.0],
            "width_cm": [5.0, 10.0, 15.0, 20.0, 25.0],
            "height_cm": [2.0, 4.0, 6.0, 8.0, 10.0],
            "weight_g": [100.0, 200.0, 300.0, 400.0, 500.0],
            "material_type": [
                "apparel",
                "standard",
                "electronics",
                "glass",
                "fragile_liquid",
            ],
            "fragility_label": [0, 1, 2, 3, 3],
            "historical_material_volume_cm3": [
                100.0, 800.0, 2700.0, 6400.0, 12500.0,
            ],
        }
    )


@pytest.fixture()
def fitted_pipeline(sample_df: pd.DataFrame):
    """Return a fitted pipeline together with the sample data."""
    pipe = build_feature_pipeline()
    pipe.fit(sample_df)
    return pipe


# ---------------------------------------------------------------------------
# DerivedFeatureTransformer tests
# ---------------------------------------------------------------------------

class TestDerivedFeatures:
    """Tests for volume and aspect-ratio computation."""

    def test_pipeline_creates_volume(
        self, sample_df: pd.DataFrame, fitted_pipeline,
    ) -> None:
        """``volume_cm3`` should equal length × width × height."""
        transformed = fitted_pipeline.transform(sample_df)

        expected_volumes = (
            sample_df["length_cm"]
            * sample_df["width_cm"]
            * sample_df["height_cm"]
        )
        # After scaling the absolute values change, so check at the
        # derived-feature step directly.
        pipe_derived_only = build_feature_pipeline()
        pipe_derived_only.steps = pipe_derived_only.steps[:1]  # keep step 0
        derived_df = pipe_derived_only.fit_transform(sample_df)

        np.testing.assert_array_almost_equal(
            derived_df["volume_cm3"].values,
            expected_volumes.values,
        )

    def test_pipeline_creates_aspect_ratio(
        self, sample_df: pd.DataFrame,
    ) -> None:
        """``aspect_ratio`` should equal max(l,w,h) / min(l,w,h)."""
        pipe = build_feature_pipeline()
        pipe.steps = pipe.steps[:1]
        derived_df = pipe.fit_transform(sample_df)

        dims = sample_df[["length_cm", "width_cm", "height_cm"]]
        expected_ar = dims.max(axis=1) / dims.min(axis=1)

        np.testing.assert_array_almost_equal(
            derived_df["aspect_ratio"].values,
            expected_ar.values,
        )


# ---------------------------------------------------------------------------
# MaterialEncoder tests
# ---------------------------------------------------------------------------

class TestMaterialEncoding:
    """Tests for ordinal risk-based encoding of material_type."""

    def test_material_encoding_order(
        self, sample_df: pd.DataFrame,
    ) -> None:
        """Encoding must follow: apparel=0 … fragile_liquid=4."""
        pipe = build_feature_pipeline()
        # Run first two steps only (derived + encoder)
        pipe.steps = pipe.steps[:2]
        encoded_df = pipe.fit_transform(sample_df)

        for idx, mat in enumerate(MATERIAL_RISK_ORDER):
            row_mask = sample_df["material_type"] == mat
            encoded_vals = encoded_df.loc[row_mask, "material_type"].values
            assert (encoded_vals == float(idx)).all(), (
                f"Expected {mat} → {idx}, got {encoded_vals}"
            )


# ---------------------------------------------------------------------------
# ContinuousScaler tests
# ---------------------------------------------------------------------------

class TestContinuousScaler:
    """Tests for standard-scaling of continuous columns."""

    def test_scaler_zero_mean(
        self, sample_df: pd.DataFrame, fitted_pipeline,
    ) -> None:
        """After transform, every continuous feature should have ≈0 mean."""
        transformed = fitted_pipeline.transform(sample_df)

        for col in CONTINUOUS_FEATURES:
            mean_val = transformed[col].mean()
            np.testing.assert_almost_equal(
                mean_val,
                0.0,
                decimal=6,
                err_msg=f"Column {col!r} has non-zero mean {mean_val}",
            )


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------

class TestSerialisation:
    """Pipeline should survive a save→load cycle without data loss."""

    def test_pipeline_serialization(
        self,
        sample_df: pd.DataFrame,
        fitted_pipeline,
        tmp_path: Path,
    ) -> None:
        """Save, reload, and compare transform outputs."""
        original_output = fitted_pipeline.transform(sample_df)

        path = tmp_path / "pipeline.joblib"
        save_pipeline(fitted_pipeline, path)
        reloaded = load_pipeline(path)
        reloaded_output = reloaded.transform(sample_df)

        pd.testing.assert_frame_equal(original_output, reloaded_output)


# ---------------------------------------------------------------------------
# Robustness / new-data test
# ---------------------------------------------------------------------------

class TestNewData:
    """A fitted pipeline must not crash on new, unseen rows."""

    def test_pipeline_inverse_not_needed(
        self, fitted_pipeline, sample_df: pd.DataFrame,
    ) -> None:
        """Transform should work on fresh data with known categories."""
        new_data = pd.DataFrame(
            {
                "product_id": ["Z99"],
                "length_cm": [12.0],
                "width_cm": [6.0],
                "height_cm": [3.0],
                "weight_g": [220.0],
                "material_type": ["electronics"],
                "fragility_label": [2],
                "historical_material_volume_cm3": [500.0],
            }
        )
        result = fitted_pipeline.transform(new_data)
        assert isinstance(result, pd.DataFrame)
        assert "volume_cm3" in result.columns
        assert "aspect_ratio" in result.columns
        assert len(result) == 1
