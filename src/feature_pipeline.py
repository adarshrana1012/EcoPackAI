"""
feature_pipeline.py — Scikit-learn feature-engineering pipeline for EcoPackAI.

This module sits immediately after schema validation in the data flow:

    validate_dataset()  ──►  build_feature_pipeline().fit_transform(df)  ──►  model

Pipeline Steps
--------------
1. **DerivedFeatureTransformer** — adds ``volume_cm3`` and ``aspect_ratio``.
2. **MaterialEncoder** — ordinal-encodes ``material_type`` by fragility risk.
3. **ContinuousScaler** — standardises all continuous features to zero-mean,
   unit-variance.

Usage
-----
>>> from src.feature_pipeline import build_feature_pipeline, save_pipeline
>>> pipe = build_feature_pipeline()
>>> transformed = pipe.fit_transform(validated_df)
>>> save_pipeline(pipe, "artifacts/feature_pipeline.joblib")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Material types ordered from *lowest* fragility-risk to *highest*.
MATERIAL_RISK_ORDER: List[str] = [
    "apparel",          # 0  — lowest risk
    "standard",         # 1
    "electronics",      # 2
    "glass",            # 3
    "fragile_liquid",   # 4  — highest risk
]

#: Continuous columns that will be scaled.
CONTINUOUS_FEATURES: List[str] = [
    "length_cm",
    "width_cm",
    "height_cm",
    "weight_g",
    "volume_cm3",
    "aspect_ratio",
    "historical_material_volume_cm3",
]


# ═══════════════════════════════════════════════════════════════════════════
# Custom Transformers
# ═══════════════════════════════════════════════════════════════════════════

class DerivedFeatureTransformer(BaseEstimator, TransformerMixin):
    """Compute physics-derived columns from raw dimensions.

    New columns
    -----------
    * ``volume_cm3``   = length_cm × width_cm × height_cm
    * ``aspect_ratio`` = max(l, w, h) / min(l, w, h)

    The transformer is *stateless* — ``fit`` is a no-op.
    """

    _DIMENSION_COLS: List[str] = ["length_cm", "width_cm", "height_cm"]

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "DerivedFeatureTransformer":
        """No-op fit (stateless transformer).

        Parameters
        ----------
        X : pd.DataFrame
            Must contain ``length_cm``, ``width_cm``, ``height_cm``.
        y : pd.Series, optional
            Ignored.

        Returns
        -------
        DerivedFeatureTransformer
            self
        """
        # Validate that the required columns are present.
        missing = set(self._DIMENSION_COLS) - set(X.columns)
        if missing:
            raise ValueError(
                f"DerivedFeatureTransformer requires columns "
                f"{self._DIMENSION_COLS}, but {missing} are missing."
            )
        self.is_fitted_ = True
        logger.debug("DerivedFeatureTransformer.fit() — no state to learn.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add ``volume_cm3`` and ``aspect_ratio`` columns.

        Parameters
        ----------
        X : pd.DataFrame
            Input data with dimension columns.

        Returns
        -------
        pd.DataFrame
            Copy of *X* with two additional columns appended.
        """
        logger.info("Computing derived features (volume, aspect_ratio).")
        df = X.copy()

        dims = df[self._DIMENSION_COLS]
        df["volume_cm3"] = dims.prod(axis=1)

        row_max = dims.max(axis=1)
        row_min = dims.min(axis=1)
        # Guard against division by zero (schema should prevent it, but
        # belt-and-suspenders).
        df["aspect_ratio"] = np.where(
            row_min > 0,
            row_max / row_min,
            np.nan,
        )

        logger.debug(
            "Derived features added — volume_cm3 range [%.2f, %.2f], "
            "aspect_ratio range [%.2f, %.2f].",
            df["volume_cm3"].min(),
            df["volume_cm3"].max(),
            df["aspect_ratio"].min(),
            df["aspect_ratio"].max(),
        )
        return df


class MaterialEncoder(BaseEstimator, TransformerMixin):
    """Ordinal-encode ``material_type`` by fragility-risk ordering.

    Encoding
    --------
    ===  ===============
    0    apparel
    1    standard
    2    electronics
    3    glass
    4    fragile_liquid
    ===  ===============

    The underlying ``OrdinalEncoder`` is fitted on the canonical list so
    that encoding is deterministic regardless of which categories appear
    in the training data.
    """

    def __init__(self) -> None:
        self._encoder: OrdinalEncoder = OrdinalEncoder(
            categories=[MATERIAL_RISK_ORDER],
            handle_unknown="error",
            dtype=np.float64,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "MaterialEncoder":
        """Fit the ordinal encoder on the canonical category list.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain ``material_type``.
        y : pd.Series, optional
            Ignored.

        Returns
        -------
        MaterialEncoder
            self
        """
        if "material_type" not in X.columns:
            raise ValueError(
                "MaterialEncoder requires a 'material_type' column."
            )
        self._encoder.fit(X[["material_type"]])
        self.is_fitted_ = True
        logger.debug("MaterialEncoder fitted — categories: %s", MATERIAL_RISK_ORDER)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace ``material_type`` strings with ordinal integers.

        Parameters
        ----------
        X : pd.DataFrame
            Input data containing ``material_type``.

        Returns
        -------
        pd.DataFrame
            Copy of *X* with ``material_type`` replaced by encoded values.
        """
        logger.info("Encoding material_type with ordinal risk order.")
        df = X.copy()
        encoded = self._encoder.transform(df[["material_type"]])
        df["material_type"] = encoded.ravel()
        return df


class ContinuousScaler(BaseEstimator, TransformerMixin):
    """Standardise continuous features to zero-mean, unit-variance.

    The scaler is applied **in-place** (on a copy) to exactly the columns
    listed in ``CONTINUOUS_FEATURES``.
    """

    def __init__(self) -> None:
        self._scaler: StandardScaler = StandardScaler()
        self._columns: List[str] = list(CONTINUOUS_FEATURES)

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "ContinuousScaler":
        """Fit the ``StandardScaler`` on continuous columns.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain all columns in ``CONTINUOUS_FEATURES``.
        y : pd.Series, optional
            Ignored.

        Returns
        -------
        ContinuousScaler
            self
        """
        missing = set(self._columns) - set(X.columns)
        if missing:
            raise ValueError(
                f"ContinuousScaler requires columns {self._columns}, "
                f"but {missing} are missing."
            )
        self._scaler.fit(X[self._columns])
        self.is_fitted_ = True
        logger.debug(
            "ContinuousScaler fitted — means=%s", self._scaler.mean_,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply standard scaling to continuous columns.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        pd.DataFrame
            Copy of *X* with continuous columns replaced by scaled values.
        """
        logger.info("Scaling %d continuous features.", len(self._columns))
        df = X.copy()
        df[self._columns] = self._scaler.transform(df[self._columns])
        return df


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Factory & Serialisation
# ═══════════════════════════════════════════════════════════════════════════

def build_feature_pipeline() -> Pipeline:
    """Construct the three-step feature-engineering pipeline.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline with steps:

        1. ``derived_features``  — :class:`DerivedFeatureTransformer`
        2. ``material_encoder``  — :class:`MaterialEncoder`
        3. ``continuous_scaler`` — :class:`ContinuousScaler`
    """
    pipeline = Pipeline(
        steps=[
            ("derived_features", DerivedFeatureTransformer()),
            ("material_encoder", MaterialEncoder()),
            ("continuous_scaler", ContinuousScaler()),
        ],
    )
    logger.info("Feature pipeline built — %d steps.", len(pipeline.steps))
    return pipeline


def save_pipeline(
    pipeline: Pipeline,
    path: Union[str, Path],
) -> Path:
    """Persist a fitted pipeline to disk via joblib.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted scikit-learn ``Pipeline``.
    path : str or Path
        Destination file path (e.g. ``"artifacts/pipeline.joblib"``).

    Returns
    -------
    Path
        The resolved, absolute path that was written.
    """
    dest = Path(path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, dest)
    logger.info("Pipeline saved to %s", dest)
    return dest


def load_pipeline(path: Union[str, Path]) -> Pipeline:
    """Load a previously saved pipeline from disk.

    Parameters
    ----------
    path : str or Path
        File path to a joblib-serialised ``Pipeline``.

    Returns
    -------
    Pipeline
        The deserialised, fitted pipeline.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Pipeline file not found: {source}")
    pipeline: Pipeline = joblib.load(source)
    logger.info("Pipeline loaded from %s", source)
    return pipeline
