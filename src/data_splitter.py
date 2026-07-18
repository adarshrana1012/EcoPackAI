"""
data_splitter.py — Stratified Train / Validation / Test Splitter
================================================================

Provides a single public function, :func:`split_dataset`, that partitions a
pandas DataFrame into three stratified subsets while verifying that the class
distribution is preserved across all splits (within a configurable tolerance).

The module is designed for the EcoPackAI pipeline where the target column
``fragility_label`` is heavily imbalanced (≈60 / 20 / 12 / 8 %).  Stratified
splitting is essential to keep minority classes represented in every fold.

Usage
-----
    from src.data_splitter import split_dataset

    train_df, val_df, test_df, report = split_dataset(df)

Author: EcoPackAI Team
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
SplitReport = Dict[str, Any]
SplitResult = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitReport]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _class_proportions(series: pd.Series) -> Dict[Any, float]:
    """Return normalised value counts as a dict.

    Parameters
    ----------
    series : pd.Series
        Categorical or integer column.

    Returns
    -------
    dict
        ``{class_label: proportion}`` sorted by label.
    """
    counts = series.value_counts(normalize=True).sort_index()
    return counts.to_dict()


def _verify_stratification(
    original_props: Dict[Any, float],
    split_props: Dict[Any, float],
    tolerance: float,
    split_name: str,
) -> Tuple[bool, Dict[str, Any]]:
    """Check whether class proportions in a split match the original.

    Parameters
    ----------
    original_props : dict
        Reference proportions from the full dataset.
    split_props : dict
        Proportions observed in the split.
    tolerance : float
        Maximum allowed absolute deviation per class.
    split_name : str
        Human-readable label for logging (e.g. ``"train"``).

    Returns
    -------
    tuple[bool, dict]
        ``(passed, details)`` where *details* maps each class to its
        deviation and pass/fail status.
    """
    details: Dict[str, Any] = {}
    all_passed = True

    for label in sorted(original_props.keys()):
        orig = original_props.get(label, 0.0)
        split = split_props.get(label, 0.0)
        deviation = abs(orig - split)
        passed = deviation <= tolerance

        details[str(label)] = {
            "original_proportion": round(orig, 6),
            "split_proportion": round(split, 6),
            "absolute_deviation": round(deviation, 6),
            "within_tolerance": passed,
        }

        if not passed:
            all_passed = False
            logger.warning(
                "Stratification drift in '%s' for class %s: "
                "original=%.4f, split=%.4f, deviation=%.4f > tolerance=%.4f",
                split_name,
                label,
                orig,
                split,
                deviation,
                tolerance,
            )

    return all_passed, details


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    stratify_col: str = "fragility_label",
    random_state: int = 42,
    tolerance: float = 0.02,
) -> SplitResult:
    """Stratified split of *df* into train / validation / test sets.

    The split is performed in two stages via
    :func:`sklearn.model_selection.train_test_split`:

    1. Separate **test** from the remainder.
    2. Separate **validation** from the remainder (which becomes **train**).

    After splitting, the function verifies that each class's proportion in
    every split is within *tolerance* of the original distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.  Must contain ``stratify_col``.
    test_size : float, optional
        Fraction of data reserved for the test set (default ``0.15``).
    val_size : float, optional
        Fraction of data reserved for the validation set (default ``0.15``).
    stratify_col : str, optional
        Column name to stratify on (default ``"fragility_label"``).
    random_state : int, optional
        Seed for reproducibility (default ``42``).
    tolerance : float, optional
        Maximum absolute deviation in class proportion before a warning is
        raised (default ``0.02``).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitReport]
        ``(train_df, val_df, test_df, split_report)``

        ``split_report`` contains:

        - ``sizes``: row counts per split.
        - ``proportions``: class distributions per split.
        - ``stratification_verification``: per-split pass/fail details.
        - ``overall_passed``: ``True`` if every class in every split is
          within tolerance.

    Raises
    ------
    ValueError
        - If *df* is empty.
        - If *stratify_col* is not in *df*.
        - If ``test_size + val_size >= 1.0``.
        - If any class has fewer samples than the number of splits (3).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "feature": range(200),
    ...     "fragility_label": [0]*120 + [1]*40 + [2]*24 + [3]*16,
    ... })
    >>> train, val, test, report = split_dataset(df)
    >>> report["overall_passed"]
    True
    """
    # ---- Input validation --------------------------------------------------
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if stratify_col not in df.columns:
        raise ValueError(
            f"Stratification column '{stratify_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    if not (0.0 < val_size < 1.0):
        raise ValueError(f"val_size must be in (0, 1), got {val_size}")

    if test_size + val_size >= 1.0:
        raise ValueError(
            f"test_size ({test_size}) + val_size ({val_size}) = "
            f"{test_size + val_size:.2f} — must be < 1.0"
        )

    # Check minimum class counts
    class_counts = df[stratify_col].value_counts()
    min_count = class_counts.min()
    if min_count < 3:
        raise ValueError(
            f"Class '{class_counts.idxmin()}' has only {min_count} sample(s). "
            "Need at least 3 samples per class for a 3-way stratified split."
        )

    logger.info(
        "Splitting %d rows → train %.0f%% / val %.0f%% / test %.0f%% "
        "(stratify='%s', seed=%d)",
        len(df),
        (1 - test_size - val_size) * 100,
        val_size * 100,
        test_size * 100,
        stratify_col,
        random_state,
    )

    # ---- Stage 1: split off test -------------------------------------------
    remainder_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col],
        random_state=random_state,
    )

    # ---- Stage 2: split remainder into train + val -------------------------
    # val_size is relative to the original dataset, so we must re-scale it
    # relative to the remainder.
    val_fraction_of_remainder = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        remainder_df,
        test_size=val_fraction_of_remainder,
        stratify=remainder_df[stratify_col],
        random_state=random_state,
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # ---- Compute proportions -----------------------------------------------
    original_props = _class_proportions(df[stratify_col])
    train_props = _class_proportions(train_df[stratify_col])
    val_props = _class_proportions(val_df[stratify_col])
    test_props = _class_proportions(test_df[stratify_col])

    # ---- Verify stratification ---------------------------------------------
    train_ok, train_detail = _verify_stratification(
        original_props, train_props, tolerance, "train"
    )
    val_ok, val_detail = _verify_stratification(
        original_props, val_props, tolerance, "validation"
    )
    test_ok, test_detail = _verify_stratification(
        original_props, test_props, tolerance, "test"
    )

    overall_passed = train_ok and val_ok and test_ok

    # ---- Build report ------------------------------------------------------
    split_report: SplitReport = {
        "sizes": {
            "original": len(df),
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "proportions": {
            "original": {str(k): round(v, 6) for k, v in original_props.items()},
            "train": {str(k): round(v, 6) for k, v in train_props.items()},
            "validation": {str(k): round(v, 6) for k, v in val_props.items()},
            "test": {str(k): round(v, 6) for k, v in test_props.items()},
        },
        "stratification_verification": {
            "tolerance": tolerance,
            "train": {"passed": train_ok, "details": train_detail},
            "validation": {"passed": val_ok, "details": val_detail},
            "test": {"passed": test_ok, "details": test_detail},
        },
        "overall_passed": overall_passed,
    }

    # ---- Log summary -------------------------------------------------------
    logger.info(
        "Split complete — train: %d | val: %d | test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    logger.info("Stratification check passed: %s", overall_passed)

    if overall_passed:
        logger.info(
            "All class proportions within ±%.1f%% tolerance.", tolerance * 100
        )
    else:
        logger.warning(
            "Some class proportions drifted beyond ±%.1f%% tolerance. "
            "Review split_report['stratification_verification'] for details.",
            tolerance * 100,
        )

    return train_df, val_df, test_df, split_report


# ---------------------------------------------------------------------------
# Main (demo / smoke-test)
# ---------------------------------------------------------------------------
def main() -> None:
    """Load the synthetic dataset, split it, and print the report."""
    from pathlib import Path
    import json

    csv_path = Path("data") / "synthetic_dataset.csv"

    if not csv_path.exists():
        logger.error("Dataset not found at %s. Run generate_dataset.py first.", csv_path)
        return

    df = pd.read_csv(csv_path)
    logger.info("Loaded dataset: %d rows × %d columns", *df.shape)

    train_df, val_df, test_df, report = split_dataset(df)

    print("\n" + "=" * 72)
    print("  EcoPackAI — Data Split Report")
    print("=" * 72)
    print(json.dumps(report, indent=2, default=str))
    print("=" * 72 + "\n")

    # Persist splits
    output_dir = Path("data")
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    logger.info("Splits exported to %s/", output_dir.resolve())


if __name__ == "__main__":
    main()
