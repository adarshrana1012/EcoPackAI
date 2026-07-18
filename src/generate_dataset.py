"""
generate_dataset.py — Synthetic E-Commerce Shipping Dataset Generator
=====================================================================

Generates a realistic e-commerce shipping dataset with 10,000 rows for the
EcoPackAI packaging-optimization pipeline.  Each product is assigned a
material type that governs its dimensional profile, weight correlation,
and fragility label distribution, producing data that mirrors real-world
warehouse telemetry.

Material profiles
-----------------
| Material        | Size      | Weight   | Fragility |
|-----------------|-----------|----------|-----------|
| glass           | small     | heavy    | 2-3       |
| electronics     | medium    | medium   | 1-2       |
| apparel         | large     | light    | 0         |
| fragile_liquid  | sm-medium | heavy    | 2-3       |
| standard        | varied    | varied   | 0-1       |

Usage
-----
    python -m src.generate_dataset          # writes data/synthetic_dataset.csv
    python src/generate_dataset.py          # same, from repo root

Author: EcoPackAI Team
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
MATERIAL_TYPES: List[str] = [
    "glass",
    "electronics",
    "apparel",
    "fragile_liquid",
    "standard",
]

# Desired overall fragility distribution (imbalanced)
# label-0 ≈ 60 %, label-1 ≈ 20 %, label-2 ≈ 12 %, label-3 ≈ 8 %
GLOBAL_FRAGILITY_WEIGHTS: Dict[int, float] = {0: 0.60, 1: 0.20, 2: 0.12, 3: 0.08}


# ---------------------------------------------------------------------------
# Per-category configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MaterialProfile:
    """Statistical profile for one material category.

    Attributes
    ----------
    name : str
        Material identifier (must be one of ``MATERIAL_TYPES``).
    proportion : float
        Share of total rows allocated to this material (0-1).
    length_mean : float
        Mean length in cm (normal distribution).
    length_std : float
        Std-dev for length.
    width_mean : float
        Mean width in cm.
    width_std : float
        Std-dev for width.
    height_mean : float
        Mean height in cm.
    height_std : float
        Std-dev for height.
    weight_density : float
        Grams-per-cm³ base density used when correlating weight to volume.
    weight_noise_std : float
        Additive Gaussian noise (grams) applied to the density-derived weight.
    fragility_weights : Dict[int, float]
        Per-label sampling probabilities for this category.  Must sum to 1.
    """

    name: str
    proportion: float
    length_mean: float
    length_std: float
    width_mean: float
    width_std: float
    height_mean: float
    height_std: float
    weight_density: float
    weight_noise_std: float
    fragility_weights: Dict[int, float] = field(default_factory=dict)


# Profiles that match the specification exactly.
PROFILES: List[MaterialProfile] = [
    MaterialProfile(
        name="glass",
        proportion=0.15,
        length_mean=12.0,
        length_std=3.0,
        width_mean=10.0,
        width_std=2.5,
        height_mean=8.0,
        height_std=2.0,
        weight_density=0.60,
        weight_noise_std=50.0,
        fragility_weights={0: 0.02, 1: 0.08, 2: 0.45, 3: 0.45},
    ),
    MaterialProfile(
        name="electronics",
        proportion=0.25,
        length_mean=25.0,
        length_std=8.0,
        width_mean=18.0,
        width_std=6.0,
        height_mean=10.0,
        height_std=4.0,
        weight_density=0.25,
        weight_noise_std=80.0,
        fragility_weights={0: 0.10, 1: 0.45, 2: 0.35, 3: 0.10},
    ),
    MaterialProfile(
        name="apparel",
        proportion=0.25,
        length_mean=35.0,
        length_std=10.0,
        width_mean=28.0,
        width_std=8.0,
        height_mean=12.0,
        height_std=5.0,
        weight_density=0.04,
        weight_noise_std=30.0,
        fragility_weights={0: 0.92, 1: 0.06, 2: 0.01, 3: 0.01},
    ),
    MaterialProfile(
        name="fragile_liquid",
        proportion=0.15,
        length_mean=14.0,
        length_std=4.0,
        width_mean=12.0,
        width_std=3.0,
        height_mean=18.0,
        height_std=5.0,
        weight_density=0.55,
        weight_noise_std=60.0,
        fragility_weights={0: 0.02, 1: 0.08, 2: 0.40, 3: 0.50},
    ),
    MaterialProfile(
        name="standard",
        proportion=0.20,
        length_mean=30.0,
        length_std=12.0,
        width_mean=22.0,
        width_std=9.0,
        height_mean=15.0,
        height_std=7.0,
        weight_density=0.15,
        weight_noise_std=100.0,
        fragility_weights={0: 0.70, 1: 0.22, 2: 0.05, 3: 0.03},
    ),
]


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------
def _generate_category_rows(
    profile: MaterialProfile,
    n_rows: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate rows for a single material category.

    Parameters
    ----------
    profile : MaterialProfile
        Statistical profile controlling dimension / weight / fragility
        distributions.
    n_rows : int
        Number of rows to synthesise for this category.
    rng : np.random.Generator
        Seeded random generator for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``product_id``, ``length_cm``, ``width_cm``,
        ``height_cm``, ``weight_g``, ``material_type``, ``fragility_label``,
        ``historical_material_volume_cm3``.
    """
    # --- Dimensions (clipped to physical minimums) --------------------------
    length = np.clip(
        rng.normal(profile.length_mean, profile.length_std, n_rows), 1.0, None
    )
    width = np.clip(
        rng.normal(profile.width_mean, profile.width_std, n_rows), 1.0, None
    )
    height = np.clip(
        rng.normal(profile.height_mean, profile.height_std, n_rows), 1.0, None
    )

    # --- Volume & weight (correlated) ---------------------------------------
    volume = length * width * height
    weight = np.clip(
        volume * profile.weight_density
        + rng.normal(0, profile.weight_noise_std, n_rows),
        10.0,  # minimum 10 g
        None,
    )

    # --- Historical volume (derived with ±10 % multiplicative noise) --------
    noise_factor = rng.normal(1.0, 0.10, n_rows)
    historical_volume = np.clip(volume * noise_factor, 1.0, None)

    # --- Fragility label (per-category imbalanced) --------------------------
    labels = list(profile.fragility_weights.keys())
    probs = list(profile.fragility_weights.values())
    fragility = rng.choice(labels, size=n_rows, p=probs)

    # --- UUIDs --------------------------------------------------------------
    product_ids = [str(uuid.uuid4()) for _ in range(n_rows)]

    return pd.DataFrame(
        {
            "product_id": product_ids,
            "length_cm": np.round(length, 2),
            "width_cm": np.round(width, 2),
            "height_cm": np.round(height, 2),
            "weight_g": np.round(weight, 2),
            "material_type": profile.name,
            "fragility_label": fragility,
            "historical_material_volume_cm3": np.round(historical_volume, 2),
        }
    )


def generate_dataset(
    n_rows: int = 10_000,
    seed: int = RANDOM_SEED,
    profiles: Optional[List[MaterialProfile]] = None,
) -> pd.DataFrame:
    """Synthesise a realistic e-commerce shipping dataset.

    The function iterates over material profiles, generates per-category
    sub-frames with independent normal distributions, then concatenates and
    shuffles them into a single DataFrame.

    Parameters
    ----------
    n_rows : int, optional
        Total number of rows to generate (default ``10_000``).
    seed : int, optional
        Random seed for full reproducibility (default ``42``).
    profiles : list[MaterialProfile], optional
        Override the built-in profiles.  If ``None``, uses ``PROFILES``.

    Returns
    -------
    pd.DataFrame
        Columns: ``product_id``, ``length_cm``, ``width_cm``, ``height_cm``,
        ``weight_g``, ``material_type``, ``fragility_label``,
        ``historical_material_volume_cm3``.

    Raises
    ------
    ValueError
        If ``n_rows < 1`` or profile proportions do not sum to 1.0.

    Examples
    --------
    >>> df = generate_dataset(n_rows=500, seed=0)
    >>> df.shape
    (500, 8)
    """
    if n_rows < 1:
        raise ValueError(f"n_rows must be >= 1, got {n_rows}")

    if profiles is None:
        profiles = PROFILES

    total_proportion = sum(p.proportion for p in profiles)
    if not np.isclose(total_proportion, 1.0, atol=1e-6):
        raise ValueError(
            f"Profile proportions must sum to 1.0, got {total_proportion:.6f}"
        )

    rng = np.random.default_rng(seed)
    logger.info(
        "Generating %d rows across %d material categories (seed=%d)",
        n_rows,
        len(profiles),
        seed,
    )

    frames: List[pd.DataFrame] = []
    allocated = 0

    for idx, profile in enumerate(profiles):
        # Last category absorbs rounding remainder
        if idx == len(profiles) - 1:
            cat_rows = n_rows - allocated
        else:
            cat_rows = int(round(n_rows * profile.proportion))
            allocated += cat_rows

        logger.info(
            "  %-15s → %5d rows (proportion=%.2f)",
            profile.name,
            cat_rows,
            profile.proportion,
        )
        frames.append(_generate_category_rows(profile, cat_rows, rng))

    df = pd.concat(frames, ignore_index=True)

    # Shuffle rows so categories are interleaved
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    logger.info("Dataset generated: %d rows × %d columns", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    """Print concise summary statistics to stdout.

    Parameters
    ----------
    df : pd.DataFrame
        The generated dataset.
    """
    separator = "=" * 72

    print(f"\n{separator}")
    print("  EcoPackAI — Synthetic Dataset Summary")
    print(separator)

    print(f"\n  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")

    # Material distribution
    print("\n  Material type distribution:")
    mat_counts = df["material_type"].value_counts()
    for mat, count in mat_counts.items():
        pct = count / len(df) * 100
        print(f"    {mat:<20s} {count:>5,}  ({pct:5.1f}%)")

    # Fragility distribution
    print("\n  Fragility label distribution:")
    frag_counts = df["fragility_label"].value_counts().sort_index()
    for label, count in frag_counts.items():
        pct = count / len(df) * 100
        print(f"    label-{label}              {count:>5,}  ({pct:5.1f}%)")

    # Continuous feature stats
    continuous_cols = [
        "length_cm",
        "width_cm",
        "height_cm",
        "weight_g",
        "historical_material_volume_cm3",
    ]
    print("\n  Continuous feature statistics:")
    print(df[continuous_cols].describe().round(2).to_string(max_cols=10))

    # Cross-tab: material × fragility
    print("\n  Material × Fragility cross-tabulation:")
    xtab = pd.crosstab(
        df["material_type"],
        df["fragility_label"],
        margins=True,
        margins_name="Total",
    )
    print(xtab.to_string())

    print(f"\n{separator}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point: generate dataset, export CSV, and print summary."""
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_dataset.csv"

    df = generate_dataset(n_rows=10_000, seed=RANDOM_SEED)
    df.to_csv(output_path, index=False)
    logger.info("Dataset exported → %s", output_path.resolve())

    print_summary(df)


if __name__ == "__main__":
    main()
