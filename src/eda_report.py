"""
eda_report.py — Exploratory Data Analysis Report Generator
==========================================================

Generates four publication-quality visualisations for the EcoPackAI synthetic
shipping dataset:

1. **Class distribution** bar chart of ``fragility_label``.
2. **Correlation heatmap** of continuous features.
3. **Box plots** of ``volume_cm3`` grouped by ``fragility_label``.
4. **Pair plot** of key numeric features coloured by ``material_type``.

All plots use a modern dark theme (``seaborn-v0_8-darkgrid`` / fallback
``darkgrid``) with a consistent colour palette and are saved at 150 DPI with
tight bounding boxes.

Usage
-----
    python -m src.eda_report            # reads data/synthetic_dataset.csv
    python src/eda_report.py            # same, from repo root

Author: EcoPackAI Team
"""

from __future__ import annotations

# Headless backend — MUST come before any other matplotlib import
import matplotlib
matplotlib.use("Agg")

import logging
import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme / style constants
# ---------------------------------------------------------------------------
_PREFERRED_STYLES = ["seaborn-v0_8-darkgrid", "seaborn-darkgrid", "darkgrid"]
PALETTE_MATERIAL = {
    "glass": "#e74c3c",
    "electronics": "#3498db",
    "apparel": "#2ecc71",
    "fragile_liquid": "#f39c12",
    "standard": "#9b59b6",
}
PALETTE_FRAGILITY = {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22", 3: "#e74c3c"}
SAVE_KW = {"dpi": 150, "bbox_inches": "tight", "facecolor": "#1e1e1e"}

# Continuous features used in the correlation heatmap
CONTINUOUS_FEATURES: List[str] = [
    "length_cm",
    "width_cm",
    "height_cm",
    "weight_g",
    "historical_material_volume_cm3",
]


# ---------------------------------------------------------------------------
# Style helper
# ---------------------------------------------------------------------------
def _apply_dark_theme() -> None:
    """Apply a professional dark theme to matplotlib / seaborn.

    Tries multiple style names for compatibility across seaborn versions.
    Falls back to manual ``rcParams`` if no named style is available.
    """
    applied = False
    for style_name in _PREFERRED_STYLES:
        try:
            plt.style.use(style_name)
            applied = True
            logger.info("Applied matplotlib style: '%s'", style_name)
            break
        except OSError:
            continue

    if not applied:
        logger.info("No named dark style found — applying manual rcParams.")

    # Override common parameters for a consistent dark look
    dark_params = {
        "figure.facecolor": "#1e1e1e",
        "axes.facecolor": "#2d2d2d",
        "axes.edgecolor": "#555555",
        "axes.labelcolor": "#cccccc",
        "text.color": "#cccccc",
        "xtick.color": "#aaaaaa",
        "ytick.color": "#aaaaaa",
        "grid.color": "#444444",
        "grid.alpha": 0.5,
        "legend.facecolor": "#2d2d2d",
        "legend.edgecolor": "#555555",
        "legend.labelcolor": "#cccccc",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    }
    plt.rcParams.update(dark_params)
    sns.set_palette("bright")


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------
def plot_class_distribution(
    df: pd.DataFrame, output_dir: Path
) -> Path:
    """Bar chart of ``fragility_label`` class distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing ``fragility_label``.
    output_dir : Path
        Directory to save the figure.

    Returns
    -------
    Path
        Absolute path of the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    counts = df["fragility_label"].value_counts().sort_index()
    colours = [PALETTE_FRAGILITY.get(lbl, "#888888") for lbl in counts.index]

    bars = ax.bar(counts.index.astype(str), counts.values, color=colours, edgecolor="#111111", linewidth=0.8)

    # Annotate bars with count + percentage
    total = counts.sum()
    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#eeeeee",
            fontweight="bold",
        )

    ax.set_xlabel("Fragility Label")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution — Fragility Label", fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.18)

    out_path = output_dir / "class_distribution.png"
    fig.savefig(out_path, **SAVE_KW)
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


def plot_correlation_heatmap(
    df: pd.DataFrame, output_dir: Path
) -> Path:
    """Annotated correlation heatmap of continuous features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing ``CONTINUOUS_FEATURES``.
    output_dir : Path
        Directory to save the figure.

    Returns
    -------
    Path
        Absolute path of the saved PNG.
    """
    available = [c for c in CONTINUOUS_FEATURES if c in df.columns]

    # Add volume_cm3 if not present
    if "volume_cm3" not in df.columns and all(
        c in df.columns for c in ["length_cm", "width_cm", "height_cm"]
    ):
        df = df.copy()
        df["volume_cm3"] = df["length_cm"] * df["width_cm"] * df["height_cm"]
        if "volume_cm3" not in available:
            available.append("volume_cm3")

    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.8,
        linecolor="#333333",
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")

    out_path = output_dir / "correlation_heatmap.png"
    fig.savefig(out_path, **SAVE_KW)
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


def plot_volume_boxplots(
    df: pd.DataFrame, output_dir: Path
) -> Path:
    """Box plots of ``volume_cm3`` grouped by ``fragility_label``.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing dimension columns or ``volume_cm3``.
    output_dir : Path
        Directory to save the figure.

    Returns
    -------
    Path
        Absolute path of the saved PNG.
    """
    df = df.copy()
    if "volume_cm3" not in df.columns:
        df["volume_cm3"] = df["length_cm"] * df["width_cm"] * df["height_cm"]

    fig, ax = plt.subplots(figsize=(10, 6))

    palette = [PALETTE_FRAGILITY.get(i, "#888888") for i in sorted(df["fragility_label"].unique())]

    sns.boxplot(
        data=df,
        x="fragility_label",
        y="volume_cm3",
        palette=palette,
        width=0.55,
        fliersize=2.5,
        linewidth=1.0,
        ax=ax,
    )

    # Overlay strip plot for density context
    sns.stripplot(
        data=df.sample(min(1000, len(df)), random_state=42),
        x="fragility_label",
        y="volume_cm3",
        color="#ffffff",
        alpha=0.15,
        size=2.5,
        jitter=True,
        ax=ax,
    )

    ax.set_xlabel("Fragility Label")
    ax.set_ylabel("Volume (cm³)")
    ax.set_title("Volume Distribution by Fragility Label", fontweight="bold")

    out_path = output_dir / "volume_boxplots.png"
    fig.savefig(out_path, **SAVE_KW)
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


def plot_pairplot_by_material(
    df: pd.DataFrame,
    output_dir: Path,
    sample_n: int = 2_000,
) -> Path:
    """Pair plot of key numeric features coloured by ``material_type``.

    A random sub-sample is drawn (default 2 000 rows) to keep rendering
    time reasonable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing dimension, weight, and material columns.
    output_dir : Path
        Directory to save the figure.
    sample_n : int, optional
        Maximum rows to plot (default ``2_000``).

    Returns
    -------
    Path
        Absolute path of the saved PNG.
    """
    plot_cols = ["length_cm", "width_cm", "height_cm", "weight_g", "material_type"]
    available_cols = [c for c in plot_cols if c in df.columns]

    sub = df[available_cols].sample(min(sample_n, len(df)), random_state=42)

    g = sns.pairplot(
        sub,
        hue="material_type",
        palette=PALETTE_MATERIAL,
        diag_kind="kde",
        plot_kws={"alpha": 0.45, "s": 12, "edgecolor": "none"},
        diag_kws={"linewidth": 1.5, "fill": True, "alpha": 0.35},
        height=2.4,
        aspect=1.0,
    )
    g.figure.suptitle(
        "Pair Plot by Material Type",
        y=1.02,
        fontsize=15,
        fontweight="bold",
        color="#cccccc",
    )

    # Dark-theme fix: paint figure & axes backgrounds
    g.figure.set_facecolor("#1e1e1e")
    for ax in g.axes.flat:
        ax.set_facecolor("#2d2d2d")

    out_path = output_dir / "pairplot_by_material.png"
    g.savefig(out_path, **SAVE_KW)
    plt.close(g.figure)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------
def generate_eda_report(
    df: pd.DataFrame,
    output_dir: str = "eda_output",
) -> List[Path]:
    """Generate the full EDA report (four plots) and return saved paths.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyse.  Expected to contain at least:
        ``length_cm``, ``width_cm``, ``height_cm``, ``weight_g``,
        ``material_type``, ``fragility_label``,
        ``historical_material_volume_cm3``.
    output_dir : str, optional
        Directory in which PNGs are saved (created if absent).
        Default ``"eda_output"``.

    Returns
    -------
    list[Path]
        Absolute paths of the four saved figures.

    Raises
    ------
    ValueError
        If *df* is empty or is missing required columns.

    Examples
    --------
    >>> import pandas as pd
    >>> paths = generate_eda_report(df, output_dir="my_eda")
    >>> len(paths)
    4
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty — nothing to plot.")

    required_cols = {
        "length_cm",
        "width_cm",
        "height_cm",
        "weight_g",
        "material_type",
        "fragility_label",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("EDA output directory: %s", out.resolve())

    _apply_dark_theme()

    saved: List[Path] = []
    logger.info("Generating Plot 1/4 — Class distribution …")
    saved.append(plot_class_distribution(df, out))

    logger.info("Generating Plot 2/4 — Correlation heatmap …")
    saved.append(plot_correlation_heatmap(df, out))

    logger.info("Generating Plot 3/4 — Volume box plots …")
    saved.append(plot_volume_boxplots(df, out))

    logger.info("Generating Plot 4/4 — Pair plot by material …")
    saved.append(plot_pairplot_by_material(df, out))

    logger.info("EDA report complete — %d figures saved to %s/", len(saved), out)
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Load synthetic dataset and generate the full EDA report."""
    csv_path = Path("data") / "synthetic_dataset.csv"

    if not csv_path.exists():
        logger.error(
            "Dataset not found at %s. Run generate_dataset.py first.", csv_path
        )
        return

    df = pd.read_csv(csv_path)
    logger.info("Loaded dataset: %d rows × %d columns", *df.shape)

    paths = generate_eda_report(df, output_dir="eda_output")

    print("\n" + "=" * 72)
    print("  EcoPackAI — EDA Report")
    print("=" * 72)
    for p in paths:
        print(f"  [OK] {p.resolve()}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
