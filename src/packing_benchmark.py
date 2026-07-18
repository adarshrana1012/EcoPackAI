"""
packing_benchmark.py — FFD vs FFD-with-Rotation Benchmark
==========================================================

Benchmarks the base First Fit Decreasing algorithm against the
rotation-enabled variant over 500 random orders from the test dataset.

Usage
-----
    python -m src.packing_benchmark

Author: EcoPackAI Team
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.box_catalogue import BoxCatalogue
from src.packing_engine import Item, PackingResult

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_ORDERS = 500
MIN_ITEMS_PER_ORDER = 1
MAX_ITEMS_PER_ORDER = 8
RANDOM_SEED = 42


def _generate_orders(
    df: pd.DataFrame,
    n_orders: int = NUM_ORDERS,
    seed: int = RANDOM_SEED,
) -> List[List[Item]]:
    """Generate random orders from the test dataset.

    Each order contains 1–8 items sampled from the DataFrame.
    """
    rng = np.random.RandomState(seed)
    orders: List[List[Item]] = []

    for i in range(n_orders):
        n_items = rng.randint(MIN_ITEMS_PER_ORDER, MAX_ITEMS_PER_ORDER + 1)
        sample = df.sample(n=n_items, random_state=rng.randint(0, 100000))

        items: List[Item] = []
        for idx, row in sample.iterrows():
            item = Item(
                item_id=str(row.get("product_id", f"item-{idx}")),
                length=float(row["length_cm"]),
                width=float(row["width_cm"]),
                height=float(row["height_cm"]),
                weight_g=float(row["weight_g"]),
                fragility_label=int(row["fragility_label"]),
            )
            items.append(item)
        orders.append(items)

    return orders


def _benchmark_approach(
    orders: List[List[Item]],
    catalogue: BoxCatalogue,
    allow_rotation: bool,
    label: str,
) -> pd.DataFrame:
    """Run packing on all orders and collect metrics."""
    records = []

    for i, items in enumerate(orders):
        start = time.perf_counter()
        result = catalogue.select_optimal_box(items, allow_rotation=allow_rotation)
        elapsed_ms = (time.perf_counter() - start) * 1000

        records.append({
            "order_idx": i,
            "approach": label,
            "n_items": len(items),
            "box_sku": result.box.sku,
            "void_volume_pct": result.void_volume_pct,
            "constraint_violations": result.constraint_violations,
            "requires_split": result.requires_split,
            "items_packed": len(result.placements),
            "separate_box_items": len(result.requires_separate_box),
            "time_ms": round(elapsed_ms, 3),
        })

    return pd.DataFrame(records)


def run_benchmark(
    data_path: str = "data/test.csv",
    n_orders: int = NUM_ORDERS,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Run the FFD vs FFD+Rotation benchmark.

    Parameters
    ----------
    data_path : str
        Path to the test dataset CSV.
    n_orders : int
        Number of random orders to benchmark.

    Returns
    -------
    tuple[DataFrame, DataFrame, str]
        ``(detailed_results, summary_df, recommendation)``
    """
    logger.info("=" * 60)
    logger.info("BENCHMARK: FFD vs FFD-with-Rotation (%d orders)", n_orders)
    logger.info("=" * 60)

    # Load data
    df = pd.read_csv(data_path)
    logger.info("Loaded %d rows from %s", len(df), data_path)

    # Generate orders
    orders = _generate_orders(df, n_orders=n_orders)
    logger.info("Generated %d random orders (1-%d items each)",
                len(orders), MAX_ITEMS_PER_ORDER)

    # Initialize catalogue
    catalogue = BoxCatalogue()

    # Run both approaches
    logger.info("Running FFD (no rotation)...")
    df_ffd = _benchmark_approach(orders, catalogue, allow_rotation=False, label="FFD")

    logger.info("Running FFD + Rotation...")
    df_rot = _benchmark_approach(orders, catalogue, allow_rotation=True, label="FFD+Rotation")

    # Combine results
    detailed = pd.concat([df_ffd, df_rot], ignore_index=True)

    # Compute summary statistics
    summary_rows = []
    for approach_df in [df_ffd, df_rot]:
        label = approach_df["approach"].iloc[0]
        summary_rows.append({
            "approach": label,
            "mean_void_pct": round(approach_df["void_volume_pct"].mean(), 2),
            "median_void_pct": round(approach_df["void_volume_pct"].median(), 2),
            "std_void_pct": round(approach_df["void_volume_pct"].std(), 2),
            "mean_time_ms": round(approach_df["time_ms"].mean(), 3),
            "median_time_ms": round(approach_df["time_ms"].median(), 3),
            "p95_time_ms": round(approach_df["time_ms"].quantile(0.95), 3),
            "split_rate_pct": round(
                approach_df["requires_split"].mean() * 100, 2
            ),
            "violation_rate_pct": round(
                (approach_df["constraint_violations"] > 0).mean() * 100, 2
            ),
        })

    summary = pd.DataFrame(summary_rows)

    # Recommendation
    ffd_void = summary.loc[summary["approach"] == "FFD", "mean_void_pct"].values[0]
    rot_void = summary.loc[summary["approach"] == "FFD+Rotation", "mean_void_pct"].values[0]
    ffd_time = summary.loc[summary["approach"] == "FFD", "mean_time_ms"].values[0]
    rot_time = summary.loc[summary["approach"] == "FFD+Rotation", "mean_time_ms"].values[0]

    void_improvement = ffd_void - rot_void
    time_increase = rot_time - ffd_time

    if rot_void < ffd_void and rot_time < 200:
        recommendation = (
            f"FFD+Rotation is recommended. It reduces mean void volume by "
            f"{void_improvement:.1f}pp at a cost of {time_increase:.1f}ms "
            f"additional latency (within the 200ms P95 SLA target)."
        )
        recommended = "FFD+Rotation"
    elif void_improvement > 3 and rot_time < 200:
        recommendation = (
            f"FFD+Rotation is recommended despite higher latency. "
            f"Void reduction of {void_improvement:.1f}pp justifies the "
            f"{time_increase:.1f}ms overhead."
        )
        recommended = "FFD+Rotation"
    else:
        recommendation = (
            f"Base FFD is recommended. Rotation adds {time_increase:.1f}ms "
            f"latency for only {void_improvement:.1f}pp void reduction, "
            f"which does not justify the computational cost."
        )
        recommended = "FFD"

    # Print report
    print("\n" + "=" * 72)
    print("  BENCHMARK RESULTS — FFD vs FFD+Rotation")
    print("=" * 72)
    print(f"\n  Orders tested: {n_orders}")
    print(f"\n{summary.to_string(index=False)}\n")
    print(f"  Void improvement (FFD -> Rotation): {void_improvement:+.2f} pp")
    print(f"  Latency increase: {time_increase:+.3f} ms")
    print(f"\n  >> RECOMMENDATION: {recommended}")
    print(f"  >> {recommendation}")
    print("=" * 72 + "\n")

    return detailed, summary, recommendation


if __name__ == "__main__":
    detailed, summary, rec = run_benchmark()
    # Save detailed results
    out_dir = Path("eda_output")
    out_dir.mkdir(exist_ok=True)
    detailed.to_csv(out_dir / "packing_benchmark_detailed.csv", index=False)
    summary.to_csv(out_dir / "packing_benchmark_summary.csv", index=False)
    print(f"Results saved to {out_dir}/")
