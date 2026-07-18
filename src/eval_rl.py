"""
eval_rl.py — PPO Policy Evaluation Script (Prompt 25)
=====================================================

Evaluates the trained PPO policy against a holdout set and compares
performance versus the FFD baseline.

Usage
-----
    python -m src.eval_rl
    python -m src.eval_rl --model models/ppo_packing_v1 --orders 1000

Author: EcoPackAI Team
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.box_catalogue import BoxCatalogue
from src.packing_engine import Item, pack_order
from src.packing_env import (
    PackingEnv,
    compute_reward,
    compute_vol_efficiency,
    make_packing_env,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_NUM_ORDERS = 1000
DEFAULT_MODEL_PATH = "models/ppo_packing_v1"


def _generate_orders(
    df: pd.DataFrame, n_orders: int, seed: int = 42,
) -> List[List[Item]]:
    """Sample random orders from the dataset."""
    rng = np.random.RandomState(seed)
    orders = []
    for _ in range(n_orders):
        n_items = rng.randint(1, 9)
        sample = df.sample(n=n_items, random_state=rng.randint(0, 100000))
        items = [
            Item(
                item_id=str(row.get("product_id", f"item-{idx}")),
                length=float(row["length_cm"]),
                width=float(row["width_cm"]),
                height=float(row["height_cm"]),
                weight_g=float(row["weight_g"]),
                fragility_label=int(row["fragility_label"]),
            )
            for idx, row in sample.iterrows()
        ]
        orders.append(items)
    return orders


def evaluate_ffd_baseline(
    orders: List[List[Item]],
    catalogue: BoxCatalogue,
) -> Dict[str, Any]:
    """Run FFD baseline on all orders and collect metrics."""
    void_pcts = []
    violations = []
    times_ms = []

    for items in orders:
        start = time.perf_counter()
        result = catalogue.select_optimal_box(items, allow_rotation=True)
        elapsed = (time.perf_counter() - start) * 1000

        void_pcts.append(result.void_volume_pct)
        violations.append(result.constraint_violations)
        times_ms.append(elapsed)

    return {
        "approach": "FFD+Rotation",
        "mean_reward": float(np.mean([
            compute_reward(
                (1 - v / 100) * 240000, 240000, c
            ) for v, c in zip(void_pcts, violations)
        ])),
        "mean_void_pct": round(float(np.mean(void_pcts)), 2),
        "median_void_pct": round(float(np.median(void_pcts)), 2),
        "std_void_pct": round(float(np.std(void_pcts)), 2),
        "violation_rate": round(
            float(np.mean([v > 0 for v in violations])) * 100, 2
        ),
        "mean_time_ms": round(float(np.mean(times_ms)), 3),
        "total_violations": int(sum(violations)),
    }


def evaluate_rl_policy(
    model_path: str,
    orders: List[List[Item]],
    catalogue: BoxCatalogue,
    data_path: str = "data/test.csv",
) -> Dict[str, Any]:
    """Evaluate the trained PPO policy on holdout orders.

    Parameters
    ----------
    model_path : str
        Path to the saved PPO model (without .zip).
    orders : list[list[Item]]
        Holdout orders.
    catalogue : BoxCatalogue
        Available boxes.
    data_path : str
        Path to the data CSV for building the env.

    Returns
    -------
    dict
        Evaluation metrics.
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        logger.error("stable-baselines3 not installed.")
        return {"approach": "PPO", "error": "stable-baselines3 not installed"}

    # Load model
    model_file = Path(model_path)
    if not model_file.with_suffix(".zip").exists():
        logger.warning("Model not found at %s.zip — skipping RL eval.", model_path)
        return {"approach": "PPO", "error": f"Model not found: {model_path}.zip"}

    model = PPO.load(str(model_path))
    logger.info("Loaded PPO model from %s", model_path)

    # Create environment
    env = make_packing_env(data_path=data_path, seed=99)

    rewards = []
    void_pcts = []
    violations = []
    times_ms = []

    for order_items in orders:
        # Override the env's items for this episode
        env._current_items = sorted(
            order_items, key=lambda i: i.volume, reverse=True
        )
        env._item_idx = 0
        env._bins = []
        env._total_violations = 0
        env._total_packed_volume = 0.0
        env._total_bin_volume = 0.0
        env._step_count = 0

        obs = env._get_obs()
        episode_reward = 0.0
        done = False
        start = time.perf_counter()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            done = terminated or truncated

        elapsed = (time.perf_counter() - start) * 1000

        vol_eff = compute_vol_efficiency(
            env._total_packed_volume, env._total_bin_volume
        )
        void_pct = (1.0 - vol_eff) * 100

        rewards.append(episode_reward)
        void_pcts.append(void_pct)
        violations.append(env._total_violations)
        times_ms.append(elapsed)

    return {
        "approach": "PPO",
        "mean_reward": round(float(np.mean(rewards)), 4),
        "std_reward": round(float(np.std(rewards)), 4),
        "mean_void_pct": round(float(np.mean(void_pcts)), 2),
        "median_void_pct": round(float(np.median(void_pcts)), 2),
        "std_void_pct": round(float(np.std(void_pcts)), 2),
        "violation_rate": round(
            float(np.mean([v > 0 for v in violations])) * 100, 2
        ),
        "mean_time_ms": round(float(np.mean(times_ms)), 3),
        "total_violations": int(sum(violations)),
    }


def run_evaluation(
    model_path: str = DEFAULT_MODEL_PATH,
    data_path: str = "data/test.csv",
    n_orders: int = DEFAULT_NUM_ORDERS,
    output_path: str = "eval_report.json",
) -> Dict[str, Any]:
    """Run full evaluation and comparison.

    Parameters
    ----------
    model_path : str
        Path to PPO model.
    data_path : str
        Path to holdout dataset.
    n_orders : int
        Number of orders to evaluate.
    output_path : str
        Path to save the evaluation report JSON.

    Returns
    -------
    dict
        Complete evaluation report.
    """
    logger.info("=" * 60)
    logger.info("POLICY EVALUATION — %d orders from %s", n_orders, data_path)
    logger.info("=" * 60)

    df = pd.read_csv(data_path)
    orders = _generate_orders(df, n_orders)
    catalogue = BoxCatalogue()

    # Evaluate FFD baseline
    logger.info("Evaluating FFD+Rotation baseline...")
    ffd_results = evaluate_ffd_baseline(orders, catalogue)

    # Evaluate RL policy
    logger.info("Evaluating PPO policy from %s...", model_path)
    rl_results = evaluate_rl_policy(model_path, orders, catalogue, data_path)

    # Build comparison report
    report = {
        "evaluation_date": pd.Timestamp.utcnow().isoformat(),
        "n_orders": n_orders,
        "data_path": data_path,
        "model_path": model_path,
        "ffd_baseline": ffd_results,
        "rl_policy": rl_results,
    }

    # Comparison
    if "error" not in rl_results:
        void_improvement = ffd_results["mean_void_pct"] - rl_results["mean_void_pct"]
        report["comparison"] = {
            "void_improvement_pp": round(void_improvement, 2),
            "rl_better_void": void_improvement > 0,
            "rl_fewer_violations": (
                rl_results["total_violations"] < ffd_results["total_violations"]
            ),
        }

    # Save report
    out = Path(output_path)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Evaluation report saved to %s", out)

    # Print comparison table
    print("\n" + "=" * 72)
    print("  POLICY EVALUATION — FFD vs PPO Comparison")
    print("=" * 72)
    print(f"  {'Metric':<25} {'FFD+Rotation':>15} {'PPO':>15}")
    print("  " + "-" * 57)

    for metric in ["mean_reward", "mean_void_pct", "violation_rate", "mean_time_ms"]:
        ffd_val = ffd_results.get(metric, "N/A")
        rl_val = rl_results.get(metric, "N/A")
        ffd_str = f"{ffd_val}" if isinstance(ffd_val, str) else f"{ffd_val:.3f}"
        rl_str = f"{rl_val}" if isinstance(rl_val, str) else f"{rl_val:.3f}"
        print(f"  {metric:<25} {ffd_str:>15} {rl_str:>15}")

    print("=" * 72 + "\n")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EcoPackAI RL policy")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=str, default="data/test.csv")
    parser.add_argument("--orders", type=int, default=DEFAULT_NUM_ORDERS)
    parser.add_argument("--output", type=str, default="eval_report.json")
    args = parser.parse_args()

    run_evaluation(
        model_path=args.model,
        data_path=args.data,
        n_orders=args.orders,
        output_path=args.output,
    )
