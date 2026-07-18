"""
ab_test_router.py — A/B Testing Framework for Packing Strategies (Prompt 28)
=============================================================================

Routes packing requests to either the RL policy or the FFD baseline
based on a configurable traffic split.  Tracks per-variant metrics and
exposes a results endpoint.

Author: EcoPackAI Team
"""

from __future__ import annotations

import logging
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.box_catalogue import BoxCatalogue
from src.packing_engine import Item, PackingResult, pack_order

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class ABTestConfig(BaseModel):
    """A/B test configuration."""
    rl_traffic_pct: float = Field(
        default=10.0, ge=0, le=100,
        description="Percentage of traffic routed to RL policy",
    )
    ffd_traffic_pct: float = Field(
        default=90.0, ge=0, le=100,
        description="Percentage of traffic routed to FFD baseline",
    )


class VariantMetrics(BaseModel):
    """Per-variant aggregated metrics."""
    variant: str
    request_count: int
    mean_void_pct: float
    median_void_pct: float
    mean_safety_violations: float
    total_violations: int
    mean_compute_time_ms: float
    p95_compute_time_ms: float
    p99_compute_time_ms: float


class ABTestResults(BaseModel):
    """Statistical summary of A/B test results."""
    test_name: str
    start_time: str
    total_requests: int
    config: ABTestConfig
    variants: List[VariantMetrics]
    recommendation: Optional[str] = None


# ---------------------------------------------------------------------------
# Variant Tracking
# ---------------------------------------------------------------------------

@dataclass
class _VariantTracker:
    """Tracks metrics for a single variant."""
    name: str
    void_pcts: List[float] = field(default_factory=list)
    violations: List[int] = field(default_factory=list)
    compute_times_ms: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.void_pcts)

    def record(self, void_pct: float, violations: int, time_ms: float) -> None:
        self.void_pcts.append(void_pct)
        self.violations.append(violations)
        self.compute_times_ms.append(time_ms)

    def summarise(self) -> VariantMetrics:
        if not self.void_pcts:
            return VariantMetrics(
                variant=self.name, request_count=0,
                mean_void_pct=0, median_void_pct=0,
                mean_safety_violations=0, total_violations=0,
                mean_compute_time_ms=0, p95_compute_time_ms=0,
                p99_compute_time_ms=0,
            )
        return VariantMetrics(
            variant=self.name,
            request_count=self.count,
            mean_void_pct=round(float(np.mean(self.void_pcts)), 2),
            median_void_pct=round(float(np.median(self.void_pcts)), 2),
            mean_safety_violations=round(float(np.mean(self.violations)), 3),
            total_violations=int(sum(self.violations)),
            mean_compute_time_ms=round(float(np.mean(self.compute_times_ms)), 3),
            p95_compute_time_ms=round(float(np.percentile(self.compute_times_ms, 95)), 3),
            p99_compute_time_ms=round(float(np.percentile(self.compute_times_ms, 99)), 3),
        )


# ═══════════════════════════════════════════════════════════════════════════
# ABTestRouter
# ═══════════════════════════════════════════════════════════════════════════

class ABTestRouter:
    """Routes packing requests between RL policy and FFD baseline.

    Parameters
    ----------
    rl_traffic_pct : float
        Percentage of traffic routed to the RL policy (0-100).
        Default is 10% (90/10 split favouring FFD).
    catalogue : BoxCatalogue, optional
        Box catalogue to use for FFD baseline.
    test_name : str
        Name identifier for this A/B test.

    Examples
    --------
    >>> router = ABTestRouter(rl_traffic_pct=10)
    >>> variant, result = router.route("order-123", items)
    >>> results = router.get_results()
    """

    def __init__(
        self,
        rl_traffic_pct: float = 10.0,
        catalogue: Optional[BoxCatalogue] = None,
        test_name: str = "packing_rl_vs_ffd",
    ) -> None:
        self.rl_pct = rl_traffic_pct
        self.ffd_pct = 100.0 - rl_traffic_pct
        self.test_name = test_name
        self.start_time = datetime.now(timezone.utc)
        self.catalogue = catalogue or BoxCatalogue()

        self._rl_tracker = _VariantTracker("rl_policy")
        self._ffd_tracker = _VariantTracker("ffd_baseline")

        self._rl_model = None
        self._rl_env = None
        self._load_rl_model()

        logger.info(
            "ABTestRouter initialised: RL=%.1f%%, FFD=%.1f%%",
            self.rl_pct, self.ffd_pct,
        )

    def _load_rl_model(self) -> None:
        """Attempt to load the PPO model for RL routing."""
        try:
            from stable_baselines3 import PPO
            from pathlib import Path

            model_path = Path("models/ppo_packing_v1")
            if model_path.with_suffix(".zip").exists():
                self._rl_model = PPO.load(str(model_path))
                logger.info("ABTest: RL model loaded.")
            else:
                logger.info("ABTest: No RL model found, RL variant will use FFD fallback.")
        except ImportError:
            logger.info("ABTest: stable-baselines3 not installed, RL uses FFD fallback.")

    def _select_variant(self, order_id: str) -> str:
        """Deterministically assign an order to a variant.

        Uses a hash of the order_id for consistent assignment
        (same order always gets the same variant).
        """
        hash_val = int(hashlib.md5(order_id.encode()).hexdigest(), 16)
        bucket = hash_val % 100
        return "rl_policy" if bucket < self.rl_pct else "ffd_baseline"

    def _run_ffd(self, items: List[Item]) -> PackingResult:
        """Run FFD+Rotation baseline."""
        return self.catalogue.select_optimal_box(items, allow_rotation=True)

    def _run_rl(self, items: List[Item]) -> PackingResult:
        """Run RL policy.  Falls back to FFD if model not available."""
        if self._rl_model is None:
            return self._run_ffd(items)

        try:
            from src.packing_env import PackingEnv, compute_vol_efficiency

            env = PackingEnv(
                items_pool=items,
                available_boxes=self.catalogue.boxes,
                max_bins=5,
                items_per_episode=len(items),
            )
            env._current_items = sorted(items, key=lambda i: i.volume, reverse=True)
            env._item_idx = 0
            env._bins = []
            env._total_violations = 0
            env._total_packed_volume = 0.0
            env._total_bin_volume = 0.0
            obs = env._get_obs()

            done = False
            while not done:
                action, _ = self._rl_model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated

            # Build PackingResult from env state
            from src.packing_engine import PackingResult as PR
            if env._bins:
                all_placements = []
                for b in env._bins:
                    all_placements.extend(b.placements)
                vol_eff = compute_vol_efficiency(
                    env._total_packed_volume, env._total_bin_volume
                )
                return PR(
                    box=env._bins[0].box,
                    placements=all_placements,
                    void_volume_pct=round((1.0 - vol_eff) * 100, 2),
                    constraint_violations=env._total_violations,
                )
            return self._run_ffd(items)

        except Exception as e:
            logger.warning("RL execution failed (%s), falling back to FFD.", e)
            return self._run_ffd(items)

    def route(
        self, order_id: str, items: List[Item],
    ) -> Tuple[str, PackingResult]:
        """Route a packing request to the appropriate variant.

        Parameters
        ----------
        order_id : str
            Unique order identifier (used for consistent bucketing).
        items : list[Item]
            Items to pack.

        Returns
        -------
        tuple[str, PackingResult]
            ``(variant_name, packing_result)``
        """
        variant = self._select_variant(order_id)

        start = time.perf_counter()
        if variant == "rl_policy":
            result = self._run_rl(items)
        else:
            result = self._run_ffd(items)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Track metrics
        tracker = self._rl_tracker if variant == "rl_policy" else self._ffd_tracker
        tracker.record(result.void_volume_pct, result.constraint_violations, elapsed_ms)

        return variant, result

    def update_config(self, rl_pct: float) -> None:
        """Update the traffic split."""
        self.rl_pct = max(0, min(100, rl_pct))
        self.ffd_pct = 100.0 - self.rl_pct
        logger.info("Traffic split updated: RL=%.1f%%, FFD=%.1f%%",
                     self.rl_pct, self.ffd_pct)

    def get_results(self) -> ABTestResults:
        """Get aggregated A/B test results."""
        rl_summary = self._rl_tracker.summarise()
        ffd_summary = self._ffd_tracker.summarise()

        # Generate recommendation
        recommendation = None
        if rl_summary.request_count >= 30 and ffd_summary.request_count >= 30:
            void_diff = ffd_summary.mean_void_pct - rl_summary.mean_void_pct
            time_diff = rl_summary.mean_compute_time_ms - ffd_summary.mean_compute_time_ms

            if void_diff > 2 and rl_summary.p95_compute_time_ms < 200:
                recommendation = (
                    f"RL policy reduces void by {void_diff:.1f}pp with acceptable "
                    f"latency. Consider increasing RL traffic."
                )
            elif void_diff < -2:
                recommendation = (
                    f"FFD baseline outperforms RL by {-void_diff:.1f}pp void. "
                    f"Keep current split or retrain RL."
                )
            else:
                recommendation = (
                    f"Results are comparable (void diff: {void_diff:.1f}pp). "
                    f"Continue testing with more samples."
                )

        return ABTestResults(
            test_name=self.test_name,
            start_time=self.start_time.isoformat(),
            total_requests=rl_summary.request_count + ffd_summary.request_count,
            config=ABTestConfig(
                rl_traffic_pct=self.rl_pct,
                ffd_traffic_pct=self.ffd_pct,
            ),
            variants=[ffd_summary, rl_summary],
            recommendation=recommendation,
        )

    def reset_metrics(self) -> None:
        """Reset all tracked metrics."""
        self._rl_tracker = _VariantTracker("rl_policy")
        self._ffd_tracker = _VariantTracker("ffd_baseline")
        self.start_time = datetime.now(timezone.utc)
        logger.info("A/B test metrics reset.")


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI Router
# ═══════════════════════════════════════════════════════════════════════════

_router_instance: Optional[ABTestRouter] = None


def get_ab_router() -> ABTestRouter:
    """Get or create the global ABTestRouter instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = ABTestRouter()
    return _router_instance


ab_router = APIRouter(prefix="/v1/ab-test", tags=["A/B Testing"])


@ab_router.get("/results", response_model=ABTestResults)
async def ab_test_results():
    """Return statistical summary of A/B test results."""
    router = get_ab_router()
    return router.get_results()


@ab_router.post("/config")
async def update_ab_config(config: ABTestConfig):
    """Update the A/B test traffic split."""
    router = get_ab_router()
    router.update_config(config.rl_traffic_pct)
    return {"status": "updated", "rl_pct": router.rl_pct, "ffd_pct": router.ffd_pct}


@ab_router.post("/reset")
async def reset_ab_metrics():
    """Reset A/B test metrics."""
    router = get_ab_router()
    router.reset_metrics()
    return {"status": "reset"}
