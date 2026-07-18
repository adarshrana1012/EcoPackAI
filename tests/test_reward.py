"""
test_reward.py — Unit Tests for the RL Reward Function (Prompt 23)
===================================================================

Tests each component of the composite reward:
    R = alpha * vol_efficiency + beta * safety_score - gamma * violations

Author: EcoPackAI Team
"""

from __future__ import annotations

import pytest

from src.packing_env import (
    ALPHA, BETA, GAMMA,
    compute_vol_efficiency,
    compute_safety_score,
    compute_reward,
)


# ═══════════════════════════════════════════════════════════════════════════
# Volume Efficiency Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestVolEfficiency:
    """Tests for compute_vol_efficiency()."""

    def test_perfect_efficiency(self) -> None:
        """100% utilisation → efficiency = 1.0."""
        assert compute_vol_efficiency(1000.0, 1000.0) == 1.0

    def test_half_efficiency(self) -> None:
        """50% utilisation → efficiency = 0.5."""
        assert compute_vol_efficiency(500.0, 1000.0) == pytest.approx(0.5)

    def test_zero_utilisation(self) -> None:
        """Empty bin → efficiency = 0.0."""
        assert compute_vol_efficiency(0.0, 1000.0) == 0.0

    def test_zero_total_volume(self) -> None:
        """No bins opened → efficiency = 0.0 (no division by zero)."""
        assert compute_vol_efficiency(0.0, 0.0) == 0.0

    def test_clamped_above_one(self) -> None:
        """Edge case: used > total should clamp to 1.0."""
        assert compute_vol_efficiency(1500.0, 1000.0) == 1.0

    def test_negative_returns_zero(self) -> None:
        """Negative used volume should clamp to 0.0."""
        assert compute_vol_efficiency(-100.0, 1000.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Safety Score Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSafetyScore:
    """Tests for compute_safety_score()."""

    def test_no_violations(self) -> None:
        """Zero violations → safety = 1.0."""
        assert compute_safety_score(0) == 1.0

    def test_one_violation(self) -> None:
        """Any violations → safety = 0.5."""
        assert compute_safety_score(1) == 0.5

    def test_many_violations(self) -> None:
        """Multiple violations still → 0.5."""
        assert compute_safety_score(10) == 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Composite Reward Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCompositeReward:
    """Tests for compute_reward()."""

    def test_perfect_packing(self) -> None:
        """Perfect packing, no violations → max reward."""
        r = compute_reward(1000.0, 1000.0, constraint_violations=0)
        expected = ALPHA * 1.0 + BETA * 1.0 - GAMMA * 0
        assert r == pytest.approx(expected)
        assert r == pytest.approx(0.9)  # 0.6 + 0.3

    def test_empty_bin_no_violations(self) -> None:
        """Empty bin, no violations → low but positive reward."""
        r = compute_reward(0.0, 1000.0, constraint_violations=0)
        expected = ALPHA * 0.0 + BETA * 1.0 - GAMMA * 0
        assert r == pytest.approx(expected)
        assert r == pytest.approx(0.3)

    def test_violation_penalty(self) -> None:
        """Each violation subtracts gamma from reward."""
        r0 = compute_reward(500.0, 1000.0, constraint_violations=0)
        r1 = compute_reward(500.0, 1000.0, constraint_violations=1)
        assert r0 - r1 == pytest.approx(GAMMA + BETA * 0.5)
        # r0 = 0.6*0.5 + 0.3*1.0 = 0.6
        # r1 = 0.6*0.5 + 0.3*0.5 - 5.0 = -4.55

    def test_negative_reward_possible(self) -> None:
        """Violations can make reward negative."""
        r = compute_reward(0.0, 1000.0, constraint_violations=2)
        assert r < 0

    def test_custom_weights(self) -> None:
        """Custom alpha/beta/gamma should work."""
        r = compute_reward(
            800.0, 1000.0, constraint_violations=0,
            alpha=1.0, beta=0.0, gamma=0.0,
        )
        assert r == pytest.approx(0.8)

    def test_half_efficiency_with_violations(self) -> None:
        """Combined scenario with violations."""
        r = compute_reward(500.0, 1000.0, constraint_violations=1)
        vol_eff = 0.5
        safety = 0.5  # 1 violation → 0.5
        expected = ALPHA * vol_eff + BETA * safety - GAMMA * 1
        assert r == pytest.approx(expected)
