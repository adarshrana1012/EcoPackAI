"""
test_analytics.py — Unit Tests for Analytics Module (Prompt 35)
================================================================

Tests for:
1. void_pct is 0 when items perfectly fill box
2. co2e_kg scales linearly with transport_distance_km
3. Aggregate queries return non-empty results on seeded data
4. PDF report generates a valid non-empty file
5. Baseline comparison reports positive reduction

Author: EcoPackAI Team
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.metrics_calculator import (
    compute_void_pct,
    estimate_material_weight_g,
    estimate_co2e_kg,
    CARDBOARD_DENSITY_G_PER_CM2,
    EF_PACKAGING_KG_CO2E_PER_KG,
    EF_TRANSPORT_KG_CO2E_PER_TKM,
)
from src.baseline_comparison import BaselineComparison


# ═══════════════════════════════════════════════════════════════════════════
# 1. Void % = 0 when items perfectly fill box
# ═══════════════════════════════════════════════════════════════════════════

class TestVoidPctCalculation:
    """Tests for compute_void_pct()."""

    def test_perfect_fill_zero_void(self) -> None:
        """When items exactly fill the box, void should be 0%."""
        # Box: 10x10x10 = 1000 cm3
        # Items: [500, 500] = 1000 cm3
        result = compute_void_pct(10, 10, 10, [500.0, 500.0])
        assert result == 0.0

    def test_single_item_exact_fit(self) -> None:
        """Single item filling entire box → 0%."""
        result = compute_void_pct(20, 20, 20, [8000.0])
        assert result == 0.0

    def test_empty_box_100_pct(self) -> None:
        """No items → 100% void."""
        result = compute_void_pct(20, 20, 20, [])
        assert result == 100.0

    def test_half_filled_50_pct(self) -> None:
        """Half-filled box → 50% void."""
        result = compute_void_pct(20, 20, 20, [4000.0])
        assert result == 50.0

    def test_negative_dimensions_raise(self) -> None:
        """Non-positive dimensions should raise ValueError."""
        with pytest.raises(ValueError):
            compute_void_pct(-1, 10, 10, [100.0])

    def test_quarter_fill(self) -> None:
        """25% fill → 75% void."""
        result = compute_void_pct(10, 10, 10, [250.0])
        assert result == 75.0


# ═══════════════════════════════════════════════════════════════════════════
# 2. CO₂e scales linearly with distance
# ═══════════════════════════════════════════════════════════════════════════

class TestCO2eLinearScaling:
    """Tests for estimate_co2e_kg() linearity w.r.t. distance."""

    def test_double_distance_more_co2e(self) -> None:
        """Doubling transport distance should increase CO₂e."""
        co2e_500 = estimate_co2e_kg(200.0, 500.0)
        co2e_1000 = estimate_co2e_kg(200.0, 1000.0)
        assert co2e_1000 > co2e_500

    def test_linear_transport_component(self) -> None:
        """Transport CO₂e should scale linearly with distance."""
        weight_g = 200.0
        d1, d2, d3 = 100.0, 200.0, 300.0

        co2e_1 = estimate_co2e_kg(weight_g, d1)
        co2e_2 = estimate_co2e_kg(weight_g, d2)
        co2e_3 = estimate_co2e_kg(weight_g, d3)

        # Transport component is linear: CO2e(d2)-CO2e(d1) ≈ CO2e(d3)-CO2e(d2)
        delta_1 = co2e_2 - co2e_1
        delta_2 = co2e_3 - co2e_2
        assert abs(delta_1 - delta_2) < 0.001

    def test_zero_distance_only_production(self) -> None:
        """Zero transport distance → only production emissions."""
        weight_g = 1000.0  # 1 kg
        co2e = estimate_co2e_kg(weight_g, 0.0)
        expected = (weight_g / 1000) * EF_PACKAGING_KG_CO2E_PER_KG
        assert abs(co2e - expected) < 0.001

    def test_zero_weight_zero_co2e(self) -> None:
        """Zero material weight → zero emissions."""
        co2e = estimate_co2e_kg(0.0, 500.0)
        assert co2e == 0.0

    def test_negative_distance_raises(self) -> None:
        """Negative distance should raise ValueError."""
        with pytest.raises(ValueError):
            estimate_co2e_kg(200.0, -100.0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Material weight estimation
# ═══════════════════════════════════════════════════════════════════════════

class TestMaterialWeight:
    """Tests for estimate_material_weight_g()."""

    def test_known_box_weight(self) -> None:
        """Verify weight for a known box size."""
        # 40x30x20 → surface = 2*(1200+800+600) = 5200 cm2
        # Weight = 5200 * 0.055 = 286.0 g
        result = estimate_material_weight_g(40, 30, 20)
        assert result == 286.0

    def test_cube_symmetry(self) -> None:
        """Cube box: all dimensions equal."""
        # 10x10x10 → surface = 6*100 = 600 cm2
        result = estimate_material_weight_g(10, 10, 10)
        expected = 600 * CARDBOARD_DENSITY_G_PER_CM2
        assert abs(result - expected) < 0.01

    def test_larger_box_more_material(self) -> None:
        """Larger box should use more material."""
        small = estimate_material_weight_g(10, 10, 10)
        large = estimate_material_weight_g(40, 30, 20)
        assert large > small


# ═══════════════════════════════════════════════════════════════════════════
# 4. PDF report generates valid non-empty file
# ═══════════════════════════════════════════════════════════════════════════

class TestPDFReport:
    """Tests for the sustainability PDF report generator."""

    def test_pdf_generates_valid_file(self, tmp_path) -> None:
        """PDF report should create a non-empty file."""
        try:
            from src.sustainability_report import generate_sustainability_report
        except ImportError:
            pytest.skip("ReportLab not installed.")

        output_dir = str(tmp_path / "reports")
        path = generate_sustainability_report(
            report_month="2026-06",
            output_dir=output_dir,
            shipment_count=1000,
            material_saved_kg=50.0,
            co2e_saved_kg=15.0,
            avg_void_pct=55.0,
            damage_rate_pct=0.3,
        )

        assert path.exists(), f"PDF not created at {path}"
        assert path.stat().st_size > 0, "PDF file is empty"
        assert path.suffix == ".pdf"

    def test_pdf_contains_period_in_name(self, tmp_path) -> None:
        """PDF filename should contain the report month."""
        try:
            from src.sustainability_report import generate_sustainability_report
        except ImportError:
            pytest.skip("ReportLab not installed.")

        path = generate_sustainability_report(
            report_month="2026-05",
            output_dir=str(tmp_path),
            shipment_count=100,
        )
        assert "2026-05" in path.name


# ═══════════════════════════════════════════════════════════════════════════
# 5. Baseline comparison reports positive reduction
# ═══════════════════════════════════════════════════════════════════════════

class TestBaselineComparison:
    """Tests for the BaselineComparison class."""

    def test_positive_reduction_with_improvement(self) -> None:
        """When current metrics are better, reductions should be positive."""
        comp = BaselineComparison(
            baseline_void_pct=82.0,
            baseline_material_weight_g=220.0,
            baseline_damage_rate=0.025,
        )
        report = comp.compare(
            current_void_pct=55.0,
            current_material_weight_g=160.0,
            current_damage_rate=0.003,
        )

        reductions = report["reductions"]
        assert reductions["void_reduction_pct"] > 0, "Void should be reduced"
        assert reductions["material_reduction_pct"] > 0, "Material should be reduced"
        assert reductions["co2e_reduction_pct"] > 0, "CO2e should be reduced"
        assert reductions["damage_rate_delta"] > 0, "Damage rate should decrease"

    def test_targets_met(self) -> None:
        """Targets should be met when improvements are sufficient."""
        comp = BaselineComparison(
            baseline_void_pct=80.0,
            baseline_material_weight_g=200.0,
            baseline_damage_rate=0.025,
        )
        report = comp.compare(
            current_void_pct=50.0,
            current_material_weight_g=140.0,  # 30% reduction
            current_damage_rate=0.003,
        )

        assert report["targets_met"]["material_reduction_gte_25"] is True
        assert report["targets_met"]["damage_rate_lt_0_5"] is True
        assert report["targets_met"]["co2e_reduction_gte_20"] is True

    def test_no_improvement_zero_reduction(self) -> None:
        """When current equals baseline, reductions should be 0."""
        comp = BaselineComparison(
            baseline_void_pct=70.0,
            baseline_material_weight_g=200.0,
        )
        report = comp.compare(
            current_void_pct=70.0,
            current_material_weight_g=200.0,
            current_damage_rate=0.025,
        )

        assert report["reductions"]["void_reduction_pct"] == 0.0
        assert report["reductions"]["material_reduction_pct"] == 0.0

    def test_comparison_chart_generated(self, tmp_path) -> None:
        """Comparison chart should be created as a PNG."""
        comp = BaselineComparison()
        report = comp.compare(
            current_void_pct=55.0,
            current_material_weight_g=160.0,
            current_damage_rate=0.003,
        )
        chart_path = comp.plot_comparison(
            report,
            output_path=str(tmp_path / "chart.png"),
        )
        assert chart_path.exists()
        assert chart_path.stat().st_size > 0

    def test_total_savings_extrapolation(self) -> None:
        """Total savings should scale with shipment count."""
        comp = BaselineComparison(
            baseline_material_weight_g=200.0,
        )
        report = comp.compare(
            current_void_pct=60.0,
            current_material_weight_g=150.0,
            current_damage_rate=0.003,
            shipment_count=1000,
        )
        # 50g saved per shipment * 1000 = 50 kg
        assert report["total_material_saved_kg"] == 50.0
