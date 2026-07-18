"""
baseline_comparison.py — Pre/Post Deployment Comparison (Prompt 33)
====================================================================

Compares pre-EcoPackAI (manual packing) metrics against post-deployment
performance and generates a bar chart visualisation.

Author: EcoPackAI Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.metrics_calculator import (
    compute_void_pct,
    estimate_material_weight_g,
    estimate_co2e_kg,
)

logger = logging.getLogger(__name__)


class BaselineComparison:
    """Compare pre-EcoPackAI vs post-deployment packing metrics.

    Parameters
    ----------
    baseline_void_pct : float
        Average void % under manual packing (pre-deployment).
    baseline_damage_rate : float
        Damage rate under manual packing (fraction, e.g. 0.02 = 2%).
    baseline_avg_distance_km : float
        Average transport distance for baseline period.

    Examples
    --------
    >>> comp = BaselineComparison(baseline_void_pct=82.0)
    >>> report = comp.compare(current_void_pct=55.0, ...)
    >>> comp.plot_comparison(report)
    """

    def __init__(
        self,
        baseline_void_pct: float = 82.0,
        baseline_damage_rate: float = 0.025,
        baseline_avg_distance_km: float = 500.0,
        baseline_material_weight_g: float = 220.0,
    ) -> None:
        self.baseline_void_pct = baseline_void_pct
        self.baseline_damage_rate = baseline_damage_rate
        self.baseline_avg_distance_km = baseline_avg_distance_km
        self.baseline_material_weight_g = baseline_material_weight_g

        # Compute baseline CO2e
        self.baseline_co2e_kg = estimate_co2e_kg(
            baseline_material_weight_g, baseline_avg_distance_km,
        )

    def compare(
        self,
        current_void_pct: float,
        current_material_weight_g: float,
        current_damage_rate: float,
        current_avg_distance_km: Optional[float] = None,
        shipment_count: int = 1,
    ) -> Dict[str, Any]:
        """Compute comparison metrics between baseline and current.

        Parameters
        ----------
        current_void_pct : float
            Current average void %.
        current_material_weight_g : float
            Current average material weight per shipment (g).
        current_damage_rate : float
            Current damage rate (fraction).
        current_avg_distance_km : float, optional
            Current avg transport distance.
        shipment_count : int
            Number of shipments to extrapolate.

        Returns
        -------
        dict
            Comparison report with reduction percentages.
        """
        distance = current_avg_distance_km or self.baseline_avg_distance_km
        current_co2e_kg = estimate_co2e_kg(current_material_weight_g, distance)

        # Reduction calculations
        void_reduction_pct = (
            (self.baseline_void_pct - current_void_pct)
            / self.baseline_void_pct * 100
        ) if self.baseline_void_pct > 0 else 0

        material_reduction_pct = (
            (self.baseline_material_weight_g - current_material_weight_g)
            / self.baseline_material_weight_g * 100
        ) if self.baseline_material_weight_g > 0 else 0

        co2e_reduction_pct = (
            (self.baseline_co2e_kg - current_co2e_kg)
            / self.baseline_co2e_kg * 100
        ) if self.baseline_co2e_kg > 0 else 0

        damage_rate_delta = self.baseline_damage_rate - current_damage_rate

        report = {
            "period": "pre-vs-post-ecopackai",
            "shipment_count": shipment_count,
            "baseline": {
                "void_pct": round(self.baseline_void_pct, 2),
                "material_weight_g": round(self.baseline_material_weight_g, 2),
                "co2e_kg": round(self.baseline_co2e_kg, 4),
                "damage_rate": round(self.baseline_damage_rate, 4),
            },
            "current": {
                "void_pct": round(current_void_pct, 2),
                "material_weight_g": round(current_material_weight_g, 2),
                "co2e_kg": round(current_co2e_kg, 4),
                "damage_rate": round(current_damage_rate, 4),
            },
            "reductions": {
                "void_reduction_pct": round(void_reduction_pct, 2),
                "material_reduction_pct": round(material_reduction_pct, 2),
                "co2e_reduction_pct": round(co2e_reduction_pct, 2),
                "damage_rate_delta": round(damage_rate_delta, 4),
            },
            "targets_met": {
                "material_reduction_gte_25": material_reduction_pct >= 25,
                "damage_rate_lt_0_5": current_damage_rate < 0.005,
                "co2e_reduction_gte_20": co2e_reduction_pct >= 20,
            },
            "total_material_saved_kg": round(
                (self.baseline_material_weight_g - current_material_weight_g)
                * shipment_count / 1000, 2
            ),
            "total_co2e_saved_kg": round(
                (self.baseline_co2e_kg - current_co2e_kg)
                * shipment_count, 4
            ),
        }

        logger.info(
            "Comparison: void %.1f%% → %.1f%% (↓%.1f%%), "
            "material %.0fg → %.0fg (↓%.1f%%), "
            "CO₂e %.4f → %.4f kg (↓%.1f%%)",
            self.baseline_void_pct, current_void_pct, void_reduction_pct,
            self.baseline_material_weight_g, current_material_weight_g,
            material_reduction_pct,
            self.baseline_co2e_kg, current_co2e_kg, co2e_reduction_pct,
        )

        return report

    def plot_comparison(
        self,
        report: Dict[str, Any],
        output_path: str = "eda_output/baseline_comparison.png",
        figsize: tuple = (12, 6),
        dpi: int = 150,
    ) -> Path:
        """Generate a bar chart comparing baseline vs current metrics.

        Parameters
        ----------
        report : dict
            Output from :meth:`compare`.
        output_path : str
            Where to save the PNG.
        figsize : tuple
            Figure size.
        dpi : int
            Resolution.

        Returns
        -------
        Path
            Path to the saved chart.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        baseline = report["baseline"]
        current = report["current"]
        reductions = report["reductions"]

        metrics = ["Void %", "Material (g)", "CO₂e (kg×1000)", "Damage Rate %"]
        baseline_vals = [
            baseline["void_pct"],
            baseline["material_weight_g"],
            baseline["co2e_kg"] * 1000,
            baseline["damage_rate"] * 100,
        ]
        current_vals = [
            current["void_pct"],
            current["material_weight_g"],
            current["co2e_kg"] * 1000,
            current["damage_rate"] * 100,
        ]

        fig, ax = plt.subplots(figsize=figsize, facecolor="#1a1a2e")
        ax.set_facecolor("#16213e")

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax.bar(x - width / 2, baseline_vals, width,
                       label="Baseline (Manual)",
                       color="#e74c3c", alpha=0.85, edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x + width / 2, current_vals, width,
                       label="EcoPackAI",
                       color="#27ae60", alpha=0.85, edgecolor="white", linewidth=0.5)

        # Add reduction labels
        reduction_vals = [
            reductions["void_reduction_pct"],
            reductions["material_reduction_pct"],
            reductions["co2e_reduction_pct"],
            reductions["damage_rate_delta"] * 100,
        ]
        for i, (b1, b2, red) in enumerate(zip(bars1, bars2, reduction_vals)):
            height = max(b1.get_height(), b2.get_height())
            ax.text(
                x[i], height * 1.05,
                f"↓{red:.1f}%",
                ha="center", fontsize=10, fontweight="bold",
                color="#f39c12",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11, color="white")
        ax.tick_params(axis="y", colors="white")
        ax.set_ylabel("Value", fontsize=12, color="white")
        ax.set_title(
            "EcoPackAI vs Manual Packing Baseline",
            fontsize=15, fontweight="bold", color="white", pad=15,
        )
        ax.legend(fontsize=11, facecolor="#0f3460", edgecolor="white",
                  labelcolor="white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("white")
        ax.spines["bottom"].set_color("white")

        plt.tight_layout()
        plt.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info("Baseline comparison chart saved to %s", out)
        return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    comp = BaselineComparison(baseline_void_pct=82.0, baseline_material_weight_g=220.0)
    report = comp.compare(
        current_void_pct=55.0,
        current_material_weight_g=160.0,
        current_damage_rate=0.003,
        shipment_count=5000,
    )
    import json
    print(json.dumps(report, indent=2))
    chart_path = comp.plot_comparison(report)
    print(f"Chart: {chart_path}")
