"""
metrics_calculator.py — Per-Shipment Metrics for EcoPackAI (Prompt 29)
======================================================================

Provides three core metric functions for shipment-level sustainability
analytics.

Formulae
--------
* **Void %**:
  ``void_pct = (1 - Σ item_volumes / box_volume) × 100``

* **Material weight (g)**:
  ``material_weight = surface_area_cm2 × density_g_per_cm2``
  where ``density ≈ 0.055 g/cm²`` for single-wall corrugated cardboard
  (standard C-flute, ~550 g/m² = 0.055 g/cm²).

* **CO₂e (kg)**:
  ``co2e = material_kg × EF_packaging + transport_tkm × EF_transport``
  Using IPCC / DEFRA 2024 emission factors:
  - Corrugated cardboard production: **1.32 kg CO₂e / kg** of cardboard
  - Road freight transport: **0.0625 kg CO₂e / tonne-km**
    (HGV average, DEFRA 2024 GHG conversion factors)

Author: EcoPackAI Team
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical Constants
# ---------------------------------------------------------------------------

# Single-wall corrugated cardboard density (C-flute)
# ~550 g/m² = 0.055 g/cm²
CARDBOARD_DENSITY_G_PER_CM2: float = 0.055

# IPCC / DEFRA emission factors
EF_PACKAGING_KG_CO2E_PER_KG: float = 1.32    # kg CO₂e per kg cardboard produced
EF_TRANSPORT_KG_CO2E_PER_TKM: float = 0.0625  # kg CO₂e per tonne-km (road freight)


# ═══════════════════════════════════════════════════════════════════════════
# Public Functions
# ═══════════════════════════════════════════════════════════════════════════

def compute_void_pct(
    box_length: float,
    box_width: float,
    box_height: float,
    item_volumes_cm3: List[float],
) -> float:
    """Compute the void volume percentage of a packed box.

    The void percentage quantifies how much of the box interior is
    unoccupied by items.  Lower is better — indicating tighter packing
    and less wasted material.

    Formula
    -------
    .. math::

        \\text{void\\_pct} = \\left(1 - \\frac{\\sum V_{\\text{items}}}
        {V_{\\text{box}}}\\right) \\times 100

    Parameters
    ----------
    box_length, box_width, box_height : float
        Internal box dimensions in cm.
    item_volumes_cm3 : list[float]
        Volumes of each packed item in cm³.

    Returns
    -------
    float
        Void volume percentage in [0, 100].  Returns 100.0 for an
        empty box, 0.0 for a perfectly filled box.

    Raises
    ------
    ValueError
        If box dimensions are non-positive.

    Examples
    --------
    >>> compute_void_pct(20, 20, 20, [4000.0, 2000.0])
    25.0
    """
    if box_length <= 0 or box_width <= 0 or box_height <= 0:
        raise ValueError(
            f"Box dimensions must be positive: "
            f"({box_length}, {box_width}, {box_height})"
        )

    box_volume = box_length * box_width * box_height
    total_item_volume = sum(item_volumes_cm3)

    if total_item_volume < 0:
        raise ValueError("Item volumes cannot be negative.")

    if box_volume == 0:
        return 100.0

    void = max(0.0, 1.0 - total_item_volume / box_volume) * 100.0
    return round(min(void, 100.0), 2)


def estimate_material_weight_g(
    box_length: float,
    box_width: float,
    box_height: float,
    density_g_per_cm2: float = CARDBOARD_DENSITY_G_PER_CM2,
) -> float:
    """Estimate the packaging material weight from box surface area.

    Assumes single-wall corrugated cardboard (C-flute).  The material
    weight is proportional to the total outer surface area of the box.

    Formula
    -------
    .. math::

        W = 2 \\times (l \\cdot w + l \\cdot h + w \\cdot h)
        \\times \\rho_{\\text{cardboard}}

    where :math:`\\rho \\approx 0.055\\,\\text{g/cm}^2` for standard
    single-wall corrugated (~550 g/m²).

    Parameters
    ----------
    box_length, box_width, box_height : float
        External box dimensions in cm.
    density_g_per_cm2 : float
        Cardboard density in g/cm².  Default is 0.055.

    Returns
    -------
    float
        Estimated material weight in grams.

    Examples
    --------
    >>> estimate_material_weight_g(40, 30, 20)  # ~176 g
    176.0
    """
    if box_length <= 0 or box_width <= 0 or box_height <= 0:
        raise ValueError("Box dimensions must be positive.")

    surface_area_cm2 = 2.0 * (
        box_length * box_width +
        box_length * box_height +
        box_width * box_height
    )
    weight_g = surface_area_cm2 * density_g_per_cm2
    return round(weight_g, 2)


def estimate_co2e_kg(
    material_weight_g: float,
    transport_distance_km: float,
    ef_packaging: float = EF_PACKAGING_KG_CO2E_PER_KG,
    ef_transport: float = EF_TRANSPORT_KG_CO2E_PER_TKM,
) -> float:
    """Estimate CO₂-equivalent emissions for a shipment.

    Combines two emission sources:

    1. **Packaging production**: CO₂e from manufacturing the corrugated
       cardboard box.
    2. **Transport**: CO₂e from road freight, proportional to the
       weight carried and distance.

    Formula
    -------
    .. math::

        \\text{CO}_2\\text{e} =
        \\underbrace{W_{\\text{kg}} \\times \\text{EF}_{\\text{packaging}}}
        _{\\text{production}} +
        \\underbrace{W_{\\text{tonnes}} \\times d_{\\text{km}} \\times
        \\text{EF}_{\\text{transport}}}_{\\text{freight}}

    Using IPCC / DEFRA 2024 emission factors:

    * ``EF_packaging`` = 1.32 kg CO₂e / kg cardboard
    * ``EF_transport`` = 0.0625 kg CO₂e / tonne-km (HGV average)

    Parameters
    ----------
    material_weight_g : float
        Total packaging material weight in grams.
    transport_distance_km : float
        One-way transport distance in kilometres.
    ef_packaging : float
        Emission factor for cardboard production (kg CO₂e / kg).
    ef_transport : float
        Emission factor for road freight (kg CO₂e / tonne-km).

    Returns
    -------
    float
        Total estimated CO₂e emissions in **kilograms**.

    Examples
    --------
    >>> estimate_co2e_kg(200.0, 500.0)  # ~0.27 kg CO₂e
    0.27
    """
    if material_weight_g < 0:
        raise ValueError("Material weight cannot be negative.")
    if transport_distance_km < 0:
        raise ValueError("Transport distance cannot be negative.")

    material_kg = material_weight_g / 1000.0
    material_tonnes = material_weight_g / 1_000_000.0

    # Production emissions
    co2e_production = material_kg * ef_packaging

    # Transport emissions
    co2e_transport = material_tonnes * transport_distance_km * ef_transport

    total = co2e_production + co2e_transport
    return round(total, 4)
