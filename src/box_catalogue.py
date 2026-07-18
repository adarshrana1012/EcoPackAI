"""
box_catalogue.py — Box SKU Catalogue for EcoPackAI
===================================================

Manages a catalogue of available shipping box SKUs and selects the
optimal box for a given set of items.

Author: EcoPackAI Team
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Union

from src.packing_engine import Box, Item, PackingResult, pack_order

logger = logging.getLogger(__name__)

# Default catalogue (embedded)
DEFAULT_CATALOGUE = [
    {"sku": "BOX-XS", "length_cm": 20, "width_cm": 15, "height_cm": 10, "cost_usd": 0.50, "material_type": "corrugated"},
    {"sku": "BOX-S",  "length_cm": 30, "width_cm": 20, "height_cm": 15, "cost_usd": 0.75, "material_type": "corrugated"},
    {"sku": "BOX-M",  "length_cm": 40, "width_cm": 30, "height_cm": 20, "cost_usd": 1.10, "material_type": "corrugated"},
    {"sku": "BOX-L",  "length_cm": 50, "width_cm": 40, "height_cm": 30, "cost_usd": 1.60, "material_type": "corrugated"},
    {"sku": "BOX-XL", "length_cm": 60, "width_cm": 50, "height_cm": 40, "cost_usd": 2.20, "material_type": "corrugated"},
    {"sku": "BOX-XXL","length_cm": 80, "width_cm": 60, "height_cm": 50, "cost_usd": 3.00, "material_type": "double-wall"},
    {"sku": "BOX-F1", "length_cm": 25, "width_cm": 20, "height_cm": 20, "cost_usd": 1.20, "material_type": "foam-lined"},
    {"sku": "BOX-F2", "length_cm": 35, "width_cm": 25, "height_cm": 25, "cost_usd": 1.80, "material_type": "foam-lined"},
    {"sku": "BOX-F3", "length_cm": 45, "width_cm": 35, "height_cm": 30, "cost_usd": 2.50, "material_type": "foam-lined"},
]


class BoxCatalogue:
    """Manages available box SKUs and selects optimal boxes.

    Parameters
    ----------
    catalogue_path : str or Path, optional
        Path to a JSON file with box definitions.  If ``None``, uses
        the built-in default catalogue.

    Examples
    --------
    >>> cat = BoxCatalogue()
    >>> box = cat.select_optimal_box(items)
    >>> cat.list_boxes()
    """

    def __init__(self, catalogue_path: Optional[Union[str, Path]] = None) -> None:
        if catalogue_path is not None:
            path = Path(catalogue_path)
            if not path.exists():
                raise FileNotFoundError(f"Catalogue not found: {path}")
            with open(path) as f:
                raw = json.load(f)
            logger.info("Loaded %d box SKUs from %s", len(raw), path)
        else:
            raw = DEFAULT_CATALOGUE
            logger.info("Using default catalogue (%d SKUs)", len(raw))

        self._boxes: List[Box] = [
            Box(
                sku=entry["sku"],
                length=entry["length_cm"],
                width=entry["width_cm"],
                height=entry["height_cm"],
                cost_usd=entry.get("cost_usd", 0.0),
            )
            for entry in raw
        ]
        self._raw = raw
        # Sort by volume ascending, then cost
        self._boxes.sort(key=lambda b: (b.volume, b.cost_usd))

    @property
    def boxes(self) -> List[Box]:
        """All boxes sorted by volume ascending."""
        return list(self._boxes)

    def list_boxes(self) -> List[dict]:
        """Return box info as a list of dicts."""
        return [
            {
                "sku": b.sku,
                "dimensions": f"{b.length}x{b.width}x{b.height}",
                "volume_cm3": b.volume,
                "cost_usd": b.cost_usd,
            }
            for b in self._boxes
        ]

    def get_box(self, sku: str) -> Optional[Box]:
        """Look up a box by SKU."""
        for b in self._boxes:
            if b.sku == sku:
                return b
        return None

    def select_optimal_box(
        self,
        items: List[Item],
        allow_rotation: bool = False,
    ) -> PackingResult:
        """Find the smallest box that fits all items with constraints.

        Parameters
        ----------
        items : list[Item]
            Items to pack.
        allow_rotation : bool
            Allow item rotation (6 orientations).

        Returns
        -------
        PackingResult
            The packing result with the optimal box.
        """
        return pack_order(
            items=items,
            available_boxes=self._boxes,
            allow_rotation=allow_rotation,
        )

    def save_catalogue(self, path: Union[str, Path]) -> Path:
        """Export the catalogue to a JSON file."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w") as f:
            json.dump(self._raw, f, indent=2)
        logger.info("Catalogue saved to %s", dest)
        return dest


# ---------------------------------------------------------------------------
# Generate default catalogue file
# ---------------------------------------------------------------------------
def create_default_catalogue(output_path: str = "data/box_catalogue.json") -> Path:
    """Write the default box catalogue to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(DEFAULT_CATALOGUE, f, indent=2)
    logger.info("Default catalogue written to %s", path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = create_default_catalogue()
    cat = BoxCatalogue(p)
    for b in cat.list_boxes():
        print(f"  {b['sku']:<10} {b['dimensions']:<20} vol={b['volume_cm3']:>8.0f} cm3  ${b['cost_usd']:.2f}")
