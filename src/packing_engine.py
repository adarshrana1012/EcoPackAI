"""
packing_engine.py — 3D Bin Packing with Fragility Constraints
=============================================================

Implements the First Fit Decreasing (FFD) heuristic for 3D bin packing
with fragility-aware placement rules for the EcoPackAI platform.

Algorithm
---------
1. Sort items descending by volume.
2. For each item, iterate open bins and attempt placement.
3. Check spatial fit AND fragility constraints.
4. If no fit, open a new bin from the available catalogue.

Fragility Constraints (Prompt 16)
---------------------------------
* Critical/High (label 2-3): 5 cm clearance buffer on all sides.
* Critical (label 3): cannot be placed below items of equal or greater weight.
* If no valid placement exists for Critical, flag ``requires_separate_box``.

Author: EcoPackAI Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FRAGILITY_BUFFER_CM: float = 5.0          # buffer for Critical/High items
FRAGILITY_HIGH_THRESHOLD: int = 2         # labels >= this get buffer
FRAGILITY_CRITICAL: int = 3               # cannot be under heavy items


# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Item:
    """A product to be packed.

    Attributes
    ----------
    item_id : str
        Unique identifier for the item.
    length : float
        Length in cm.
    width : float
        Width in cm.
    height : float
        Height in cm.
    weight_g : float
        Weight in grams.
    fragility_label : int
        Fragility tier (0=None, 1=Low, 2=Medium/High, 3=Critical).
    """
    item_id: str
    length: float
    width: float
    height: float
    weight_g: float
    fragility_label: int = 0

    @property
    def volume(self) -> float:
        """Volume in cm^3."""
        return self.length * self.width * self.height

    def orientations(self) -> List[Tuple[float, float, float]]:
        """Return all 6 possible (l, w, h) orientations."""
        dims = [self.length, self.width, self.height]
        seen = set()
        result = []
        from itertools import permutations
        for perm in permutations(dims):
            if perm not in seen:
                seen.add(perm)
                result.append(perm)
        return result


@dataclass
class Box:
    """A shipping box.

    Attributes
    ----------
    sku : str
        Box SKU identifier.
    length : float
        Internal length in cm.
    width : float
        Internal width in cm.
    height : float
        Internal height in cm.
    cost_usd : float
        Cost per box in USD.
    """
    sku: str
    length: float
    width: float
    height: float
    cost_usd: float = 0.0

    @property
    def volume(self) -> float:
        """Internal volume in cm^3."""
        return self.length * self.width * self.height


@dataclass
class Placement:
    """Placement coordinates for a packed item.

    Attributes
    ----------
    item : Item
        The item being placed.
    x, y, z : float
        Bottom-left-back corner coordinates within the box.
    placed_length, placed_width, placed_height : float
        Oriented dimensions as placed.
    """
    item: Item
    x: float
    y: float
    z: float
    placed_length: float
    placed_width: float
    placed_height: float

    @property
    def x_end(self) -> float:
        return self.x + self.placed_length

    @property
    def y_end(self) -> float:
        return self.y + self.placed_width

    @property
    def z_end(self) -> float:
        return self.z + self.placed_height


@dataclass
class PackingResult:
    """Result of a packing operation.

    Attributes
    ----------
    box : Box
        The chosen box.
    placements : list[Placement]
        Where each item was placed.
    unpacked_items : list[Item]
        Items that could not fit.
    void_volume_pct : float
        Percentage of box volume not occupied by items.
    constraint_violations : int
        Number of fragility constraint violations detected.
    requires_separate_box : list[Item]
        Critical items that need their own box.
    requires_split : bool
        True if items exceed all available box sizes.
    """
    box: Box
    placements: List[Placement] = field(default_factory=list)
    unpacked_items: List[Item] = field(default_factory=list)
    void_volume_pct: float = 0.0
    constraint_violations: int = 0
    requires_separate_box: List[Item] = field(default_factory=list)
    requires_split: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# Internal Spatial Tracking
# ═══════════════════════════════════════════════════════════════════════════

class _BinState:
    """Tracks placed items and available spaces within a single box."""

    def __init__(self, box: Box) -> None:
        self.box = box
        self.placements: List[Placement] = []
        self._used_volume: float = 0.0

    @property
    def remaining_volume(self) -> float:
        return self.box.volume - self._used_volume

    def _overlaps(self, x: float, y: float, z: float,
                  l: float, w: float, h: float) -> bool:
        """Check if a proposed placement overlaps any existing item."""
        for p in self.placements:
            if (x < p.x_end and x + l > p.x and
                y < p.y_end and y + w > p.y and
                z < p.z_end and z + h > p.z):
                return True
        return False

    def _check_fragility_constraints(
        self, item: Item, x: float, y: float, z: float,
        l: float, w: float, h: float,
    ) -> Tuple[bool, int]:
        """Check fragility placement constraints.

        Returns
        -------
        tuple[bool, int]
            (is_valid, violation_count)
        """
        violations = 0

        # --- Buffer constraint for High/Critical items ---------------------
        if item.fragility_label >= FRAGILITY_HIGH_THRESHOLD:
            buf = FRAGILITY_BUFFER_CM
            # Check buffer from box walls
            if (x < buf or y < buf or z < buf or
                x + l + buf > self.box.length or
                y + w + buf > self.box.width or
                z + h + buf > self.box.height):
                # Only enforce if there's enough space in the box at all
                if (self.box.length >= l + 2 * buf and
                    self.box.width >= w + 2 * buf and
                    self.box.height >= h + 2 * buf):
                    violations += 1
                    return False, violations

            # Check buffer from other items
            for p in self.placements:
                gap_x = max(0, max(x, p.x) - min(x + l, p.x_end))
                gap_y = max(0, max(y, p.y) - min(y + w, p.y_end))
                gap_z = max(0, max(z, p.z) - min(z + h, p.z_end))

                # If items overlap in 2D, check the gap in the 3rd dimension
                x_overlap = x < p.x_end and x + l > p.x
                y_overlap = y < p.y_end and y + w > p.y
                z_overlap = z < p.z_end and z + h > p.z

                if x_overlap and y_overlap:
                    # Adjacent in Z
                    z_gap = max(0, z - p.z_end) if z >= p.z_end else max(0, p.z - (z + h))
                    if z_gap < buf and z_gap >= 0 and not z_overlap:
                        violations += 1

                if x_overlap and z_overlap:
                    y_gap = max(0, y - p.y_end) if y >= p.y_end else max(0, p.y - (y + w))
                    if y_gap < buf and y_gap >= 0 and not y_overlap:
                        violations += 1

                if y_overlap and z_overlap:
                    x_gap = max(0, x - p.x_end) if x >= p.x_end else max(0, p.x - (x + l))
                    if x_gap < buf and x_gap >= 0 and not x_overlap:
                        violations += 1

        # --- Critical item: cannot be below heavier items ------------------
        if item.fragility_label >= FRAGILITY_CRITICAL:
            item_top = z + h
            for p in self.placements:
                # Check if any existing item is directly above this placement
                if (p.z >= item_top and
                    x < p.x_end and x + l > p.x and
                    y < p.y_end and y + w > p.y):
                    if p.item.weight_g >= item.weight_g:
                        violations += 1
                        return False, violations

            # Also check: would placing this item mean a previously placed
            # Critical item is now below it?
            for p in self.placements:
                if p.item.fragility_label >= FRAGILITY_CRITICAL:
                    if (z >= p.z_end and
                        x < p.x_end and x + l > p.x and
                        y < p.y_end and y + w > p.y):
                        if item.weight_g >= p.item.weight_g:
                            violations += 1
                            return False, violations

        return violations == 0, violations

    def try_place(
        self, item: Item, allow_rotation: bool = False,
    ) -> Optional[Placement]:
        """Attempt to place an item using a grid-scan approach.

        Parameters
        ----------
        item : Item
            Item to place.
        allow_rotation : bool
            If True, try all 6 orientations.

        Returns
        -------
        Placement or None
            The placement if successful, None otherwise.
        """
        orientations = item.orientations() if allow_rotation else [
            (item.length, item.width, item.height)
        ]

        best_placement: Optional[Placement] = None
        best_z: float = float("inf")  # prefer lower placements

        for l, w, h in orientations:
            # Quick volume check
            if l > self.box.length or w > self.box.width or h > self.box.height:
                continue

            # Generate candidate positions: corners of existing items + origin
            candidate_positions = [(0.0, 0.0, 0.0)]
            for p in self.placements:
                candidate_positions.extend([
                    (p.x_end, p.y, p.z),
                    (p.x, p.y_end, p.z),
                    (p.x, p.y, p.z_end),
                    (p.x_end, p.y_end, p.z),
                    (p.x_end, p.y, p.z_end),
                    (p.x, p.y_end, p.z_end),
                ])

            for x, y, z in candidate_positions:
                # Bounds check
                if (x + l > self.box.length + 1e-9 or
                    y + w > self.box.width + 1e-9 or
                    z + h > self.box.height + 1e-9):
                    continue

                if x < 0 or y < 0 or z < 0:
                    continue

                # Overlap check
                if self._overlaps(x, y, z, l, w, h):
                    continue

                # Fragility constraint check
                valid, _ = self._check_fragility_constraints(
                    item, x, y, z, l, w, h,
                )
                if not valid:
                    continue

                # Prefer lowest z, then smallest x, then smallest y
                if z < best_z or (z == best_z and best_placement is not None
                                  and (x < best_placement.x or
                                       (x == best_placement.x and y < best_placement.y))):
                    best_z = z
                    best_placement = Placement(
                        item=item, x=x, y=y, z=z,
                        placed_length=l, placed_width=w, placed_height=h,
                    )

        if best_placement is not None:
            self.placements.append(best_placement)
            self._used_volume += best_placement.item.volume
            return best_placement

        return None


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def pack_order(
    items: List[Item],
    available_boxes: List[Box],
    allow_rotation: bool = False,
) -> PackingResult:
    """Pack items into the smallest suitable box using FFD.

    Parameters
    ----------
    items : list[Item]
        Items to pack (will be sorted by volume descending).
    available_boxes : list[Box]
        Available box SKUs sorted by volume ascending internally.
    allow_rotation : bool
        If True, items may be placed in any of 6 orientations.

    Returns
    -------
    PackingResult
        Contains box choice, placements, void %, constraint violations,
        and any items requiring separate boxes.
    """
    if not items:
        logger.warning("Empty item list passed to pack_order.")
        return PackingResult(
            box=available_boxes[0] if available_boxes else Box("NONE", 0, 0, 0),
            void_volume_pct=100.0,
        )

    if not available_boxes:
        raise ValueError("No available boxes provided.")

    # Sort items by volume descending (FFD)
    sorted_items = sorted(items, key=lambda i: i.volume, reverse=True)

    # Sort boxes by volume ascending
    sorted_boxes = sorted(available_boxes, key=lambda b: (b.volume, b.cost_usd))

    total_item_volume = sum(i.volume for i in sorted_items)

    # Try each box starting from smallest
    for box in sorted_boxes:
        if box.volume < total_item_volume * 0.8:
            # Skip obviously too-small boxes
            continue

        bin_state = _BinState(box)
        unpacked: List[Item] = []
        separate_box_items: List[Item] = []
        violations = 0

        for item in sorted_items:
            placement = bin_state.try_place(item, allow_rotation=allow_rotation)

            if placement is None:
                if item.fragility_label >= FRAGILITY_CRITICAL:
                    separate_box_items.append(item)
                    logger.debug(
                        "Critical item %s requires separate box.", item.item_id
                    )
                else:
                    unpacked.append(item)

        # If all non-critical items are packed, accept this box
        if not unpacked:
            packed_volume = sum(
                p.item.volume for p in bin_state.placements
            )
            void_pct = (1.0 - packed_volume / box.volume) * 100.0

            result = PackingResult(
                box=box,
                placements=bin_state.placements,
                unpacked_items=unpacked,
                void_volume_pct=round(void_pct, 2),
                constraint_violations=violations,
                requires_separate_box=separate_box_items,
                requires_split=False,
            )
            logger.info(
                "Packed %d/%d items into box %s (void=%.1f%%)",
                len(bin_state.placements), len(sorted_items),
                box.sku, void_pct,
            )
            return result

    # No single box fits — return largest box with partial packing
    largest_box = sorted_boxes[-1]
    bin_state = _BinState(largest_box)
    separate_box_items = []

    for item in sorted_items:
        placement = bin_state.try_place(item, allow_rotation=allow_rotation)
        if placement is None:
            if item.fragility_label >= FRAGILITY_CRITICAL:
                separate_box_items.append(item)
            # else stays unpacked

    all_unpacked = [
        item for item in sorted_items
        if not any(p.item.item_id == item.item_id for p in bin_state.placements)
        and item not in separate_box_items
    ]

    packed_volume = sum(p.item.volume for p in bin_state.placements)
    void_pct = ((1.0 - packed_volume / largest_box.volume) * 100.0
                if largest_box.volume > 0 else 100.0)

    result = PackingResult(
        box=largest_box,
        placements=bin_state.placements,
        unpacked_items=all_unpacked,
        void_volume_pct=round(void_pct, 2),
        constraint_violations=0,
        requires_separate_box=separate_box_items,
        requires_split=True,
    )
    logger.warning(
        "Order requires split — packed %d/%d items into %s.",
        len(bin_state.placements), len(sorted_items), largest_box.sku,
    )
    return result
