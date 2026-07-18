"""
test_packing_engine.py — Unit Tests for the 3D Bin Packing Engine
==================================================================

Covers the five required test cases plus additional edge cases.

Author: EcoPackAI Team
"""

from __future__ import annotations

import pytest

from src.packing_engine import (
    Box,
    FRAGILITY_CRITICAL,
    Item,
    Placement,
    PackingResult,
    pack_order,
)
from src.box_catalogue import BoxCatalogue


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_boxes() -> list[Box]:
    """Standard set of test boxes, ascending by volume."""
    return [
        Box("SMALL", 20, 15, 10, 0.50),
        Box("MEDIUM", 40, 30, 20, 1.10),
        Box("LARGE", 60, 50, 40, 2.20),
    ]


@pytest.fixture()
def catalogue() -> BoxCatalogue:
    """BoxCatalogue with default SKUs."""
    return BoxCatalogue()


# ═══════════════════════════════════════════════════════════════════════════
# 1. Single item fits in smallest valid box
# ═══════════════════════════════════════════════════════════════════════════

class TestSingleItemFit:
    """A single small item should be packed into the smallest box."""

    def test_single_item_chooses_smallest_box(self, default_boxes) -> None:
        """One small item should fit in the SMALL box."""
        item = Item("single", 10, 8, 5, 200, fragility_label=0)
        result = pack_order([item], default_boxes)

        assert result.box.sku == "SMALL"
        assert len(result.placements) == 1
        assert not result.requires_split

    def test_item_larger_than_small_box_uses_medium(self, default_boxes) -> None:
        """An item that doesn't fit in SMALL should use MEDIUM."""
        item = Item("bigger", 35, 25, 15, 800, fragility_label=0)
        result = pack_order([item], default_boxes)

        assert result.box.sku == "MEDIUM"
        assert len(result.placements) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 2. Critical item is never placed at the bottom under heavy items
# ═══════════════════════════════════════════════════════════════════════════

class TestCriticalPlacement:
    """Critical items must not be placed under items of equal/greater weight."""

    def test_critical_item_not_under_heavy(self) -> None:
        """A Critical item should not be placed below a heavier item."""
        # Use a box large enough for both items stacked
        box = Box("TEST", 30, 30, 40, 1.0)

        # Critical item (light) + heavy non-critical item
        critical_item = Item("fragile", 20, 20, 10, 200, fragility_label=3)
        heavy_item = Item("heavy", 20, 20, 10, 5000, fragility_label=0)

        result = pack_order([heavy_item, critical_item], [box])

        # Find the critical item's placement
        critical_placement = None
        heavy_placement = None
        for p in result.placements:
            if p.item.item_id == "fragile":
                critical_placement = p
            elif p.item.item_id == "heavy":
                heavy_placement = p

        if critical_placement and heavy_placement:
            # If both are placed and overlap in X/Y, critical must be on top
            x_overlap = (critical_placement.x < heavy_placement.x_end and
                         critical_placement.x_end > heavy_placement.x)
            y_overlap = (critical_placement.y < heavy_placement.y_end and
                         critical_placement.y_end > heavy_placement.y)

            if x_overlap and y_overlap:
                assert critical_placement.z >= heavy_placement.z_end, (
                    f"Critical item z={critical_placement.z} is below "
                    f"heavy item z_end={heavy_placement.z_end}"
                )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Order exceeding all box sizes returns requires_split=True
# ═══════════════════════════════════════════════════════════════════════════

class TestOversizedOrder:
    """Orders that exceed all available boxes must flag requires_split."""

    def test_oversized_order_requires_split(self) -> None:
        """Many large items that can't fit in any single box."""
        tiny_boxes = [Box("TINY", 10, 10, 10, 0.25)]

        items = [
            Item(f"big-{i}", 15, 15, 15, 500, fragility_label=0)
            for i in range(5)
        ]

        result = pack_order(items, tiny_boxes)
        assert result.requires_split is True

    def test_single_oversized_item(self) -> None:
        """A single item bigger than all boxes."""
        small_boxes = [Box("MINI", 5, 5, 5, 0.10)]
        item = Item("huge", 50, 50, 50, 10000, fragility_label=0)

        result = pack_order([item], small_boxes)
        assert result.requires_split is True


# ═══════════════════════════════════════════════════════════════════════════
# 4. void_volume_pct is calculated correctly
# ═══════════════════════════════════════════════════════════════════════════

class TestVoidCalculation:
    """Void volume percentage must be mathematically correct."""

    def test_known_void_volume(self) -> None:
        """Pack a known-size item in a known-size box and verify void %."""
        # Box: 20x20x20 = 8000 cm3
        box = Box("EXACT", 20, 20, 20, 1.0)
        # Item: 10x10x10 = 1000 cm3
        item = Item("cube", 10, 10, 10, 500, fragility_label=0)

        result = pack_order([item], [box])

        # Expected void = (8000 - 1000) / 8000 * 100 = 87.5%
        expected_void = (1.0 - 1000 / 8000) * 100
        assert abs(result.void_volume_pct - expected_void) < 0.1, (
            f"Expected {expected_void:.1f}%, got {result.void_volume_pct:.1f}%"
        )

    def test_two_items_void_volume(self) -> None:
        """Two items: void should account for total packed volume."""
        box = Box("DUAL", 20, 20, 20, 1.0)
        item1 = Item("a", 10, 10, 10, 300, fragility_label=0)
        item2 = Item("b", 10, 10, 10, 300, fragility_label=0)

        result = pack_order([item1, item2], [box])

        # 2 x 1000 = 2000 packed out of 8000 -> 75% void
        if len(result.placements) == 2:
            expected_void = (1.0 - 2000 / 8000) * 100
            assert abs(result.void_volume_pct - expected_void) < 0.1


# ═══════════════════════════════════════════════════════════════════════════
# 5. All coordinates are within box bounds
# ═══════════════════════════════════════════════════════════════════════════

class TestBoundsCheck:
    """All (x,y,z) placements must be within the box dimensions."""

    def test_all_placements_within_bounds(self, default_boxes) -> None:
        """Every placement must be non-negative and within box limits."""
        items = [
            Item("i1", 10, 8, 5, 200, 0),
            Item("i2", 12, 10, 6, 350, 1),
            Item("i3", 8, 6, 4, 150, 0),
            Item("i4", 15, 12, 8, 500, 2),
        ]

        result = pack_order(items, default_boxes)

        for p in result.placements:
            assert p.x >= 0, f"x={p.x} is negative"
            assert p.y >= 0, f"y={p.y} is negative"
            assert p.z >= 0, f"z={p.z} is negative"

            assert p.x_end <= result.box.length + 1e-9, (
                f"x_end={p.x_end} exceeds box length={result.box.length}"
            )
            assert p.y_end <= result.box.width + 1e-9, (
                f"y_end={p.y_end} exceeds box width={result.box.width}"
            )
            assert p.z_end <= result.box.height + 1e-9, (
                f"z_end={p.z_end} exceeds box height={result.box.height}"
            )

    def test_no_placements_overlap(self, default_boxes) -> None:
        """No two placed items should occupy the same space."""
        items = [
            Item("o1", 10, 8, 5, 200, 0),
            Item("o2", 12, 10, 6, 350, 0),
            Item("o3", 8, 6, 4, 150, 0),
        ]
        result = pack_order(items, default_boxes)

        placements = result.placements
        for i in range(len(placements)):
            for j in range(i + 1, len(placements)):
                a, b = placements[i], placements[j]
                x_overlap = a.x < b.x_end and a.x_end > b.x
                y_overlap = a.y < b.y_end and a.y_end > b.y
                z_overlap = a.z < b.z_end and a.z_end > b.z

                assert not (x_overlap and y_overlap and z_overlap), (
                    f"Items {a.item.item_id} and {b.item.item_id} overlap"
                )


# ═══════════════════════════════════════════════════════════════════════════
# Additional edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Additional robustness tests."""

    def test_empty_item_list(self, default_boxes) -> None:
        """Empty item list should return a result without crashing."""
        result = pack_order([], default_boxes)
        assert result.void_volume_pct == 100.0
        assert len(result.placements) == 0

    def test_rotation_packs_more_items(self) -> None:
        """Rotation should enable packing items that don't fit without it."""
        # Box: 30x10x10 — long and narrow
        box = Box("NARROW", 30, 10, 10, 1.0)
        # Item: 10x10x25 — only fits rotated as 25x10x10
        item = Item("tall", 10, 10, 25, 400, fragility_label=0)

        result_no_rot = pack_order([item], [box], allow_rotation=False)
        result_rot = pack_order([item], [box], allow_rotation=True)

        # Without rotation it may not fit; with rotation it should
        if len(result_no_rot.placements) == 0:
            assert len(result_rot.placements) == 1

    def test_catalogue_select_optimal(self, catalogue) -> None:
        """BoxCatalogue.select_optimal_box should return valid result."""
        items = [Item("cat-1", 15, 10, 8, 300, 0)]
        result = catalogue.select_optimal_box(items)

        assert result.box is not None
        assert len(result.placements) == 1
        assert not result.requires_split
