"""
packing_visualizer.py — 3D Packing Visualization
=================================================

Renders a 3D box with packed items as colored rectangular prisms,
colored by fragility tier.

Author: EcoPackAI Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.packing_engine import PackingResult, Placement

logger = logging.getLogger(__name__)

# Fragility tier colors
TIER_COLORS = {
    0: "#4CAF50",   # Green  — None
    1: "#FFC107",   # Yellow — Low
    2: "#FF9800",   # Orange — Medium
    3: "#F44336",   # Red    — Critical
}

TIER_LABELS = {
    0: "None (0)",
    1: "Low (1)",
    2: "Medium (2)",
    3: "Critical (3)",
}


def _draw_box_wireframe(ax: Axes3D, l: float, w: float, h: float) -> None:
    """Draw the outer box as a wireframe."""
    # 12 edges of a rectangular box
    edges = [
        [(0,0,0),(l,0,0)], [(0,w,0),(l,w,0)], [(0,0,h),(l,0,h)], [(0,w,h),(l,w,h)],
        [(0,0,0),(0,w,0)], [(l,0,0),(l,w,0)], [(0,0,h),(0,w,h)], [(l,0,h),(l,w,h)],
        [(0,0,0),(0,0,h)], [(l,0,0),(l,0,h)], [(0,w,0),(0,w,h)], [(l,w,0),(l,w,h)],
    ]
    for start, end in edges:
        ax.plot3D(*zip(start, end), color="#555555", linewidth=0.8, linestyle="--")


def _draw_item(ax: Axes3D, p: Placement, color: str, alpha: float = 0.6) -> None:
    """Draw a single item as a colored rectangular prism."""
    x, y, z = p.x, p.y, p.z
    dx, dy, dz = p.placed_length, p.placed_width, p.placed_height

    # 6 faces of the rectangular prism
    vertices = [
        # Bottom
        [[x,y,z],[x+dx,y,z],[x+dx,y+dy,z],[x,y+dy,z]],
        # Top
        [[x,y,z+dz],[x+dx,y,z+dz],[x+dx,y+dy,z+dz],[x,y+dy,z+dz]],
        # Front
        [[x,y,z],[x+dx,y,z],[x+dx,y,z+dz],[x,y,z+dz]],
        # Back
        [[x,y+dy,z],[x+dx,y+dy,z],[x+dx,y+dy,z+dz],[x,y+dy,z+dz]],
        # Left
        [[x,y,z],[x,y+dy,z],[x,y+dy,z+dz],[x,y,z+dz]],
        # Right
        [[x+dx,y,z],[x+dx,y+dy,z],[x+dx,y+dy,z+dz],[x+dx,y,z+dz]],
    ]

    faces = Poly3DCollection(vertices, alpha=alpha,
                              facecolors=color, edgecolors="#333333",
                              linewidths=0.5)
    ax.add_collection3d(faces)


def visualize_packing(
    result: PackingResult,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 9),
    dpi: int = 150,
) -> Path:
    """Render a 3D visualization of a packing result.

    Parameters
    ----------
    result : PackingResult
        The packing result to visualize.
    output_path : str, optional
        Where to save the PNG.  Defaults to ``eda_output/packing_3d.png``.
    title : str, optional
        Plot title.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Output resolution.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    if output_path is None:
        output_path = "eda_output/packing_3d.png"

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=figsize, facecolor="#1a1a2e")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#16213e")

    box = result.box

    # Draw box wireframe
    _draw_box_wireframe(ax, box.length, box.width, box.height)

    # Draw each item
    for placement in result.placements:
        color = TIER_COLORS.get(placement.item.fragility_label, "#9E9E9E")
        _draw_item(ax, placement, color, alpha=0.65)

    # Labels
    ax.set_xlabel("Length (cm)", fontsize=10, color="white", labelpad=10)
    ax.set_ylabel("Width (cm)", fontsize=10, color="white", labelpad=10)
    ax.set_zlabel("Height (cm)", fontsize=10, color="white", labelpad=10)

    ax.set_xlim(0, box.length)
    ax.set_ylim(0, box.width)
    ax.set_zlim(0, box.height)

    ax.tick_params(colors="white", labelsize=8)

    # Title
    if title is None:
        title = (f"EcoPackAI Packing — Box {box.sku} "
                 f"({box.length}x{box.width}x{box.height} cm)\n"
                 f"Items: {len(result.placements)} | "
                 f"Void: {result.void_volume_pct:.1f}%")
    ax.set_title(title, fontsize=13, color="white", pad=20, fontweight="bold")

    # Legend
    legend_patches = [
        mpatches.Patch(color=TIER_COLORS[t], label=TIER_LABELS[t])
        for t in sorted(TIER_COLORS.keys())
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9,
              facecolor="#0f3460", edgecolor="white", labelcolor="white")

    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    logger.info("Packing visualization saved to %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.packing_engine import Item, Box, pack_order

    # Demo items
    items = [
        Item("A", 15, 10, 8, 500, 0),
        Item("B", 12, 8, 6, 300, 1),
        Item("C", 10, 10, 5, 200, 2),
        Item("D", 8, 6, 4, 150, 3),
        Item("E", 18, 12, 10, 800, 0),
    ]
    boxes = [Box("DEMO", 40, 30, 25, 1.50)]
    result = pack_order(items, boxes)
    path = visualize_packing(result)
    print(f"Saved: {path}")
