"""
packing_env.py — Custom Gymnasium Environment for RL Packing (Prompts 22-23)
=============================================================================

Implements ``PackingEnv(gymnasium.Env)`` for training RL agents to optimize
3D bin packing decisions.  Compatible with Stable Baselines3.

State Vector (12-dim)
---------------------
* ``used_volume``        — fraction of current bin volume used
* ``item_count``         — number of items placed so far
* ``fragility_count[4]`` — count of items per fragility tier in current bin
* ``next_length``        — next item's length (normalised)
* ``next_width``         — next item's width  (normalised)
* ``next_height``        — next item's height (normalised)
* ``next_weight``        — next item's weight (normalised)
* ``next_fragility``     — next item's fragility label (0-3)
* ``next_volume``        — next item's volume (normalised)

Action Space
------------
Discrete — index selects which open bin to place into, or opens a new bin.

Reward (Prompt 23)
------------------
``R = alpha * vol_efficiency + beta * safety_score - gamma * violations``

Author: EcoPackAI Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.packing_engine import (
    Box,
    Item,
    PackingResult,
    _BinState,
    FRAGILITY_CRITICAL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reward Constants (Prompt 23)
# ---------------------------------------------------------------------------
ALPHA: float = 0.6    # volume efficiency weight
BETA: float = 0.3     # safety score weight
GAMMA: float = 5.0    # violation penalty weight


# ═══════════════════════════════════════════════════════════════════════════
# Reward Function (Prompt 23)
# ═══════════════════════════════════════════════════════════════════════════

def compute_vol_efficiency(used_volume: float, total_volume: float) -> float:
    """Compute volume efficiency: 1 - void_volume_pct / 100.

    Parameters
    ----------
    used_volume : float
        Total volume of packed items (cm^3).
    total_volume : float
        Total volume of all bins used (cm^3).

    Returns
    -------
    float
        Volume efficiency in [0, 1].  Higher is better.
    """
    if total_volume <= 0:
        return 0.0
    return min(1.0, max(0.0, used_volume / total_volume))


def compute_safety_score(constraint_violations: int) -> float:
    """Safety score: 1.0 if no violations, 0.5 otherwise.

    Parameters
    ----------
    constraint_violations : int
        Number of fragility constraint violations in the episode.

    Returns
    -------
    float
        1.0 or 0.5.
    """
    return 1.0 if constraint_violations == 0 else 0.5


def compute_reward(
    used_volume: float,
    total_volume: float,
    constraint_violations: int,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> float:
    """Compute the composite reward.

    ``R = alpha * vol_efficiency + beta * safety_score - gamma * violations``

    Parameters
    ----------
    used_volume : float
        Total volume of packed items (cm^3).
    total_volume : float
        Total volume of all bins used (cm^3).
    constraint_violations : int
        Number of fragility constraint violations.
    alpha, beta, gamma : float
        Weight parameters.

    Returns
    -------
    float
        Composite reward value.
    """
    vol_eff = compute_vol_efficiency(used_volume, total_volume)
    safety = compute_safety_score(constraint_violations)
    reward = alpha * vol_eff + beta * safety - gamma * constraint_violations
    return reward


# ═══════════════════════════════════════════════════════════════════════════
# Normalisation helpers
# ═══════════════════════════════════════════════════════════════════════════
_MAX_DIM = 80.0        # max box dimension
_MAX_WEIGHT = 10000.0  # max item weight (grams)
_MAX_VOLUME = 240000.0 # max box volume
_MAX_ITEMS = 20        # max items per episode


# ═══════════════════════════════════════════════════════════════════════════
# Gymnasium Environment (Prompt 22)
# ═══════════════════════════════════════════════════════════════════════════

class PackingEnv(gym.Env):
    """Gymnasium environment for 3D bin packing optimisation.

    Compatible with Stable Baselines3 algorithms (PPO, A2C, DQN, etc.).

    Parameters
    ----------
    items_pool : list[Item]
        Pool of items to sample orders from.
    available_boxes : list[Box]
        Available box SKUs.
    max_bins : int
        Maximum number of bins the agent can open.
    items_per_episode : int
        Number of items in each episode (order).
    seed : int, optional
        Random seed for reproducibility.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        items_pool: List[Item],
        available_boxes: List[Box],
        max_bins: int = 5,
        items_per_episode: int = 8,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.items_pool = items_pool
        self.available_boxes = sorted(available_boxes, key=lambda b: b.volume)
        self.max_bins = max_bins
        self.items_per_episode = items_per_episode
        self.render_mode = render_mode

        # --- Observation space: 12-dim continuous vector ---
        # [used_vol_frac, item_count_norm, frag_0, frag_1, frag_2, frag_3,
        #  next_l, next_w, next_h, next_wt, next_frag, next_vol]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32,
        )

        # --- Action space: select bin index (0..max_bins-1) or open new ---
        # Action i means "place into bin i".  If bin i doesn't exist yet,
        # the agent is implicitly opening a new bin.
        self.action_space = spaces.Discrete(max_bins)

        # Episode state
        self._rng = np.random.RandomState(seed)
        self._current_items: List[Item] = []
        self._item_idx: int = 0
        self._bins: List[_BinState] = []
        self._total_violations: int = 0
        self._total_packed_volume: float = 0.0
        self._total_bin_volume: float = 0.0
        self._step_count: int = 0

    # -------------------------------------------------------------------
    # Observation builder
    # -------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Build the 12-dim observation vector."""
        # Aggregate bin state
        used_vol_frac = (
            self._total_packed_volume / self._total_bin_volume
            if self._total_bin_volume > 0 else 0.0
        )
        item_count = sum(len(b.placements) for b in self._bins)
        item_count_norm = min(item_count / _MAX_ITEMS, 1.0)

        frag_counts = [0, 0, 0, 0]
        for b in self._bins:
            for p in b.placements:
                fl = min(p.item.fragility_label, 3)
                frag_counts[fl] += 1
        frag_norm = [min(c / _MAX_ITEMS, 1.0) for c in frag_counts]

        # Next item features
        if self._item_idx < len(self._current_items):
            item = self._current_items[self._item_idx]
            next_l = item.length / _MAX_DIM
            next_w = item.width / _MAX_DIM
            next_h = item.height / _MAX_DIM
            next_wt = item.weight_g / _MAX_WEIGHT
            next_frag = item.fragility_label / 3.0
            next_vol = item.volume / _MAX_VOLUME
        else:
            next_l = next_w = next_h = next_wt = next_frag = next_vol = 0.0

        obs = np.array([
            used_vol_frac, item_count_norm,
            frag_norm[0], frag_norm[1], frag_norm[2], frag_norm[3],
            next_l, next_w, next_h, next_wt, next_frag, next_vol,
        ], dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)

    # -------------------------------------------------------------------
    # Core Gym methods
    # -------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode.

        Samples a random order from the items pool.
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        # Sample items for this episode
        n = min(self.items_per_episode, len(self.items_pool))
        indices = self._rng.choice(len(self.items_pool), size=n, replace=False)
        self._current_items = [self.items_pool[i] for i in indices]

        # Sort by volume descending (FFD heuristic hint)
        self._current_items.sort(key=lambda i: i.volume, reverse=True)

        # Reset state
        self._item_idx = 0
        self._bins = []
        self._total_violations = 0
        self._total_packed_volume = 0.0
        self._total_bin_volume = 0.0
        self._step_count = 0

        return self._get_obs(), {"items": len(self._current_items)}

    def step(
        self, action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Place the current item into the selected bin.

        Parameters
        ----------
        action : int
            Bin index (0..max_bins-1).  If the bin doesn't exist, a new
            bin is opened using the smallest available box.

        Returns
        -------
        tuple
            ``(observation, reward, terminated, truncated, info)``
        """
        self._step_count += 1
        truncated = False

        if self._item_idx >= len(self._current_items):
            # Episode already done
            return self._get_obs(), 0.0, True, False, {}

        item = self._current_items[self._item_idx]
        step_violations = 0
        placed = False

        # Clamp action to valid range
        action = max(0, min(action, self.max_bins - 1))

        # Open new bin if needed
        if action >= len(self._bins):
            # Select the smallest box that can fit this item
            selected_box = None
            for box in self.available_boxes:
                if (box.length >= item.length and
                    box.width >= item.width and
                    box.height >= item.height):
                    selected_box = box
                    break
            if selected_box is None:
                selected_box = self.available_boxes[-1]  # largest

            new_bin = _BinState(selected_box)
            self._bins.append(new_bin)
            self._total_bin_volume += selected_box.volume
            action = len(self._bins) - 1

        # Try to place item in selected bin
        bin_state = self._bins[action]
        placement = bin_state.try_place(item, allow_rotation=True)

        if placement is not None:
            self._total_packed_volume += item.volume
            placed = True
        else:
            # Couldn't fit — try opening a new bin
            if len(self._bins) < self.max_bins:
                selected_box = self.available_boxes[-1]
                new_bin = _BinState(selected_box)
                placement = new_bin.try_place(item, allow_rotation=True)
                if placement is not None:
                    self._bins.append(new_bin)
                    self._total_bin_volume += selected_box.volume
                    self._total_packed_volume += item.volume
                    placed = True
                else:
                    step_violations += 1
            else:
                step_violations += 1

        self._total_violations += step_violations
        self._item_idx += 1

        # Check if episode is done
        terminated = self._item_idx >= len(self._current_items)

        # Compute step reward
        reward = compute_reward(
            used_volume=self._total_packed_volume,
            total_volume=self._total_bin_volume,
            constraint_violations=self._total_violations,
        )

        # Scale reward per step (give partial credit)
        if not terminated:
            reward *= 0.1  # small intermediate reward

        info = {
            "placed": placed,
            "step_violations": step_violations,
            "total_violations": self._total_violations,
            "bins_used": len(self._bins),
            "items_packed": self._item_idx,
            "vol_efficiency": compute_vol_efficiency(
                self._total_packed_volume, self._total_bin_volume
            ),
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """Render current environment state."""
        if self.render_mode == "ansi":
            lines = [
                f"Step {self._step_count} | "
                f"Items: {self._item_idx}/{len(self._current_items)} | "
                f"Bins: {len(self._bins)} | "
                f"Vol eff: {compute_vol_efficiency(self._total_packed_volume, self._total_bin_volume):.1%} | "
                f"Violations: {self._total_violations}",
            ]
            for i, b in enumerate(self._bins):
                lines.append(
                    f"  Bin {i} ({b.box.sku}): "
                    f"{len(b.placements)} items, "
                    f"used={b._used_volume:.0f}/{b.box.volume:.0f} cm3"
                )
            output = "\n".join(lines)
            print(output)
            return output
        elif self.render_mode == "human":
            self.render_mode = "ansi"
            return self.render()
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass


# ---------------------------------------------------------------------------
# Convenience: build env from CSV data
# ---------------------------------------------------------------------------
def make_packing_env(
    data_path: str = "data/train.csv",
    items_per_episode: int = 8,
    max_bins: int = 5,
    seed: int = 42,
) -> PackingEnv:
    """Create a PackingEnv from a training CSV.

    Parameters
    ----------
    data_path : str
        Path to the dataset CSV.
    items_per_episode : int
        Items per episode.
    max_bins : int
        Max bins the agent can open.
    seed : int
        Random seed.

    Returns
    -------
    PackingEnv
    """
    import pandas as pd
    from src.box_catalogue import BoxCatalogue

    df = pd.read_csv(data_path)
    items = [
        Item(
            item_id=str(row.get("product_id", f"item-{idx}")),
            length=float(row["length_cm"]),
            width=float(row["width_cm"]),
            height=float(row["height_cm"]),
            weight_g=float(row["weight_g"]),
            fragility_label=int(row["fragility_label"]),
        )
        for idx, row in df.iterrows()
    ]

    catalogue = BoxCatalogue()

    env = PackingEnv(
        items_pool=items,
        available_boxes=catalogue.boxes,
        max_bins=max_bins,
        items_per_episode=items_per_episode,
        seed=seed,
    )
    logger.info(
        "PackingEnv created: %d items in pool, %d boxes, %d items/episode",
        len(items), len(catalogue.boxes), items_per_episode,
    )
    return env
