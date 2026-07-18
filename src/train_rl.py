"""
train_rl.py — PPO Training Script for EcoPackAI Packing Policy (Prompt 24)
===========================================================================

Trains a PPO agent via Stable Baselines3 on the PackingEnv.

Usage
-----
    python -m src.train_rl                       # default 100K steps
    python -m src.train_rl --timesteps 1000000   # full training

Author: EcoPackAI Team
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TIMESTEPS = 100_000       # use 1_000_000 for full training
DEFAULT_LR = 3e-4
DEFAULT_N_STEPS = 2048
DEFAULT_BATCH_SIZE = 64
CHECKPOINT_FREQ = 100_000
MODEL_DIR = Path("models")
LOG_DIR = Path("logs/tensorboard")


def train_ppo(
    timesteps: int = DEFAULT_TIMESTEPS,
    lr: float = DEFAULT_LR,
    n_steps: int = DEFAULT_N_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    data_path: str = "data/train.csv",
    version: str = "1",
    seed: int = 42,
    checkpoint_freq: int = CHECKPOINT_FREQ,
) -> Path:
    """Train a PPO agent on the PackingEnv.

    Parameters
    ----------
    timesteps : int
        Total training timesteps.
    lr : float
        Learning rate.
    n_steps : int
        Steps per rollout buffer collection.
    batch_size : int
        Minibatch size for PPO updates.
    data_path : str
        Path to training CSV.
    version : str
        Model version string.
    seed : int
        Random seed.
    checkpoint_freq : int
        Save checkpoint every N steps.

    Returns
    -------
    Path
        Path to saved model.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        logger.error(
            "stable-baselines3 not installed. "
            "Run: pip install stable-baselines3"
        )
        raise

    from src.packing_env import make_packing_env

    # --- Setup directories --------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = MODEL_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Create environment -------------------------------------------------
    logger.info("Creating PackingEnv from %s", data_path)
    env = make_packing_env(data_path=data_path, seed=seed)
    env = Monitor(env)

    # Create eval environment
    eval_env = make_packing_env(data_path="data/val.csv", seed=seed + 1)
    eval_env = Monitor(eval_env)

    # --- Callbacks ----------------------------------------------------------
    checkpoint_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq, n_steps),
        save_path=str(checkpoint_dir),
        name_prefix=f"ppo_packing_v{version}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR / "best_rl"),
        log_path=str(LOG_DIR),
        eval_freq=max(checkpoint_freq // 2, n_steps),
        n_eval_episodes=20,
        deterministic=True,
    )

    # --- PPO Model ----------------------------------------------------------
    logger.info(
        "Initialising PPO: lr=%.1e, n_steps=%d, batch_size=%d, "
        "timesteps=%d",
        lr, n_steps, batch_size, timesteps,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
    )

    # --- Train --------------------------------------------------------------
    logger.info("Starting PPO training for %d timesteps...", timesteps)
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=False,
        tb_log_name=f"ppo_v{version}",
    )

    # --- Save final model ---------------------------------------------------
    model_path = MODEL_DIR / f"ppo_packing_v{version}"
    model.save(str(model_path))
    logger.info("Final model saved to %s.zip", model_path)

    # Save training metadata
    import json
    meta = {
        "version": version,
        "training_date": datetime.utcnow().isoformat(),
        "n_timesteps": timesteps,
        "learning_rate": lr,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "seed": seed,
        "data_path": data_path,
    }
    meta_path = MODEL_DIR / f"ppo_packing_v{version}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Training complete. Metadata saved to %s", meta_path)
    return model_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EcoPackAI PPO agent")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--version", type=str, default="1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="data/train.csv")
    args = parser.parse_args()

    train_ppo(
        timesteps=args.timesteps,
        lr=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        data_path=args.data,
        version=args.version,
        seed=args.seed,
    )
