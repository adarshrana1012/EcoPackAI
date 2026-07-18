"""
model_registry.py
=================

ML Model Registry for the EcoPackAI platform.

Provides a :class:`ModelRegistry` that manages versioned model artifacts
on the local filesystem.  Each version is stored as a joblib-serialised
file with a JSON metadata sidecar.  The registry supports promotion of
a single version to *production* status, listing / loading / deleting
versions, and safe concurrent access via file-system conventions.

Typical usage
-------------
>>> from src.model_registry import ModelRegistry
>>> registry = ModelRegistry("models/registry")
>>> path = registry.save(model, "1.0.0", metadata)
>>> model, meta = registry.load("1.0.0")
>>> registry.promote_to_production("1.0.0")
>>> prod_model, prod_meta = registry.get_production_model()

Author:  EcoPackAI Team
Created: 2026-06-12
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_FILENAME: str = "model.joblib"
_PRODUCTION_POINTER: str = "production.json"

_REQUIRED_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "training_date",
        "metrics",
        "feature_names",
        "model_type",
        "hyperparameters",
    }
)


class ModelRegistry:
    """Manages versioned ML model artifacts on the local filesystem.

    Each model version is stored under ``{base_dir}/v{version}/`` and
    comprises:

    * ``model.joblib`` – the serialised model (or pipeline) artifact.
    * ``{version}_metadata.json`` – sidecar metadata including training
      date, evaluation metrics, feature list, model type, and hyper-
      parameters.

    A single version can be *promoted* to production.  The pointer to the
    current production version lives in ``{base_dir}/production.json``.

    Parameters
    ----------
    base_dir : str, optional
        Root directory for the registry.  Created if it does not exist.
        Defaults to ``"models/registry"``.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self, base_dir: str = "models/registry") -> None:
        self._base_dir = Path(base_dir)
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Model registry initialised at '%s'.", self._base_dir.resolve())
        except OSError as exc:
            logger.error(
                "Failed to create registry directory '%s': %s",
                self._base_dir,
                exc,
            )
            raise

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    @property
    def base_dir(self) -> Path:
        """Return the resolved base directory of the registry."""
        return self._base_dir.resolve()

    # ------------------------------------------------------------------ #
    # save
    # ------------------------------------------------------------------ #
    def save(self, model: Any, version: str, metadata: Dict[str, Any]) -> Path:
        """Persist a model artifact and its metadata to the registry.

        The model (or an entire sklearn ``Pipeline``) is serialised with
        :func:`joblib.dump`.  A sidecar JSON file stores the supplied
        metadata enriched with a ``registered_at`` timestamp and an
        ``is_production`` flag (initially ``False``).

        Parameters
        ----------
        model : Any
            The trained model or preprocessing pipeline to serialise.
        version : str
            Semantic version string, e.g. ``"1.2.0"``.
        metadata : dict
            Must contain the keys ``training_date``, ``metrics`` (dict),
            ``feature_names`` (list), ``model_type`` (str), and
            ``hyperparameters`` (dict).

        Returns
        -------
        pathlib.Path
            Absolute path to the saved ``model.joblib`` file.

        Raises
        ------
        ValueError
            If required metadata keys are missing.
        OSError
            If writing to the filesystem fails.
        """
        # --- Validate metadata ------------------------------------------------
        missing = _REQUIRED_METADATA_KEYS - set(metadata.keys())
        if missing:
            raise ValueError(
                f"Metadata is missing required keys: {sorted(missing)}"
            )

        version_dir = self._version_dir(version)
        if version_dir.exists():
            logger.warning(
                "Version '%s' already exists – overwriting artifacts.", version
            )
        version_dir.mkdir(parents=True, exist_ok=True)

        # --- Serialise model ---------------------------------------------------
        model_path = version_dir / _MODEL_FILENAME
        try:
            joblib.dump(model, model_path)
            logger.info(
                "Model artifact saved to '%s' (%.2f MB).",
                model_path,
                model_path.stat().st_size / (1024 * 1024),
            )
        except Exception as exc:
            logger.error("Failed to serialise model for version '%s': %s", version, exc)
            raise

        # --- Write metadata sidecar --------------------------------------------
        enriched_metadata: Dict[str, Any] = {
            **metadata,
            "version": version,
            "is_production": False,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = version_dir / f"{version}_metadata.json"
        try:
            meta_path.write_text(
                json.dumps(enriched_metadata, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("Metadata sidecar written to '%s'.", meta_path)
        except Exception as exc:
            logger.error(
                "Failed to write metadata for version '%s': %s", version, exc
            )
            raise

        return model_path.resolve()

    # ------------------------------------------------------------------ #
    # load
    # ------------------------------------------------------------------ #
    def load(self, version: str) -> Tuple[Any, Dict[str, Any]]:
        """Load a model and its metadata by version string.

        Parameters
        ----------
        version : str
            The version to load, e.g. ``"1.0.0"``.

        Returns
        -------
        tuple[Any, dict]
            ``(model, metadata_dict)``

        Raises
        ------
        FileNotFoundError
            If the requested version directory or its artifacts are missing.
        """
        version_dir = self._version_dir(version)
        model_path = version_dir / _MODEL_FILENAME
        meta_path = version_dir / f"{version}_metadata.json"

        if not version_dir.exists():
            raise FileNotFoundError(
                f"Version '{version}' not found in registry at '{self._base_dir}'."
            )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact missing for version '{version}': expected '{model_path}'."
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata file missing for version '{version}': expected '{meta_path}'."
            )

        try:
            model = joblib.load(model_path)
            logger.info("Loaded model artifact from '%s'.", model_path)
        except Exception as exc:
            logger.error("Failed to load model for version '%s': %s", version, exc)
            raise

        try:
            metadata: Dict[str, Any] = json.loads(
                meta_path.read_text(encoding="utf-8")
            )
            logger.info("Loaded metadata from '%s'.", meta_path)
        except Exception as exc:
            logger.error("Failed to read metadata for version '%s': %s", version, exc)
            raise

        return model, metadata

    # ------------------------------------------------------------------ #
    # list_versions
    # ------------------------------------------------------------------ #
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all registered model versions with summary metadata.

        Returns
        -------
        list[dict]
            Each dict contains ``version``, ``training_date``,
            ``is_production``, and ``metrics``.  The list is sorted by
            version in descending order.
        """
        versions: List[Dict[str, Any]] = []

        if not self._base_dir.exists():
            logger.warning("Registry base directory does not exist.")
            return versions

        for child in sorted(self._base_dir.iterdir(), reverse=True):
            if not child.is_dir() or not child.name.startswith("v"):
                continue

            version_str = child.name[1:]  # strip leading 'v'
            meta_path = child / f"{version_str}_metadata.json"
            if not meta_path.exists():
                logger.warning(
                    "Skipping directory '%s' – no metadata sidecar found.", child
                )
                continue

            try:
                meta: Dict[str, Any] = json.loads(
                    meta_path.read_text(encoding="utf-8")
                )
                versions.append(
                    {
                        "version": meta.get("version", version_str),
                        "training_date": meta.get("training_date"),
                        "is_production": meta.get("is_production", False),
                        "metrics": meta.get("metrics", {}),
                    }
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.error(
                    "Failed to read metadata in '%s': %s", meta_path, exc
                )

        logger.info("Found %d registered model version(s).", len(versions))
        return versions

    # ------------------------------------------------------------------ #
    # promote_to_production
    # ------------------------------------------------------------------ #
    def promote_to_production(self, version: str) -> None:
        """Mark *version* as the production model.

        Any previously-promoted version is automatically demoted.  The
        metadata sidecar of both the old and new production version is
        updated, and a ``production.json`` pointer file is written (or
        updated) at the registry root.

        Parameters
        ----------
        version : str
            The version to promote.

        Raises
        ------
        FileNotFoundError
            If *version* does not exist in the registry.
        """
        version_dir = self._version_dir(version)
        if not version_dir.exists():
            raise FileNotFoundError(
                f"Cannot promote version '{version}' – it does not exist."
            )

        # --- Demote the current production version (if any) --------------------
        current_prod = self._read_production_pointer()
        if current_prod is not None and current_prod != version:
            self._set_production_flag(current_prod, is_production=False)
            logger.info("Demoted version '%s' from production.", current_prod)

        # --- Promote the requested version -------------------------------------
        self._set_production_flag(version, is_production=True)

        # --- Write/update the production pointer -------------------------------
        pointer_path = self._base_dir / _PRODUCTION_POINTER
        pointer_data = {
            "production_version": version,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        pointer_path.write_text(
            json.dumps(pointer_data, indent=2), encoding="utf-8"
        )
        logger.info(
            "Version '%s' promoted to production. Pointer updated at '%s'.",
            version,
            pointer_path,
        )

    # ------------------------------------------------------------------ #
    # get_production_model
    # ------------------------------------------------------------------ #
    def get_production_model(self) -> Tuple[Any, Dict[str, Any]]:
        """Load the current production model and its metadata.

        Returns
        -------
        tuple[Any, dict]
            ``(model, metadata_dict)``

        Raises
        ------
        RuntimeError
            If no model version has been promoted to production.
        """
        prod_version = self._read_production_pointer()
        if prod_version is None:
            raise RuntimeError(
                "No model is currently in production. "
                "Use promote_to_production() first."
            )

        logger.info("Loading production model (version '%s').", prod_version)
        return self.load(prod_version)

    # ------------------------------------------------------------------ #
    # delete_version
    # ------------------------------------------------------------------ #
    def delete_version(self, version: str) -> None:
        """Delete a model version and all associated artifacts.

        The production version cannot be deleted; demote it first by
        promoting a different version.

        Parameters
        ----------
        version : str
            The version to delete.

        Raises
        ------
        FileNotFoundError
            If *version* does not exist.
        RuntimeError
            If *version* is the current production version.
        """
        version_dir = self._version_dir(version)
        if not version_dir.exists():
            raise FileNotFoundError(
                f"Cannot delete version '{version}' – it does not exist."
            )

        # Guard against deleting production
        prod_version = self._read_production_pointer()
        if prod_version is not None and prod_version == version:
            raise RuntimeError(
                f"Cannot delete version '{version}' because it is the "
                "current production model. Promote another version first."
            )

        try:
            shutil.rmtree(version_dir)
            logger.info(
                "Deleted version '%s' and its artifacts from '%s'.",
                version,
                version_dir,
            )
        except OSError as exc:
            logger.error("Failed to delete version '%s': %s", version, exc)
            raise

    # ================================================================== #
    # Private helpers
    # ================================================================== #
    def _version_dir(self, version: str) -> Path:
        """Return the directory path for a given version string."""
        return self._base_dir / f"v{version}"

    def _read_production_pointer(self) -> Optional[str]:
        """Read the production pointer file and return the version string.

        Returns ``None`` if no pointer file exists or it is malformed.
        """
        pointer_path = self._base_dir / _PRODUCTION_POINTER
        if not pointer_path.exists():
            return None
        try:
            data = json.loads(pointer_path.read_text(encoding="utf-8"))
            return data.get("production_version")
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Corrupt production pointer file: %s", exc)
            return None

    def _set_production_flag(self, version: str, *, is_production: bool) -> None:
        """Update the ``is_production`` flag in a version's metadata sidecar.

        Parameters
        ----------
        version : str
            Target version.
        is_production : bool
            New value for the flag.
        """
        meta_path = self._version_dir(version) / f"{version}_metadata.json"
        if not meta_path.exists():
            logger.warning(
                "Metadata sidecar for version '%s' not found; "
                "skipping flag update.",
                version,
            )
            return

        try:
            metadata: Dict[str, Any] = json.loads(
                meta_path.read_text(encoding="utf-8")
            )
            metadata["is_production"] = is_production
            meta_path.write_text(
                json.dumps(metadata, indent=2, default=str), encoding="utf-8"
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(
                "Failed to update production flag for version '%s': %s",
                version,
                exc,
            )

    # ------------------------------------------------------------------ #
    # RL Policy Support (Prompt 27)
    # ------------------------------------------------------------------ #

    def save_rl_policy(
        self,
        policy_path: str,
        version: str,
        metadata: Dict[str, Any],
    ) -> Path:
        """Save an RL policy (.zip file) to the registry.

        Parameters
        ----------
        policy_path : str
            Path to the PPO ``.zip`` file on disk.
        version : str
            Version identifier.
        metadata : dict
            Must include ``training_date``, ``n_timesteps``,
            ``eval_reward``, and optionally ``is_active``.

        Returns
        -------
        Path
            Destination path of the stored policy.
        """
        from pathlib import Path as _P

        version_dir = self._version_dir(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        src = _P(policy_path)
        if not src.exists():
            # Try with .zip extension
            src = _P(policy_path + ".zip") if not policy_path.endswith(".zip") else src
        if not src.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_path}")

        dest = version_dir / f"policy_v{version}.zip"
        shutil.copy2(str(src), str(dest))

        # Enrich metadata
        enriched = {
            **metadata,
            "version": version,
            "model_type": metadata.get("model_type", "PPO"),
            "artifact_type": "rl_policy",
            "is_production": metadata.get("is_active", False),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        meta_path = version_dir / f"{version}_metadata.json"
        meta_path.write_text(
            json.dumps(enriched, indent=2, default=str), encoding="utf-8"
        )

        logger.info(
            "RL policy v%s saved to %s (reward=%.4f).",
            version, dest,
            metadata.get("eval_reward", metadata.get("metrics", {}).get("mean_reward", 0)),
        )
        return dest

    def load_rl_policy(self, version: str) -> Tuple[Any, Dict[str, Any]]:
        """Load an RL policy by version.

        Parameters
        ----------
        version : str
            Version identifier.

        Returns
        -------
        tuple[Any, dict]
            ``(PPO_model, metadata)``

        Raises
        ------
        FileNotFoundError
            If the policy version doesn't exist.
        """
        version_dir = self._version_dir(version)
        policy_path = version_dir / f"policy_v{version}.zip"

        if not policy_path.exists():
            raise FileNotFoundError(
                f"RL policy v{version} not found at {policy_path}"
            )

        try:
            from stable_baselines3 import PPO
            model = PPO.load(str(policy_path))
        except ImportError:
            logger.warning("stable-baselines3 not installed; returning path.")
            model = str(policy_path)

        meta_path = version_dir / f"{version}_metadata.json"
        metadata = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        logger.info("Loaded RL policy v%s.", version)
        return model, metadata

    def rollback_policy(self, version: str) -> None:
        """Rollback to a specific policy version.

        Deactivates the current production policy and promotes the
        specified version.

        Parameters
        ----------
        version : str
            The version to promote as the new production policy.

        Raises
        ------
        FileNotFoundError
            If the target version doesn't exist.
        """
        version_dir = self._version_dir(version)
        if not version_dir.exists():
            raise FileNotFoundError(
                f"Cannot rollback: version '{version}' not found."
            )

        # Demote current production
        current = self._read_production_pointer()
        if current is not None:
            self._set_production_flag(current, is_production=False)
            logger.info("Demoted current production v%s.", current)

        # Promote target version
        self._set_production_flag(version, is_production=True)

        # Update production pointer
        pointer_path = self._base_dir / _PRODUCTION_POINTER
        pointer_path.write_text(
            json.dumps(
                {
                    "production_version": version,
                    "promoted_at": datetime.now(timezone.utc).isoformat(),
                    "action": "rollback",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.info("Rolled back to policy v%s.", version)

    def policy_history(self) -> List[Dict[str, Any]]:
        """Return a chronological list of all RL policy versions.

        Returns
        -------
        list[dict]
            Sorted by ``training_date`` ascending.  Each dict contains
            ``version``, ``training_date``, ``n_timesteps``,
            ``eval_reward``, ``is_production``, ``model_type``.
        """
        versions = self.list_versions()
        rl_versions = [
            v for v in versions
            if v.get("model_type") in ("PPO", "rl_policy")
            or v.get("artifact_type") == "rl_policy"
        ]

        # If no explicit RL filtering, return all
        if not rl_versions:
            rl_versions = versions

        # Sort by training date
        rl_versions.sort(
            key=lambda v: v.get("training_date", ""),
        )

        result = []
        for v in rl_versions:
            metrics = v.get("metrics", {})
            result.append({
                "version": v.get("version", "unknown"),
                "training_date": v.get("training_date", "unknown"),
                "n_timesteps": v.get("hyperparameters", v.get("n_timesteps", {})).get(
                    "n_timesteps", v.get("n_timesteps", "N/A")
                ) if isinstance(v.get("hyperparameters", {}), dict) else "N/A",
                "eval_reward": v.get("eval_reward", metrics.get("mean_reward", "N/A")),
                "is_production": v.get("is_production", False),
                "model_type": v.get("model_type", "unknown"),
            })

        logger.info("Policy history: %d versions found.", len(result))
        return result

