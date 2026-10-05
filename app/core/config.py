"""Load settings from YAML and environment variables.

Code guide: section "Configuration" (sec:config).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def _load_yaml(name: str) -> dict[str, Any]:
    with (CONFIG_DIR / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply UPLIFT_* environment variables on top of the YAML values.

    Only a few keys are overridable; CI uses them to run a smaller job.
    """
    env = os.environ
    if "UPLIFT_N" in env:
        cfg["n_subscribers"] = int(env["UPLIFT_N"])
    if "UPLIFT_SEED" in env:
        cfg["seed"] = int(env["UPLIFT_SEED"])
    if "UPLIFT_BOOTSTRAP" in env:
        cfg["benchmark"]["bootstrap_reps"] = int(env["UPLIFT_BOOTSTRAP"])
    if "UPLIFT_TREES" in env:
        cfg["benchmark"]["forest_trees"] = int(env["UPLIFT_TREES"])
    if "UPLIFT_DRAWS" in env:
        cfg["benchmark"]["n_draws"] = int(env["UPLIFT_DRAWS"])
    if "UPLIFT_ARTIFACTS" in env:
        # Point every artifact path at another folder (used by tests).
        base = Path(env["UPLIFT_ARTIFACTS"])
        cfg["paths"]["artifacts"] = str(base)
        cfg["paths"]["database"] = str(base / "uplift.db")
        cfg["paths"]["report"] = str(base / "reports" / "latest_run.json")
        cfg["paths"]["model"] = str(base / "models" / "uplift_scorer.joblib")
        cfg["mlflow"]["tracking_uri"] = f"sqlite:///{(base / 'mlflow.db').as_posix()}"
    if "AGENT_PROVIDER" in env:
        cfg["agent"]["provider"] = env["AGENT_PROVIDER"]
    if "MLFLOW_TRACKING_URI" in env:
        cfg["mlflow"]["tracking_uri"] = env["MLFLOW_TRACKING_URI"]
    return cfg


@lru_cache(maxsize=1)
def get_settings() -> dict[str, Any]:
    cfg = _env_overrides(_load_yaml("base.yaml"))
    cfg["sim"] = _load_yaml("sim.yaml")
    # Relative paths are relative to the project root, not the working dir.
    for key, value in cfg["paths"].items():
        path = Path(value)
        cfg["paths"][key] = path if path.is_absolute() else ROOT / path
    return cfg


def reset_settings() -> None:
    """Clear the cache so new environment variables take effect (tests)."""
    get_settings.cache_clear()
