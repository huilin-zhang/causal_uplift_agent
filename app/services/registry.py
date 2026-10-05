"""MLflow tracking, registration, promotion, and rollback.

Promotion rule: a new model version gets the alias `champion` only if the CI
gates passed. The version it replaces keeps the alias `previous_champion`, so
rollback is one alias move. Serving always loads `models:/<name>@champion`
and falls back to the local file if MLflow is not reachable.
Code guide: section "MLflow registry" (sec:registry).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from app.causal.scorer import UpliftScorer
from app.core.config import ROOT

CHAMPION, PREVIOUS = "champion", "previous_champion"


def _tracking_uri(uri: str) -> str:
    """Resolve a relative sqlite path against the project root, not the working dir."""
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not Path(uri[len(prefix):]).is_absolute():
        return prefix + (ROOT / uri[len(prefix):]).as_posix()
    return uri


def _client(cfg: dict):
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = _tracking_uri(cfg["mlflow"]["tracking_uri"])
    Path(uri.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    return mlflow, MlflowClient(tracking_uri=uri)


def log_and_register(
    scorer_path: Path,
    params: dict[str, Any],
    metrics: dict[str, float],
    report_path: Path | None,
    cfg: dict,
    promote: bool,
) -> dict[str, Any]:
    """Log one run, register the scorer as a new model version, promote if allowed."""
    mlflow, client = _client(cfg)
    name = cfg["mlflow"]["model_name"]
    exp_name = cfg["mlflow"]["experiment"]
    if client.get_experiment_by_name(exp_name) is None:
        # Keep run artifacts next to the database, not in ./mlruns of whatever folder we run from.
        artifact_root = Path(cfg["paths"]["artifacts"]) / "mlartifacts"
        client.create_experiment(exp_name, artifact_location=artifact_root.resolve().as_uri())
    mlflow.set_experiment(exp_name)
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        mlflow.log_artifact(str(scorer_path), artifact_path="scorer")
        if report_path and Path(report_path).exists():
            mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.set_tag("gates_passed", str(promote))

    version = client.create_model_version(
        name=_ensure_model(client, name), source=f"runs:/{run.info.run_id}/scorer", run_id=run.info.run_id
    ).version

    previous = None
    if promote:
        try:
            previous = client.get_model_version_by_alias(name, CHAMPION).version
        except Exception:  # no champion yet
            previous = None
        if previous and previous != version:
            client.set_registered_model_alias(name, PREVIOUS, previous)
        client.set_registered_model_alias(name, CHAMPION, version)
    return {"model_name": name, "version": str(version), "run_id": run.info.run_id,
            "promoted": promote, "previous_champion": previous}


def _ensure_model(client, name: str) -> str:
    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(name)
    return name


def rollback(cfg: dict) -> dict[str, Any]:
    """Point `champion` back at `previous_champion`, and swap the two aliases."""
    _, client = _client(cfg)
    name = cfg["mlflow"]["model_name"]
    current = client.get_model_version_by_alias(name, CHAMPION).version
    previous = client.get_model_version_by_alias(name, PREVIOUS).version
    client.set_registered_model_alias(name, CHAMPION, previous)
    client.set_registered_model_alias(name, PREVIOUS, current)
    return {"model_name": name, "champion": str(previous), "previous_champion": str(current)}


def load_champion(cfg: dict) -> UpliftScorer:
    """Load the champion scorer from MLflow; fall back to the local artifact."""
    try:
        mlflow, client = _client(cfg)
        name = cfg["mlflow"]["model_name"]
        mv = client.get_model_version_by_alias(name, CHAMPION)
        with tempfile.TemporaryDirectory() as tmp:
            folder = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="scorer", dst_path=tmp)
            scorer = UpliftScorer.load(next(Path(folder).glob("*.joblib")))
        scorer.version = f"{name}@{CHAMPION}=v{mv.version}"
        return scorer
    except Exception:
        return UpliftScorer.load(cfg["paths"]["model"])


def registry_status(cfg: dict) -> dict[str, Any]:
    try:
        _, client = _client(cfg)
        name = cfg["mlflow"]["model_name"]
        out = {"model_name": name}
        for alias in (CHAMPION, PREVIOUS):
            try:
                out[alias] = str(client.get_model_version_by_alias(name, alias).version)
            except Exception:
                out[alias] = None
        return out
    except Exception as exc:
        return {"error": str(exc)}


def dump(obj: dict) -> str:
    return json.dumps(obj, indent=2, default=str)
