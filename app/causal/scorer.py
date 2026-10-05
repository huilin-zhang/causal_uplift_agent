"""The deployable scorer: one file holding the uplift model, the risk model,
and the metadata needed to use them.

Serving and the agent only call UpliftScorer.score(); they never need to know
which learner is inside. Code guide: section "Scorer bundle" (sec:scorer).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.data.simulate import feature_matrix


@dataclass
class UpliftScorer:
    uplift_model: Any
    risk_model: Any
    method: str
    feature_columns: list[str]
    version: str = "local"
    metrics: dict[str, float] = field(default_factory=dict)

    def score(self, customers: pd.DataFrame) -> pd.DataFrame:
        """Add tau_hat (churn reduction), churn_risk, and net_value columns."""
        x = feature_matrix(customers)[self.feature_columns]
        out = customers.copy()
        out["tau_hat"] = self.uplift_model.predict(x)
        out["churn_risk"] = self.risk_model.predict(x)
        out["net_value"] = out["tau_hat"] * out["value"] - out["offer_cost"]
        out["model_version"] = self.version
        return out

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> UpliftScorer:
        return joblib.load(path)


def data_fingerprint(df: pd.DataFrame) -> str:
    """Short hash of the training data, logged with the model for traceability."""
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return digest.hexdigest()[:12]


def config_fingerprint(cfg: dict) -> str:
    text = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]
