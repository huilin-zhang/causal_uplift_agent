"""Loaders for three public uplift datasets, mapped to one common format.

The files are not in the repository (size and licence). Download them into
data/external/ (or set UPLIFT_DATA_DIR) as described in the operation manual:

  orange/churn_uplift_anonymized.csv   Verhelst et al. (2023), telecom churn, RCT
  hillstrom/*.csv                      Hillstrom MineThatData e-mail test (2008), 3 arms
  criteo/criteo_sample.csv.gz          made by scripts/sample_criteo.py from Criteo Uplift v2.1

Every loader returns a RealDataset. `good` is the outcome we want more of
(retained, visited). Learners model `bad = 1 - good`, matching the churn
convention used in the simulation. Code guide: section "Real datasets" (sec:real).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import ROOT


def data_dir() -> Path:
    return Path(os.environ.get("UPLIFT_DATA_DIR", ROOT / "data" / "external"))


@dataclass
class RealDataset:
    name: str
    x: pd.DataFrame
    t: np.ndarray            # 1 = treated, 0 = control (for one arm vs control)
    good: np.ndarray         # outcome we want more of
    design: str
    arm_labels: dict[str, np.ndarray] = field(default_factory=dict)  # multi-arm only
    raw: pd.DataFrame | None = None
    source: str = ""


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """Trees accept integer codes for categories; keep it simple and reproducible."""
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype("category").cat.codes
    return out.astype(float)


def load_orange(path: Path | None = None) -> RealDataset:
    """Telecom churn uplift data. y = churn, t = retention campaign. ~76% treated."""
    path = path or data_dir() / "orange" / "churn_uplift_anonymized.csv"
    raw = pd.read_csv(path)
    x = _encode(raw.drop(columns=["y", "t"]))
    return RealDataset(
        name="orange_churn",
        x=x,
        t=raw["t"].to_numpy(int),
        good=1 - raw["y"].to_numpy(int),
        design="rct",
        raw=raw,
        source="Verhelst et al. (2023), https://github.com/TheoVerhelst/Churn-Uplift-Dataset-Paper",
    )


def load_hillstrom(path: Path | None = None, arm: str = "Womens E-Mail") -> RealDataset:
    """E-mail test with three arms. Default: one arm vs 'No E-Mail', outcome = visit."""
    if path is None:
        files = sorted((data_dir() / "hillstrom").glob("*.csv"))
        if not files:
            raise FileNotFoundError("no Hillstrom csv in data/external/hillstrom")
        path = files[0]
    raw = pd.read_csv(path)
    features = ["recency", "history", "mens", "womens", "zip_code", "newbie", "channel"]
    keep = raw["segment"].isin([arm, "No E-Mail"]).to_numpy()
    sub = raw[keep].reset_index(drop=True)
    arms = {a: (raw["segment"] == a).to_numpy() for a in raw["segment"].unique()}
    return RealDataset(
        name="hillstrom",
        x=_encode(sub[features]),
        t=(sub["segment"] == arm).to_numpy(int),
        good=sub["visit"].to_numpy(int),
        design="rct",
        arm_labels=arms,
        raw=raw,
        source="Hillstrom (2008), https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html",
    )


def load_criteo(path: Path | None = None) -> RealDataset:
    """Ad uplift data, ~85% treated. Outcome = visit (conversion is very rare)."""
    path = path or data_dir() / "criteo" / "criteo_sample.csv.gz"
    raw = pd.read_csv(path)
    features = [f"f{i}" for i in range(12)]
    return RealDataset(
        name="criteo_sample",
        x=raw[features].astype(float),
        t=raw["treatment"].to_numpy(int),
        good=raw["visit"].to_numpy(int),
        design="rct",
        raw=raw,
        source="Diemert et al. (2018), https://ailab.criteo.com/criteo-uplift-prediction-dataset/",
    )


LOADERS = {"orange_churn": load_orange, "hillstrom": load_hillstrom, "criteo_sample": load_criteo}


def available() -> list[str]:
    """Names of datasets whose files are present."""
    names = []
    base = data_dir()
    if (base / "orange" / "churn_uplift_anonymized.csv").exists():
        names.append("orange_churn")
    if list((base / "hillstrom").glob("*.csv")) if (base / "hillstrom").exists() else []:
        names.append("hillstrom")
    if (base / "criteo" / "criteo_sample.csv.gz").exists():
        names.append("criteo_sample")
    return names
