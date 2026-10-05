"""Shared fixtures. `pipeline_env` runs a small end-to-end pipeline once per
test session in a temporary folder; API, agent, and registry tests use it."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from app.core.config import get_settings, reset_settings
from app.data.simulate import feature_matrix, simulate


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def sim_cfg(settings):
    return settings["sim"]


@pytest.fixture(scope="session")
def rct(sim_cfg) -> pd.DataFrame:
    return simulate(20_000, 1, sim_cfg)


@pytest.fixture(scope="session")
def rct_x(rct) -> pd.DataFrame:
    return feature_matrix(rct)


@pytest.fixture(scope="session")
def pipeline_env(tmp_path_factory):
    folder = tmp_path_factory.mktemp("artifacts")
    env = {"UPLIFT_ARTIFACTS": str(folder), "UPLIFT_N": "20000", "UPLIFT_DRAWS": "1",
           "UPLIFT_TREES": "48", "UPLIFT_BOOTSTRAP": "50", "AGENT_PROVIDER": "deterministic",
           "MLFLOW_DISABLE_AGENT_HINT": "1"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    reset_settings()
    from pipelines.run_pipeline import force_register, run

    cfg = get_settings()
    run(["generate", "validate", "analyze", "benchmark", "train", "gates"], cfg)
    # The tiny test run may miss a gate by chance; serving tests need a champion either way.
    force_register(cfg)
    run(["score"], cfg)
    yield cfg
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    reset_settings()


def toy_rct(n: int = 4000, seed: int = 0):
    """Tiny dataset with a strong, known effect: x0 > 0 halves churn; x0 <= 0 has no effect."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"x0": rng.normal(size=n), "x1": rng.normal(size=n)})
    t = rng.binomial(1, 0.5, size=n)
    base = 0.4
    tau = np.where(x["x0"] > 0, 0.2, 0.0)
    y = rng.binomial(1, np.where(t == 1, base - tau, base))
    return x, t, y, tau
