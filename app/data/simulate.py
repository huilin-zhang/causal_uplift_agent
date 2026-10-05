"""Simulate subscribers, a randomized outreach test, and 30-day churn.

We know the true effect of outreach for every simulated customer, so we can
check how well each uplift model recovers it (PEHE). Real data never gives us
that. Code guide: section "Simulation" (sec:simulate).

Notation used in comments:
    mu0(x)  churn probability without outreach
    tau(x)  churn reduction from outreach (positive = outreach helps)
    mu1(x)  churn probability with outreach = mu0(x) - tau(x)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "tenure_months",
    "plan_annual",
    "sessions_30d",
    "days_since_login",
    "price_sensitivity",
    "payment_failures_90d",
    "support_tickets_90d",
    "monthly_fee",
]
REGIONS = ["north", "south", "east", "west"]
CATEGORICAL_FEATURES = ["region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that only exist because the data is simulated. Models must never see them.
ORACLE_COLUMNS = ["true_mu0", "true_cate", "true_propensity"]

# Gauss-Hermite nodes to average the sigmoid over the unobserved noise term.
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(20)
_GH_WEIGHTS = _GH_WEIGHTS / _GH_WEIGHTS.sum()


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _std(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-12)


def simulate_covariates(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw customer attributes. Distributions are plain guesses for a
    consumer subscription product."""
    tenure = rng.integers(1, 61, size=n)
    annual = rng.binomial(1, 0.35, size=n)
    sessions = rng.poisson(rng.lognormal(mean=2.0, sigma=0.8, size=n))
    # Fewer sessions -> longer since last login.
    days = np.minimum(90, np.round(rng.exponential(60.0 / (1.0 + sessions / 2.0))))
    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "tenure_months": tenure,
            "plan_annual": annual,
            "sessions_30d": sessions,
            "days_since_login": days.astype(int),
            "price_sensitivity": rng.beta(2, 3, size=n).round(3),
            "payment_failures_90d": rng.binomial(3, 0.05, size=n),
            "support_tickets_90d": rng.poisson(0.4, size=n),
            "monthly_fee": rng.choice([8.99, 13.99, 17.99], size=n, p=[0.4, 0.4, 0.2]),
            "region": rng.choice(REGIONS, size=n),
        }
    )


def risk_index(df: pd.DataFrame) -> np.ndarray:
    """Observed part of churn risk (log-odds scale, before the intercept)."""
    return (
        -0.5 * _std(np.log1p(df["sessions_30d"].to_numpy()))
        + 0.5 * _std(df["days_since_login"].to_numpy(float))
        - 0.4 * _std(df["tenure_months"].to_numpy(float))
        - 0.5 * df["plan_annual"].to_numpy()
        + 0.4 * df["payment_failures_90d"].to_numpy()
        + 0.2 * df["support_tickets_90d"].to_numpy()
        + 0.3 * _std(df["price_sensitivity"].to_numpy())
    )


def expected_mu0(index: np.ndarray, intercept: float, noise_sd: float) -> np.ndarray:
    """E[sigmoid(a + index + noise)] over noise ~ N(0, noise_sd^2)."""
    z = intercept + index[:, None] + noise_sd * _GH_NODES[None, :]
    return _sigmoid(z) @ _GH_WEIGHTS


def solve_intercept(index: np.ndarray, target: float, noise_sd: float) -> float:
    """Find the intercept so the mean churn rate without outreach equals target.

    This sets the base rate only. The treatment effect is not calibrated.
    """
    lo, hi = -10.0, 5.0
    for _ in range(60):  # bisection; mean churn rises with the intercept
        mid = (lo + hi) / 2
        if expected_mu0(index, mid, noise_sd).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def relative_effect(df: pd.DataFrame, sim: dict[str, Any]) -> np.ndarray:
    """r(x): share of churn risk removed by outreach (negative = harm).

    Persuadables: price-sensitive monthly customers who are drifting away
    (last login 7-30 days ago). A discount reaches them in time.
    Lost causes: no login for 45+ days. Outreach changes nothing.
    Sleeping dogs: long-tenure annual customers who barely use the product.
    A message reminds them they are paying, and some cancel.
    """
    days = df["days_since_login"].to_numpy(float)
    # 0.3 when very active, 1.0 inside 7-30 days, falls to 0 at 45 days.
    timing = np.where(days < 7, 0.3, np.where(days <= 30, 1.0, np.clip((45 - days) / 15, 0, 1)))
    persuadable = (
        df["price_sensitivity"].to_numpy() * timing * (1 - df["plan_annual"].to_numpy())
    )
    sleeping_dog = (
        df["plan_annual"].to_numpy()
        * (df["tenure_months"].to_numpy() > 24)
        * (df["sessions_30d"].to_numpy() <= 3)
    )
    return sim["persuadable_effect"] * persuadable - sim["sleeping_dog_effect"] * sleeping_dog


def simulate(
    n: int,
    seed: int,
    sim: dict[str, Any],
    design: str = "rct",
) -> pd.DataFrame:
    """Return one row per subscriber with features, assignment, outcome, and oracle columns.

    design="rct": each customer is assigned to outreach by an independent coin flip.
    design="observational": customer-success reps contact risky customers more often,
    so the naive treated-vs-control difference is biased.
    """
    rng = np.random.default_rng(seed)
    df = simulate_covariates(n, rng)

    index = risk_index(df)
    noise_sd = sim["risk_noise_sd"]
    intercept = solve_intercept(index, sim["base_churn_rate"], noise_sd)
    r = relative_effect(df, sim)

    # Individual churn probabilities include unobserved noise.
    noise = rng.normal(0.0, noise_sd, size=n)
    mu0_i = _sigmoid(intercept + index + noise)
    mu1_i = np.clip(mu0_i * (1 - r), 0, 1)

    # The CATE given observed x averages over the noise: tau(x) = E[mu0 | x] * r(x).
    mu0_x = expected_mu0(index, intercept, noise_sd)

    if design == "rct":
        e = np.full(n, sim["treat_share"])
    elif design == "observational":
        lo, hi = sim["propensity_clip"]
        e = np.clip(_sigmoid(sim["obs_intercept"] + sim["obs_risk_slope"] * _std(index)), lo, hi)
    else:
        raise ValueError(f"unknown design: {design}")

    treatment = rng.binomial(1, e)
    churn = rng.binomial(1, np.where(treatment == 1, mu1_i, mu0_i))

    df["treatment"] = treatment
    df["churn_30d"] = churn
    df["value"] = (sim["margin"] * df["monthly_fee"] * sim["value_months"]).round(2)
    df["offer_cost"] = (sim["contact_cost"] + sim["discount_share"] * df["monthly_fee"]).round(2)
    df["true_mu0"] = mu0_x
    df["true_cate"] = mu0_x * r
    df["true_propensity"] = e
    return df


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric features plus one-hot region, in a fixed column order."""
    x = df[NUMERIC_FEATURES].astype(float).copy()
    for region in REGIONS[1:]:  # drop the first level to avoid a redundant column
        x[f"region_{region}"] = (df["region"] == region).astype(float)
    return x
