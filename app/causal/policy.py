"""Turn effect estimates into a contact list.

net value = tau_hat * customer value - offer cost

A customer is worth contacting only if outreach is expected to save more
than it costs. Customers with tau_hat <= 0 (sleeping dogs or no effect)
are never contacted, whatever their churn risk.
Code guide: section "Targeting policy" (sec:policy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OFFER_NAME = "retention_discount"


def net_value(tau_hat: np.ndarray, value: np.ndarray, cost: np.ndarray) -> np.ndarray:
    return np.asarray(tau_hat) * np.asarray(value) - np.asarray(cost)


def rank_by_net_value(
    scored: pd.DataFrame,
    budget: int,
    exclude_ids: set[int] | None = None,
    min_net_value: float = 0.0,
) -> pd.DataFrame:
    """Pick up to `budget` customers with the highest positive net value.

    scored needs: customer_id, tau_hat, value, offer_cost.
    exclude_ids: customers contacted recently (cooldown), from memory.
    """
    df = scored.copy()
    df["net_value"] = net_value(df["tau_hat"], df["value"], df["offer_cost"])
    if exclude_ids:
        df = df[~df["customer_id"].isin(exclude_ids)]
    df = df[(df["tau_hat"] > 0) & (df["net_value"] > min_net_value)]
    df = df.sort_values("net_value", ascending=False).head(budget).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["offer"] = OFFER_NAME
    return df


def rank_by_tau(scored: pd.DataFrame, budget: int, exclude_ids: set[int] | None = None) -> pd.DataFrame:
    """Fallback when value or cost is unknown: rank by tau_hat alone (never by risk)."""
    df = scored.copy()
    if exclude_ids:
        df = df[~df["customer_id"].isin(exclude_ids)]
    df = df[df["tau_hat"] > 0].sort_values("tau_hat", ascending=False).head(budget)
    df = df.reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["offer"] = OFFER_NAME
    return df


def rank_by_risk(scored: pd.DataFrame, budget: int) -> pd.DataFrame:
    """The status-quo policy, kept only for comparison: highest churn risk first."""
    df = scored.sort_values("churn_risk", ascending=False).head(budget).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def policy_summary(offers: pd.DataFrame) -> dict[str, float]:
    if offers.empty:
        return {"n_offers": 0, "expected_saved": 0.0, "expected_net_value": 0.0, "total_cost": 0.0}
    return {
        "n_offers": int(len(offers)),
        "expected_saved": float(offers["tau_hat"].sum()),
        "expected_net_value": float(offers.get("net_value", pd.Series(dtype=float)).sum()),
        "total_cost": float(offers["offer_cost"].sum()),
    }
