"""Causal Targeting Agent: evidence and unconstrained predictions, never allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from app.agents.ml_agent import run_ml_agent
from app.agents.state import AgentState, Deps


def load_scores(deps: Deps) -> pd.DataFrame:
    current = deps.store.read_table("current_subscribers")
    if "scores" in deps.store.tables():
        scores = deps.store.read_table("scores")
        if deps.scorer is None or (
            "model_version" in scores and scores["model_version"].eq(deps.scorer.version).all()
        ):
            return current.merge(scores, on="customer_id", validate="one_to_one")
    if deps.scorer is None:
        raise RuntimeError("no usable scores and no model; run the pipeline first")
    return deps.scorer.score(current)


def estimate_segment_uplift(deps: Deps, plan_annual: int, queries: int) -> dict:
    """Bonferroni-adjusted RCT readout on a separate discovery cohort.

    This cohort is never used by step_train. Fixed segments avoid data-driven
    subgroup search on the same outcomes. Cost is the simulator's expected
    per-contact cost, not a coupon face value or an estimated redemption rate.
    """
    if "discovery_experiment" not in deps.store.tables():
        return {"estimable": False, "reason": "independent discovery experiment unavailable"}
    df = deps.store.read_table("discovery_experiment")
    df = df[df["plan_annual"] == plan_annual]
    t = df["treatment"].to_numpy() == 1
    if min(int(t.sum()), int((~t).sum())) < 30:
        return {"estimable": False, "reason": "insufficient randomized observations"}
    from app.causal.experiment import srm_test, standardized_mean_diff
    from app.data.simulate import feature_matrix

    srm = srm_test(int(t.sum()), int((~t).sum()), deps.settings["sim"]["treat_share"])
    smd = standardized_mean_diff(feature_matrix(df), t).abs().max()
    if srm["p_value"] < deps.settings["experiment"]["srm_alpha"] or smd >= deps.settings["experiment"]["smd_threshold"]:
        return {"estimable": False, "reason": "discovery experiment validity checks failed"}
    z = stats.norm.ppf(1 - 0.05 / (2 * queries))

    def difference(values):
        est = float(values[t].mean() - values[~t].mean())
        se = float(np.sqrt(values[t].var(ddof=1) / t.sum() + values[~t].var(ddof=1) / (~t).sum()))
        return {"estimate": est, "ci_low": est - z * se, "ci_high": est + z * se}

    retained = 1 - df["churn_30d"].to_numpy(float)
    reward = retained * df["value"].to_numpy() - t * df["offer_cost"].to_numpy()
    return {"estimable": True, "uplift": difference(retained), "net_value": difference(reward),
            "n": len(df), "queries": queries, "source": "discovery_experiment",
            "expected_cost_per_contact": float(df["offer_cost"].mean())}


def run_causal_targeting_agent(state: AgentState, deps: Deps) -> AgentState:
    try:
        out = run_ml_agent(state, deps)
        usable = bool(out["experiment"]["valid"])
        scores = load_scores(deps) if usable and state["intent"] in {"campaign", "discovery"} else pd.DataFrame()
    except (RuntimeError, ValueError, KeyError, pd.errors.DatabaseError) as exc:
        return {"targeting": {"usable": False, "reason": str(exc)}, "uplift_table": [],
                "trace": state.get("trace", []) + ["causal_targeting: evidence unavailable"]}
    cols = ["customer_id", "plan_annual", "tenure_months", "tau_hat", "churn_risk", "value",
            "offer_cost", "net_value", "model_version"]
    out.update({"uplift_table": scores[[c for c in cols if c in scores]].to_dict("records"),
                "targeting": {"usable": usable, "supported_interventions": ["retention_discount"],
                              "individual_intervals": None, "message_attribute_effects": None,
                              "limitation": "single intervention point estimates; no individual confidence intervals"}})
    out["trace"] += ["causal_targeting: supplied evidence and unconstrained uplift table"]
    return out
