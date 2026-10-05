"""Policy sub-agent: rank current subscribers by net value and build the offer list.

Code guide: section "Policy agent" (sec:policy-agent).
"""

from __future__ import annotations

import time

import pandas as pd

from app.agents.state import AgentState, Deps
from app.causal.policy import policy_summary, rank_by_net_value, rank_by_tau


def load_scores(deps: Deps) -> pd.DataFrame:
    """Scores from the batch job; score on the fly if the batch table is missing."""
    current = deps.store.read_table("current_subscribers")
    if "scores" in deps.store.tables():
        scores = deps.store.read_table("scores")
        return current.merge(scores, on="customer_id", how="inner")
    if deps.scorer is None:
        raise RuntimeError("no scores and no model; run the pipeline first")
    return deps.scorer.score(current)


def run_policy_agent(state: AgentState, deps: Deps) -> AgentState:
    constraints = state.get("constraints") or {}
    memory = state.get("memory") or {}
    exclude = set(memory.get("cooldown_ids", [])) | set(memory.get("excluded_ids", []))
    exclude |= set(constraints.get("exclude_ids", []))

    scored = load_scores(deps)
    policy = (state.get("selection") or {}).get("policy", "net_value_top_k")
    budget = int(state.get("budget") or deps.settings["agent"]["default_budget"])
    if policy.startswith("tau"):
        offers = rank_by_tau(scored, budget, exclude)
    else:
        offers = rank_by_net_value(scored, budget, exclude, constraints.get("min_net_value", 0.0))

    cols = ["customer_id", "rank", "tau_hat", "churn_risk", "value", "offer_cost", "net_value", "offer"]
    records = offers[[c for c in cols if c in offers]].to_dict("records")
    summary = policy_summary(offers)
    summary.update({"budget": budget, "eligible_positive_net": int((scored["net_value"] > 0).sum()),
                    "excluded": len(exclude), "policy": policy})
    return {
        "offers": records,
        "policy_summary": summary,
        "campaign_id": state.get("campaign_id") or f"cmp-{time.strftime('%Y%m%d-%H%M%S')}",
        "trace": state.get("trace", []) + [
            f"policy_agent: {summary['n_offers']} offers (budget {budget}, "
            f"{summary['eligible_positive_net']} with positive net value, {len(exclude)} excluded)"
        ],
    }
