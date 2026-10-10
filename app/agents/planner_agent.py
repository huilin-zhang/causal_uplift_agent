"""Planner Agent: designs candidates and allocates existing effect estimates."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from app.agents.contracts import CampaignDesign, PolicyTargets
from app.agents.providers import get_provider
from app.agents.state import AgentState, Deps
from app.causal.policy import policy_summary


def allocate(scored: pd.DataFrame, policy: PolicyTargets) -> pd.DataFrame:
    """Binary optimization: money, count, aggregate ROI and new-user share.

    Every coefficient comes from targeting. New users have tenure <= 3 months.
    An empty feasible set is reported to the human; no constraint is relaxed.
    """
    if scored.empty or policy.max_contacts == 0:
        return scored.iloc[:0].copy()
    df = scored[(scored["tau_hat"] > 0) & (scored["net_value"] > 0)].copy().reset_index(drop=True)
    if df.empty:
        return df
    cost = df["offer_cost"].to_numpy(float)
    gross = (df["tau_hat"] * df["value"]).to_numpy(float)
    new = (df["tenure_months"] <= 3).to_numpy(float)
    matrix = np.vstack([np.ones(len(df)), cost, gross - policy.min_roi * cost, new - policy.min_new_user_share])
    bounds = [policy.max_contacts, policy.total_budget if policy.total_budget is not None else np.inf, np.inf, np.inf]
    objective = df["tau_hat" if policy.objective == "incremental_retention" else "net_value"].to_numpy(float)
    result = milp(-objective, integrality=np.ones(len(df)), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, [0, 0, 0, 0], bounds),
                  options={"time_limit": 10})
    if not result.success:
        raise ValueError("optimizer did not find a certified optimum; human review required")
    return df.loc[result.x > 0.5].sort_values("tau_hat", ascending=False).reset_index(drop=True)


def run_planner_agent(state: AgentState, deps: Deps) -> AgentState:
    policy = PolicyTargets.model_validate(state.get("policy_targets") or {
        "max_contacts": state.get("budget", deps.settings["agent"]["default_budget"]),
    })
    policy_dict = policy.model_dump()
    base = {"campaign_id": state.get("campaign_id") or f"cmp-{uuid4().hex}", "policy_targets": policy_dict}
    if not (state.get("targeting") or {}).get("usable"):
        return {**base, "offers": [], "plan": {"status": "blocked", "reason": "causal evidence is unusable"},
                "policy_summary": policy_summary(pd.DataFrame()),
                "trace": state.get("trace", []) + ["planner: blocked unusable causal evidence"]}
    scored = pd.DataFrame(state.get("uplift_table") or [])
    memory = state.get("memory") or {}
    exclude = set(memory.get("cooldown_ids", [])) | set(memory.get("excluded_ids", []))
    exclude |= set((state.get("constraints") or {}).get("exclude_ids", []))
    exclude |= deps.memory.contact_limit_ids(policy.max_contacts_per_week)
    from app.services.campaigns import active_assignment_ids

    exclude |= active_assignment_ids(deps)
    if not scored.empty:
        scored = scored[~scored["customer_id"].isin(exclude)]
        numeric = scored[["tau_hat", "value", "offer_cost", "net_value", "tenure_months"]]
        invalid = (not np.isfinite(numeric.to_numpy(float)).all() or (scored["offer_cost"] < 0).any()
                   or (scored["value"] <= 0).any() or (scored["tau_hat"].abs() > 1).any())
        if invalid:
            return {**base, "offers": [], "plan": {"status": "blocked", "reason": "invalid targeting numbers"},
                    "policy_summary": policy_summary(pd.DataFrame()),
                    "trace": state.get("trace", []) + ["planner: invalid numeric evidence requires human review"]}
    provider = get_provider(deps.settings["agent"]["provider"])
    evidence = []
    if not scored.empty:
        for audience, subset in (("all", scored), ("monthly", scored[scored["plan_annual"] == 0])):
            evidence.append({"audience": audience, **policy_summary(subset), "confidence_interval": None})
    designs = [CampaignDesign.model_validate(d) for d in provider.design_campaigns(policy_dict, evidence)]
    comparison, allocations = [], []
    for design in designs[:4]:
        subset = scored
        if design.audience == "monthly" and not scored.empty:
            subset = scored[scored["plan_annual"] == 0]
        try:
            chosen = allocate(subset, policy)
            reason = None if len(chosen) >= 2 else "insufficient eligible users for treatment and holdout"
        except ValueError as exc:
            chosen, reason = scored.iloc[:0], str(exc)
        summary = policy_summary(chosen)
        score = summary["expected_saved" if policy.objective == "incremental_retention" else "expected_net_value"]
        comparison.append({"design": design.model_dump(), **summary, "score": score,
                           "feasible": reason is None, "reason": reason, "confidence_interval": None})
        allocations.append(chosen)
    order = sorted(range(len(comparison)), key=lambda i: (comparison[i]["feasible"], comparison[i]["score"]), reverse=True)
    best = order[0]
    chosen = allocations[best] if comparison[best]["feasible"] else scored.iloc[:0]
    chosen = chosen.copy()
    chosen["rank"] = np.arange(1, len(chosen) + 1)
    chosen["offer"] = "retention_discount"
    cols = ["customer_id", "rank", "tau_hat", "churn_risk", "value", "offer_cost", "net_value", "offer"]
    summary = policy_summary(chosen)
    summary.update({"budget": policy.max_contacts, "total_budget": policy.total_budget,
                    "eligible_positive_net": int((scored["net_value"] > 0).sum()) if not scored.empty else 0,
                    "excluded": len(exclude), "policy": policy.objective})
    plan = {"status": "pending_approval" if not chosen.empty else "infeasible",
            "selected": comparison[best], "alternative": comparison[order[1]] if len(order) > 1 else None,
            "candidates": comparison, "holdout_fraction": policy.holdout_fraction,
            "limitation": "point-estimate planning on the existing intervention; new attributes require an experiment"}
    return {**base, "offers": chosen[[c for c in cols if c in chosen]].to_dict("records"),
            "policy_summary": summary, "plan": plan,
            "trace": state.get("trace", []) + [f"planner: optimized {len(comparison)} supported candidates; {plan['status']}"]}
