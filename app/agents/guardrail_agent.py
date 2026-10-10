"""Guardrail Agent: independently check evidence, allocations and rendered copy."""

import math

from app.agents.contracts import PolicyTargets
from app.agents.marketing_content_agent import render_copy
from app.agents.reviewer_agent import review
from app.agents.state import AgentState, Deps


def run_guardrail_agent(state: AgentState, deps: Deps) -> AgentState:
    result = review(state, deps.settings)
    failures = result["blocking"]
    policy = PolicyTargets.model_validate(state["policy_targets"])
    offers = state.get("offers") or []
    if not (state.get("targeting") or {}).get("usable"):
        failures.append("Causal Targeting evidence is unusable")
    if (state.get("plan") or {}).get("status") != "pending_approval" or len(offers) < 2:
        failures.append("No feasible campaign with treatment and holdout; human decision required")
    cost = sum(o["offer_cost"] for o in offers)
    gross = sum(o["tau_hat"] * o["value"] for o in offers)
    if not math.isfinite(cost + gross):
        failures.append("Non-finite campaign economics")
    if policy.total_budget is not None and cost > policy.total_budget + 1e-6:
        failures.append("Total monetary budget exceeded")
    if gross + 1e-6 < policy.min_roi * cost:
        failures.append("ROI constraint violated")
    if len(offers) > policy.max_contacts:
        failures.append("Contact limit exceeded")
    rows = {r["customer_id"]: r for r in state.get("uplift_table", [])}
    for offer in offers:
        evidence = rows.get(offer["customer_id"])
        if evidence is None or any(offer[k] != evidence[k] for k in ("tau_hat", "value", "offer_cost", "net_value")):
            failures.append("Plan numbers differ from Causal Targeting evidence")
            break
    n_new = sum(rows.get(o["customer_id"], {}).get("tenure_months", 999) <= 3 for o in offers)
    if n_new + 1e-6 < policy.min_new_user_share * len(offers):
        failures.append("New-user share constraint violated")
    weekly = deps.memory.contact_limit_ids(policy.max_contacts_per_week)
    if weekly.intersection(o["customer_id"] for o in offers):
        failures.append("Weekly contact frequency exceeded")
    design = ((state.get("plan") or {}).get("selected") or {}).get("design", {})
    messages = state.get("messages") or []
    if offers and not messages:
        failures.append("Campaign copy is missing")
    banned = deps.settings["agent"].get("prohibited_claims", ["guaranteed", "risk-free", "only today"])
    for message in messages:
        try:
            if render_copy(message["template"], message["values"]) != message["text"]:
                failures.append("Rendered copy differs from approved values")
            # Current plan has no approved customer-facing numbers.
            if message["values"]:
                failures.append("No customer-facing numeric values are approved by this plan")
        except (ValueError, KeyError):
            failures.append("Invalid numeric placeholders")
        if any(term.casefold() in message["text"].casefold() for term in banned):
            failures.append("Prohibited claim in campaign copy")
        if any(message.get(k) != design.get(k) for k in ("channel", "message_type", "tone", "length", "timing")):
            failures.append("Message attributes differ from the approved design")
    result["passed"] = not failures and not result["fixable"]
    result["requires_human_copy_review"] = True
    out = {"review": result, "trace": state.get("trace", []) + [f"guardrail: {'passed' if result['passed'] else 'blocked or revision required'}"]}
    if result["fixable"] and not failures:
        constraints = dict(state.get("constraints") or {})
        constraints["exclude_ids"] = sorted(set(constraints.get("exclude_ids", [])) | set(result["bad_ids"]))
        out.update({"constraints": constraints, "retries": state.get("retries", 0) + 1})
    return out
