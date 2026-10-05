"""Report node: turn the state into a short brief.

The template text is built only from numbers in the state. An optional LLM
may rephrase it, but providers.numbers_match() rejects any new number.
Code guide: section "Report" (sec:report).
"""

from __future__ import annotations

import pandas as pd

from app.agents.policy_agent import load_scores
from app.agents.providers import get_provider
from app.agents.state import AgentState, Deps


def _pp(x: float) -> str:
    return f"{100 * x:.2f} pp"


def _usd(x: float) -> str:
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def _experiment_text(exp: dict) -> str:
    lin = exp["ate_lin"]
    sig = "significant" if lin["ci_low"] > 0 or lin["ci_high"] < 0 else "not significant at 5%"
    return (
        f"A/B test: SRM p = {exp['srm_p']:.3f}, max |SMD| = {exp['max_abs_smd']:.3f} "
        f"({'valid' if exp['valid'] else 'NOT valid'}). Outreach reduced 30-day churn by "
        f"{_pp(lin['estimate'])} (Lin-adjusted; 95% CI for the reduction {_pp(lin['ci_low'])} to "
        f"{_pp(lin['ci_high'])}, {sig})."
    )


def _campaign_text(state: AgentState) -> str:
    s, review = state.get("policy_summary") or {}, state.get("review") or {}
    model = state.get("model") or {}
    lines = [_experiment_text(state["experiment"])]
    lines.append(
        f"Model: {model.get('method')} ({model.get('version')}), chosen by rule: "
        + "; ".join((state.get("selection") or {}).get("reasons", [])[:1])
    )
    head = model.get("headline")
    if head:
        rel = head["relative_gain_ratio_of_totals"]
        lines.append(
            f"In simulation with known effects, this model kept {100 * rel['estimate']:.1f}% more customers "
            f"than churn-risk ranking at {head['description'].split('top ')[1].split(' ')[0]} coverage "
            f"(95% CI {100 * rel['ci_low']:.1f}% to {100 * rel['ci_high']:.1f}%)."
        )
    lines.append(
        f"Offer list: {s.get('n_offers', 0)} customers (budget {s.get('budget')}). The model expects "
        f"{s.get('expected_saved', 0):.1f} extra customers retained and a net value of "
        f"${s.get('expected_net_value', 0):,.0f} after ${s.get('total_cost', 0):,.0f} of offer cost. "
        f"{s.get('eligible_positive_net', 0)} current subscribers had positive predicted net value."
    )
    if review.get("passed"):
        lines.append(f"Reviewer checks passed after {state.get('retries', 0)} retries. Campaign {state.get('campaign_id')} awaits human approval.")
    else:
        issues = review.get("blocking") or review.get("fixable") or []
        lines.append("Reviewer did not pass the list: " + "; ".join(issues[:3]))
    return "\n".join(lines)


def _customer_text(state: AgentState, deps: Deps) -> str:
    cid = state.get("customer_id")
    try:
        scored = load_scores(deps)
        row = scored[scored["customer_id"] == cid]
    except Exception:
        row = pd.DataFrame()
    if row.empty:
        return f"Customer {cid} was not found among current subscribers."
    r = row.iloc[0]
    decision = "send an offer" if r["tau_hat"] > 0 and r["net_value"] > 0 else "do not contact"
    return (
        f"Customer {cid}: churn risk {100 * r['churn_risk']:.1f}%, predicted churn reduction from outreach "
        f"{_pp(r['tau_hat'])}, value {_usd(r['value'])}, offer cost {_usd(r['offer_cost'])}, "
        f"net value {_usd(r['net_value'])}. Recommendation: {decision}."
    )


def build_text(state: AgentState, deps: Deps) -> str:
    intent = state["intent"]
    if intent == "campaign":
        text = _campaign_text(state)
    elif intent == "experiment":
        text = _experiment_text(state["experiment"])
    elif intent == "customer":
        text = _customer_text(state, deps)
    else:
        text = "Relevant notes:"
    hits = state.get("knowledge") or []
    if hits:
        text += "\nSources: " + "; ".join(h["citation"] for h in hits)
        if intent == "method":
            text += "\n" + hits[0]["text"]
    return text


def run_report(state: AgentState, deps: Deps) -> AgentState:
    provider = get_provider(deps.settings["agent"]["provider"])
    template = build_text(state, deps)
    facts = {k: state.get(k) for k in ("intent", "experiment", "policy_summary", "review", "model")}
    narrative = provider.narrate(facts, template)
    return {
        "narrative": narrative,
        "provider": provider.name,
        "trace": state.get("trace", []) + [f"report: written by {provider.name} provider"],
    }
