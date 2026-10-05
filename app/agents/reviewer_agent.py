"""Reviewer sub-agent: rule checks before any list reaches a human.

Two kinds of problems:
  blocking  the evidence is not trustworthy (broken A/B test, estimator not
            allowed for this data, model no better than random). No retry;
            the run ends with "blocked".
  fixable   individual offers break a rule (negative effect, negative net
            value, cooldown, over budget). The reviewer names the customers
            and sends the list back to the policy agent once more.
Code guide: section "Reviewer agent" (sec:reviewer).
"""

from __future__ import annotations

from app.agents.state import AgentState, Deps


def review(state: AgentState, settings: dict) -> dict:
    blocking, fixable, bad_ids = [], [], set()
    exp = state.get("experiment") or {}
    exp_cfg = settings["experiment"]

    if exp.get("srm_p", 1.0) < exp_cfg["srm_alpha"]:
        blocking.append(f"SRM: arm sizes differ from the plan (p = {exp['srm_p']:.2g})")
    if exp.get("max_abs_smd", 0.0) >= exp_cfg["smd_threshold"]:
        blocking.append(f"Imbalance: max |SMD| = {exp['max_abs_smd']:.3f}")

    model = state.get("model") or {}
    allowed = (state.get("selection") or {}).get("estimators", [])
    if model.get("method") and model["method"] not in allowed:
        blocking.append(f"Estimator {model['method']} is not allowed for this data ({', '.join(allowed)})")
    qini = (model.get("metrics") or {}).get("qini_true_normalized")
    if qini is not None and qini <= 0:
        blocking.append("Model ranks no better than random (normalized Qini <= 0)")

    offers = state.get("offers") or []
    memory = state.get("memory") or {}
    cooldown = set(memory.get("cooldown_ids", [])) | set(memory.get("excluded_ids", []))
    seen = set()
    for o in offers:
        cid = int(o["customer_id"])
        if o["tau_hat"] <= 0:
            fixable.append(f"customer {cid}: predicted effect <= 0 (sleeping dog or no effect)")
            bad_ids.add(cid)
        elif o["net_value"] <= 0:
            fixable.append(f"customer {cid}: net value <= 0")
            bad_ids.add(cid)
        if cid in cooldown:
            fixable.append(f"customer {cid}: contacted recently or excluded by a reviewer")
            bad_ids.add(cid)
        if cid in seen:
            fixable.append(f"customer {cid}: listed twice")
        seen.add(cid)
    budget = int(state.get("budget") or settings["agent"]["default_budget"])
    if len(offers) > budget:
        fixable.append(f"{len(offers)} offers exceed the budget of {budget}")

    return {
        "passed": not blocking and not fixable,
        "blocking": blocking,
        "fixable": fixable[:20],
        "n_fixable": len(fixable),
        "bad_ids": sorted(bad_ids),
    }


def run_reviewer(state: AgentState, deps: Deps) -> AgentState:
    result = review(state, deps.settings)
    out: AgentState = {"review": result}
    if result["fixable"] and not result["blocking"]:
        constraints = dict(state.get("constraints") or {})
        constraints["exclude_ids"] = sorted(set(constraints.get("exclude_ids", [])) | set(result["bad_ids"]))
        out["constraints"] = constraints
        out["retries"] = state.get("retries", 0) + 1
    if result["passed"]:
        verdict = "passed"
    elif result["blocking"]:
        verdict = "blocked"
    else:
        verdict = f"{result['n_fixable']} fixable issues"
    out["trace"] = state.get("trace", []) + [f"reviewer_agent: {verdict}"]
    return out
