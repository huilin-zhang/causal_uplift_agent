"""Opportunity Discovery Agent: bounded, evidence-checked segment briefs."""

from app.agents.causal_targeting_agent import estimate_segment_uplift
from app.agents.state import AgentState, Deps


def run_opportunity_discovery_agent(state: AgentState, deps: Deps) -> AgentState:
    briefs, rejected, explorations = [], [], []
    history = deps.memory.recall("campaign_evaluation", limit=10)
    # Exactly two pre-specified hypotheses; no repeated outcome-driven slicing.
    for annual, audience in ((0, "monthly"), (1, "annual")):
        evidence = estimate_segment_uplift(deps, annual, queries=2)
        item = {"audience": audience, "intervention": "retention_discount", "evidence": evidence}
        if not evidence["estimable"]:
            explorations.append({**item, "status": "exploration_requires_approval"})
        elif evidence["uplift"]["ci_low"] > 0 and evidence["net_value"]["ci_low"] > 0:
            briefs.append({**item, "status": "brief_requires_approval",
                           "risks": ["segment averages do not establish individual significance"]})
        else:
            rejected.append({**item, "reason": "positive incremental net value not established"})
    briefs.sort(key=lambda b: b["evidence"]["net_value"]["ci_low"], reverse=True)
    result = {"briefs": briefs, "rejected": rejected, "explorations": explorations,
              "campaign_findings": history, "query_count": 2,
              "mode": "fixed-segment discovery; briefs require human approval"}
    # Store evidence and rejections for audit; reruns refresh rather than launch.
    deps.memory.remember("opportunity_discovery", result)
    return {"opportunities": result,
            "trace": state.get("trace", []) + ["opportunity_discovery: checked two independent RCT segment hypotheses"]}
