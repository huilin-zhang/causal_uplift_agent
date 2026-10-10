"""Chat Agent: read-only answers about historical, current and planned work."""

import json

from app.agents.providers import get_provider
from app.agents.report import _experiment_text, build_text
from app.agents.state import AgentState, Deps


def run_chat_agent(state: AgentState, deps: Deps) -> AgentState:
    intent = state["intent"]
    context = {}
    if intent == "chat":
        if "campaigns" in deps.store.tables():
            campaigns = deps.store.query(
                "SELECT campaign_id, status, snapshot, evaluation FROM campaigns ORDER BY created_at DESC LIMIT 5"
            ).to_dict("records")
            for campaign in campaigns:
                snapshot = json.loads(campaign.pop("snapshot"))
                campaign.update({k: snapshot[k] for k in ("policy_targets", "plan", "messages", "review")})
                campaign["evaluation"] = json.loads(campaign["evaluation"]) if isinstance(campaign["evaluation"], str) else None
            context["campaigns"] = campaigns
        context["opportunities"] = deps.memory.recall("opportunity_discovery", limit=1)
        context["evaluations"] = deps.memory.recall("campaign_evaluation", limit=3)
        context["historical_experiment"] = state.get("experiment")
        lines = []
        for campaign in context.get("campaigns", []):
            policy = campaign["policy_targets"]
            lines.append(f"Campaign {campaign['campaign_id']}: {campaign['status']}. "
                         f"Policy objective: {policy['objective']}; maximum audience: {policy['max_contacts']}; "
                         f"monetary budget: {policy['total_budget']}; minimum gross ROI: {policy['min_roi']}.")
            lines.append("Guardrail checks " + ("passed." if campaign["review"]["passed"] else "failed."))
            lines.extend("Draft message: " + message["text"] for message in campaign["messages"])
            if campaign["evaluation"]:
                lines.append(campaign["evaluation"]["report"])
        if not lines:
            lines.append("No campaigns have been recorded yet.")
        if context["historical_experiment"]:
            lines.append(_experiment_text(context["historical_experiment"]))
        if context["opportunities"]:
            recent = context["opportunities"][0]["content"]
            lines.append(f"Latest discovery: {len(recent['briefs'])} briefs awaiting approval, "
                         f"{len(recent['explorations'])} exploration proposals, {len(recent['rejected'])} rejected hypotheses.")
        template = "\n".join(lines)
    elif intent == "discovery":
        template = "Opportunity discovery (briefs and explorations require human approval):\n" + json.dumps(state["opportunities"], indent=2)
    elif intent in {"experiment", "campaign"} and not state.get("experiment"):
        template = "Causal evidence is unavailable. Run the pipeline before planning a campaign."
    else:
        template = build_text(state, deps)
    provider = get_provider(deps.settings["agent"]["provider"])
    return {"narrative": provider.narrate(context, template), "provider": provider.name, "chat_context": context,
            "trace": state.get("trace", []) + ["chat: answered from read-only workflow records"]}
