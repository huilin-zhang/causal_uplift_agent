"""Marketing Content Agent: copy attributes come from the plan; workflow renders values."""

from string import Formatter

from app.agents.state import AgentState, Deps


def render_copy(template: str, values: dict[str, str]) -> str:
    fields = {name for _, name, _, _ in Formatter().parse(template) if name is not None}
    if fields - values.keys() or any(character.isdigit() for character in template):
        raise ValueError("copy must use approved placeholders for numeric values")
    return template.format_map(values)


def run_marketing_content_agent(state: AgentState, deps: Deps) -> AgentState:
    plan = state.get("plan") or {}
    messages = []
    if plan.get("status") == "pending_approval":
        design = plan["selected"]["design"]
        # offer_cost is an economic cost, not the customer's discount amount.
        # Do not put it into copy. No face value or expiry is supplied by this RCT.
        template = "An offer is available for your subscription. View the terms in your account."
        messages.append({"channel": design["channel"], "message_type": design["message_type"],
                         "tone": design["tone"], "length": design["length"], "timing": design["timing"],
                         "template": template, "values": {}, "text": render_copy(template, {}),
                         "status": "draft_requires_human_approval"})
    return {"messages": messages, "trace": state.get("trace", []) + ["Marketing Content Agent: drafted copy with plan attributes"]}
