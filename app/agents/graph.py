"""Growth-agent orchestration; evaluation and reporting run after approved launch.

Code guide: docs/manual/growth_agents_implementation.md.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.causal_targeting_agent import run_causal_targeting_agent
from app.agents.chat_agent import run_chat_agent
from app.agents.contracts import PolicyTargets
from app.agents.guardrail_agent import run_guardrail_agent
from app.agents.memory_agent import load_memory, save_memory
from app.agents.marketing_content_agent import run_marketing_content_agent
from app.agents.opportunity_discovery_agent import run_opportunity_discovery_agent
from app.agents.planner_agent import run_planner_agent
from app.agents.providers import get_provider
from app.agents.rag_agent import run_rag_agent
from app.agents.sql_agent import run_sql_agent
from app.agents.state import AgentState, Deps, build_deps
from app.services.campaigns import save_campaign


def route(state: AgentState, deps: Deps) -> AgentState:
    import re

    question = state.get("question", "")
    intent = get_provider(deps.settings["agent"]["provider"]).route(question)
    match = re.search(r"customer\s*#?\s*(\d+)", question, re.I)
    return {
        "intent": intent,
        "customer_id": int(match.group(1)) if match else state.get("customer_id"),
        "retries": 0,
        "trace": [f"router: intent = {intent}"],
    }


def save_offers(state: AgentState, deps: Deps) -> AgentState:
    """Store the reviewed list as pending; contacts are logged only after human approval."""
    offers = state.get("offers") or []
    if offers:
        save_campaign(state, deps)
    return {"trace": state.get("trace", []) + [f"save_offers: {len(offers)} offers pending approval"]}


def after_sql(state: AgentState) -> str:
    return "rag" if state["intent"] == "method" else "causal_targeting"


def after_rag(state: AgentState) -> str:
    return "opportunity_discovery" if state["intent"] in {"campaign", "discovery"} else "chat"


def after_review(state: AgentState, max_retries: int) -> str:
    review = state["review"]
    if review["passed"]:
        return "save_offers"
    if review["blocking"] or state.get("retries", 0) > max_retries:
        return "chat"
    return "planner"


def build_graph(deps: Deps):
    g = StateGraph(AgentState)
    nodes = {
        "router": route,
        "memory_load": load_memory,
        "sql": run_sql_agent,
        "causal_targeting": run_causal_targeting_agent,
        "rag": run_rag_agent,
        "opportunity_discovery": run_opportunity_discovery_agent,
        "planner": run_planner_agent,
        "marketing_content": run_marketing_content_agent,
        "guardrail": run_guardrail_agent,
        "save_offers": save_offers,
        "chat": run_chat_agent,
        "memory_save": save_memory,
    }
    for name, fn in nodes.items():
        g.add_node(name, partial(fn, deps=deps))

    g.add_edge(START, "router")
    g.add_edge("router", "memory_load")
    g.add_edge("memory_load", "sql")
    g.add_conditional_edges("sql", after_sql, {"rag": "rag", "causal_targeting": "causal_targeting"})
    g.add_edge("causal_targeting", "rag")
    g.add_conditional_edges("rag", after_rag, {"opportunity_discovery": "opportunity_discovery", "chat": "chat"})
    g.add_conditional_edges("opportunity_discovery", lambda s: "planner" if s["intent"] == "campaign" else "chat")
    g.add_edge("planner", "marketing_content")
    g.add_edge("marketing_content", "guardrail")
    max_retries = min(2, deps.settings["agent"]["max_review_retries"])
    g.add_conditional_edges(
        "guardrail", partial(after_review, max_retries=max_retries),
        {"save_offers": "save_offers", "chat": "chat", "planner": "planner"},
    )
    g.add_edge("save_offers", "chat")
    g.add_conditional_edges("chat", lambda s: "memory_save" if s["intent"] in {"campaign", "discovery"} else END)
    g.add_edge("memory_save", END)
    return g.compile()


def run_agent(question: str, budget: int | None = None, settings: dict | None = None,
              deps: Deps | None = None, policy_targets: dict | None = None) -> dict[str, Any]:
    from app.core.config import get_settings

    settings = settings or get_settings()
    deps = deps or build_deps(settings)
    graph = build_graph(deps)
    targets = PolicyTargets.model_validate(policy_targets or {
        "max_contacts": budget if budget is not None else settings["agent"]["default_budget"],
    })
    return graph.invoke({"question": question, "budget": targets.max_contacts,
                         "policy_targets": targets.model_dump()}, {"recursion_limit": 50})
