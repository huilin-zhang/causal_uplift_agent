"""LangGraph wiring of the sub-agents.

    START -> router -> memory(load) -> sql -+-> ml -> rag -+-> policy -> reviewer -+-> save_offers -> report
                                            |              |      ^               |
                                            +-> rag (method questions)  +-- retry -+ (fixable issues, max 2)
                                                           +-> report (experiment / customer / method)
    report -> memory(save) -> END

Code guide: section "Agent graph" (sec:graph).
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd
from langgraph.graph import END, START, StateGraph

from app.agents.memory_agent import load_memory, save_memory
from app.agents.ml_agent import run_ml_agent
from app.agents.policy_agent import run_policy_agent
from app.agents.providers import get_provider
from app.agents.rag_agent import run_rag_agent
from app.agents.report import run_report
from app.agents.reviewer_agent import run_reviewer
from app.agents.sql_agent import run_sql_agent
from app.agents.state import AgentState, Deps, build_deps


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
    offers = pd.DataFrame(state.get("offers") or [])
    if not offers.empty:
        offers["campaign_id"] = state["campaign_id"]
        offers["status"] = "pending_approval"
        with deps.store.connect() as conn:
            offers.to_sql("pending_offers", conn, if_exists="append", index=False)
    return {"trace": state.get("trace", []) + [f"save_offers: {len(offers)} offers pending approval"]}


def after_sql(state: AgentState) -> str:
    return "rag" if state["intent"] == "method" else "ml"


def after_rag(state: AgentState) -> str:
    return "policy" if state["intent"] == "campaign" else "report"


def after_review(state: AgentState, max_retries: int) -> str:
    review = state["review"]
    if review["passed"]:
        return "save_offers"
    if review["blocking"] or state.get("retries", 0) > max_retries:
        return "report"
    return "policy"


def build_graph(deps: Deps):
    g = StateGraph(AgentState)
    nodes = {
        "router": route,
        "memory_load": load_memory,
        "sql": run_sql_agent,
        "ml": run_ml_agent,
        "rag": run_rag_agent,
        "policy": run_policy_agent,
        "reviewer": run_reviewer,
        "save_offers": save_offers,
        "report": run_report,
        "memory_save": save_memory,
    }
    for name, fn in nodes.items():
        g.add_node(name, partial(fn, deps=deps))

    g.add_edge(START, "router")
    g.add_edge("router", "memory_load")
    g.add_edge("memory_load", "sql")
    g.add_conditional_edges("sql", after_sql, {"rag": "rag", "ml": "ml"})
    g.add_edge("ml", "rag")
    g.add_conditional_edges("rag", after_rag, {"policy": "policy", "report": "report"})
    g.add_edge("policy", "reviewer")
    max_retries = deps.settings["agent"]["max_review_retries"]
    g.add_conditional_edges(
        "reviewer", partial(after_review, max_retries=max_retries),
        {"save_offers": "save_offers", "report": "report", "policy": "policy"},
    )
    g.add_edge("save_offers", "report")
    g.add_edge("report", "memory_save")
    g.add_edge("memory_save", END)
    return g.compile()


def run_agent(question: str, budget: int | None = None, settings: dict | None = None,
              deps: Deps | None = None) -> dict[str, Any]:
    from app.core.config import get_settings

    settings = settings or get_settings()
    deps = deps or build_deps(settings)
    graph = build_graph(deps)
    return graph.invoke({"question": question, "budget": budget or settings["agent"]["default_budget"]})
