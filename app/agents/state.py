"""Shared state passed between LangGraph nodes, and the services the nodes use.

Code guide: section "Agent state" (sec:state).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from app.causal.scorer import UpliftScorer
from app.data.store import Store
from app.services.knowledge import KnowledgeBase
from app.services.memory import MemoryService
from app.services.sql_service import SQLService


class AgentState(TypedDict, total=False):
    # Input
    question: str
    budget: int
    customer_id: int | None
    # Router
    intent: str                     # campaign | experiment | method | customer
    # Memory agent
    memory: dict[str, Any]          # recent runs, reviewer feedback, cooldown ids
    # SQL agent
    sql: dict[str, Any]             # template name -> rows
    # ML agent
    experiment: dict[str, Any]
    profile: dict[str, Any]
    selection: dict[str, Any]
    model: dict[str, Any]           # method, version, headline metrics
    # RAG agent
    knowledge: list[dict[str, Any]]
    # Policy agent
    constraints: dict[str, Any]     # set by the reviewer on a retry
    offers: list[dict[str, Any]]
    policy_summary: dict[str, Any]
    campaign_id: str
    # Reviewer agent
    review: dict[str, Any]
    retries: int
    # Report
    narrative: str
    provider: str
    trace: list[str]


@dataclass
class Deps:
    """Services injected into every node, built once per run."""

    settings: dict[str, Any]
    store: Store
    sql: SQLService
    kb: KnowledgeBase
    memory: MemoryService
    scorer: UpliftScorer | None


def build_deps(settings: dict[str, Any]) -> Deps:
    from app.services.registry import load_champion

    store = Store(settings["paths"]["database"])
    try:
        scorer = load_champion(settings)
    except Exception:  # no model trained yet; the ML agent reports this
        scorer = None
    return Deps(
        settings=settings,
        store=store,
        sql=SQLService(store),
        kb=KnowledgeBase(settings["paths"]["knowledge"]),
        memory=MemoryService(store),
        scorer=scorer,
    )
