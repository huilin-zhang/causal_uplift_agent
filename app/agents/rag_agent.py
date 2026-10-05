"""RAG sub-agent: retrieve method cards that explain or constrain the decision.

The query is built from the question plus what the ML agent found (selected
estimators, warnings), so the cited rules match the situation.
Code guide: section "RAG agent" (sec:rag-agent).
"""

from __future__ import annotations

from app.agents.state import AgentState, Deps

EXTRA_QUERY = {
    "campaign": "net value budget offer rules cooldown sleeping dogs",
    "experiment": "sample ratio mismatch covariate balance average treatment effect power",
    "method": "",
    "customer": "net value uplift churn risk",
}


def run_rag_agent(state: AgentState, deps: Deps) -> AgentState:
    selection = state.get("selection") or {}
    query = " ".join(
        [state.get("question", ""), EXTRA_QUERY.get(state["intent"], "")]
        + selection.get("estimators", [])[:1]
        + selection.get("warnings", [])
    )
    hits = deps.kb.search(query, k=3)
    return {
        "knowledge": [{"citation": h.citation(), "text": h.text, "score": round(h.score, 3)} for h in hits],
        "trace": state.get("trace", []) + [f"rag_agent: {len(hits)} passages ({', '.join(h.citation() for h in hits)})"],
    }
