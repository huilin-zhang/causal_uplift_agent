"""Memory sub-agent: load context at the start of a run, save a summary at the end.

Code guide: section "Memory agent" (sec:memory-agent).
"""

from __future__ import annotations

from app.agents.state import AgentState, Deps


def load_memory(state: AgentState, deps: Deps) -> AgentState:
    days = deps.settings["agent"]["contact_cooldown_days"]
    feedback = deps.memory.recall("feedback", limit=5)
    # Reviewer feedback can carry customer exclusions for future campaigns.
    excluded = {int(i) for f in feedback for i in f["content"].get("exclude_ids", [])}
    return {
        "memory": {
            "recent_runs": deps.memory.recall("run", limit=3),
            "feedback": feedback,
            "cooldown_ids": sorted(deps.memory.recently_contacted(days)),
            "excluded_ids": sorted(excluded),
        },
        "trace": state.get("trace", []) + ["memory_agent: loaded runs, feedback, cooldown list"],
    }


def save_memory(state: AgentState, deps: Deps) -> AgentState:
    deps.memory.remember(
        "run",
        {
            "question": state.get("question"),
            "intent": state.get("intent"),
            "campaign_id": state.get("campaign_id"),
            "method": (state.get("model") or {}).get("method"),
            "n_offers": (state.get("policy_summary") or {}).get("n_offers"),
            "review_passed": (state.get("review") or {}).get("passed"),
        },
    )
    return {"trace": state.get("trace", []) + ["memory_agent: saved run summary"]}
