"""SQL sub-agent: pull the facts each intent needs, through named read-only queries.

Code guide: section "SQL agent" (sec:sql-agent).
"""

from __future__ import annotations

from app.agents.state import AgentState, Deps

QUERIES_BY_INTENT = {
    "campaign": ["arm_summary", "current_population"],
    "experiment": ["arm_summary", "churn_by_plan"],
    "customer": ["customer"],
    "method": [],
}


def run_sql_agent(state: AgentState, deps: Deps) -> AgentState:
    results, notes = {}, []
    for name in QUERIES_BY_INTENT.get(state["intent"], []):
        params = (state.get("customer_id"),) if name == "customer" else ()
        try:
            results[name] = deps.sql.run_template(name, params).to_dict("records")
        except Exception as exc:  # missing table before the first pipeline run
            notes.append(f"{name} failed: {exc}")
    trace = f"sql_agent: ran {', '.join(results) or 'no queries'}" + (f" ({'; '.join(notes)})" if notes else "")
    return {"sql": results, "trace": state.get("trace", []) + [trace]}
