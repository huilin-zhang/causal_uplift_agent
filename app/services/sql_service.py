"""Read-only SQL access for the SQL agent.

Two layers of safety:
  1. Named query templates cover the common questions; parameters are bound,
     never pasted into the SQL string.
  2. Free-form SQL (from an LLM, optional) must be a single SELECT, and it runs
     on a read-only SQLite connection, so a write fails even if the check is fooled.
Code guide: section "SQL service" (sec:sql).
"""

from __future__ import annotations

import re

import pandas as pd

from app.data.store import Store

TEMPLATES: dict[str, str] = {
    "arm_summary": """
        SELECT treatment, COUNT(*) AS n, AVG(churn_30d) AS churn_rate
        FROM experiment GROUP BY treatment ORDER BY treatment
    """,
    "churn_by_plan": """
        SELECT plan_annual, treatment, COUNT(*) AS n, AVG(churn_30d) AS churn_rate
        FROM experiment GROUP BY plan_annual, treatment ORDER BY plan_annual, treatment
    """,
    "current_population": """
        SELECT COUNT(*) AS n, AVG(monthly_fee) AS avg_fee, AVG(value) AS avg_value,
               AVG(offer_cost) AS avg_cost
        FROM current_subscribers
    """,
    "recent_contacts": """
        SELECT customer_id, MAX(contacted_at) AS last_contacted
        FROM contacts WHERE contacted_at >= ? GROUP BY customer_id
    """,
    "customer": "SELECT * FROM current_subscribers WHERE customer_id = ?",
}

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum)\b", re.I
)


class SQLGuardError(ValueError):
    pass


def check_sql(sql: str) -> str:
    """Accept one SELECT (or WITH ... SELECT) statement; reject anything else."""
    stripped = sql.strip().rstrip(";").strip()
    if ";" in stripped:
        raise SQLGuardError("only one statement is allowed")
    if not re.match(r"^(select|with)\b", stripped, re.I):
        raise SQLGuardError("only SELECT queries are allowed")
    if _FORBIDDEN.search(stripped):
        raise SQLGuardError("query contains a forbidden keyword")
    return stripped


class SQLService:
    def __init__(self, store: Store):
        self.store = store

    def run_template(self, name: str, params: tuple = ()) -> pd.DataFrame:
        if name not in TEMPLATES:
            raise KeyError(f"unknown template: {name}")
        return self.store.query(TEMPLATES[name], params)

    def run_sql(self, sql: str, limit: int = 1000) -> pd.DataFrame:
        safe = check_sql(sql)
        return self.store.query(f"SELECT * FROM ({safe}) LIMIT {int(limit)}")
