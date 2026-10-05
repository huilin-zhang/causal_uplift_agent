"""Long-term memory for the agent, stored in SQLite.

Three kinds of memory:
  run       summary of each agent run (what was chosen, what the reviewer said)
  feedback  notes from human reviewers, e.g. "exclude annual-plan customers"
  contact   handled by the contacts table: who was contacted and when (cooldown)
Code guide: section "Memory" (sec:memory).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from app.data.store import Store

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def utc_now() -> datetime:
    return datetime.now(UTC)


class MemoryService:
    def __init__(self, store: Store):
        self.store = store
        with store.connect() as conn:
            conn.executescript(SCHEMA)

    def remember(self, kind: str, content: dict, at: datetime | None = None) -> None:
        stamp = (at or utc_now()).isoformat()
        with self.store.connect() as conn:
            conn.execute(
                "INSERT INTO memory (kind, content, created_at) VALUES (?, ?, ?)",
                (kind, json.dumps(content, default=str), stamp),
            )

    def recall(self, kind: str | None = None, limit: int = 5) -> list[dict]:
        sql = "SELECT kind, content, created_at FROM memory"
        params: tuple = ()
        if kind:
            sql += " WHERE kind = ?"
            params = (kind,)
        sql += " ORDER BY id DESC LIMIT ?"
        with self.store.connect() as conn:
            rows = conn.execute(sql, params + (limit,)).fetchall()
        return [{"kind": k, "content": json.loads(c), "created_at": t} for k, c, t in rows]

    def recently_contacted(self, days: int, now: datetime | None = None) -> set[int]:
        """Customer ids contacted within the last `days` days (cooldown list)."""
        since = ((now or utc_now()) - timedelta(days=days)).isoformat()
        with self.store.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT customer_id FROM contacts WHERE contacted_at >= ?", (since,)
            ).fetchall()
        return {int(r[0]) for r in rows}
