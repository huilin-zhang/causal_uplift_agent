"""SQLite storage for subscribers, scores, offers, and contact history.

The SQL agent reads these tables. Code guide: section "Data store" (sec:store).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS contacts (
    customer_id INTEGER,
    campaign_id TEXT,
    contacted_at TEXT,
    offer TEXT
);
"""


class Store:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self, read_only: bool = False) -> Iterator[sqlite3.Connection]:
        if read_only:
            uri = f"file:{self.path.as_posix()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
        else:
            conn = sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def write_table(self, name: str, df: pd.DataFrame) -> None:
        with self.connect() as conn:
            df.to_sql(name, conn, if_exists="replace", index=False)

    def read_table(self, name: str) -> pd.DataFrame:
        with self.connect(read_only=True) as conn:
            return pd.read_sql(f"SELECT * FROM {name}", conn)  # noqa: S608 (internal names only)

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        """Run a SELECT on a read-only connection. Writes fail at the driver level."""
        with self.connect(read_only=True) as conn:
            return pd.read_sql(sql, conn, params=params)

    def tables(self) -> list[str]:
        with self.connect(read_only=True) as conn:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return sorted(r[0] for r in rows)

    def log_contacts(self, offers: pd.DataFrame, campaign_id: str, contacted_at: str) -> int:
        rows = offers.assign(campaign_id=campaign_id, contacted_at=contacted_at)
        rows = rows[["customer_id", "campaign_id", "contacted_at", "offer"]]
        with self.connect() as conn:
            rows.to_sql("contacts", conn, if_exists="append", index=False)
        return len(rows)
