import sqlite3

import pandas as pd
import pytest

from app.core.config import ROOT
from app.data.store import Store
from app.services.knowledge import KnowledgeBase
from app.services.memory import MemoryService
from app.services.sql_service import SQLGuardError, SQLService, check_sql


@pytest.mark.parametrize("bad", [
    "DROP TABLE experiment",
    "SELECT 1; DELETE FROM contacts",
    "UPDATE experiment SET churn_30d = 0",
    "PRAGMA table_info(experiment)",
])
def test_sql_guard_blocks_writes(bad):
    with pytest.raises(SQLGuardError):
        check_sql(bad)


def test_sql_guard_allows_select():
    assert check_sql("SELECT * FROM experiment;").startswith("SELECT")


def test_read_only_connection_blocks_writes_even_past_the_guard(tmp_path):
    store = Store(tmp_path / "t.db")
    store.write_table("x", pd.DataFrame({"a": [1]}))
    with pytest.raises(sqlite3.OperationalError), store.connect(read_only=True) as conn:
        conn.execute("DELETE FROM x")


def test_sql_template_runs(tmp_path):
    store = Store(tmp_path / "t.db")
    store.write_table("experiment", pd.DataFrame({"treatment": [0, 1, 1], "churn_30d": [1, 0, 1]}))
    rows = SQLService(store).run_template("arm_summary")
    assert list(rows["n"]) == [1, 2]


def test_knowledge_search_cites_source():
    hits = KnowledgeBase(ROOT / "docs" / "knowledge").search("sample ratio mismatch chi-square")
    assert hits and hits[0].source == "ab_test_checks.md"


def test_memory_round_trip_and_cooldown(tmp_path):
    store = Store(tmp_path / "t.db")
    mem = MemoryService(store)
    mem.remember("feedback", {"exclude_ids": [5]})
    assert mem.recall("feedback")[0]["content"]["exclude_ids"] == [5]
    store.log_contacts(pd.DataFrame({"customer_id": [9], "offer": ["x"]}), "c1", "2999-01-01T00:00:00+00:00")
    assert mem.recently_contacted(30) == {9}
