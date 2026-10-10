"""Local campaign approval, randomized assignments, and immutable evaluation records.

Approval simulates launch by recording contacts; this module sends no messages.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np

from app.agents.contracts import PolicyTargets
from app.agents.state import Deps

SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL, snapshot TEXT NOT NULL, decision TEXT, evaluation TEXT
);
CREATE TABLE IF NOT EXISTS campaign_assignments (
    campaign_id TEXT NOT NULL, customer_id INTEGER NOT NULL, treatment INTEGER NOT NULL,
    tau_hat REAL NOT NULL, PRIMARY KEY (campaign_id, customer_id)
);
CREATE TABLE IF NOT EXISTS campaign_outcomes (
    campaign_id TEXT NOT NULL, customer_id INTEGER NOT NULL, churn_30d INTEGER NOT NULL,
    PRIMARY KEY (campaign_id, customer_id)
);
CREATE TABLE IF NOT EXISTS pending_offers (
    customer_id INTEGER, rank INTEGER, tau_hat REAL, churn_risk REAL,
    value REAL, offer_cost REAL, net_value REAL, offer TEXT,
    campaign_id TEXT, status TEXT
);
"""


def save_campaign(state: dict, deps: Deps) -> None:
    if not state["review"]["passed"]:
        raise ValueError("only a guardrail-approved draft can be saved")
    now = datetime.now(UTC)
    policy = PolicyTargets.model_validate(state["policy_targets"])
    snapshot = {k: state.get(k) for k in ("policy_targets", "plan", "messages", "review", "model", "offers", "opportunities")}
    with deps.store.connect() as conn:
        conn.executescript(SCHEMA)
        conn.execute("INSERT INTO campaigns VALUES (?, ?, ?, ?, ?, NULL, NULL)", (
            state["campaign_id"], "pending_approval", now.isoformat(),
            (now + timedelta(hours=policy.approval_ttl_hours)).isoformat(), json.dumps(snapshot),
        ))
        conn.executemany(
            "INSERT INTO pending_offers "
            "(customer_id, rank, tau_hat, churn_risk, value, offer_cost, net_value, offer, campaign_id, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [tuple(o.get(k) for k in ("customer_id", "rank", "tau_hat", "churn_risk", "value", "offer_cost", "net_value", "offer"))
             + (state["campaign_id"], "pending_approval") for o in state["offers"]],
        )


def get_campaign(campaign_id: str, deps: Deps) -> dict:
    if "campaigns" not in deps.store.tables():
        raise KeyError(campaign_id)
    df = deps.store.query("SELECT * FROM campaigns WHERE campaign_id = ?", (campaign_id,))
    if df.empty:
        raise KeyError(campaign_id)
    row = df.iloc[0].to_dict()
    for key in ("snapshot", "decision", "evaluation"):
        row[key] = json.loads(row[key]) if isinstance(row[key], str) else None
    return row


def decide_campaign(campaign_id: str, approve: bool, reviewer: str, exclude_ids: list[int], note: str, deps: Deps) -> dict:
    campaign = get_campaign(campaign_id, deps)
    if exclude_ids and approve:
        raise ValueError("exclusions change policy constraints; reject and request a revised plan")
    now = datetime.now(UTC)
    snapshot = campaign["snapshot"]
    policy = PolicyTargets.model_validate(snapshot["policy_targets"])
    offers = snapshot["offers"]
    # Recheck freshness at approval, including other approved holdouts.
    excluded = deps.memory.recently_contacted(deps.settings["agent"]["contact_cooldown_days"])
    excluded |= deps.memory.contact_limit_ids(policy.max_contacts_per_week)
    excluded |= active_assignment_ids(deps)
    decision = {"reviewer": reviewer, "approve": approve, "note": note, "exclude_ids": exclude_ids}
    with deps.store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        status = conn.execute("SELECT status FROM campaigns WHERE campaign_id = ?", (campaign_id,)).fetchone()[0]
        if status != "pending_approval":
            raise ValueError(f"campaign is {status}; decision cannot be repeated")
        if datetime.fromisoformat(campaign["expires_at"]) <= now:
            conn.execute("UPDATE campaigns SET status = 'expired' WHERE campaign_id = ?", (campaign_id,))
            conn.execute("UPDATE pending_offers SET status = 'expired' WHERE campaign_id = ?", (campaign_id,))
            return {"campaign_id": campaign_id, "status": "expired", "contacts_logged": 0}
        if approve and excluded.intersection(o["customer_id"] for o in offers):
            raise ValueError("audience is no longer eligible; request a revised plan")
        # Repeat the eligibility check under the write lock to serialize
        # approval of overlapping drafts from concurrent API requests.
        if approve:
            since = (now - timedelta(days=deps.settings["agent"]["contact_cooldown_days"])).isoformat()
            recent = conn.execute("SELECT DISTINCT customer_id FROM contacts WHERE contacted_at >= ?", (since,)).fetchall()
            active = conn.execute(
                "SELECT a.customer_id FROM campaign_assignments a JOIN campaigns c USING (campaign_id) "
                "WHERE c.status IN ('approved', 'paused')"
            ).fetchall()
            week = (now - timedelta(days=7)).isoformat()
            weekly = conn.execute("SELECT customer_id FROM contacts WHERE contacted_at >= ? "
                                  "GROUP BY customer_id HAVING COUNT(*) >= ?",
                                  (week, policy.max_contacts_per_week)).fetchall()
            ineligible = {row[0] for row in recent + active + weekly}
            if ineligible.intersection(o["customer_id"] for o in offers):
                raise ValueError("audience changed during approval; request a revised plan")
        status = "approved" if approve else "rejected"
        sent = 0
        if approve:
            n_holdout = max(1, min(len(offers) - 1, round(len(offers) * policy.holdout_fraction)))
            holdout = set(np.random.default_rng().choice(len(offers), n_holdout, replace=False).tolist())
            for i, offer in enumerate(offers):
                treatment = int(i not in holdout)
                conn.execute("INSERT INTO campaign_assignments VALUES (?, ?, ?, ?)",
                             (campaign_id, offer["customer_id"], treatment, offer["tau_hat"]))
                if treatment:
                    conn.execute("INSERT INTO contacts VALUES (?, ?, ?, ?)",
                                 (offer["customer_id"], campaign_id, now.isoformat(), offer["offer"]))
                    sent += 1
        conn.execute("UPDATE campaigns SET status = ?, decision = ? WHERE campaign_id = ?",
                     (status, json.dumps(decision), campaign_id))
        conn.execute("UPDATE pending_offers SET status = ? WHERE campaign_id = ?", (status, campaign_id))
    deps.memory.remember("feedback", {"campaign_id": campaign_id, **decision})
    return {"campaign_id": campaign_id, "status": status, "contacts_logged": sent,
            "holdout_size": len(offers) - sent if approve else 0}


def active_assignment_ids(deps: Deps) -> set[int]:
    if "campaign_assignments" not in deps.store.tables():
        return set()
    rows = deps.store.query(
        "SELECT a.customer_id FROM campaign_assignments a JOIN campaigns c USING (campaign_id) "
        "WHERE c.status IN ('approved', 'paused')"
    )
    return set(rows["customer_id"].astype(int))
