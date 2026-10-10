"""Campaign Evaluation Agent: a complete, fixed-horizon randomized readout.

The response is intention-to-treat churn reduction. This implementation does
not do repeated significance peeking or claim to implement sequential tests.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from app.agents.state import Deps
from app.services.campaigns import get_campaign


def estimate_campaign(outcomes: pd.DataFrame, predictions: np.ndarray) -> dict:
    t = outcomes["treatment"].to_numpy() == 1
    y = outcomes["churn_30d"].to_numpy(float)
    if min(t.sum(), (~t).sum()) < 2:
        raise ValueError("at least two complete outcomes per randomized arm are required")
    a, b = y[~t], y[t]
    effect = float(a.mean() - b.mean())
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se else len(y) - 2
    margin = float(stats.t.ppf(0.975, df) * se)
    predicted = float(np.mean(predictions))
    return {"estimate": effect, "ci_low": effect - margin, "ci_high": effect + margin,
            "se": se, "n_treatment": int(t.sum()), "n_holdout": int((~t).sum()),
            "predicted_uplift": predicted, "prediction_error": effect - predicted,
            "incremental_retained": effect * int(t.sum()),
            "pause_required": effect + margin < 0,
            "method": "fixed-horizon randomized difference in means; Welch interval",
            "outcome": "churn_30d", "direction": "positive means reduced churn"}


def evaluate_campaign(campaign_id: str, records: list[dict], deps: Deps) -> dict:
    campaign = get_campaign(campaign_id, deps)
    if campaign["evaluation"] is not None:
        raise ValueError("campaign already evaluated; read the stored report")
    if campaign["status"] != "approved":
        raise ValueError("only an approved campaign can be evaluated")
    assignments = deps.store.query("SELECT * FROM campaign_assignments WHERE campaign_id = ?", (campaign_id,))
    outcomes = pd.DataFrame(records)
    if outcomes.empty or outcomes["customer_id"].duplicated().any():
        raise ValueError("outcomes must contain unique assigned customers")
    if set(outcomes["customer_id"]) != set(assignments["customer_id"]):
        raise ValueError("complete outcomes for every randomized customer are required")
    if not outcomes["churn_30d"].isin([0, 1]).all():
        raise ValueError("churn_30d must be binary")
    merged = assignments.merge(outcomes, on="customer_id", validate="one_to_one")
    evaluation = {"campaign_id": campaign_id, **estimate_campaign(merged, merged["tau_hat"].to_numpy()),
                  "feedback": {"causal_targeting": "campaign_outcomes joined to campaign_assignments",
                               "opportunity_discovery": "campaign_evaluation memory",
                               "planner": "prediction_error calibration; no automatic model or policy change"}}
    from app.agents.report import run_report

    report = run_report({"evaluation": evaluation}, deps)
    evaluation["report"] = report["narrative"]
    with deps.store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute("SELECT evaluation FROM campaigns WHERE campaign_id = ?", (campaign_id,)).fetchone()
        if current[0] is not None:
            raise ValueError("campaign already evaluated")
        conn.executemany("INSERT INTO campaign_outcomes VALUES (?, ?, ?)",
                         [(campaign_id, int(r["customer_id"]), int(r["churn_30d"])) for r in records])
        conn.execute("UPDATE campaigns SET status = ?, evaluation = ? WHERE campaign_id = ?",
                     ("paused" if evaluation["pause_required"] else "evaluated", json.dumps(evaluation), campaign_id))
        conn.execute("UPDATE pending_offers SET status = ? WHERE campaign_id = ?",
                     ("paused" if evaluation["pause_required"] else "evaluated", campaign_id))
    deps.memory.remember("campaign_evaluation", evaluation)
    return evaluation
