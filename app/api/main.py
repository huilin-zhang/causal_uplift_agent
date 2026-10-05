"""FastAPI service.

    uvicorn app.api.main:app --reload

Every response that depends on a model carries model_version, so a score can
be traced back to the MLflow version that produced it.
Code guide: section "API" (sec:api).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException

from app.agents.graph import run_agent
from app.agents.state import Deps, build_deps
from app.api.schemas import (
    AgentRequest,
    AgentResponse,
    ApproveRequest,
    RecommendRequest,
    ScoreRequest,
    ScoreResponse,
    ScoreRow,
)
from app.causal.policy import policy_summary, rank_by_net_value
from app.core.config import get_settings
from app.services.registry import registry_status

app = FastAPI(title="Causal Uplift Agent", version="0.1.0")
_DEPS: Deps | None = None


def get_deps() -> Deps:
    """Build services once per process; reload after a rollback via /model/reload."""
    global _DEPS
    if _DEPS is None:
        _DEPS = build_deps(get_settings())
    return _DEPS


def _report(deps: Deps, name: str) -> dict:
    path = Path(deps.settings["paths"]["artifacts"]) / "reports" / name
    if not path.exists():
        raise HTTPException(404, f"{name} not found; run the pipeline first")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/health")
def health(deps: Deps = Depends(get_deps)) -> dict:
    return {
        "status": "ok",
        "model_loaded": deps.scorer is not None,
        "model_version": deps.scorer.version if deps.scorer else None,
        "provider": deps.settings["agent"]["provider"],
        "tables": deps.store.tables(),
    }


@app.get("/model")
def model_info(deps: Deps = Depends(get_deps)) -> dict:
    if deps.scorer is None:
        raise HTTPException(503, "no model loaded")
    return {"method": deps.scorer.method, "model_version": deps.scorer.version,
            "metrics": deps.scorer.metrics, "registry": registry_status(deps.settings)}


@app.post("/model/reload")
def model_reload() -> dict:
    global _DEPS
    _DEPS = None
    deps = get_deps()
    return {"model_version": deps.scorer.version if deps.scorer else None}


@app.get("/experiment")
def experiment(deps: Deps = Depends(get_deps)) -> dict:
    return _report(deps, "experiment.json")


@app.get("/benchmark")
def benchmark(deps: Deps = Depends(get_deps)) -> dict:
    return _report(deps, "benchmark_summary.json")


@app.post("/uplift/score", response_model=ScoreResponse)
def score(req: ScoreRequest, deps: Deps = Depends(get_deps)) -> ScoreResponse:
    if deps.scorer is None:
        raise HTTPException(503, "no model loaded")
    df = pd.DataFrame([s.model_dump() for s in req.subscribers])
    scored = deps.scorer.score(df)
    rows = [
        ScoreRow(customer_id=int(r.customer_id), tau_hat=float(r.tau_hat), churn_risk=float(r.churn_risk),
                 net_value=float(r.net_value), recommend_offer=bool(r.tau_hat > 0 and r.net_value > 0))
        for r in scored.itertuples()
    ]
    return ScoreResponse(model_version=deps.scorer.version, scores=rows)


@app.post("/policy/recommend")
def recommend(req: RecommendRequest, deps: Deps = Depends(get_deps)) -> dict:
    """Plain ranking without the agent (no memory, no reviewer). For the full flow use /agent/run."""
    from app.agents.policy_agent import load_scores

    offers = rank_by_net_value(load_scores(deps), req.budget)
    return {"model_version": deps.scorer.version if deps.scorer else None,
            "summary": policy_summary(offers),
            "top": offers.head(50).to_dict("records")}


@app.post("/agent/run", response_model=AgentResponse)
def agent_run(req: AgentRequest, deps: Deps = Depends(get_deps)) -> AgentResponse:
    state = run_agent(req.question, req.budget, deps.settings, deps)
    return AgentResponse(
        intent=state["intent"], narrative=state["narrative"], review=state.get("review"),
        policy_summary=state.get("policy_summary"), campaign_id=state.get("campaign_id"),
        trace=state["trace"], provider=state["provider"],
        model_version=deps.scorer.version if deps.scorer else None,
    )


@app.post("/campaigns/{campaign_id}/decision")
def campaign_decision(campaign_id: str, req: ApproveRequest, deps: Deps = Depends(get_deps)) -> dict:
    """Human sign-off. Approval logs contacts (which start the cooldown); exclusions go to memory."""
    if "pending_offers" not in deps.store.tables():
        raise HTTPException(404, "no pending campaigns")
    pending = deps.store.query("SELECT * FROM pending_offers WHERE campaign_id = ?", (campaign_id,))
    if pending.empty:
        raise HTTPException(404, f"campaign {campaign_id} not found")
    deps.memory.remember("feedback", {"campaign_id": campaign_id, "reviewer": req.reviewer,
                                      "approve": req.approve, "exclude_ids": req.exclude_ids, "note": req.note})
    sent = 0
    if req.approve:
        to_send = pending[~pending["customer_id"].isin(req.exclude_ids)]
        sent = deps.store.log_contacts(to_send, campaign_id, datetime.now(UTC).isoformat())
    status = "approved" if req.approve else "rejected"
    with deps.store.connect() as conn:
        conn.execute("UPDATE pending_offers SET status = ? WHERE campaign_id = ?", (status, campaign_id))
    return {"campaign_id": campaign_id, "status": status, "contacts_logged": sent}
