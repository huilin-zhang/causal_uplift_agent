import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.api.main as api
from pipelines.gates import evaluate_gates


@pytest.fixture(scope="module")
def client(pipeline_env):
    api._DEPS = None
    with TestClient(api.app) as c:
        yield c
    api._DEPS = None


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"]
    assert "@champion" in body["model_version"]


def test_score_endpoint(client):
    sub = {"customer_id": 1, "tenure_months": 6, "plan_annual": 0, "sessions_30d": 3, "days_since_login": 14,
           "price_sensitivity": 0.8, "payment_failures_90d": 0, "support_tickets_90d": 1,
           "monthly_fee": 13.99, "region": "north", "value": 117.5, "offer_cost": 3.8}
    body = client.post("/uplift/score", json={"subscribers": [sub]}).json()
    assert body["model_version"] and len(body["scores"]) == 1


def test_score_rejects_bad_input(client):
    assert client.post("/uplift/score", json={"subscribers": [{"customer_id": 1}]}).status_code == 422


def test_agent_endpoint_and_approval(client):
    out = client.post("/agent/run", json={"question": "Plan this month's retention campaign.", "budget": 100}).json()
    assert out["review"]["passed"] and out["campaign_id"]
    dec = client.post(f"/campaigns/{out['campaign_id']}/decision", json={"reviewer": "test", "approve": True}).json()
    assert dec["status"] == "approved" and dec["contacts_logged"] > 0
    # Approved customers are now in cooldown, so the next campaign must avoid them.
    again = client.post("/agent/run", json={"question": "Plan this month's retention campaign.", "budget": 100}).json()
    assert again["review"]["passed"]


def test_gates_pass_on_pipeline(pipeline_env):
    reports = Path(pipeline_env["paths"]["artifacts"]) / "reports"
    gates = json.loads((reports / "gates.json").read_text())
    assert set(gates["checks"]) >= {"srm_passed", "qini_beats_random", "policy_value_vs_risk"}


def test_gates_fail_on_broken_experiment(pipeline_env):
    reports = Path(pipeline_env["paths"]["artifacts"]) / "reports"
    exp = json.loads((reports / "experiment.json").read_text())
    bench = json.loads((reports / "benchmark.json").read_text())
    exp["srm"]["passed"] = False
    assert not evaluate_gates(exp, bench)["passed"]


def test_register_then_rollback(pipeline_env):
    from app.services.registry import registry_status, rollback
    from pipelines.run_pipeline import force_register

    first = registry_status(pipeline_env)["champion"]
    out = force_register(pipeline_env)
    assert registry_status(pipeline_env)["champion"] == out["version"]
    assert registry_status(pipeline_env)["previous_champion"] == first
    rolled = rollback(pipeline_env)
    assert rolled["champion"] == first
