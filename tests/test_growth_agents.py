import copy
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.agents.campaign_evaluation_agent import estimate_campaign, evaluate_campaign
from app.agents.contracts import PolicyTargets
from app.agents.graph import run_agent
from app.agents.marketing_content_agent import render_copy
from app.agents.planner_agent import allocate
from app.agents.providers import DeterministicProvider
from app.agents.report import run_report
from app.agents.state import build_deps
from app.services.campaigns import decide_campaign, get_campaign


def test_plain_workflow_questions_never_create_campaigns():
    provider = DeterministicProvider()
    for question in ("What campaigns are running?", "Show planned messages", "What are our policy targets?",
                     "Tell me about guardrails", "What is the plan for the campaign?", "当前活动是什么？"):
        assert provider.route(question) != "campaign"


def test_optimizer_enforces_money_roi_and_audience():
    scored = pd.DataFrame([
        {"customer_id": 1, "tau_hat": .5, "value": 20, "offer_cost": 3, "net_value": 7, "tenure_months": 10},
        {"customer_id": 2, "tau_hat": .3, "value": 20, "offer_cost": 2, "net_value": 4, "tenure_months": 1},
        {"customer_id": 3, "tau_hat": .2, "value": 20, "offer_cost": 3, "net_value": 1, "tenure_months": 1},
    ])
    policy = PolicyTargets(max_contacts=3, total_budget=5, min_roi=3, min_new_user_share=.5)
    chosen = allocate(scored, policy)
    assert set(chosen.customer_id) == {1, 2}
    assert chosen.offer_cost.sum() <= policy.total_budget
    assert allocate(scored, PolicyTargets(max_contacts=0)).empty
    assert allocate(scored, PolicyTargets(min_roi=100)).empty


def test_numeric_copy_uses_only_workflow_placeholders():
    assert render_copy("Save {amount}.", {"amount": "$5"}) == "Save $5."
    with pytest.raises(ValueError):
        render_copy("Save $100.", {})
    with pytest.raises(ValueError):
        render_copy("Save {amount}.", {})


def test_evaluation_recovers_known_lift_and_detects_harm():
    outcomes = pd.DataFrame({"treatment": [0] * 100 + [1] * 100,
                             "churn_30d": [1] * 60 + [0] * 40 + [1] * 20 + [0] * 80})
    result = estimate_campaign(outcomes, np.full(200, .3))
    assert result["estimate"] == pytest.approx(.4)
    assert result["ci_low"] < .4 < result["ci_high"]
    assert result["prediction_error"] == pytest.approx(.1)
    assert not result["pause_required"]
    outcomes["treatment"] = 1 - outcomes["treatment"]
    assert estimate_campaign(outcomes, np.full(200, .3))["pause_required"]


def test_report_requires_evaluation(pipeline_env):
    with pytest.raises(ValueError, match="completed campaign evaluation"):
        run_report({"intent": "campaign"}, build_deps(pipeline_env))


def test_chat_does_not_write_any_database_rows(pipeline_env):
    deps = build_deps(pipeline_env)
    before = {table: deps.store.read_table(table).to_json() for table in deps.store.tables()}
    state = run_agent("Show current campaigns and planned messages", deps=deps, settings=pipeline_env)
    assert state["intent"] == "chat"
    assert "offers" not in state
    after = {table: deps.store.read_table(table).to_json() for table in deps.store.tables()}
    assert before == after


def test_campaign_holdout_and_immutable_evaluation(pipeline_env):
    deps = build_deps(pipeline_env)
    state = run_agent("Plan a retention campaign", 100, pipeline_env, deps)
    assert state["review"]["passed"]
    campaign_id = state["campaign_id"]
    assert state["plan"]["alternative"]
    result = decide_campaign(campaign_id, True, "tester", [], "", deps)
    assert result["holdout_size"] == 10
    assert result["contacts_logged"] == 90
    with pytest.raises(ValueError, match="cannot be repeated"):
        decide_campaign(campaign_id, True, "tester", [], "", deps)
    assignments = deps.store.query("SELECT * FROM campaign_assignments WHERE campaign_id = ?", (campaign_id,))
    contacts = deps.store.query("SELECT * FROM contacts WHERE campaign_id = ?", (campaign_id,))
    assert not set(assignments.loc[assignments.treatment == 0, "customer_id"]) & set(contacts.customer_id)
    # Known adverse effect: all treated churn, no holdout users churn.
    outcomes = [{"customer_id": int(r.customer_id), "churn_30d": int(r.treatment)} for r in assignments.itertuples()]
    with pytest.raises(ValueError, match="complete outcomes"):
        evaluate_campaign(campaign_id, outcomes[:-1], deps)
    result = evaluate_campaign(campaign_id, outcomes, deps)
    assert result["pause_required"]
    assert get_campaign(campaign_id, deps)["status"] == "paused"
    assert "Campaign paused" in result["report"]
    assert deps.memory.recall("campaign_evaluation", 1)[0]["content"]["campaign_id"] == campaign_id
    with pytest.raises(ValueError, match="already evaluated"):
        evaluate_campaign(campaign_id, outcomes, deps)
    chat = run_agent("Show campaign evaluation records", settings=pipeline_env, deps=deps)
    assert chat["intent"] == "chat"
    assert chat["chat_context"]["campaigns"]
    again = run_agent("Plan a retention campaign", 100, pipeline_env, deps)
    assert not set(assignments.customer_id) & {o["customer_id"] for o in again["offers"]}


def test_approval_timeout_does_not_contact(pipeline_env):
    deps = build_deps(pipeline_env)
    state = run_agent("Plan a retention campaign", 20, pipeline_env, deps)
    campaign_id = state["campaign_id"]
    with deps.store.connect() as conn:
        conn.execute("UPDATE campaigns SET expires_at = ? WHERE campaign_id = ?",
                     ((datetime.now(UTC) - timedelta(hours=1)).isoformat(), campaign_id))
    result = decide_campaign(campaign_id, True, "tester", [], "", deps)
    assert result["status"] == "expired" and result["contacts_logged"] == 0


def test_guardrail_rejects_modified_copy_and_evidence(pipeline_env):
    from app.agents.guardrail_agent import run_guardrail_agent

    deps = build_deps(pipeline_env)
    state = run_agent("Plan a retention campaign", 20, pipeline_env, deps)
    altered = copy.deepcopy(state)
    altered["messages"][0]["text"] += " Guaranteed $100."
    assert not run_guardrail_agent(altered, deps)["review"]["passed"]
    altered = copy.deepcopy(state)
    altered["offers"][0]["tau_hat"] += .5
    assert not run_guardrail_agent(altered, deps)["review"]["passed"]


def test_infeasible_policy_is_not_relaxed(pipeline_env):
    state = run_agent("Plan a retention campaign", settings=pipeline_env,
                      policy_targets={"max_contacts": 100, "total_budget": 0})
    assert not state["review"]["passed"]
    assert state["offers"] == []
    assert state["plan"]["status"] == "infeasible"


def test_failed_experiment_never_reaches_optimizer(pipeline_env, monkeypatch):
    from app.agents.planner_agent import run_planner_agent

    def unexpected(*args):
        raise AssertionError("unusable evidence reached optimizer")

    monkeypatch.setattr("app.agents.planner_agent.allocate", unexpected)
    out = run_planner_agent({"targeting": {"usable": False}, "budget": 100}, build_deps(pipeline_env))
    assert out["offers"] == [] and out["plan"]["status"] == "blocked"
