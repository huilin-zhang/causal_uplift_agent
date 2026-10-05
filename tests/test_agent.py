from app.agents.graph import run_agent
from app.agents.providers import DeterministicProvider, numbers_match
from app.agents.reviewer_agent import review


def test_router_intents():
    p = DeterministicProvider()
    assert p.route("Plan the campaign") == "campaign"
    assert p.route("Did the test pass the SRM check?") == "experiment"
    assert p.route("Explain the X-learner") == "method"
    assert p.route("What about customer 17?") == "customer"


def test_numbers_guard_rejects_new_numbers():
    assert numbers_match("kept 12.5 customers", "We kept 12.5 customers")
    assert not numbers_match("kept 99 customers", "We kept 12.5 customers")


def test_campaign_run_produces_reviewed_offers(pipeline_env):
    state = run_agent("Plan this month's retention campaign.", 300, pipeline_env)
    assert state["intent"] == "campaign" and state["review"]["passed"]
    assert 0 < len(state["offers"]) <= 300
    assert all(o["tau_hat"] > 0 and o["net_value"] > 0 for o in state["offers"])
    assert state["trace"][0].startswith("router") and state["trace"][-1].startswith("memory_agent")


def test_reviewer_catches_injected_bad_offers(settings):
    state = {"experiment": {"srm_p": 0.5, "max_abs_smd": 0.0}, "model": {"method": "causal_forest"},
             "selection": {"estimators": ["causal_forest"]}, "budget": 5, "memory": {"cooldown_ids": [3]},
             "offers": [{"customer_id": 1, "tau_hat": -0.01, "net_value": -1.0},
                        {"customer_id": 3, "tau_hat": 0.05, "net_value": 2.0}]}
    r = review(state, settings)
    assert not r["passed"] and r["bad_ids"] == [1, 3] and not r["blocking"]


def test_reviewer_blocks_disallowed_estimator(settings):
    state = {"experiment": {"srm_p": 0.5, "max_abs_smd": 0.0}, "model": {"method": "t_learner"},
             "selection": {"estimators": ["causal_forest"]}, "offers": []}
    assert review(state, settings)["blocking"]


def test_experiment_question_skips_policy(pipeline_env):
    state = run_agent("Is the A/B test valid? Check SRM.", None, pipeline_env)
    assert state["intent"] == "experiment" and "offers" not in state
