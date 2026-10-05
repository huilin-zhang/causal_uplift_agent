import numpy as np
import pandas as pd

from app.causal.policy import rank_by_net_value, rank_by_tau
from app.causal.selector import DataProfile, profile_data, select


def _profile(**kw):
    base = dict(n=50_000, design="rct", n_arms=2, treat_share=0.5, outcome_type="binary",
                positive_rate=0.12, has_value_cost=True)
    return DataProfile(**{**base, **kw})


def test_balanced_rct_allows_all_learners():
    s = select(_profile())
    assert set(s.estimators) == {"causal_forest", "x_learner", "t_learner", "ipw"}
    assert s.policy == "net_value_top_k" and s.propensity == "known"


def test_unbalanced_arms_prefer_x_learner():
    assert select(_profile(treat_share=0.85)).estimators[0] == "x_learner"


def test_observational_estimates_propensity_and_bans_t_learner():
    s = select(_profile(design="observational"))
    assert s.propensity == "estimated" and "t_learner" not in s.estimators and s.warnings


def test_rare_outcome_warns_and_drops_t_learner():
    s = select(_profile(positive_rate=0.03))
    assert "t_learner" not in s.estimators and any("Rare" in w for w in s.warnings)


def test_large_n_skips_forest():
    assert "causal_forest" not in select(_profile(n=1_000_000)).estimators


def test_multi_arm_policy():
    assert select(_profile(n_arms=3)).policy == "best_arm_net_value"


def test_missing_value_cost_ranks_by_tau():
    assert select(_profile(has_value_cost=False)).policy == "tau_top_k"


def test_continuous_outcome_reason():
    s = select(_profile(outcome_type="continuous", positive_rate=None))
    assert any("Continuous" in r for r in s.reasons)


def test_profile_data_reads_table(rct):
    p = profile_data(rct, "treatment", "churn_30d", "rct", value_col="value", cost_col="offer_cost")
    assert p.outcome_type == "binary" and 0.45 < p.treat_share < 0.55 and p.has_value_cost


def _scored():
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "tau_hat": [0.10, 0.05, -0.02, 0.001, 0.08],
        "value": [100.0, 100.0, 100.0, 100.0, 100.0],
        "offer_cost": [3.0, 3.0, 3.0, 3.0, 3.0],
    })


def test_net_value_ranking_skips_negative_and_respects_budget():
    offers = rank_by_net_value(_scored(), budget=2)
    assert list(offers["customer_id"]) == [1, 5]
    assert np.allclose(offers["net_value"], [7.0, 5.0])


def test_net_value_ranking_excludes_cooldown():
    offers = rank_by_net_value(_scored(), budget=10, exclude_ids={1})
    assert 1 not in set(offers["customer_id"]) and 3 not in set(offers["customer_id"])


def test_tau_ranking_drops_sleeping_dogs():
    offers = rank_by_tau(_scored(), budget=10)
    assert (offers["tau_hat"] > 0).all() and len(offers) == 4
