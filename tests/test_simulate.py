import numpy as np

from app.data.simulate import FEATURES, ORACLE_COLUMNS, feature_matrix, simulate


def test_shape_and_columns(rct):
    assert len(rct) == 20_000
    for col in FEATURES + ORACLE_COLUMNS + ["treatment", "churn_30d", "value", "offer_cost"]:
        assert col in rct


def test_base_rate_matches_config(rct, sim_cfg):
    assert abs(rct["true_mu0"].mean() - sim_cfg["base_churn_rate"]) < 1e-3


def test_same_seed_same_data(sim_cfg):
    a, b = simulate(500, 7, sim_cfg), simulate(500, 7, sim_cfg)
    assert a.equals(b)


def test_sleeping_dogs_have_negative_effect(rct):
    dogs = (rct["plan_annual"] == 1) & (rct["tenure_months"] > 24) & (rct["sessions_30d"] <= 3)
    assert (rct.loc[dogs, "true_cate"] < 0).all()


def test_lost_causes_have_no_effect(rct):
    lost = (rct["days_since_login"] >= 45) & (rct["plan_annual"] == 0)
    assert np.allclose(rct.loc[lost, "true_cate"], 0)


def test_probabilities_valid(rct):
    mu1 = rct["true_mu0"] - rct["true_cate"]
    assert mu1.between(0, 1).all()


def test_observational_assignment_depends_on_risk(sim_cfg):
    obs = simulate(5000, 0, sim_cfg, "observational")
    assert obs["true_propensity"].std() > 0.1
    assert np.corrcoef(obs["true_propensity"], obs["true_mu0"])[0, 1] > 0.3


def test_feature_matrix_one_hot(rct):
    x = feature_matrix(rct)
    assert {"region_south", "region_east", "region_west"} <= set(x.columns)
    assert "region" not in x
