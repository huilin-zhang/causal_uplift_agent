import numpy as np
import pytest

from app.causal.learners import make_learner
from tests.conftest import toy_rct


@pytest.mark.parametrize("name", ["t_learner", "x_learner", "causal_forest", "ipw"])
def test_learner_finds_strong_effect(name):
    x, t, y, tau = toy_rct()
    model = make_learner(name, seed=0, forest_trees=40).fit(x, t, y)
    pred = model.predict(x)
    # Customers with the real effect must score higher on average.
    assert pred[tau > 0].mean() > pred[tau == 0].mean() + 0.05


def test_learners_return_reduction_sign():
    x, t, y, _ = toy_rct()
    assert make_learner("t_learner").fit(x, t, y).predict(x).mean() > 0


def test_kmeans_gives_one_value_per_cluster():
    x, t, y, _ = toy_rct()
    model = make_learner("kmeans", kmeans_k=4).fit(x, t, y)
    assert len(np.unique(model.predict(x))) <= 4


def test_constant_ate_is_constant():
    x, t, y, _ = toy_rct()
    pred = make_learner("constant_ate").fit(x, t, y).predict(x)
    assert np.ptp(pred) == 0 and 0.05 < pred[0] < 0.15


def test_risk_model_trained_on_control_only():
    x, t, y, _ = toy_rct()
    risk = make_learner("churn_risk").fit(x, t, y).predict(x)
    assert 0.3 < risk.mean() < 0.5  # control churn is 0.4


def test_unknown_learner_raises():
    with pytest.raises(ValueError):
        make_learner("magic")
