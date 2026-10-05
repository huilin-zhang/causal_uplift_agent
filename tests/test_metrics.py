import numpy as np

from app.causal import metrics as M
from app.causal.bootstrap import bootstrap_gain, stratified_weights, summarize


def test_pehe_zero_for_oracle():
    tau = np.array([0.1, 0.2, -0.1])
    assert M.pehe(tau, tau) == 0


def test_true_qini_oracle_is_one_and_random_near_zero():
    rng = np.random.default_rng(0)
    tau = rng.normal(0.01, 0.02, 5000)
    assert M.true_qini_normalized(tau, tau) == 1.0
    assert abs(M.true_qini_normalized(rng.random(5000), tau)) < 0.1


def test_constant_score_has_zero_true_qini():
    tau = np.random.default_rng(1).normal(size=1000)
    assert abs(M.true_qini(np.zeros(1000), tau)) < 1e-12


def test_observed_qini_ranks_oracle_above_random():
    rng = np.random.default_rng(2)
    n = 20_000
    tau = np.where(rng.random(n) < 0.3, 0.3, 0.0)
    t = rng.binomial(1, 0.5, n)
    retained = rng.binomial(1, np.where(t == 1, 0.5 + tau, 0.5))
    assert M.qini_coefficient(tau, retained, t) > M.qini_coefficient(rng.random(n), retained, t) + 0.01


def test_policy_value_hand_computed():
    score = np.array([4, 3, 2, 1])
    tau = np.array([0.5, 0.1, 0.2, 0.3])
    assert np.isclose(M.expected_policy_value(score, tau, 0.5), 0.6)


def test_ties_average_inside_group():
    # Two tied rows at the top; contacting one of them is worth their average.
    score = np.array([1, 1, 0, 0])
    tau = np.array([1.0, 0.0, 5.0, 5.0])
    assert np.isclose(M.expected_policy_value(score, tau, 0.25), 0.5)


def test_policy_value_estimated_from_ab():
    mask = np.array([True, True, True, True, False])
    retained = np.array([1, 1, 0, 1, 0])
    t = np.array([1, 1, 0, 0, 1])
    assert np.isclose(M.policy_value_estimated(mask, retained, t), (1.0 - 0.5) * 4)


def test_dr_loss_prefers_truth_on_rct():
    rng = np.random.default_rng(3)
    n = 50_000
    tau = np.where(rng.random(n) < 0.5, 0.2, 0.0)
    t = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, np.where(t == 1, 0.4 - tau, 0.4))
    m0, m1 = np.full(n, 0.4), 0.4 - tau
    phi = M.dr_pseudo_outcome(y, t, m0, m1, 0.5)
    assert M.dr_loss(tau, phi) < M.dr_loss(np.full(n, tau.mean()), phi)


def test_net_value_policy_counts():
    out = M.net_value_policy(np.array([0.1, 0.01]), np.array([0.1, 0.01]), np.array([100, 100]), np.array([2, 2]))
    assert out["n_contacted"] == 1 and np.isclose(out["true_net_value"], 8.0)


def test_bootstrap_keeps_arm_sizes():
    t = np.array([1] * 30 + [0] * 70)
    w = stratified_weights(t, reps=20, seed=0)
    assert (w[:, t == 1].sum(1) == 30).all() and (w[:, t == 0].sum(1) == 70).all()


def test_bootstrap_reproducible_and_ordered():
    rng = np.random.default_rng(4)
    t = rng.binomial(1, 0.5, 2000)
    r = rng.binomial(1, 0.8, 2000)
    m1, m2 = rng.random(2000) < 0.2, rng.random(2000) < 0.2
    a = bootstrap_gain(m1, m2, r, t, reps=50, seed=1)["abs_gain"]
    b = bootstrap_gain(m1, m2, r, t, reps=50, seed=1)["abs_gain"]
    s = summarize(a)
    assert np.array_equal(a, b) and s["ci_low"] <= s["median"] <= s["ci_high"]
