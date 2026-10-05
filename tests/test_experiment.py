import numpy as np
import pandas as pd

from app.causal.experiment import (
    analyze_experiment,
    ate_diff_in_means,
    ate_ipw,
    ate_lin_adjusted,
    srm_test,
    standardized_mean_diff,
)


def test_srm_flags_broken_split():
    assert srm_test(52_000, 48_000)["p_value"] < 1e-3


def test_srm_passes_fair_split():
    assert srm_test(25_064, 24_936)["p_value"] > 0.05


def test_smd_small_under_randomization(rct, rct_x):
    assert standardized_mean_diff(rct_x, rct["treatment"]).abs().max() < 0.1


def test_smd_detects_shift():
    x = pd.DataFrame({"a": np.r_[np.zeros(100), np.ones(100)] + np.random.default_rng(0).normal(size=200)})
    t = np.r_[np.zeros(100), np.ones(100)]
    assert standardized_mean_diff(x, t)["a"] > 0.5


def test_ate_ci_covers_truth_on_large_sample(sim_cfg):
    from app.data.simulate import simulate

    df = simulate(200_000, 3, sim_cfg)
    res = ate_diff_in_means(df["churn_30d"], df["treatment"])
    assert res["ci_low"] < df["true_cate"].mean() < res["ci_high"]


def test_lin_close_to_diff_in_means(rct, rct_x):
    dim = ate_diff_in_means(rct["churn_30d"], rct["treatment"])["estimate"]
    lin = ate_lin_adjusted(rct["churn_30d"], rct["treatment"], rct_x)
    assert abs(lin["estimate"] - dim) < 0.005
    assert lin["se"] <= ate_diff_in_means(rct["churn_30d"], rct["treatment"])["se"] * 1.01


def test_ipw_with_true_propensity_removes_confounding(sim_cfg):
    from app.data.simulate import simulate

    obs = simulate(100_000, 0, sim_cfg, "observational")
    naive = ate_diff_in_means(obs["churn_30d"], obs["treatment"])["estimate"]
    ipw = ate_ipw(obs["churn_30d"], obs["treatment"], obs["true_propensity"])
    truth = obs["true_cate"].mean()
    assert abs(ipw - truth) < abs(naive - truth) / 5


def test_analyze_marks_valid(rct, rct_x):
    out = analyze_experiment(rct, rct_x)
    assert out["valid"] and out["ate_primary"] == "lin"
