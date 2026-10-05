"""Check that the A/B test is valid, then estimate the average effect.

Order matters: if the split is broken (SRM) or the groups differ before
treatment (imbalance), the effect estimate cannot be trusted.
Code guide: section "Experiment analysis" (sec:experiment).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def srm_test(n_treat: int, n_control: int, expected_share: float = 0.5) -> dict[str, float]:
    """Sample ratio mismatch: chi-square test of arm sizes against the planned split.

    A tiny p-value means assignment or logging is broken (e.g. a bug drops
    some treated users). Industry practice alerts at p < 0.001.
    """
    n = n_treat + n_control
    expected = np.array([n * expected_share, n * (1 - expected_share)])
    chi2, p = stats.chisquare([n_treat, n_control], f_exp=expected)
    return {
        "n_treat": int(n_treat),
        "n_control": int(n_control),
        "treat_share": n_treat / n,
        "chi2": float(chi2),
        "p_value": float(p),
    }


def standardized_mean_diff(x: pd.DataFrame, t: np.ndarray) -> pd.Series:
    """SMD per column: (mean_T - mean_C) / sqrt((var_T + var_C) / 2).

    Unlike a t-test, SMD does not shrink with sample size, so it measures
    whether the gap is large enough to matter. |SMD| < 0.1 is the usual bar.
    """
    t = np.asarray(t).astype(bool)
    xt, xc = x[t], x[~t]
    pooled = np.sqrt((xt.var(ddof=1) + xc.var(ddof=1)) / 2).replace(0, np.nan)
    return ((xt.mean() - xc.mean()) / pooled).fillna(0.0)


def ate_diff_in_means(y: np.ndarray, t: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """Effect on churn reduction: mean(y | control) - mean(y | treated).

    Neyman variance p(1-p)/n per arm. Positive = outreach lowered churn.
    """
    y, t = np.asarray(y, float), np.asarray(t).astype(bool)
    p1, p0 = y[t].mean(), y[~t].mean()
    se = np.sqrt(p1 * (1 - p1) / t.sum() + p0 * (1 - p0) / (~t).sum())
    z = stats.norm.ppf(1 - alpha / 2)
    est = p0 - p1
    return {
        "estimate": float(est),
        "se": float(se),
        "ci_low": float(est - z * se),
        "ci_high": float(est + z * se),
        "p_value": float(2 * stats.norm.sf(abs(est) / se)),
        "churn_treat": float(p1),
        "churn_control": float(p0),
    }


def ate_lin_adjusted(
    y: np.ndarray, t: np.ndarray, x: pd.DataFrame, alpha: float = 0.05
) -> dict[str, float]:
    """Lin (2013) regression adjustment: y ~ t + x_c + t * x_c, x centered, HC2 errors.

    Using covariates that predict churn shrinks the standard error without
    adding bias in a randomized test. The coefficient on t is the ATE.
    """
    t = np.asarray(t, float)
    xc = (x - x.mean()).to_numpy(float)
    design = np.column_stack([np.ones_like(t), t, xc, xc * t[:, None]])
    fit = sm.OLS(np.asarray(y, float), design).fit(cov_type="HC2")
    est, se = -fit.params[1], fit.bse[1]  # negate: report churn reduction
    z = stats.norm.ppf(1 - alpha / 2)
    return {
        "estimate": float(est),
        "se": float(se),
        "ci_low": float(est - z * se),
        "ci_high": float(est + z * se),
    }


def ate_ipw(y: np.ndarray, t: np.ndarray, e: np.ndarray) -> float:
    """Hajek IPW estimate of the churn reduction when assignment was NOT random.

    Each treated unit is weighted by 1/e(x), each control by 1/(1-e(x)), so
    both arms are reweighted to look like the full population. Normalizing
    the weights (Hajek) is more stable than the plain Horvitz-Thompson sum.
    """
    y, t, e = np.asarray(y, float), np.asarray(t).astype(bool), np.asarray(e, float)
    w1, w0 = t / e, (~t) / (1 - e)
    mu1 = np.sum(w1 * y) / np.sum(w1)
    mu0 = np.sum(w0 * y) / np.sum(w0)
    return float(mu0 - mu1)


def weighted_smd(x: pd.DataFrame, t: np.ndarray, w: np.ndarray) -> pd.Series:
    """SMD after inverse probability weighting: did the weights balance the arms?"""
    t = np.asarray(t).astype(bool)
    out = {}
    for col in x.columns:
        v = x[col].to_numpy(float)
        m1 = np.average(v[t], weights=w[t])
        m0 = np.average(v[~t], weights=w[~t])
        s = np.sqrt((v[t].var() + v[~t].var()) / 2) or 1.0
        out[col] = (m1 - m0) / s
    return pd.Series(out)


def aipw_ate(x: pd.DataFrame, t: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> dict:
    """Cross-fitted AIPW (doubly robust) estimate of the churn reduction, with a standard error.

    For each fold, the propensity model and the two outcome models are fit on
    the other folds and used to score this fold, so no unit is scored by a
    model that saw it. The estimate is the mean of the DR scores; its SE is
    their standard deviation / sqrt(n).

    The propensity model is a plain logistic regression, chosen for how well
    it predicts treatment, not borrowed from the heavily regularized CATE
    settings. Those settings pull e(x) toward 0.5, under-weight the rare
    high-risk controls, and leave bias (an earlier version missed by ~4 SE).
    Whether the weights balance the arms is checked afterwards (weighted SMD).
    """
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold

    from app.causal.learners import LGBM_PARAMS, Propensity
    from app.causal.metrics import dr_pseudo_outcome

    t, y = np.asarray(t), np.asarray(y, float)
    n = len(t)
    e, m0, m1 = np.zeros(n), np.zeros(n), np.zeros(n)
    for k, (fit_idx, score_idx) in enumerate(
        StratifiedKFold(folds, shuffle=True, random_state=seed).split(x, t)
    ):
        xf, tf, yf = x.iloc[fit_idx], t[fit_idx], y[fit_idx]
        xs = x.iloc[score_idx]
        e[score_idx] = Propensity(known=None).fit(xf, tf).predict(xs)  # clipped to [0.05, 0.95]
        c0 = LGBMClassifier(random_state=seed + k, **LGBM_PARAMS).fit(xf[tf == 0], yf[tf == 0])
        c1 = LGBMClassifier(random_state=seed + k, **LGBM_PARAMS).fit(xf[tf == 1], yf[tf == 1])
        m0[score_idx] = c0.predict_proba(xs)[:, 1]
        m1[score_idx] = c1.predict_proba(xs)[:, 1]
    phi = dr_pseudo_outcome(y, t, m0, m1, e)
    est, se = float(phi.mean()), float(phi.std(ddof=1) / np.sqrt(n))
    return {"estimate": est, "se": se, "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se, "propensity": e}


def observational_check(df: pd.DataFrame, x: pd.DataFrame, smd_threshold: float = 0.1) -> dict[str, float]:
    """On the observational variant: naive difference vs IPW vs AIPW vs the true average effect.

    Pass condition stated in advance: after weighting, every |SMD| < smd_threshold.
    It needs no ground truth, so the same check works on real observational data."""
    t, y = df["treatment"].to_numpy(), df["churn_30d"].to_numpy()
    aipw = aipw_ate(x, t, y)
    e = aipw.pop("propensity")
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    return {
        "true_ate": float(df["true_cate"].mean()),
        "naive_diff": float(ate_diff_in_means(y, t)["estimate"]),
        "ipw": ate_ipw(y, t, e),
        "aipw": aipw,
        "max_abs_smd_before": float(standardized_mean_diff(x, t).abs().max()),
        "max_abs_smd_after_weighting": float(weighted_smd(x, t, w).abs().max()),
        "weighting_balance_passed": bool(weighted_smd(x, t, w).abs().max() < smd_threshold),
        "propensity_min": float(e.min()),
        "propensity_max": float(e.max()),
    }


def analyze_experiment(
    df: pd.DataFrame,
    x: pd.DataFrame,
    srm_alpha: float = 0.001,
    smd_threshold: float = 0.1,
    expected_share: float = 0.5,
) -> dict[str, Any]:
    """Run SRM, balance, and both ATE estimators. Returns a JSON-ready dict."""
    t = df["treatment"].to_numpy()
    y = df["churn_30d"].to_numpy()
    srm = srm_test(int(t.sum()), int(len(t) - t.sum()), expected_share)
    smd = standardized_mean_diff(x, t)
    result = {
        "n": int(len(df)),
        "srm": {**srm, "passed": srm["p_value"] >= srm_alpha},
        "balance": {
            "max_abs_smd": float(smd.abs().max()),
            "worst_feature": str(smd.abs().idxmax()),
            "smd": {k: round(float(v), 4) for k, v in smd.items()},
            "passed": bool(smd.abs().max() < smd_threshold),
        },
        # Lin is the pre-registered primary estimate; difference in means is reported beside it.
        "ate_primary": "lin",
        "ate_lin": ate_lin_adjusted(y, t, x),
        "ate_diff_in_means": ate_diff_in_means(y, t),
    }
    result["valid"] = result["srm"]["passed"] and result["balance"]["passed"]
    return result
