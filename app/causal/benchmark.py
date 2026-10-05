"""Benchmark all estimators over many fresh simulation draws.

For each draw d = 0..D-1:
  1. Simulate a new set of subscribers (seed = base seed + d) and split it
     50/20/30 into train / validation / test, stratified by arm.
  2. Fit every model on train. For the LightGBM learners, pick settings from
     a small grid by the doubly robust (DR) loss on validation. The DR loss
     needs no true tau, so the same step would work on real data.
  3. Score the test set: PEHE, observed and true Qini, policy value at the
     coverage level, and the net-value policy.
The primary model is fixed in the config before any run (causal forest), so
the headline result involves no data-driven selection. The DR-loss ranking
is reported beside it.

Uncertainty for the headline comes from the spread across independent draws.
The test-set bootstrap (draw 0 only) shows how noisy an observed A/B readout
of the same comparison would be. Code guide: section "Benchmark" (sec:benchmark).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split

from app.causal import metrics as M
from app.causal.bootstrap import bootstrap_gain, summarize
from app.causal.learners import GRID, LGBM_PARAMS, TUNABLE, UPLIFT_LEARNERS, make_learner
from app.data.simulate import feature_matrix, simulate

BASELINES = ["churn_risk", "kmeans", "constant_ate"]


def split_indices(t: np.ndarray, fractions: list[float], seed: int) -> tuple[np.ndarray, ...]:
    """Stratified train / validation / test indices."""
    idx = np.arange(len(t))
    train_frac, valid_frac, _ = fractions
    train, rest = train_test_split(idx, train_size=train_frac, stratify=t, random_state=seed)
    valid_share = valid_frac / (1 - train_frac)
    valid, test = train_test_split(rest, train_size=valid_share, stratify=t[rest], random_state=seed)
    return train, valid, test


def validation_dr_scores(x: pd.DataFrame, t: np.ndarray, y: np.ndarray, e: float, seed: int) -> np.ndarray:
    """DR pseudo-outcomes on the validation set, with 2-fold cross-fitted outcome models."""
    m0, m1 = np.zeros(len(t)), np.zeros(len(t))
    for fit_idx, score_idx in StratifiedKFold(2, shuffle=True, random_state=seed).split(x, t):
        xf, tf, yf = x.iloc[fit_idx], t[fit_idx], y[fit_idx]
        c0 = LGBMClassifier(random_state=seed, **LGBM_PARAMS).fit(xf[tf == 0], yf[tf == 0])
        c1 = LGBMClassifier(random_state=seed, **LGBM_PARAMS).fit(xf[tf == 1], yf[tf == 1])
        m0[score_idx] = c0.predict_proba(x.iloc[score_idx])[:, 1]
        m1[score_idx] = c1.predict_proba(x.iloc[score_idx])[:, 1]
    return M.dr_pseudo_outcome(y, t, m0, m1, e)


def fit_models(x_tr, t_tr, y_tr, x_va, phi_va, seed: int, cfg: dict[str, Any], treat_share: float):
    """Fit baselines and uplift learners; grid-search the tunable ones by validation DR loss."""
    models, info = {}, {}
    for name in BASELINES + UPLIFT_LEARNERS:
        start = time.perf_counter()
        candidates = GRID if name in TUNABLE else [None]
        best = None
        for params in candidates:
            model = make_learner(
                name, seed=seed, treat_share=treat_share, forest_trees=cfg["forest_trees"],
                kmeans_k=cfg["kmeans_k"], params=params,
            ).fit(x_tr, t_tr, y_tr)
            loss = M.dr_loss(model.predict(x_va), phi_va) if name != "churn_risk" else None
            if best is None or (loss is not None and loss < best[1]):
                best = (model, loss, params)
        models[name] = best[0]
        info[name] = {"dr_loss": best[1], "params": best[2], "fit_seconds": round(time.perf_counter() - start, 2)}
    return models, info


def evaluate_draw(draw: int, n: int, base_seed: int, sim: dict, cfg: dict) -> dict[str, Any]:
    seed = base_seed + draw
    df = simulate(n, seed, sim)
    x = feature_matrix(df)
    t, y = df["treatment"].to_numpy(), df["churn_30d"].to_numpy()
    retained, tau = 1 - y, df["true_cate"].to_numpy()
    tr, va, te = split_indices(t, cfg["split"], seed)

    share = sim["treat_share"]
    phi_va = validation_dr_scores(x.iloc[va], t[va], y[va], share, seed)
    models, info = fit_models(x.iloc[tr], t[tr], y[tr], x.iloc[va], phi_va, seed, cfg, share)

    # Secondary selection: lowest validation DR loss, falling back to the primary
    # model unless the winner is better by more than one standard error.
    primary = cfg["primary_model"]
    p_pred = models[primary].predict(x.iloc[va])
    dr_choice = primary
    for name in UPLIFT_LEARNERS:
        diff = (phi_va - models[name].predict(x.iloc[va])) ** 2 - (phi_va - p_pred) ** 2
        if diff.mean() + diff.std(ddof=1) / np.sqrt(len(diff)) < 0:
            if dr_choice == primary or info[name]["dr_loss"] < info[dr_choice]["dr_loss"]:
                dr_choice = name

    x_te, t_te, r_te, tau_te = x.iloc[te], t[te], retained[te], tau[te]
    value_te, cost_te = df["value"].to_numpy()[te], df["offer_cost"].to_numpy()[te]
    cov = cfg["coverage"]
    rows, scores = {}, {}
    for name in BASELINES + UPLIFT_LEARNERS + ["oracle", "random"]:
        if name == "oracle":
            score = tau_te
        elif name == "random":
            score = np.random.default_rng(seed).random(len(te))
        else:
            score = models[name].predict(x_te)
        scores[name] = score
        is_effect = name not in ("churn_risk", "random")
        rows[name] = {
            "pehe_pp": 100 * M.pehe(score, tau_te) if is_effect else None,
            "qini_observed": M.qini_coefficient(score, r_te, t_te),
            "qini_true": M.true_qini(score, tau_te),
            "qini_true_normalized": M.true_qini_normalized(score, tau_te),
            "policy_true": M.expected_policy_value(score, tau_te, cov),
            "policy_observed": M.policy_value_estimated(M.top_k_mask(score, cov), r_te, t_te),
            "dr_loss_valid": info.get(name, {}).get("dr_loss"),
        }
        if is_effect:
            rows[name].update(
                {f"netpolicy_{k}": v for k, v in M.net_value_policy(score, tau_te, value_te, cost_te).items()}
            )

    out = {
        "draw": draw,
        "seed": seed,
        "n_test": int(len(te)),
        "dr_choice": dr_choice,
        "params": {k: v["params"] for k, v in info.items() if v["params"]},
        "fit_seconds": {k: v["fit_seconds"] for k, v in info.items()},
        "metrics": rows,
    }
    if draw == 0:
        out["models"] = models
        out["test_index"] = te
        out["scores"] = scores
        out["observed_bootstrap"] = bootstrap_gain(
            M.top_k_mask(scores[primary], cov), M.top_k_mask(scores["churn_risk"], cov),
            r_te, t_te, cfg["bootstrap_reps"], seed,
        )
    return out


def _mean_ci(values: list[float]) -> dict[str, float]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    mean = float(v.mean())
    if len(v) < 2:
        return {"mean": mean, "sd": 0.0, "ci_low": mean, "ci_high": mean, "n": int(len(v))}
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return {"mean": mean, "sd": float(v.std(ddof=1)), "ci_low": mean - half, "ci_high": mean + half, "n": int(len(v))}


def ratio_of_totals(model_vals, risk_vals, reps: int = 2000, seed: int = 0) -> dict[str, float]:
    """sum(model) / sum(risk) - 1 across draws, with a percentile interval from resampling draws."""
    a, b = np.asarray(model_vals, float), np.asarray(risk_vals, float)
    point = a.sum() / b.sum() - 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(reps, len(a)))
    boot = a[idx].sum(1) / b[idx].sum(1) - 1
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"estimate": float(point), "ci_low": float(lo), "ci_high": float(hi), "n_draws": int(len(a))}


def median_with_ci(values, reps: int = 2000, seed: int = 0) -> dict[str, float]:
    """Median of per-draw values with a percentile interval from resampling draws."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    boot = np.median(v[rng.integers(0, len(v), size=(reps, len(v)))], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"median": float(np.median(v)), "ci_low": float(lo), "ci_high": float(hi), "n_draws": int(len(v))}


def run_benchmark(n: int, base_seed: int, sim: dict, cfg: dict, log=print) -> dict[str, Any]:
    draws = []
    for d in range(cfg["n_draws"]):
        start = time.perf_counter()
        draws.append(evaluate_draw(d, n, base_seed, sim, cfg))
        log(f"draw {d + 1}/{cfg['n_draws']} done in {time.perf_counter() - start:.1f}s")

    names = list(draws[0]["metrics"])
    table = {}
    for name in names:
        table[name] = {}
        for key in draws[0]["metrics"][name]:
            vals = [r["metrics"][name][key] for r in draws if r["metrics"][name].get(key) is not None]
            table[name][key] = _mean_ci(vals) if vals else None

    primary = cfg["primary_model"]
    up = [r["metrics"][primary]["policy_true"] for r in draws]
    risk = [r["metrics"]["churn_risk"]["policy_true"] for r in draws]
    boot0 = draws[0]["observed_bootstrap"]
    return {
        "n_subscribers": n,
        "n_draws": cfg["n_draws"],
        "coverage": cfg["coverage"],
        "primary_model": primary,
        "dr_choice_counts": {k: int(v) for k, v in pd.Series([r["dr_choice"] for r in draws]).value_counts().items()},
        "table": table,
        "headline": {
            "description": f"{primary} vs churn-risk ranking, top {cfg['coverage']:.0%} of test set, true effects",
            "abs_gain_customers": _mean_ci(np.subtract(up, risk)),
            "relative_gain_ratio_of_totals": ratio_of_totals(up, risk),
            # Per-draw ratio is stable here: the risk baseline's true value is always well above 0.
            "median_relative_gain": median_with_ci([u / r - 1 for u, r in zip(up, risk, strict=True)]),
            "per_draw_relative_gain": [float(u / r - 1) for u, r in zip(up, risk, strict=True)],
            "share_of_draws_uplift_wins": float(np.mean(np.subtract(up, risk) > 0)),
        },
        "observed_readout_draw0": {
            "note": "Test-set bootstrap with frozen models: evaluation noise only.",
            "reps": cfg["bootstrap_reps"],
            "abs_gain_customers": summarize(boot0["abs_gain"]),
            "uplift_incremental": summarize(boot0["i_uplift"]),
            "risk_incremental": summarize(boot0["i_risk"]),
        },
        "_draw0": draws[0],  # fitted models and scores; not saved to JSON
        "_draws": draws,
    }
