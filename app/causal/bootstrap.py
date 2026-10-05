"""Bootstrap the test set to put error bars on the uplift-vs-risk comparison.

Models are trained once and frozen. We only resample test rows, so the
interval covers evaluation noise, not training noise. Spread across split
seeds is reported separately. Code guide: section "Bootstrap" (sec:bootstrap).
"""

from __future__ import annotations

import numpy as np


def stratified_weights(t: np.ndarray, reps: int, seed: int) -> np.ndarray:
    """Return a (reps, n) matrix of resample counts.

    Rows are drawn with replacement separately inside the treated and control
    arms, so each replicate keeps the same arm sizes as the real test set.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t)
    w = np.zeros((reps, len(t)), dtype=np.int32)
    for arm in (0, 1):
        idx = np.flatnonzero(t == arm)
        draws = rng.integers(0, len(idx), size=(reps, len(idx)))
        for b in range(reps):
            np.add.at(w[b], idx[draws[b]], 1)
    return w


def incremental_retained(w: np.ndarray, mask: np.ndarray, retained: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per replicate: (treated retention - control retention) in the selected group * group size."""
    y = np.asarray(retained, float)
    sel_t = (mask & (t == 1)).astype(float)
    sel_c = (mask & (t == 0)).astype(float)
    n_t = w @ sel_t
    n_c = w @ sel_c
    rate_t = (w @ (sel_t * y)) / np.maximum(n_t, 1)
    rate_c = (w @ (sel_c * y)) / np.maximum(n_c, 1)
    return (rate_t - rate_c) * (n_t + n_c)


def bootstrap_gain(
    mask_uplift: np.ndarray,
    mask_risk: np.ndarray,
    retained: np.ndarray,
    t: np.ndarray,
    reps: int = 1000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Relative and absolute gain of uplift targeting over risk targeting, per replicate.

    relative gain = (I_uplift - I_risk) / |I_risk|. This ratio blows up when
    I_risk is near zero, so the absolute difference is kept as well.
    """
    t = np.asarray(t)
    w = stratified_weights(t, reps, seed)
    i_up = incremental_retained(w, mask_uplift, retained, t)
    i_risk = incremental_retained(w, mask_risk, retained, t)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (i_up - i_risk) / np.abs(i_risk)
    return {"i_uplift": i_up, "i_risk": i_risk, "abs_gain": i_up - i_risk, "rel_gain": rel}


def summarize(values: np.ndarray) -> dict[str, float]:
    """Median and 95% percentile interval, ignoring non-finite replicates."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    lo, med, hi = np.percentile(v, [2.5, 50, 97.5])
    return {"median": float(med), "ci_low": float(lo), "ci_high": float(hi), "n": int(len(v))}
