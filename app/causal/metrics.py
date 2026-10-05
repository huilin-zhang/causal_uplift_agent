"""Evaluation metrics for uplift models.

Three questions, three metrics:
  PEHE          How close is tau_hat to the true tau? (simulation only)
  Qini          Does the ranking put high-effect customers first?
  Policy value  How many extra customers do we keep if we contact the top k%?

Qini and policy value come in two versions:
  observed  uses only outcomes and assignment, so it works on real A/B data
  true      sums the known tau (simulation only); no outcome noise

Tied scores (k-means gives only 8 distinct values) are handled as if ties were
broken at random, by averaging inside each tie group, so results do not
depend on row order. Code guide: section "Metrics" (sec:metrics).
"""

from __future__ import annotations

import numpy as np


def pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """Root mean squared error of the effect: sqrt(mean((tau_hat - tau)^2))."""
    return float(np.sqrt(np.mean((np.asarray(tau_hat) - np.asarray(tau_true)) ** 2)))


def _order_and_groups(score) -> tuple[np.ndarray, np.ndarray]:
    """Row order by score (highest first) and the tie-group id of each sorted row."""
    score = np.asarray(score, float)
    order = np.argsort(-score, kind="stable")
    s = score[order]
    group = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])])
    return order, group


def _interpolate_ties(cum: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Replace a cumulative curve inside each tie group by a straight line
    between the group's start and end. This is its expected value under
    random tie-breaking. cum has length n+1 with cum[0] = 0."""
    n = len(group)
    ends = np.flatnonzero(np.r_[group[1:] != group[:-1], True]) + 1  # exclusive end per group
    starts = np.r_[0, ends[:-1]]
    pos = np.arange(1, n + 1)
    g_start, g_end = starts[group], ends[group]
    frac = (pos - g_start) / (g_end - g_start)
    out = cum.copy()
    out[1:] = cum[g_start] + frac * (cum[g_end] - cum[g_start])
    return out


def expected_gain_curve(score, tau) -> np.ndarray:
    """G(k) = expected sum of true tau over the top k rows, k = 0..n."""
    order, group = _order_and_groups(score)
    cum = np.r_[0.0, np.cumsum(np.asarray(tau, float)[order])]
    return _interpolate_ties(cum, group)


def _grid(n: int, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    phi = np.arange(1, n_points + 1) / n_points
    return phi, np.round(phi * n).astype(int)


def true_qini(score, tau, n_points: int = 100) -> float:
    """Noise-free Qini: Q* = (1 / (M n)) * sum_m [G(k_m) - phi_m * G(n)].

    Random targeting gives 0. Sorting by the true tau gives the maximum.
    """
    g = expected_gain_curve(score, tau)
    n = len(g) - 1
    phi, k = _grid(n, n_points)
    return float(np.mean(g[k] - phi * g[n]) / n)


def true_qini_normalized(score, tau, n_points: int = 100) -> float:
    """Q*(score) / Q*(tau). At most 1; 0 = random; negative = worse than random."""
    best = true_qini(tau, tau, n_points)
    return float(true_qini(score, tau, n_points) / best) if best > 0 else float("nan")


def qini_curve(score, retained, t, n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Observed incremental retained customers when contacting the top phi share.

    Q(k) = R_T(k) - R_C(k) * N_T(k) / N_C(k)
    R_T, R_C: retained counts in the treated / control part of the top k.
    N_T, N_C: treated / control counts in the top k.
    The control count is rescaled to the treated group size (Radcliffe 2007).
    """
    order, group = _order_and_groups(score)
    y = np.asarray(retained, float)[order]
    tt = np.asarray(t)[order]
    rt = np.cumsum(y * (tt == 1))
    rc = np.cumsum(y * (tt == 0))
    nt = np.cumsum(tt == 1)
    nc = np.cumsum(tt == 0)
    q = rt - np.divide(rc * nt, nc, out=np.zeros_like(rc), where=nc > 0)
    q = _interpolate_ties(np.r_[0.0, q], group)
    phi, k = _grid(len(y), n_points)
    return np.r_[0.0, phi], np.r_[0.0, q[k]]


def qini_coefficient(score, retained, t, n_points: int = 100) -> float:
    """Average gap between the observed Qini curve and random targeting,
    divided by the number of treated units. 0 = no better than random."""
    phi, q = qini_curve(score, retained, t, n_points)
    gap = q - phi * q[-1]
    n_treat = max(int(np.sum(np.asarray(t) == 1)), 1)
    return float(np.mean(gap[1:]) / n_treat)


def top_k_mask(score, coverage: float, seed: int = 0) -> np.ndarray:
    """Boolean mask for the top `coverage` share by score.

    Ties at the cut-off are broken at random. Use expected_policy_value for
    the tie-free true value; this mask is for observed-outcome estimates."""
    score = np.asarray(score, float)
    k = int(round(coverage * len(score)))
    jitter = np.random.default_rng(seed).random(len(score))
    order = np.lexsort((jitter, -score))
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:k]] = True
    return mask


def expected_policy_value(score, tau, coverage: float) -> float:
    """Expected extra retained customers if we contact the top `coverage` share (true tau)."""
    g = expected_gain_curve(score, tau)
    return float(g[int(round(coverage * (len(g) - 1)))])


def policy_value_estimated(mask, retained, t) -> float:
    """Extra retained customers estimated from the A/B data inside the selected group.

    (retention rate of treated - retention rate of control) * group size.
    Works on real data because assignment inside the group is still random.
    """
    y = np.asarray(retained, float)[mask]
    tt = np.asarray(t)[mask]
    if (tt == 1).sum() == 0 or (tt == 0).sum() == 0:
        return 0.0
    return float((y[tt == 1].mean() - y[tt == 0].mean()) * mask.sum())


def net_value_policy(tau_hat, tau_true, value, cost) -> dict[str, float]:
    """Contact only customers with tau_hat * value - cost > 0; report the true outcome."""
    tau_hat, tau_true = np.asarray(tau_hat), np.asarray(tau_true)
    value, cost = np.asarray(value), np.asarray(cost)
    chosen = tau_hat * value - cost > 0
    return {
        "n_contacted": int(chosen.sum()),
        "share_contacted": float(chosen.mean()),
        "true_saved": float(tau_true[chosen].sum()),
        "true_net_value": float((tau_true * value - cost)[chosen].sum()),
    }


def dr_pseudo_outcome(y, t, m0, m1, e) -> np.ndarray:
    """Doubly robust (AIPW) score for the churn REDUCTION of each unit.

    phi = (m0 - m1) + (1-t)(y - m0)/(1-e) - t(y - m1)/e
    E[phi | x] equals the true reduction if either the outcome models or the
    propensity are right; in an RCT e is known, so phi is unbiased.
    """
    y, t = np.asarray(y, float), np.asarray(t, float)
    return (m0 - m1) + (1 - t) * (y - m0) / (1 - e) - t * (y - m1) / e


def dr_loss(tau_hat, phi) -> float:
    """mean((phi - tau_hat)^2). Equals PEHE^2 plus a constant shared by all models,
    so it ranks models like PEHE does, without knowing the true tau."""
    return float(np.mean((np.asarray(phi) - np.asarray(tau_hat)) ** 2))
