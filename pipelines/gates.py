"""Quality gates. A model is promoted (and CI passes) only if every gate holds.

These are sanity checks written before the first full run, not targets
chosen after seeing results. Code guide: section "Quality gates" (sec:gates).
"""

from __future__ import annotations

import math
from typing import Any


def _finite(*values) -> bool:
    return all(v is not None and math.isfinite(v) for v in values)


def evaluate_gates(experiment: dict[str, Any], bench: dict[str, Any]) -> dict[str, Any]:
    table = bench["table"]
    primary = bench["primary_model"]
    lin = experiment["ate_lin"]

    checks = {
        # The A/B test itself must be valid before anything downstream counts.
        "srm_passed": experiment["srm"]["passed"],
        "balance_passed": experiment["balance"]["passed"],
        "ate_ci_finite": _finite(lin["estimate"], lin["ci_low"], lin["ci_high"]),
        # Qini check: the primary model ranks better than random and better than k-means.
        "qini_beats_random": table[primary]["qini_true_normalized"]["mean"] > 0,
        "qini_beats_kmeans": table[primary]["qini_true_normalized"]["mean"]
        > table["kmeans"]["qini_true_normalized"]["mean"],
        # Policy-value check: at the planned coverage, uplift targeting keeps at least
        # as many customers as churn-risk targeting, on average over draws.
        "policy_value_vs_risk": table[primary]["policy_true"]["mean"] >= table["churn_risk"]["policy_true"]["mean"],
        "no_nan_metrics": all(
            _finite(cell["mean"]) for row in table.values() for cell in row.values() if cell is not None
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}
