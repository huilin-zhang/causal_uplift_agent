"""Choose estimators and a targeting policy from what the data looks like.

The ML agent calls profile_data() then select(). Each rule is written out
with its reason, so the agent can explain its choice and the reviewer can
check it. Code guide: section "Estimator selector" (sec:selector).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

LARGE_N = 200_000
RARE_OUTCOME = 0.05


@dataclass
class DataProfile:
    n: int
    design: str               # "rct" or "observational"
    n_arms: int
    treat_share: float        # share in the largest non-control arm vs control
    outcome_type: str         # "binary" or "continuous"
    positive_rate: float | None
    has_value_cost: bool


@dataclass
class Selection:
    estimators: list[str]
    policy: str
    propensity: str           # "known" or "estimated"
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def profile_data(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    design: str,
    control_label=0,
    value_col: str | None = None,
    cost_col: str | None = None,
) -> DataProfile:
    """Summarize the facts the selection rules need. `design` comes from the
    data owner: whether assignment was randomized cannot be read off the data."""
    arms = df[treatment].unique()
    y = df[outcome]
    binary = set(pd.unique(y.dropna())) <= {0, 1}
    counts = df[treatment].value_counts()
    n_control = counts.get(control_label, 0)
    largest_treated = counts.drop(control_label, errors="ignore").max()
    return DataProfile(
        n=len(df),
        design=design,
        n_arms=len(arms),
        treat_share=float(largest_treated / (largest_treated + n_control)) if n_control else 1.0,
        outcome_type="binary" if binary else "continuous",
        positive_rate=float(y.mean()) if binary else None,
        has_value_cost=bool(value_col and cost_col and value_col in df and cost_col in df),
    )


def select(p: DataProfile) -> Selection:
    """Rule table. Rules are checked in order; later rules adjust earlier choices."""
    s = Selection(estimators=[], policy="net_value_top_k", propensity="known")

    if p.design == "observational":
        s.estimators = ["causal_forest", "ipw", "x_learner"]
        s.propensity = "estimated"
        s.reasons.append("Not randomized: treatment depends on x, so we model the propensity "
                         "and use estimators that adjust for it. A plain T-learner is not allowed.")
        s.warnings.append("Effects rely on no unmeasured confounding; check propensity overlap.")
    elif not 0.3 <= p.treat_share <= 0.7:
        s.estimators = ["x_learner", "causal_forest", "ipw", "t_learner"]
        s.reasons.append(f"Arms are unbalanced (treated share {p.treat_share:.2f}); the X-learner "
                         "borrows strength from the large arm to model the small one.")
    else:
        s.estimators = ["causal_forest", "x_learner", "t_learner", "ipw"]
        s.reasons.append("Balanced randomized test: all learners are valid; propensity is known.")

    if p.positive_rate is not None and p.positive_rate < RARE_OUTCOME:
        s.estimators = [e for e in s.estimators if e != "t_learner"]
        s.warnings.append(f"Rare outcome ({p.positive_rate:.1%}): effect estimates are noisy; "
                          "expect wide intervals and prefer pooled learners.")
        s.reasons.append("Rare outcome: dropped the T-learner, whose two separate models each see few events.")

    if p.n > LARGE_N:
        s.estimators = [e for e in s.estimators if e in ("t_learner", "x_learner", "ipw")] or ["t_learner"]
        s.reasons.append(f"N = {p.n:,} is large: skip the causal forest for run time; fit on a subsample if needed.")

    if p.n_arms > 2:
        s.policy = "best_arm_net_value"
        s.reasons.append(f"{p.n_arms} arms: fit each treatment arm against control and give each "
                         "customer the arm with the highest net value, or no contact.")

    if p.outcome_type == "continuous":
        s.reasons.append("Continuous outcome: learners use regressors and the effect is in outcome units.")

    if not p.has_value_cost:
        s.policy = "tau_top_k" if p.n_arms <= 2 else "best_arm_tau"
        s.reasons.append("No value or cost columns: rank by estimated effect. Never rank by risk.")

    return s


def describe(profile: DataProfile, selection: Selection) -> str:
    lines = [
        f"Data: n={profile.n:,}, design={profile.design}, arms={profile.n_arms}, "
        f"treated share={profile.treat_share:.2f}, outcome={profile.outcome_type}"
        + (f" (rate {profile.positive_rate:.1%})" if profile.positive_rate is not None else ""),
        f"Estimators: {', '.join(selection.estimators)}; policy: {selection.policy}",
    ]
    lines += [f"- {r}" for r in selection.reasons]
    lines += [f"! {w}" for w in selection.warnings]
    return "\n".join(lines)


def is_allowed(estimator: str, selection: Selection) -> bool:
    return estimator in selection.estimators

