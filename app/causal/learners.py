"""Uplift estimators with one interface: fit(x, t, y) then predict(x) -> tau_hat.

y is churn (1 = churned). Every learner returns the predicted churn
REDUCTION from outreach, so a larger score means "contact this customer".
Code guide: section "Uplift learners" (sec:learners).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Strongly regularized trees: the true effect varies by only ~1.5 pp across
# customers, so flexible trees fit noise. Final settings per learner come from
# GRID, scored by the doubly robust loss on validation (see benchmark.py).
LGBM_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.02,
    num_leaves=8,
    min_child_samples=500,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
)
GRID = [
    {"num_leaves": leaves, "min_child_samples": child}
    for leaves in (4, 8)
    for child in (500, 1000)
]


def _clf(seed: int, params: dict | None = None) -> LGBMClassifier:
    return LGBMClassifier(random_state=seed, **{**LGBM_PARAMS, **(params or {})})


def _reg(seed: int, params: dict | None = None) -> LGBMRegressor:
    return LGBMRegressor(random_state=seed, **{**LGBM_PARAMS, **(params or {})})


class Propensity:
    """P(treated | x). Known in an RCT; estimated by logistic regression otherwise.

    Clipping to [0.05, 0.95] keeps inverse weights bounded when overlap is poor.
    """

    def __init__(self, known: float | None = None, clip: tuple[float, float] = (0.05, 0.95)):
        self.known, self.clip = known, clip
        self.model: LogisticRegression | None = None
        self.scaler: StandardScaler | None = None

    def fit(self, x: pd.DataFrame, t: np.ndarray) -> Propensity:
        if self.known is None:
            self.scaler = StandardScaler().fit(x)
            self.model = LogisticRegression(max_iter=1000).fit(self.scaler.transform(x), t)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.known is not None:
            return np.full(len(x), self.known)
        e = self.model.predict_proba(self.scaler.transform(x))[:, 1]
        return np.clip(e, *self.clip)


class RiskModel:
    """Baseline policy: rank by predicted churn without outreach.

    Trained on the control arm only, so the score is mu0(x). This is what most
    retention teams do today. It finds people who will churn, not people who
    can be saved.
    """

    name = "churn_risk"

    def __init__(self, seed: int = 0):
        self.model = _clf(seed)

    def fit(self, x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> RiskModel:
        control = np.asarray(t) == 0
        self.model.fit(x[control], np.asarray(y)[control])
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]


class KMeansUplift:
    """Segment baseline: cluster customers, then use each cluster's A/B difference.

    This is the "segment-level" approach many teams start with. Every customer
    in a cluster gets the same effect estimate.
    """

    name = "kmeans"

    def __init__(self, k: int = 8, seed: int = 0):
        self.k, self.seed = k, seed

    def fit(self, x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> KMeansUplift:
        t, y = np.asarray(t), np.asarray(y, float)
        self.scaler = StandardScaler().fit(x)
        self.km = KMeans(n_clusters=self.k, n_init=10, random_state=self.seed)
        labels = self.km.fit_predict(self.scaler.transform(x))
        self.cluster_tau = np.zeros(self.k)
        for c in range(self.k):
            in_c = labels == c
            treated, control = in_c & (t == 1), in_c & (t == 0)
            if treated.any() and control.any():
                self.cluster_tau[c] = y[control].mean() - y[treated].mean()
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.cluster_tau[self.km.predict(self.scaler.transform(x))]


class TLearner:
    """Two models: churn without outreach (control arm) and with it (treated arm).

    tau_hat = mu0_hat(x) - mu1_hat(x). Simple, but each model is fit on half
    the data and their errors do not cancel.
    """

    name = "t_learner"

    def __init__(self, seed: int = 0, params: dict | None = None):
        self.m0, self.m1 = _clf(seed, params), _clf(seed + 1, params)

    def fit(self, x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> TLearner:
        t, y = np.asarray(t), np.asarray(y)
        self.m0.fit(x[t == 0], y[t == 0])
        self.m1.fit(x[t == 1], y[t == 1])
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.m0.predict_proba(x)[:, 1] - self.m1.predict_proba(x)[:, 1]


class XLearner:
    """Kuenzel et al. (2019). Impute each unit's effect, then model the effect directly.

    Step 1: fit mu0, mu1 as in the T-learner.
    Step 2: imputed effects on churn,
            treated units: D1 = y - mu0(x);  control units: D0 = mu1(x) - y.
    Step 3: regress D1 and D0 on x to get tau1(x), tau0(x).
    Step 4: blend, tau(x) = e(x) * tau0(x) + (1 - e(x)) * tau1(x).
    Helps most when one arm is much larger than the other.
    """

    name = "x_learner"

    def __init__(self, seed: int = 0, propensity: Propensity | None = None, params: dict | None = None):
        self.seed, self.params = seed, params
        self.propensity = propensity or Propensity(known=0.5)

    def fit(self, x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> XLearner:
        t, y = np.asarray(t), np.asarray(y, float)
        x0, x1 = x[t == 0], x[t == 1]
        m0 = _clf(self.seed, self.params).fit(x0, y[t == 0])
        m1 = _clf(self.seed + 1, self.params).fit(x1, y[t == 1])
        d1 = y[t == 1] - m0.predict_proba(x1)[:, 1]
        d0 = m1.predict_proba(x0)[:, 1] - y[t == 0]
        self.tau1 = _reg(self.seed + 2, self.params).fit(x1, d1)
        self.tau0 = _reg(self.seed + 3, self.params).fit(x0, d0)
        self.propensity.fit(x, t)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        e = self.propensity.predict(x)
        effect_on_churn = e * self.tau0.predict(x) + (1 - e) * self.tau1.predict(x)
        return -effect_on_churn


class CausalForest:
    """Generalized random forest (Athey, Tibshirani, Wager 2019) via econml CausalForestDML.

    First removes the part of y and t that x predicts (double ML), then grows
    trees that split on where the effect differs. Honest splitting gives
    valid confidence intervals.
    """

    name = "causal_forest"

    def __init__(self, seed: int = 0, n_estimators: int = 300, known_propensity: bool = True):
        from econml.dml import CausalForestDML
        from sklearn.dummy import DummyClassifier

        # In an RCT the propensity is a constant; the class prior estimates it exactly.
        model_t = DummyClassifier(strategy="prior") if known_propensity else LogisticRegression(max_iter=1000)
        self.est = CausalForestDML(
            model_y=_reg(seed),
            model_t=model_t,
            discrete_treatment=True,
            n_estimators=max(4, n_estimators // 4 * 4),  # econml grows trees in groups of 4
            min_samples_leaf=100,
            cv=2,
            random_state=seed,
        )

    def fit(self, x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> CausalForest:
        self.est.fit(np.asarray(y, float), np.asarray(t), X=x.to_numpy(float))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return -self.est.effect(x.to_numpy(float)).ravel()


class IPWLearner:
    """Transformed-outcome learner (Athey and Imbens 2016), an IPW method for CATE.

    Y* = y * (t - e) / (e * (1 - e)) has E[Y* | x] = mu1(x) - mu0(x), so any
    regression of Y* on x estimates the effect. Unbiased, but Y* is very noisy
    (values like +2 or -2 for every churner), so it needs a lot of data.
    """

    name = "ipw"

    def __init__(self, seed: int = 0, propensity: Propensity | None = None, params: dict | None = None):
        self.seed, self.params = seed, params
        self.propensity = propensity or Propensity(known=0.5)

    def fit(self, x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> IPWLearner:
        t, y = np.asarray(t, float), np.asarray(y, float)
        e = self.propensity.fit(x, t).predict(x)
        y_star = y * (t - e) / (e * (1 - e))
        self.model = _reg(self.seed, self.params).fit(x, y_star)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return -self.model.predict(x)


class ConstantATE:
    """Reference model: the same effect (the train-set A/B difference) for everyone.

    Any learner that cannot beat this has not found real differences between
    customers. It ranks nobody above anybody, so it has no targeting value.
    """

    name = "constant_ate"

    def fit(self, x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> ConstantATE:
        t, y = np.asarray(t), np.asarray(y, float)
        self.ate = y[t == 0].mean() - y[t == 1].mean()
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.ate)


UPLIFT_LEARNERS = ["t_learner", "x_learner", "causal_forest", "ipw"]
TUNABLE = {"t_learner", "x_learner", "ipw"}


def make_learner(
    name: str,
    seed: int = 0,
    treat_share: float | None = 0.5,
    forest_trees: int = 300,
    kmeans_k: int = 8,
    params: dict | None = None,
):
    """Factory. treat_share=None means the propensity is unknown (observational data).
    params: LightGBM overrides for the tunable learners (one entry of GRID)."""
    known = treat_share is not None
    if name == "churn_risk":
        return RiskModel(seed)
    if name == "kmeans":
        return KMeansUplift(kmeans_k, seed)
    if name == "constant_ate":
        return ConstantATE()
    if name == "t_learner":
        return TLearner(seed, params)
    if name == "x_learner":
        return XLearner(seed, Propensity(known=treat_share), params)
    if name == "causal_forest":
        return CausalForest(seed, forest_trees, known_propensity=known)
    if name == "ipw":
        return IPWLearner(seed, Propensity(known=treat_share), params)
    raise ValueError(f"unknown learner: {name}")
