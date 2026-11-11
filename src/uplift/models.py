"""Uplift (heterogeneous treatment effect) models.

Every model scores subscribers by expected *incremental net revenue* from sending the offer:

    uplift_revenue(x) = price * [(1 - discount) * P(renew | offer, x) - P(renew | no offer, x)]

This is the right target for the business. A subscriber who would renew anyway has positive
renewal probability but negative revenue uplift, because the offer only gives away margin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

FEATURES = ["tenure_months", "plan_code", "annual_price", "usage_days_30", "usage_trend",
            "support_tickets_90d", "price_increase", "auto_renew", "past_discounts",
            "email_engagement", "channel_code"]
DISCOUNT = 0.20
PROPENSITY = 0.5  # randomized 50/50


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["plan_code"] = out["plan"].map({"basic": 0, "standard": 1, "premium": 2})
    out["channel_code"] = out["signup_channel"].map({"web": 0, "app": 1, "partner": 2})
    out["revenue"] = out["renewed"] * out["annual_price"] * np.where(out["treatment"] == 1, 1 - DISCOUNT, 1)
    return out


def _clf(seed):
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                          min_samples_leaf=80, l2_regularization=1.0, random_state=seed)


def _reg(seed):
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                         min_samples_leaf=150, l2_regularization=1.0, random_state=seed)


class ChurnRiskTargeting:
    """Common practice: offer the discount to whoever looks most likely to leave.
    Scored as negative renewal probability without an offer (a response model, not uplift)."""

    name = "churn_risk"

    def fit(self, d: pd.DataFrame, seed: int = 42):
        c = d[d.treatment == 0]
        self.m = _clf(seed).fit(c[FEATURES], c.renewed)
        return self

    def score(self, d: pd.DataFrame) -> np.ndarray:
        return -self.m.predict_proba(d[FEATURES])[:, 1]


class TLearner:
    """Separate renewal models for each arm; uplift is the difference in expected revenue."""

    name = "t_learner"

    def fit(self, d: pd.DataFrame, seed: int = 42):
        self.m1 = _clf(seed).fit(d.loc[d.treatment == 1, FEATURES], d.loc[d.treatment == 1, "renewed"])
        self.m0 = _clf(seed).fit(d.loc[d.treatment == 0, FEATURES], d.loc[d.treatment == 0, "renewed"])
        return self

    def score(self, d: pd.DataFrame) -> np.ndarray:
        p1 = self.m1.predict_proba(d[FEATURES])[:, 1]
        p0 = self.m0.predict_proba(d[FEATURES])[:, 1]
        return d["annual_price"].to_numpy() * ((1 - DISCOUNT) * p1 - p0)


class XLearner:
    """Künzel et al. (2019). Imputes individual effects from the other arm's outcome model,
    then learns those effects directly. Often better when effects are smaller than the
    baseline outcome variation."""

    name = "x_learner"

    def fit(self, d: pd.DataFrame, seed: int = 42):
        t, c = d[d.treatment == 1], d[d.treatment == 0]
        mu1 = _reg(seed).fit(t[FEATURES], t.revenue)
        mu0 = _reg(seed).fit(c[FEATURES], c.revenue)
        d1 = t.revenue - mu0.predict(t[FEATURES])     # observed treated - predicted control
        d0 = mu1.predict(c[FEATURES]) - c.revenue     # predicted treated - observed control
        self.tau1 = _reg(seed).fit(t[FEATURES], d1)
        self.tau0 = _reg(seed).fit(c[FEATURES], d0)
        return self

    def score(self, d: pd.DataFrame) -> np.ndarray:
        return PROPENSITY * self.tau0.predict(d[FEATURES]) + (1 - PROPENSITY) * self.tau1.predict(d[FEATURES])


class TransformedOutcome:
    """Single regression on Z = Y (T - e) / (e (1 - e)), whose expectation is the uplift.
    Simple and unbiased, but noisy, so it needs more data than the meta-learners."""

    name = "transformed_outcome"

    def fit(self, d: pd.DataFrame, seed: int = 42):
        e = PROPENSITY
        z = d.revenue * (d.treatment - e) / (e * (1 - e))
        self.m = _reg(seed).fit(d[FEATURES], z)
        return self

    def score(self, d: pd.DataFrame) -> np.ndarray:
        return self.m.predict(d[FEATURES])


def all_models():
    return [ChurnRiskTargeting(), TLearner(), XLearner(), TransformedOutcome()]
