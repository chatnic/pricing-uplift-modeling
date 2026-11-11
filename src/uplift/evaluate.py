"""Uplift evaluation on a randomized holdout.

Individual effects are never observed, so models are judged by targeting curves:
rank subscribers by score, and for each top-k% compare treated vs. control outcomes
inside that group. Randomization makes the comparison unbiased.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONTACT_COST = 0.50


def targeting_curve(d: pd.DataFrame, score: np.ndarray, fractions=None) -> pd.DataFrame:
    """Estimated incremental net revenue from treating the top-k% by score (vs. treating nobody)."""
    if fractions is None:
        fractions = np.round(np.arange(0, 1.0001, 0.05), 2)
    order = np.argsort(-score)
    T = d["treatment"].to_numpy()[order]
    R = d["revenue"].to_numpy()[order]
    n = len(d)
    rows = []
    for f in fractions:
        k = int(round(f * n))
        if k == 0:
            rows.append({"fraction": f, "incremental_revenue": 0.0})
            continue
        t, r = T[:k], R[:k]
        if t.sum() == 0 or (1 - t).sum() == 0:
            continue
        uplift_per_person = r[t == 1].mean() - r[t == 0].mean()
        rows.append({"fraction": f, "incremental_revenue": uplift_per_person * k - CONTACT_COST * k})
    out = pd.DataFrame(rows)
    out["per_1k_subscribers"] = out["incremental_revenue"] / n * 1000
    return out


def qini_coefficient(curve: pd.DataFrame) -> float:
    """Area between the model's curve and the random-targeting line (normalized per 1k)."""
    x = curve["fraction"].to_numpy()
    y = curve["per_1k_subscribers"].to_numpy()
    random_line = x * y[-1]
    return float(np.trapezoid(y - random_line, x))


def true_policy_value(d: pd.DataFrame, truth: pd.DataFrame, score: np.ndarray, fraction: float) -> float:
    """Validation only: exact expected incremental net revenue per 1k, using the simulator's ground truth."""
    g = d[["subscriber_id", "annual_price"]].merge(truth, on="subscriber_id")
    from uplift.models import DISCOUNT
    gain = g.annual_price * ((1 - DISCOUNT) * (g.p_renew_control + g.true_uplift) - g.p_renew_control) - CONTACT_COST
    k = int(round(fraction * len(g)))
    top = np.argsort(-score)[:k]
    return float(gain.to_numpy()[top].sum() / len(g) * 1000)
