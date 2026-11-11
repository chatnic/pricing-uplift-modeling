import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from generate_data import generate  # noqa: E402
from uplift.evaluate import qini_coefficient, targeting_curve  # noqa: E402
from uplift.models import DISCOUNT, XLearner, prepare  # noqa: E402


@pytest.fixture(scope="module")
def data():
    df, truth = generate(n=20_000, seed=9)
    return prepare(df), truth


def test_randomization_balanced(data):
    df, _ = data
    assert abs(df.treatment.mean() - 0.5) < 0.02
    # Pre-treatment features should not differ between arms.
    g = df.groupby("treatment")[["tenure_months", "usage_days_30", "auto_renew"]].mean()
    assert (g.diff().abs().iloc[1] / g.iloc[0] < 0.05).all()


def test_revenue_applies_discount(data):
    df, _ = data
    t = df[(df.treatment == 1) & (df.renewed == 1)].iloc[0]
    assert t.revenue == pytest.approx(t.annual_price * (1 - DISCOUNT))


def test_targeting_curve_endpoints(data):
    df, _ = data
    cur = targeting_curve(df, np.random.default_rng(0).random(len(df)))
    assert cur.iloc[0]["incremental_revenue"] == 0
    assert cur.iloc[-1]["fraction"] == 1.0


def test_perfect_score_beats_random(data):
    df, truth = data
    true_tau = df[["subscriber_id"]].merge(truth, on="subscriber_id").true_uplift.to_numpy()
    perfect = qini_coefficient(targeting_curve(df, true_tau))
    random = qini_coefficient(targeting_curve(df, np.random.default_rng(1).random(len(df))))
    assert perfect > random


def test_x_learner_recovers_effect_direction(data):
    df, truth = data
    s = XLearner().fit(df).score(df)
    tau = df[["subscriber_id"]].merge(truth, on="subscriber_id").true_uplift.to_numpy()
    assert np.corrcoef(s, tau)[0, 1] > 0.3
