"""Simulate a randomized retention-discount experiment for a subscription business.

Subscribers coming up for annual renewal were randomly assigned to receive a 20% renewal
discount offer (treatment) or nothing (control). The simulator gives each subscriber a
hidden individual treatment effect, so the data contains all four classic uplift groups:

  persuadables   renew only if offered the discount (positive effect)
  sure things    renew either way (effect near zero, so the discount is wasted margin)
  lost causes    leave either way
  sleeping dogs  the offer reminds them to reconsider, and they become LESS likely to renew

The true effect is saved in a separate file for validation only. Models never see it.

Usage:
    python data/generate_data.py --n 60000 --seed 42 --out data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int = 60_000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    tenure_m = np.clip(rng.gamma(2.2, 11, n), 1, 120).round()
    plan = rng.choice(["basic", "standard", "premium"], n, p=[0.35, 0.45, 0.20])
    price = np.select([plan == "basic", plan == "standard"], [96.0, 144.0], 228.0)
    usage_days_30 = np.clip(rng.beta(2, 2.2, n) * 30 + rng.normal(0, 2, n), 0, 30).round()
    usage_trend = rng.normal(0, 1, n)                      # standardized change in usage, last 90 days
    support_tickets_90 = rng.poisson(0.6, n)
    price_increase = rng.random(n) < 0.3                   # renewal price went up this year
    auto_renew = rng.random(n) < 0.55
    discount_history = rng.poisson(0.5, n)                 # past discounts accepted
    email_engagement = np.clip(rng.beta(2, 4, n), 0, 1)
    channel = rng.choice(["web", "app", "partner"], n, p=[0.5, 0.35, 0.15])

    # Baseline renewal probability without any offer.
    z = (-0.2 + 0.035 * np.minimum(tenure_m, 48) + 0.06 * (usage_days_30 - 12) + 0.45 * usage_trend
         - 0.35 * support_tickets_90 - 0.7 * price_increase + 1.4 * auto_renew
         + np.select([plan == "basic", plan == "standard"], [-0.2, 0.0], 0.25)
         - 0.15 * (channel == "partner"))
    p0 = 1 / (1 + np.exp(-z))

    # Individual treatment effect (on the probability scale).
    persuadable = (0.16 * price_increase + 0.07 * (discount_history > 0) + 0.06 * (usage_days_30 > 8)
                   - 0.02 * (tenure_m > 36))
    # Auto-renewers who rarely use the product get "woken up" by the offer email.
    sleeping_dog = -0.16 * (auto_renew & (usage_days_30 < 9) & (email_engagement > 0.25))
    tau = (persuadable + sleeping_dog) * (4 * p0 * (1 - p0)) ** 0.5  # effects shrink near 0 or 1
    tau = tau + rng.normal(0, 0.01, n)

    treated = rng.random(n) < 0.5
    p = np.clip(p0 + treated * tau, 0.001, 0.999)
    renewed = rng.random(n) < p

    df = pd.DataFrame({
        "subscriber_id": [f"S{300000 + i}" for i in range(n)],
        "tenure_months": tenure_m.astype(int),
        "plan": plan,
        "annual_price": price,
        "usage_days_30": usage_days_30.astype(int),
        "usage_trend": usage_trend.round(3),
        "support_tickets_90d": support_tickets_90,
        "price_increase": price_increase.astype(int),
        "auto_renew": auto_renew.astype(int),
        "past_discounts": discount_history,
        "email_engagement": email_engagement.round(3),
        "signup_channel": channel,
        "treatment": treated.astype(int),
        "renewed": renewed.astype(int),
    })
    truth = pd.DataFrame({"subscriber_id": df.subscriber_id, "p_renew_control": p0.round(4),
                          "true_uplift": tau.round(4)})
    return df, truth


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    a = p.parse_args()
    df, truth = generate(a.n, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "experiment.csv", index=False)
    truth.to_csv(a.out / "ground_truth_validation_only.csv", index=False)
    t, c = df[df.treatment == 1], df[df.treatment == 0]
    print(f"Wrote {len(df):,} subscribers. Renewal: control {c.renewed.mean():.1%}, "
          f"treatment {t.renewed.mean():.1%}. True uplift < 0 for {(truth.true_uplift < -0.01).mean():.0%}.")


if __name__ == "__main__":
    main()
