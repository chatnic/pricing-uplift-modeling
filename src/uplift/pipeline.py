"""Fit uplift models on the experiment, evaluate targeting on a randomized holdout.

    PYTHONPATH=src python -m uplift.pipeline --data data/raw --out reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from uplift.evaluate import CONTACT_COST, qini_coefficient, targeting_curve, true_policy_value  # noqa: E402
from uplift.models import DISCOUNT, all_models, prepare  # noqa: E402

SEED = 42
LABELS = {"churn_risk": "Churn-risk targeting (status quo)", "t_learner": "T-learner",
          "x_learner": "X-learner", "transformed_outcome": "Transformed outcome"}
COLORS = {"churn_risk": "#c62828", "t_learner": "#6c8ebf", "x_learner": "#1f4e8c",
          "transformed_outcome": "#9aa5b1"}


def run(data: Path, out: Path) -> dict:
    df = prepare(pd.read_csv(data / "experiment.csv"))
    truth_path = data / "ground_truth_validation_only.csv"
    truth = pd.read_csv(truth_path) if truth_path.exists() else None
    train, test = train_test_split(df, test_size=0.3, stratify=df[["treatment", "renewed"]], random_state=SEED)

    t, c = test[test.treatment == 1], test[test.treatment == 0]
    ate = {
        "renewal_control": float(c.renewed.mean()), "renewal_treatment": float(t.renewed.mean()),
        "revenue_per_sub_control": float(c.revenue.mean()), "revenue_per_sub_treatment": float(t.revenue.mean()),
    }
    ate["treat_all_net_per_1k"] = (ate["revenue_per_sub_treatment"] - ate["revenue_per_sub_control"] - CONTACT_COST) * 1000
    print(f"Holdout: renewal {ate['renewal_control']:.1%} -> {ate['renewal_treatment']:.1%}, "
          f"net revenue per 1k if everyone gets the offer: ${ate['treat_all_net_per_1k']:,.0f}")

    curves, results, scores = {}, {}, {}
    for m in all_models():
        m.fit(train, SEED)
        s = m.score(test)
        scores[m.name] = s
        cur = targeting_curve(test, s)
        curves[m.name] = cur
        best = cur.loc[cur.per_1k_subscribers.idxmax()]
        res = {"qini": qini_coefficient(cur), "best_fraction": float(best.fraction),
               "best_net_per_1k": float(best.per_1k_subscribers)}
        if truth is not None:
            tr = test[["subscriber_id"]].merge(truth, on="subscriber_id")
            res["corr_with_true_uplift"] = float(np.corrcoef(s, tr.true_uplift)[0, 1]) if m.name != "churn_risk" else None
            res["true_value_at_best_fraction_per_1k"] = true_policy_value(test, truth, s, float(best.fraction))
        results[m.name] = res
        print(f"  {m.name:<20} qini {res['qini']:8.1f}  best: target top {best.fraction:.0%} -> "
              f"${best.per_1k_subscribers:,.0f} per 1k")

    # Who is in each targeting group? Profile the X-learner's top vs. bottom deciles.
    xs = scores["x_learner"]
    test = test.assign(uplift_score=xs, decile=pd.qcut(-xs, 10, labels=range(1, 11)))
    profile = test.groupby("decile", observed=True).agg(
        subscribers=("renewed", "size"),
        mean_score=("uplift_score", "mean"),
        renewal_control=("renewed", lambda s: s[test.loc[s.index, "treatment"] == 0].mean()),
        renewal_treated=("renewed", lambda s: s[test.loc[s.index, "treatment"] == 1].mean()),
        price_increase=("price_increase", "mean"),
        auto_renew=("auto_renew", "mean"),
        usage_days_30=("usage_days_30", "mean"),
        past_discounts=("past_discounts", "mean"),
    )
    profile["observed_lift_pts"] = (profile.renewal_treated - profile.renewal_control) * 100

    metrics = {"discount": DISCOUNT, "contact_cost": CONTACT_COST, "holdout_size": len(test),
               "average_effect": ate, "models": results}
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    pd.concat({k: v for k, v in curves.items()}, names=["model"]).reset_index(level=0) \
        .round(2).to_csv(out / "tables" / "targeting_curves.csv", index=False)
    profile.round(3).to_csv(out / "tables" / "x_learner_deciles.csv")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    figures(curves, profile, ate, out / "figures")
    print(profile.round(2).to_string())
    return metrics


def figures(curves, profile, ate, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name, cur in curves.items():
        ax.plot(cur.fraction, cur.per_1k_subscribers, lw=2.4 if name == "x_learner" else 1.6,
                color=COLORS[name], label=LABELS[name])
    end = curves["x_learner"].per_1k_subscribers.iloc[-1]
    ax.plot([0, 1], [0, end], ls="--", color="grey", lw=1, label="Random targeting")
    ax.axhline(0, color="black", lw=0.7)
    ax.set(xlabel="Share of renewing subscribers sent the offer (highest score first)",
           ylabel="Incremental net revenue per 1,000 subscribers ($)",
           title="Targeting curves on the randomized holdout")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout(); fig.savefig(fig_dir / "targeting_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = np.where(profile.observed_lift_pts > 0, "#1f4e8c", "#c62828")
    ax.bar(profile.index.astype(int), profile.observed_lift_pts, color=colors)
    ax.axhline(0, color="black", lw=0.7)
    ax.set(xlabel="X-learner uplift decile (1 = highest predicted uplift)",
           ylabel="Observed renewal lift (pts)", title="Observed effect of the offer by predicted-uplift decile")
    ax.set_xticks(range(1, 11))
    fig.tight_layout(); fig.savefig(fig_dir / "lift_by_decile.png", dpi=150); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    a = p.parse_args()
    run(a.data, a.out)


if __name__ == "__main__":
    main()
