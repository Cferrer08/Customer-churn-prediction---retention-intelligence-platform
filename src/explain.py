"""
Model explainability with SHAP.

A churn score on its own is not actionable - "82% likely to leave" tells a retention agent
nothing about what to say on the call. This module produces three layers:

1. Global drivers      - which features move churn risk across the whole book.
2. Per-customer drivers - the signed contribution of each feature for one customer.
3. Plain-English reasons - the same contributions rendered as sentences a human can act on
                           ("Support tickets up 3 in the last quarter"), which is what the
                           dashboard and the recommendation engine consume.

SHAP values are additive in log-odds space, so the per-customer contributions genuinely sum
to that customer's score. Nothing here is a post-hoc rationalisation of the prediction.
"""

from __future__ import annotations

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR  # noqa: E402

TOP_POSITIVE = 6
TOP_NEGATIVE = 4

# How each feature should be described to a human. {v} is the customer's own value.
PHRASES = {
    "usage_trend_3m": ("Usage {dir} {pct} versus the prior quarter", "pct_change"),
    "login_trend_3m": ("Logins {dir} {pct} versus the prior quarter", "pct_change"),
    "usage_hours_l3": ("Recent usage averages {v:.1f} hours per month", "raw"),
    "usage_hours_l1": ("Last month's usage was {v:.1f} hours", "raw"),
    "usage_hours_l6": ("Six-month average usage of {v:.1f} hours per month", "raw"),
    "usage_hours_std": ("Usage swings by {v:.1f} hours month to month", "raw"),
    "usage_hours_max": ("Peak usage of {v:.1f} hours in a month", "raw"),
    "logins_l3": ("{v:.0f} logins per month recently", "raw"),
    "logins_mean": ("{v:.0f} logins per month on average", "raw"),
    "spend_l3": ("Recent monthly bill of ${v:.2f}", "raw"),
    "data_gb_l3": ("Uses {v:.1f} GB per month recently", "raw"),
    "data_gb_mean": ("Uses {v:.1f} GB per month on average", "raw"),
    "discount_months": ("Has taken a discount in {v:.0f} months", "raw"),
    "months_active": ("{v:.0f} months of recorded activity", "raw"),
    "usage_vs_lifetime": ("Recent usage is {pct} of this customer's own norm", "ratio"),
    "usage_hours_mean": ("Lifetime average usage of {v:.1f} hours per month", "raw"),
    "tickets_l3": ("{v:.1f} support tickets per month recently", "raw"),
    "tickets_l6": ("{v:.1f} support tickets per month over six months", "raw"),
    "tickets_total": ("{v:.0f} support tickets raised in total", "raw"),
    "ticket_trend_3m": ("Support contacts {dir} by {v_abs:.1f} per month", "delta"),
    "critical_tickets_total": ("{v:.0f} high-priority tickets on record", "raw"),
    "support_frequency": ("{v:.2f} tickets per active month", "raw"),
    "satisfaction_l3": ("Recent satisfaction score of {v:.1f} out of 5", "raw"),
    "satisfaction_mean": ("Lifetime satisfaction score of {v:.1f} out of 5", "raw"),
    "satisfaction_trend_3m": ("Satisfaction {dir} {v_abs:.1f} points", "delta"),
    "satisfaction_min": ("Lowest recorded satisfaction of {v:.1f}", "raw"),
    "tenure_months": ("{v:.0f} months as a customer", "raw"),
    "months_active": ("{v:.0f} months of activity on record", "raw"),
    "monthly_price": ("Pays ${v:.2f} per month", "raw"),
    "avg_monthly_spend": ("Average monthly bill of ${v:.2f}", "raw"),
    "revenue_per_usage_hour": ("Pays ${v:.2f} per hour of actual usage", "raw"),
    "price_per_tier": ("Price of ${v:.2f} per plan tier", "raw"),
    "is_month_to_month": ("On a month-to-month contract", "flag"),
    "contract_type_Month-to-month": ("On a month-to-month contract", "flag"),
    "contract_type_One year": ("On a one-year contract", "flag"),
    "contract_type_Two year": ("On a two-year contract", "flag"),
    "late_payment_rate": ("Late on {pct} of payments", "ratio"),
    "late_rate_l3": ("Late on {pct} of recent payments", "ratio"),
    "late_payments_total": ("{v:.0f} late payments on record", "raw"),
    "discount_rate": ("Has used a discount in {pct} of months", "ratio"),
    "feature_usage_l3": ("Uses {v:.1f} product features recently", "raw"),
    "feature_usage_mean": ("Uses {v:.1f} product features on average", "raw"),
    "data_utilisation": ("Using {pct} of the plan's data allowance", "ratio"),
    "low_engagement_flag": ("Engagement in the bottom quartile", "flag"),
    "new_customer_flag": ("Still inside the first six months", "flag"),
    "high_touch_flag": ("Contacted support in the last quarter", "flag"),
    "avg_resolution_hours": ("Tickets take {v:.1f} hours to resolve on average", "raw"),
    "fcr_rate": ("{pct} of tickets resolved on first contact", "ratio"),
    "usage_volatility": ("Usage varies by {pct} month to month", "ratio"),
    "spend_trend_3m": ("Spend {dir} {pct} versus the prior quarter", "pct_change"),
    "age": ("Age {v:.0f}", "raw"),
    "tier_rank": ("On plan tier {v:.0f} of 6", "raw"),
    "total_revenue": ("${v:,.0f} of revenue contributed to date", "raw"),
}


def humanise(feature: str, value: float, shap_value: float) -> str:
    """Render one feature contribution as a sentence."""
    spec = PHRASES.get(feature)
    if spec is None:
        pretty = feature.replace("_", " ").capitalize()
        if "_" in feature and feature.split("_")[0] in {"customer", "income", "region", "gender", "plan"}:
            return f"{pretty.rsplit(' ', 1)[0]}: {feature.rsplit('_', 1)[-1]}"
        return f"{pretty}: {value:,.2f}"

    template, kind = spec
    if kind == "flag":
        return template if value >= 0.5 else f"Not: {template.lower()}"
    if kind == "pct_change":
        return template.format(dir="down" if value < 0 else "up", pct=f"{abs(value):.0%}")
    if kind == "ratio":
        return template.format(pct=f"{value:.0%}")
    if kind == "delta":
        return template.format(dir="up" if value > 0 else "down", v_abs=abs(value))
    return template.format(v=value)


def transform(model, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    prep = model.named_steps["prep"]
    Xt = prep.transform(X)
    return Xt, list(prep.get_feature_names_out())


def compute_shap(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    Xt, names = transform(model, X)
    explainer = shap.TreeExplainer(model.named_steps["clf"])
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):          # older SHAP returns one array per class
        sv = sv[1]
    base = float(np.ravel(explainer.expected_value)[0])
    return sv, Xt, names, base


def top_drivers(sv: np.ndarray, Xt: np.ndarray, names: list[str], customer_ids: pd.Series,
                n_pos: int = TOP_POSITIVE, n_neg: int = TOP_NEGATIVE) -> pd.DataFrame:
    """Tidy table of the strongest positive and negative contributions per customer."""
    order = np.argsort(-sv, axis=1)
    rows = []
    ids = customer_ids.to_numpy()
    for i in range(sv.shape[0]):
        picks = list(order[i, :n_pos]) + list(order[i, -n_neg:])
        for rank, j in enumerate(picks):
            rows.append(
                (ids[i], names[j], float(Xt[i, j]), float(sv[i, j]),
                 "increases risk" if sv[i, j] > 0 else "reduces risk", rank)
            )
    out = pd.DataFrame(rows, columns=["customer_id", "feature", "feature_value", "shap_value",
                                      "direction", "rank"])
    out["explanation"] = [humanise(f, v, s) for f, v, s in
                          zip(out["feature"], out["feature_value"], out["shap_value"])]
    return out


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def save_global_figures(sv: np.ndarray, Xt: np.ndarray, names: list[str], sample: int = 4000) -> pd.DataFrame:
    idx = np.random.default_rng(0).choice(len(sv), size=min(sample, len(sv)), replace=False)

    plt.figure(figsize=(9, 7))
    shap.summary_plot(sv[idx], Xt[idx], feature_names=names, max_display=18, show=False)
    plt.title("Global churn drivers (SHAP)", fontsize=13, pad=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=140)
    plt.close()

    plt.figure(figsize=(9, 7))
    shap.summary_plot(sv[idx], Xt[idx], feature_names=names, plot_type="bar", max_display=18, show=False)
    plt.title("Mean absolute SHAP value by feature", fontsize=13, pad=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_importance.png", dpi=140)
    plt.close()

    importance = (
        pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    for feat in ["usage_trend_3m", "tickets_l6", "revenue_per_usage_hour", "tenure_months"]:
        if feat not in names:
            continue
        j = names.index(feat)
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(j, sv[idx], Xt[idx], feature_names=names, show=False,
                             interaction_index=None)
        plt.title(f"How {feat} moves churn risk", fontsize=12)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_dependence_{feat}.png", dpi=140)
        plt.close()

    return importance


def save_customer_waterfall(sv_row: np.ndarray, x_row: np.ndarray, names: list[str],
                            base: float, customer_id: str) -> None:
    expl = shap.Explanation(values=sv_row, base_values=base, data=x_row, feature_names=names)
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(expl, max_display=12, show=False)
    plt.title(f"Why {customer_id} is predicted to churn", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_waterfall_example.png", dpi=140, bbox_inches="tight")
    plt.close()


def use_headless_backend() -> None:
    """Switch to a file-only backend when running as a script.

    Deliberately not done at import time: these modules are also imported by the
    notebooks, where forcing Agg would silently stop every inline chart from rendering.
    """
    matplotlib.use("Agg")


def main() -> None:
    use_headless_backend()
    model = joblib.load(MODELS_DIR / "churn_model.pkl")
    meta = joblib.load(MODELS_DIR / "model_metadata.pkl")
    feature_cols = meta["feature_columns"]

    score = pd.read_parquet(PROCESSED_DIR / "features_score.parquet")
    print(f"Explaining {len(score):,} active customers ...", flush=True)

    sv, Xt, names, base = compute_shap(model, score[feature_cols])

    print("Saving global driver figures ...", flush=True)
    importance = save_global_figures(sv, Xt, names)
    importance.to_csv(REPORTS_DIR / "shap_feature_importance.csv", index=False)
    print(importance.head(12).to_string(index=False))

    print("Building per-customer driver table ...", flush=True)
    drivers = top_drivers(sv, Xt, names, score["customer_id"])
    drivers.to_parquet(PROCESSED_DIR / "customer_drivers.parquet", index=False)

    # Waterfall for the single highest-risk customer, used in the README and dashboard
    proba = model.predict_proba(score[feature_cols])[:, 1]
    i = int(np.argmax(proba))
    save_customer_waterfall(sv[i], Xt[i], names, base, score["customer_id"].iloc[i])
    print(f"Example explained: {score['customer_id'].iloc[i]} at {proba[i]:.1%} risk")

    np.save(MODELS_DIR / "shap_base_value.npy", np.array([base]))
    print("Done.")


if __name__ == "__main__":
    main()
