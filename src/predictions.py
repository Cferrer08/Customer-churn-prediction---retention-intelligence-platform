"""
Scoring and the retention decision layer.

Turns a probability into something a retention team can run on Monday morning:

    probability -> risk band -> revenue exposure -> a named play with an owner and a cost

CLV is modelled as a discounted, survival-weighted stream rather than the flat
"monthly revenue x tenure" shortcut, because a customer with a 60% chance of leaving this
quarter is not worth the same as a two-year contract on the same price point:

    monthly hazard h   = 1 - (1 - p_horizon) ** (1 / horizon_months)
    survival factor q  = (1 - h) / (1 + discount_rate)
    CLV                = monthly_margin * q / (1 - q)

Every recommendation carries an expected value, so the programme can be ranked by payoff
rather than by gut feel.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from config import (
    ASSUMED_SAVE_RATE,
    DISCOUNT_RATE_MONTHLY,
    GROSS_MARGIN,
    HORIZON_MONTHS,
    MODELS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    RETENTION_OFFER_COST,
    RISK_BANDS,
)

CONTACT_COST_MONTHS = 3
BASELINE_MONTHLY_CHURN = 0.02      # hazard a healthy, retained customer reverts to
CSM_COST_PER_CUSTOMER = 180.0      # loaded cost of a customer-success touch
CAMPAIGN_COST_PER_CUSTOMER = 12.0  # automated onboarding / adoption campaign


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def score_customers(score_tbl: pd.DataFrame, model, feature_cols: list[str]) -> pd.DataFrame:
    out = score_tbl.copy()
    out["churn_probability"] = model.predict_proba(out[feature_cols])[:, 1]
    bands, lows, highs = zip(*RISK_BANDS)
    out["risk_band"] = pd.cut(
        out["churn_probability"], bins=[lows[0], *highs], labels=list(bands), include_lowest=True
    ).astype(str)
    return out


def add_value_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    p = out["churn_probability"].clip(0, 0.999)

    out["monthly_hazard"] = 1 - (1 - p) ** (1 / HORIZON_MONTHS)
    q = (1 - out["monthly_hazard"]) / (1 + DISCOUNT_RATE_MONTHLY)
    q = q.clip(upper=0.999)

    out["expected_remaining_months"] = (1 / out["monthly_hazard"]).clip(upper=120).round(1)
    out["monthly_margin"] = out["avg_monthly_spend"] * GROSS_MARGIN
    out["clv"] = (out["monthly_margin"] * q / (1 - q)).round(2)
    out["historic_revenue"] = out["total_revenue"].round(2)

    # What the customer would be worth *if the save lands*. A customer who is about to
    # leave has a low CLV precisely because they are leaving, so using it to value a
    # retention offer would tell the business to ignore its most urgent accounts. The
    # counterfactual value assumes a rescued customer reverts to healthy-book behaviour.
    q_healthy = (1 - BASELINE_MONTHLY_CHURN) / (1 + DISCOUNT_RATE_MONTHLY)
    out["clv_if_retained"] = (out["monthly_margin"] * q_healthy / (1 - q_healthy)).round(2)

    # ARR exposure: annualised revenue weighted by the probability of losing it
    out["annual_revenue"] = out["avg_monthly_spend"] * 12
    out["arr_at_risk"] = (out["annual_revenue"] * p).round(2)
    return out


# --------------------------------------------------------------------------------------
# Retention playbook
# --------------------------------------------------------------------------------------
PLAYS = {
    "price_relief": ("Offer a 20% loyalty discount or right-size the plan", "Retention desk"),
    "service_recovery": ("Assign a customer success manager and run a support review", "Customer Success"),
    "reengagement": ("Trigger a re-onboarding campaign with a guided feature walkthrough", "Lifecycle Marketing"),
    "contract_lock": ("Offer an annual contract with two months free", "Retention desk"),
    "billing_fix": ("Move to auto-pay with a one-off billing credit", "Billing operations"),
    "adoption": ("Send a targeted feature-adoption sequence", "Lifecycle Marketing"),
    "check_in": ("Add to the quarterly satisfaction check-in list", "Account management"),
    "monitor": ("No action - keep under standard monitoring", "None"),
    "upsell": ("Flag as an expansion candidate for the next review", "Sales"),
}


def recommend(row: pd.Series) -> tuple[str, str, str, float]:
    """
    Rule-based play selection. Rules are ordered by expected impact and read the same
    features SHAP surfaces as drivers, so the recommendation always matches the stated
    reason for the risk.
    """
    p = row["churn_probability"]
    high, medium = p >= 0.70, 0.30 <= p < 0.70

    value_gap = row["revenue_per_usage_hour"]
    heavy_support = row["tickets_l3"] >= 1.0 or row["critical_tickets_total"] >= 2
    dropping = row["usage_trend_3m"] <= -0.20
    unhappy = row["satisfaction_l3"] < 3.3
    late = row["late_payment_rate"] >= 0.25
    m2m = row["is_month_to_month"] == 1
    low_adoption = row["feature_usage_l3"] <= 2

    if high and (value_gap >= 8 or (row["monthly_price"] >= 90 and dropping)):
        key = "price_relief"
    elif high and (heavy_support or unhappy):
        key = "service_recovery"
    elif high and (dropping or low_adoption):
        key = "reengagement"
    elif high and m2m:
        key = "contract_lock"
    elif high and late:
        key = "billing_fix"
    elif high:
        key = "service_recovery"
    elif medium and late:
        key = "billing_fix"
    elif medium and (dropping or low_adoption):
        key = "adoption"
    elif medium and m2m:
        key = "contract_lock"
    elif medium:
        key = "check_in"
    elif row["clv"] >= 3000 and row["data_utilisation"] >= 0.8:
        key = "upsell"
    else:
        key = "monitor"

    action, owner = PLAYS[key]
    cost = {
        "price_relief": row["avg_monthly_spend"] * RETENTION_OFFER_COST * CONTACT_COST_MONTHS,
        "service_recovery": CSM_COST_PER_CUSTOMER,
        "reengagement": CAMPAIGN_COST_PER_CUSTOMER * 2,
        "contract_lock": row["avg_monthly_spend"] * 2 * 0.5,
        "billing_fix": 25.0,
        "adoption": CAMPAIGN_COST_PER_CUSTOMER,
        "check_in": 20.0,
        "monitor": 0.0,
        "upsell": 0.0,
    }[key]
    return key, action, owner, round(float(cost), 2)


def apply_playbook(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    picks = out.apply(recommend, axis=1, result_type="expand")
    picks.columns = ["play", "recommended_action", "action_owner", "action_cost"]
    out = pd.concat([out, picks], axis=1)

    # Expected value of acting: margin preserved if the save lands, minus the cost of trying
    # Only customers we actually intervene on can be credited with a save.
    intervened = ~out["play"].isin(["monitor", "upsell"])
    out["expected_saved_margin"] = (
        out["churn_probability"] * ASSUMED_SAVE_RATE * out["clv_if_retained"] * intervened
    ).round(2)
    out["action_expected_value"] = (out["expected_saved_margin"] - out["action_cost"]).round(2)
    out["action_priority"] = out["action_expected_value"].rank(ascending=False, method="first").astype(int)
    return out


def attach_reasons(df: pd.DataFrame, drivers: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Join the top SHAP explanations onto each customer as a single readable string."""
    top = (
        drivers[drivers["direction"] == "increases risk"]
        .sort_values(["customer_id", "shap_value"], ascending=[True, False])
        .groupby("customer_id")
        .head(n)
        .groupby("customer_id")["explanation"]
        .apply(lambda s: " | ".join(s))
        .rename("top_risk_reasons")
    )
    prot = (
        drivers[drivers["direction"] == "reduces risk"]
        .sort_values(["customer_id", "shap_value"])
        .groupby("customer_id")
        .head(2)
        .groupby("customer_id")["explanation"]
        .apply(lambda s: " | ".join(s))
        .rename("top_protective_factors")
    )
    return df.merge(top, on="customer_id", how="left").merge(prot, on="customer_id", how="left")


# --------------------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------------------
def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("risk_band").agg(
        customers=("customer_id", "count"),
        avg_probability=("churn_probability", "mean"),
        avg_monthly_spend=("avg_monthly_spend", "mean"),
        annual_revenue=("annual_revenue", "sum"),
        arr_at_risk=("arr_at_risk", "sum"),
        total_clv=("clv", "sum"),
        avg_tenure_months=("tenure_months", "mean"),
        month_to_month_share=("is_month_to_month", "mean"),
    )
    order = [b[0] for b in RISK_BANDS]
    g = g.reindex([b for b in order if b in g.index])
    g["share_of_customers"] = (g["customers"] / g["customers"].sum()).round(4)
    return g.round(2)


def business_impact(df: pd.DataFrame) -> dict:
    high = df[df["risk_band"] == "High"]
    med = df[df["risk_band"] == "Medium"]
    targeted = df[df["churn_probability"] >= 0.30]

    expected_losses = float((df["churn_probability"] * df["annual_revenue"]).sum())
    programme_cost = float(targeted["action_cost"].sum())
    expected_saved = float((targeted["churn_probability"] * ASSUMED_SAVE_RATE * targeted["annual_revenue"]).sum())

    return {
        "active_customers": int(len(df)),
        "high_risk_customers": int(len(high)),
        "medium_risk_customers": int(len(med)),
        "annual_recurring_revenue": round(float(df["annual_revenue"].sum()), 2),
        "arr_at_risk_total": round(float(df["arr_at_risk"].sum()), 2),
        "arr_at_risk_high_band": round(float(high["arr_at_risk"].sum()), 2),
        "high_band_annual_revenue": round(float(high["annual_revenue"].sum()), 2),
        "avg_monthly_spend": round(float(df["avg_monthly_spend"].mean()), 2),
        "avg_expected_remaining_months": round(float(df["expected_remaining_months"].mean()), 1),
        "total_clv": round(float(df["clv"].sum()), 2),
        "customers_targeted": int(len(targeted)),
        "programme_cost": round(programme_cost, 2),
        "expected_revenue_saved": round(expected_saved, 2),
        "net_benefit": round(expected_saved - programme_cost, 2),
        "roi_multiple": round(expected_saved / programme_cost, 2) if programme_cost else None,
        "expected_annual_revenue_lost_if_no_action": round(expected_losses, 2),
        "assumptions": {
            "gross_margin": GROSS_MARGIN,
            "save_rate": ASSUMED_SAVE_RATE,
            "horizon_months": HORIZON_MONTHS,
            "monthly_discount_rate": DISCOUNT_RATE_MONTHLY,
        },
    }


def main() -> None:
    model = joblib.load(MODELS_DIR / "churn_model.pkl")
    meta = joblib.load(MODELS_DIR / "model_metadata.pkl")

    score_tbl = pd.read_parquet(PROCESSED_DIR / "features_score.parquet")
    print(f"Scoring {len(score_tbl):,} active customers with {meta['model_name']} ...", flush=True)

    scored = score_customers(score_tbl, model, meta["feature_columns"])
    scored = add_value_metrics(scored)
    scored = apply_playbook(scored)

    drivers_path = PROCESSED_DIR / "customer_drivers.parquet"
    if drivers_path.exists():
        scored = attach_reasons(scored, pd.read_parquet(drivers_path))

    keep = [
        "customer_id", "customer_name", "customer_segment", "region", "state", "city", "age",
        "gender", "income_level", "plan_id", "plan_name", "plan_tier", "contract_type",
        "monthly_price", "tenure_months", "months_active", "avg_monthly_spend", "annual_revenue",
        "total_revenue", "usage_hours_l3", "usage_trend_3m", "tickets_l3", "tickets_total",
        "satisfaction_l3", "late_payment_rate", "feature_usage_l3", "data_utilisation",
        "revenue_per_usage_hour", "is_month_to_month", "churn_probability", "risk_band",
        "monthly_hazard", "expected_remaining_months", "clv", "clv_if_retained", "arr_at_risk",
        "top_risk_reasons", "top_protective_factors", "play", "recommended_action",
        "action_owner", "action_cost", "expected_saved_margin", "action_expected_value",
        "action_priority",
    ]
    keep = [c for c in keep if c in scored.columns]
    scored_out = scored[keep].sort_values("churn_probability", ascending=False)

    scored_out.to_parquet(PROCESSED_DIR / "scored_customers.parquet", index=False)
    scored_out.to_csv(PROCESSED_DIR / "scored_customers.csv", index=False)

    seg = segment_summary(scored)
    seg.to_csv(REPORTS_DIR / "risk_segment_summary.csv")

    plays = (
        scored.groupby(["risk_band", "recommended_action", "action_owner"])
        .agg(customers=("customer_id", "count"),
             arr_at_risk=("arr_at_risk", "sum"),
             programme_cost=("action_cost", "sum"),
             expected_value=("action_expected_value", "sum"))
        .round(2)
        .sort_values("expected_value", ascending=False)
    )
    plays.to_csv(REPORTS_DIR / "retention_playbook_summary.csv")

    impact = business_impact(scored)
    with open(REPORTS_DIR / "business_impact.json", "w") as f:
        json.dump(impact, f, indent=2)

    print("\nRisk segments:")
    print(seg.to_string())
    print("\nHeadline numbers:")
    print(f"  Active customers        {impact['active_customers']:,}")
    print(f"  High-risk customers     {impact['high_risk_customers']:,}")
    print(f"  ARR in the high band    ${impact['high_band_annual_revenue']:,.0f}")
    print(f"  Probability-weighted ARR at risk  ${impact['arr_at_risk_total']:,.0f}")
    print(f"  Programme cost          ${impact['programme_cost']:,.0f}")
    print(f"  Expected revenue saved  ${impact['expected_revenue_saved']:,.0f}")
    print(f"  Net benefit             ${impact['net_benefit']:,.0f}  (ROI {impact['roi_multiple']}x)")


if __name__ == "__main__":
    main()
