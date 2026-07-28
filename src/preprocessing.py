"""
Transform layer: clean the raw extracts and engineer a modelling table.

Two things happen here.

1. Cleaning. The raw CSVs carry the defects a production extract really has - duplicate
   customer rows, five spellings of "Credit card", impossible ages, sensor spikes in usage,
   refunds booked as negative payments, missing readings. Each is handled explicitly and
   counted, so the notebook can report exactly what was repaired.

2. Feature engineering with a time cutoff. Features are built only from months <= cutoff;
   the label looks at the following `horizon` months. Nothing that happens after the cutoff
   can leak into a feature. The same function produces:
       - the training table  (cutoff = month 21, label = churn in months 22-24)
       - the scoring table   (cutoff = month 24, label unknown - this is "today")
   which is exactly how the model would run in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    HORIZON_MONTHS,
    OBSERVATION_MONTHS,
    PROCESSED_DIR,
    RAW_DIR,
)

# US state -> census region, used to turn 50 high-cardinality states into a usable feature
_REGION = {
    "Northeast": ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont",
                  "New Jersey", "New York", "Pennsylvania"],
    "Midwest": ["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin", "Iowa", "Kansas", "Minnesota",
                "Missouri", "Nebraska", "North Dakota", "South Dakota"],
    "South": ["Delaware", "Florida", "Georgia", "Maryland", "North Carolina", "South Carolina", "Virginia",
              "District of Columbia", "West Virginia", "Alabama", "Kentucky", "Mississippi", "Tennessee",
              "Arkansas", "Louisiana", "Oklahoma", "Texas"],
    "West": ["Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Utah", "Wyoming",
             "Alaska", "California", "Hawaii", "Oregon", "Washington"],
}
STATE_TO_REGION = {s: r for r, states in _REGION.items() for s in states}

GENDER_MAP = {"m": "Male", "male": "Male", "f": "Female", "female": "Female", "other": "Other"}

PAYMENT_MAP = {
    "cc": "Credit card", "credit card": "Credit card",
    "bank transfer": "Bank transfer",
    "e-check": "Electronic check", "electronic check": "Electronic check",
    "mailed check": "Mailed check",
    "dd": "Direct debit", "direct debit": "Direct debit",
}


# --------------------------------------------------------------------------------------
# Extract
# --------------------------------------------------------------------------------------
def load_raw() -> dict[str, pd.DataFrame]:
    names = ["customers", "plans", "dim_date", "usage", "payments", "support", "status"]
    return {n: pd.read_csv(RAW_DIR / f"{n}.csv") for n in names}


# --------------------------------------------------------------------------------------
# Clean
# --------------------------------------------------------------------------------------
def clean_customers(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    out = df.copy()

    before = len(out)
    out = out.drop_duplicates(subset="customer_id", keep="first").reset_index(drop=True)
    report["duplicate_customer_rows_removed"] = before - len(out)

    out["gender"] = out["gender"].astype(str).str.strip().str.lower().map(GENDER_MAP).fillna("Unknown")
    out["city"] = out["city"].astype(str).str.strip().str.title()
    out["state"] = out["state"].astype(str).str.strip()
    out["region"] = out["state"].map(STATE_TO_REGION).fillna("Other")

    impossible = (out["age"] < 16) | (out["age"] > 100)
    report["impossible_ages_nulled"] = int(impossible.sum())
    out.loc[impossible, "age"] = np.nan
    report["age_missing_imputed"] = int(out["age"].isna().sum())
    out["age"] = out.groupby("customer_segment")["age"].transform(lambda s: s.fillna(s.median()))
    out["age"] = out["age"].round().astype(int)

    report["income_missing_filled"] = int(out["income_level"].isna().sum())
    out["income_level"] = out["income_level"].fillna("Unknown")
    out["occupation"] = out["occupation"].fillna("Unknown")

    out["signup_date"] = pd.to_datetime(out["signup_date"])
    return out


def clean_payments(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    out = df.copy()
    # Parquet round-trips can hand back float32; widen before imputing so median fills
    # cannot fail on a narrower dtype.
    out["payment_amount"] = out["payment_amount"].astype("float64")
    out["payment_method"] = (
        out["payment_method"].astype(str).str.strip().str.lower().map(PAYMENT_MAP).fillna("Unknown")
    )
    report["payment_method_variants_collapsed"] = int(df["payment_method"].nunique() - out["payment_method"].nunique())

    neg = out["payment_amount"] < 0
    report["negative_payments_corrected"] = int(neg.sum())
    out.loc[neg, "payment_amount"] = out.loc[neg, "payment_amount"].abs()
    out["is_refund"] = neg.astype(int)

    missing = out["payment_amount"].isna()
    report["payment_amount_imputed"] = int(missing.sum())
    out["payment_amount"] = out.groupby("customer_id")["payment_amount"].transform(lambda s: s.fillna(s.median()))
    out["payment_amount"] = out["payment_amount"].fillna(out["payment_amount"].median())
    return out


def clean_usage(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    out = df.copy()
    for col in ("usage_hours", "data_consumption", "satisfaction_score"):
        if col in out.columns:
            out[col] = out[col].astype("float64")

    missing = out["usage_hours"].isna()
    report["usage_hours_imputed"] = int(missing.sum())
    out["usage_hours"] = out.groupby("customer_id")["usage_hours"].transform(lambda s: s.fillna(s.median()))
    out["usage_hours"] = out["usage_hours"].fillna(out["usage_hours"].median())

    # Winsorise instrumentation spikes at the 99.9th percentile rather than deleting rows:
    # the customer-month still happened, only the magnitude is untrustworthy.
    cap = out["usage_hours"].quantile(0.999)
    spikes = out["usage_hours"] > cap
    report["usage_outliers_winsorised"] = int(spikes.sum())
    out.loc[spikes, "usage_hours"] = cap

    out["support_tickets"] = out["support_tickets"].clip(lower=0)
    return out


def clean_all(raw: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    report: dict = {}
    clean = {
        "customers": clean_customers(raw["customers"], report),
        "payments": clean_payments(raw["payments"], report),
        "usage": clean_usage(raw["usage"], report),
        "plans": raw["plans"].copy(),
        "dim_date": raw["dim_date"].copy(),
        "support": raw["support"].copy(),
        "status": raw["status"].copy(),
    }
    clean["status"]["churn_date"] = pd.to_datetime(clean["status"]["churn_date"])
    clean["status"]["cancellation_reason"] = clean["status"]["cancellation_reason"].fillna("Still active")
    report["rows_customers"] = len(clean["customers"])
    report["rows_usage"] = len(clean["usage"])
    report["rows_payments"] = len(clean["payments"])
    return clean, report


# --------------------------------------------------------------------------------------
# Monthly panel
# --------------------------------------------------------------------------------------
def build_panel(clean: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per customer-month: engagement + billing + ticket detail, with month_index."""
    date_map = clean["dim_date"][["date_id", "month_index", "date"]]
    panel = clean["usage"].merge(
        clean["payments"].drop(columns=["payment_method"]), on=["customer_id", "date_id"], how="outer"
    )

    tickets = (
        clean["support"]
        .assign(is_critical=lambda d: d["priority"].isin(["High", "Critical"]).astype(int))
        .groupby(["customer_id", "date_id"], as_index=False)
        .agg(
            avg_resolution_hours=("resolution_hours", "mean"),
            critical_tickets=("is_critical", "sum"),
            fcr_rate=("first_contact_resolved", "mean"),
        )
    )
    panel = panel.merge(tickets, on=["customer_id", "date_id"], how="left")
    panel = panel.merge(date_map, on="date_id", how="left").sort_values(["customer_id", "month_index"])
    return panel.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------------------
def _window_agg(panel: pd.DataFrame, lo: int, hi: int, suffix: str) -> pd.DataFrame:
    """Aggregate a [lo, hi] month window into per-customer columns."""
    w = panel[(panel["month_index"] >= lo) & (panel["month_index"] <= hi)]
    g = w.groupby("customer_id")
    out = g.agg(
        **{
            f"usage_hours_{suffix}": ("usage_hours", "mean"),
            f"logins_{suffix}": ("login_frequency", "mean"),
            f"tickets_{suffix}": ("support_tickets", "mean"),
            f"data_gb_{suffix}": ("data_consumption", "mean"),
            f"feature_usage_{suffix}": ("feature_usage", "mean"),
            f"satisfaction_{suffix}": ("satisfaction_score", "mean"),
            f"spend_{suffix}": ("payment_amount", "mean"),
            f"late_rate_{suffix}": ("late_payment", "mean"),
        }
    )
    return out


def build_features(
    clean: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    cutoff_index: int,
    horizon: int | None = HORIZON_MONTHS,
) -> pd.DataFrame:
    """
    Build one row per customer who is still active at the end of `cutoff_index`.

    Features use months [0, cutoff_index]. If `horizon` is given, a `churned` label marks
    customers who cancel in months (cutoff_index, cutoff_index + horizon].
    """
    customers = clean["customers"]
    plans = clean["plans"]
    status = clean["status"]
    dates = clean["dim_date"]

    cutoff_date = pd.to_datetime(dates.loc[dates["month_index"] == cutoff_index, "date"].iloc[0])
    cutoff_end = cutoff_date + pd.offsets.MonthEnd(0)

    # ---- population: active at the end of the cutoff month
    st = status.copy()
    st["churn_month_index"] = np.where(
        st["churn_flag"] == 1,
        (st["churn_date"].dt.year - dates["year"].iloc[0]) * 12 + (st["churn_date"].dt.month - dates["month"].iloc[0]),
        np.nan,
    )
    base = customers.merge(st, on="customer_id", how="left")
    active_at_cutoff = base["churn_flag"].eq(0) | base["churn_month_index"].gt(cutoff_index)
    base = base[active_at_cutoff & (base["signup_date"] <= cutoff_end)].copy()

    hist = panel[panel["month_index"] <= cutoff_index]
    hist = hist[hist["customer_id"].isin(base["customer_id"])]

    # ---- lifetime aggregates
    life = hist.groupby("customer_id").agg(
        months_active=("month_index", "nunique"),
        usage_hours_mean=("usage_hours", "mean"),
        usage_hours_std=("usage_hours", "std"),
        usage_hours_max=("usage_hours", "max"),
        logins_mean=("login_frequency", "mean"),
        data_gb_mean=("data_consumption", "mean"),
        feature_usage_mean=("feature_usage", "mean"),
        satisfaction_mean=("satisfaction_score", "mean"),
        satisfaction_min=("satisfaction_score", "min"),
        tickets_total=("support_tickets", "sum"),
        critical_tickets_total=("critical_tickets", "sum"),
        avg_resolution_hours=("avg_resolution_hours", "mean"),
        fcr_rate=("fcr_rate", "mean"),
        total_revenue=("payment_amount", "sum"),
        avg_monthly_spend=("payment_amount", "mean"),
        late_payments_total=("late_payment", "sum"),
        discount_months=("discount_used", "sum"),
    )

    # ---- recency windows
    last1 = _window_agg(hist, cutoff_index, cutoff_index, "l1")
    last3 = _window_agg(hist, cutoff_index - 2, cutoff_index, "l3")
    prev3 = _window_agg(hist, cutoff_index - 5, cutoff_index - 3, "p3")
    last6 = _window_agg(hist, cutoff_index - 5, cutoff_index, "l6")

    df = (
        base[["customer_id", "customer_name", "gender", "age", "city", "state", "region",
              "signup_date", "customer_segment", "occupation", "income_level", "plan_id",
              "churn_flag", "churn_date", "churn_month_index", "cancellation_reason"]]
        .merge(plans[["plan_id", "plan_name", "plan_tier", "tier_rank", "monthly_price",
                      "contract_type", "data_limit_gb"]], on="plan_id", how="left")
        .merge(life, on="customer_id", how="left")
        .merge(last1, on="customer_id", how="left")
        .merge(last3, on="customer_id", how="left")
        .merge(prev3, on="customer_id", how="left")
        .merge(last6, on="customer_id", how="left")
    )

    # ---- derived business features
    df["tenure_months"] = ((cutoff_end - df["signup_date"]).dt.days / 30.44).round().astype(int)
    df["tenure_months"] = df["tenure_months"].clip(lower=0)

    df["usage_trend_3m"] = (df["usage_hours_l3"] - df["usage_hours_p3"]) / (df["usage_hours_p3"] + 0.5)
    df["login_trend_3m"] = (df["logins_l3"] - df["logins_p3"]) / (df["logins_p3"] + 0.5)
    df["spend_trend_3m"] = (df["spend_l3"] - df["spend_p3"]) / (df["spend_p3"] + 0.5)
    df["ticket_trend_3m"] = df["tickets_l3"] - df["tickets_p3"]
    df["satisfaction_trend_3m"] = df["satisfaction_l3"] - df["satisfaction_p3"]

    df["usage_vs_lifetime"] = df["usage_hours_l3"] / (df["usage_hours_mean"] + 0.5)
    df["usage_volatility"] = df["usage_hours_std"] / (df["usage_hours_mean"] + 0.5)
    df["support_frequency"] = df["tickets_total"] / df["months_active"].clip(lower=1)
    df["late_payment_rate"] = df["late_payments_total"] / df["months_active"].clip(lower=1)
    df["discount_rate"] = df["discount_months"] / df["months_active"].clip(lower=1)
    df["feature_adoption"] = df["feature_usage_l3"] / df["feature_usage_mean"].clip(lower=0.1)
    df["revenue_per_usage_hour"] = df["avg_monthly_spend"] / (df["usage_hours_l3"] + 0.5)
    df["price_per_tier"] = df["monthly_price"] / df["tier_rank"]
    df["data_utilisation"] = df["data_gb_l3"] / df["data_limit_gb"].clip(lower=1)
    df["is_month_to_month"] = (df["contract_type"] == "Month-to-month").astype(int)
    df["low_engagement_flag"] = (df["usage_hours_l3"] < df["usage_hours_l3"].quantile(0.25)).astype(int)
    df["new_customer_flag"] = (df["tenure_months"] < 6).astype(int)
    df["high_touch_flag"] = (df["tickets_l3"] >= 1).astype(int)

    # ---- label
    if horizon is not None:
        df["churned"] = (
            df["churn_month_index"].notna()
            & df["churn_month_index"].le(cutoff_index + horizon)
        ).astype(int)
        # Reason is only known for customers who actually left inside the horizon
        df["horizon_reason"] = np.where(df["churned"] == 1, df["cancellation_reason"], "Still active")
    else:
        df["churned"] = np.nan
        df["horizon_reason"] = "Unknown"

    df["cutoff_index"] = cutoff_index
    df["cutoff_date"] = cutoff_end.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)


# Columns the model is allowed to see (identifiers and post-cutoff info are excluded)
FEATURE_COLUMNS_NUMERIC = [
    "age", "tenure_months", "months_active", "monthly_price", "tier_rank", "data_limit_gb",
    "usage_hours_mean", "usage_hours_std", "usage_hours_max", "usage_hours_l1", "usage_hours_l3",
    "usage_hours_l6", "logins_mean", "logins_l3", "data_gb_mean", "data_gb_l3",
    "feature_usage_mean", "feature_usage_l3", "satisfaction_mean", "satisfaction_min",
    "satisfaction_l3", "tickets_total", "tickets_l3", "tickets_l6", "critical_tickets_total",
    "avg_resolution_hours", "fcr_rate", "total_revenue", "avg_monthly_spend", "spend_l3",
    "late_payments_total", "late_rate_l3", "discount_months", "usage_trend_3m", "login_trend_3m",
    "spend_trend_3m", "ticket_trend_3m", "satisfaction_trend_3m", "usage_vs_lifetime",
    "usage_volatility", "support_frequency", "late_payment_rate", "discount_rate",
    "revenue_per_usage_hour", "price_per_tier", "data_utilisation",
    "is_month_to_month", "low_engagement_flag", "new_customer_flag", "high_touch_flag",
]
FEATURE_COLUMNS_CATEGORICAL = [
    "gender", "customer_segment", "income_level", "region", "contract_type", "plan_tier",
]
FEATURE_COLUMNS = FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL


def main() -> None:
    print("Loading raw extracts ...", flush=True)
    raw = load_raw()

    print("Cleaning ...", flush=True)
    clean, report = clean_all(raw)
    print("Cleaning report:")
    for k, v in report.items():
        print(f"  {k:<38} {v:,}")

    print("Building customer-month panel ...", flush=True)
    panel = build_panel(clean)

    print(f"Building training table (cutoff month {OBSERVATION_MONTHS}, horizon {HORIZON_MONTHS}m) ...", flush=True)
    train_tbl = build_features(clean, panel, cutoff_index=OBSERVATION_MONTHS - 1, horizon=HORIZON_MONTHS)

    print("Building scoring table (cutoff = latest month, label unknown) ...", flush=True)
    last_index = int(clean["dim_date"]["month_index"].max())
    score_tbl = build_features(clean, panel, cutoff_index=last_index, horizon=None)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in clean.items():
        df.to_parquet(PROCESSED_DIR / f"clean_{name}.parquet", index=False)
    panel.to_parquet(PROCESSED_DIR / "panel_customer_month.parquet", index=False)
    train_tbl.to_parquet(PROCESSED_DIR / "features_train.parquet", index=False)
    score_tbl.to_parquet(PROCESSED_DIR / "features_score.parquet", index=False)
    pd.Series(report).to_csv(PROCESSED_DIR / "cleaning_report.csv", header=["value"])

    print(
        f"\nTraining table : {len(train_tbl):,} customers | churn in horizon = "
        f"{train_tbl['churned'].mean():.2%}"
    )
    print(f"Scoring table  : {len(score_tbl):,} active customers")


if __name__ == "__main__":
    main()
