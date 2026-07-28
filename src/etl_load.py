"""
Load step of the ETL: push the cleaned tables into the analytical warehouse.

The project targets PostgreSQL (database/schema.sql is the real DDL), but it defaults to a
local SQLite file so anyone cloning the repo can run the whole pipeline and execute the
SQL layer without standing up a server:

    python src/etl_load.py                                  # -> database/churn_warehouse.db
    CHURN_DB_URL=postgresql+psycopg2://user:pw@host/db \
        python src/etl_load.py                              # -> PostgreSQL

Table names match schema.sql exactly, so the same queries run against either backend.
"""

from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import create_engine, text

from config import DEFAULT_DB_URL, PROCESSED_DIR, REPORTS_DIR

TABLE_MAP = {
    "dim_plan": "clean_plans.parquet",
    "dim_date": "clean_dim_date.parquet",
    "dim_customer": "clean_customers.parquet",
    "fact_customer_usage": "clean_usage.parquet",
    "fact_payments": "clean_payments.parquet",
    "fact_support_tickets": "clean_support.parquet",
    "fact_customer_status": "clean_status.parquet",
}

COLUMNS = {
    "dim_plan": ["plan_id", "plan_name", "plan_tier", "tier_rank", "monthly_price",
                 "contract_type", "data_limit", "data_limit_gb", "features"],
    "dim_date": ["date_id", "date", "month", "month_name", "quarter", "year", "month_index"],
    "dim_customer": ["customer_id", "customer_name", "gender", "age", "location", "city", "state",
                     "region", "signup_date", "customer_segment", "occupation", "income_level", "plan_id"],
    "fact_customer_usage": ["customer_id", "date_id", "login_frequency", "support_tickets",
                            "usage_hours", "data_consumption", "feature_usage", "satisfaction_score"],
    "fact_payments": ["customer_id", "date_id", "payment_amount", "payment_method",
                      "late_payment", "discount_used", "is_refund"],
    "fact_support_tickets": ["ticket_id", "customer_id", "date_id", "category", "priority",
                             "resolution_hours", "first_contact_resolved"],
    "fact_customer_status": ["customer_id", "churn_flag", "churn_date", "cancellation_reason"],
}

SCORE_COLUMNS = ["customer_id", "churn_probability", "risk_band", "expected_remaining_months",
                 "clv", "arr_at_risk", "recommended_action", "action_owner", "action_expected_value"]


def get_engine(url: str | None = None):
    return create_engine(url or os.environ.get("CHURN_DB_URL", DEFAULT_DB_URL))


def load_all(engine, chunksize: int = 50_000) -> dict[str, int]:
    counts = {}
    for table, filename in TABLE_MAP.items():
        df = pd.read_parquet(PROCESSED_DIR / filename)
        if table == "dim_customer":
            df["location"] = df["city"] + ", " + df["state"]
        df = df[[c for c in COLUMNS[table] if c in df.columns]]
        for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
            df[col] = df[col].dt.strftime("%Y-%m-%d")
        df.to_sql(table, engine, if_exists="replace", index=False, chunksize=chunksize)
        counts[table] = len(df)
        print(f"  loaded {table:<22} {len(df):>9,} rows", flush=True)

    scores_path = PROCESSED_DIR / "scored_customers.parquet"
    if scores_path.exists():
        scores = pd.read_parquet(scores_path)
        scores = scores[[c for c in SCORE_COLUMNS if c in scores.columns]].copy()
        scores["scoring_date"] = pd.Timestamp.today().strftime("%Y-%m-%d")
        scores.to_sql("fact_churn_scores", engine, if_exists="replace", index=False, chunksize=chunksize)
        counts["fact_churn_scores"] = len(scores)
        print(f"  loaded {'fact_churn_scores':<22} {len(scores):>9,} rows", flush=True)
    return counts


# SQLite-compatible smoke tests: prove the warehouse answers real questions after loading.
VALIDATION_QUERIES = {
    "churn_by_contract": """
        SELECT p.contract_type,
               COUNT(*) AS customers,
               ROUND(AVG(CAST(s.churn_flag AS REAL)), 4) AS churn_rate,
               ROUND(SUM(s.churn_flag * p.monthly_price) * 12, 2) AS annual_revenue_lost
        FROM dim_customer c
        JOIN dim_plan p ON p.plan_id = c.plan_id
        JOIN fact_customer_status s ON s.customer_id = c.customer_id
        GROUP BY p.contract_type
        ORDER BY churn_rate DESC
    """,
    "top_churn_reasons_by_revenue": """
        SELECT s.cancellation_reason,
               COUNT(*) AS customers,
               ROUND(SUM(p.monthly_price) * 12, 2) AS annual_revenue_lost
        FROM fact_customer_status s
        JOIN dim_customer c ON c.customer_id = s.customer_id
        JOIN dim_plan p ON p.plan_id = c.plan_id
        WHERE s.churn_flag = 1
        GROUP BY s.cancellation_reason
        ORDER BY annual_revenue_lost DESC
    """,
    "usage_quintile_vs_churn": """
        WITH per_customer AS (
            SELECT customer_id, AVG(usage_hours) AS avg_usage
            FROM fact_customer_usage GROUP BY customer_id
        ),
        banded AS (
            SELECT u.customer_id,
                   NTILE(5) OVER (ORDER BY u.avg_usage) AS usage_quintile,
                   s.churn_flag
            FROM per_customer u
            JOIN fact_customer_status s ON s.customer_id = u.customer_id
        )
        SELECT usage_quintile, COUNT(*) AS customers,
               ROUND(AVG(CAST(churn_flag AS REAL)), 4) AS churn_rate
        FROM banded GROUP BY usage_quintile ORDER BY usage_quintile
    """,
    "revenue_at_risk_by_band": """
        SELECT risk_band, COUNT(*) AS customers,
               ROUND(SUM(arr_at_risk), 2) AS arr_at_risk,
               ROUND(SUM(action_expected_value), 2) AS expected_value_of_action
        FROM fact_churn_scores
        GROUP BY risk_band
    """,
}


def validate(engine) -> dict[str, pd.DataFrame]:
    results = {}
    for name, sql in VALIDATION_QUERIES.items():
        try:
            results[name] = pd.read_sql(text(sql), engine)
        except Exception as exc:                      # pragma: no cover - backend differences
            print(f"  ! {name} failed: {exc}")
    return results


def main() -> None:
    engine = get_engine()
    print(f"Loading warehouse at {engine.url} ...")
    load_all(engine)

    print("\nValidating with analytical queries:")
    results = validate(engine)
    for name, df in results.items():
        print(f"\n-- {name}")
        print(df.to_string(index=False))
        df.to_csv(REPORTS_DIR / f"sql_{name}.csv", index=False)


if __name__ == "__main__":
    main()
