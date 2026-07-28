"""
Tests for the churn pipeline.

These are not decorative. Each one guards a property that, if it broke silently, would
invalidate the results rather than merely crash the code:

    * the generator produces internally consistent history
    * cleaning actually removes the defects it claims to
    * no feature can see past the cutoff (the leakage guard)
    * CLV and the retention economics stay directionally sane
    * the trained model still beats a random ranker on held-out data

    pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import data_generator as gen           # noqa: E402
from preprocessing import (            # noqa: E402
    FEATURE_COLUMNS,
    build_features,
    build_panel,
    clean_all,
)

PROCESSED = ROOT / "data" / "processed"


# --------------------------------------------------------------------------------------
# Fixtures: one small simulated world, reused across tests
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def raw() -> dict[str, pd.DataFrame]:
    return gen.generate(n_customers=1200, seed=7, write=False, verbose=False)


@pytest.fixture(scope="module")
def cleaned(raw):
    clean, report = clean_all({k: v.copy() for k, v in raw.items()})
    return clean, report


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------
def test_every_customer_has_a_plan(raw):
    assert set(raw["customers"]["plan_id"]).issubset(set(raw["plans"]["plan_id"]))


def test_churn_rate_is_plausible(raw):
    rate = raw["status"]["churn_flag"].mean()
    assert 0.15 < rate < 0.45, f"churn rate {rate:.2%} is outside a believable range"


def test_churned_customers_have_a_reason_and_a_date(raw):
    churned = raw["status"][raw["status"]["churn_flag"] == 1]
    assert churned["churn_date"].ne("").all()
    assert churned["cancellation_reason"].ne("").all()


def test_active_customers_have_no_churn_date(raw):
    active = raw["status"][raw["status"]["churn_flag"] == 0]
    assert active["churn_date"].eq("").all()


def test_no_usage_recorded_after_cancellation(raw):
    """The most important consistency check: a cancelled customer stops generating rows."""
    status = raw["status"]
    churned = status[status["churn_flag"] == 1].copy()
    churned["churn_month"] = pd.to_datetime(churned["churn_date"]).dt.to_period("M")

    usage = raw["usage"].merge(churned[["customer_id", "churn_month"]], on="customer_id")
    usage["usage_month"] = pd.to_datetime(usage["date_id"], format="%Y%m%d").dt.to_period("M")
    assert (usage["usage_month"] <= usage["churn_month"]).all()


def test_usage_and_payments_cover_the_same_customer_months(raw):
    u = set(map(tuple, raw["usage"][["customer_id", "date_id"]].to_numpy()))
    p = set(map(tuple, raw["payments"][["customer_id", "date_id"]].to_numpy()))
    assert u == p


def test_disengagement_precedes_cancellation(raw):
    """Churners should show materially lower usage in their final months than survivors."""
    status, usage = raw["status"], raw["usage"]
    last = usage.sort_values("date_id").groupby("customer_id").tail(3)
    mean_usage = last.groupby("customer_id")["usage_hours"].mean()
    joined = status.set_index("customer_id").join(mean_usage.rename("recent_usage")).dropna()
    churned = joined.loc[joined["churn_flag"] == 1, "recent_usage"].mean()
    stayed = joined.loc[joined["churn_flag"] == 0, "recent_usage"].mean()
    assert churned < stayed * 0.8, f"churners {churned:.1f}h vs survivors {stayed:.1f}h"


# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------
def test_duplicates_are_removed(cleaned):
    clean, report = cleaned
    assert clean["customers"]["customer_id"].is_unique
    assert report["duplicate_customer_rows_removed"] > 0


def test_categories_are_standardised(cleaned):
    clean, _ = cleaned
    assert set(clean["customers"]["gender"]) <= {"Male", "Female", "Other", "Unknown"}
    assert "CC" not in set(clean["payments"]["payment_method"])
    assert set(clean["payments"]["payment_method"]) <= {
        "Credit card", "Bank transfer", "Electronic check", "Mailed check", "Direct debit", "Unknown"
    }


def test_impossible_values_are_gone(cleaned):
    clean, _ = cleaned
    ages = clean["customers"]["age"]
    assert ages.between(16, 100).all()
    assert clean["customers"]["age"].isna().sum() == 0
    assert (clean["payments"]["payment_amount"] >= 0).all()
    assert clean["usage"]["usage_hours"].isna().sum() == 0


def test_cleaning_preserves_row_counts_for_facts(cleaned, raw):
    clean, _ = cleaned
    assert len(clean["usage"]) == len(raw["usage"])
    assert len(clean["payments"]) == len(raw["payments"])


# --------------------------------------------------------------------------------------
# Feature engineering: the leakage guard
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def features(cleaned):
    clean, _ = cleaned
    panel = build_panel(clean)
    return clean, panel, build_features(clean, panel, cutoff_index=20, horizon=3)


def test_population_excludes_already_churned_customers(features):
    _, _, feats = features
    already_gone = feats["churn_month_index"].notna() & feats["churn_month_index"].le(20)
    assert not already_gone.any(), "customers who left before the cutoff must not be scored"


def test_label_only_covers_the_horizon(features):
    _, _, feats = features
    labelled = feats[feats["churned"] == 1]
    assert labelled["churn_month_index"].between(21, 23).all()


def test_features_use_no_post_cutoff_data(features):
    """
    Rebuild the features from a panel truncated at the cutoff. If any feature secretly read
    a later month, the two tables would differ.
    """
    clean, panel, feats = features
    truncated = panel[panel["month_index"] <= 20]
    rebuilt = build_features(clean, truncated, cutoff_index=20, horizon=3)

    numeric = [c for c in FEATURE_COLUMNS if pd.api.types.is_numeric_dtype(feats.get(c))]
    a = feats.set_index("customer_id")[numeric].sort_index()
    b = rebuilt.set_index("customer_id")[numeric].sort_index()
    pd.testing.assert_frame_equal(a, b, check_dtype=False, rtol=1e-9)


def test_no_identifier_columns_leak_into_the_feature_list():
    banned = {"churn_flag", "churn_date", "churn_month_index", "cancellation_reason",
              "customer_id", "customer_name", "churned"}
    assert banned.isdisjoint(FEATURE_COLUMNS)


def test_scoring_table_has_no_label(features):
    clean, panel, _ = features
    last = int(clean["dim_date"]["month_index"].max())
    score = build_features(clean, panel, cutoff_index=last, horizon=None)
    assert score["churned"].isna().all()
    assert (score["churn_flag"] == 0).all()


# --------------------------------------------------------------------------------------
# Business layer
# --------------------------------------------------------------------------------------
def test_risk_bands_partition_the_probability_range():
    from predictions import RISK_BANDS
    lows = [b[1] for b in RISK_BANDS]
    highs = [b[2] for b in RISK_BANDS]
    assert lows[0] == 0.0
    for i in range(len(RISK_BANDS) - 1):
        assert highs[i] == lows[i + 1], "risk bands must be contiguous with no gaps"


def test_clv_falls_as_churn_risk_rises():
    from predictions import add_value_metrics
    df = pd.DataFrame({
        "churn_probability": [0.05, 0.50, 0.95],
        "avg_monthly_spend": [100.0, 100.0, 100.0],
        "total_revenue": [1200.0, 1200.0, 1200.0],
    })
    out = add_value_metrics(df)
    assert out["clv"].is_monotonic_decreasing
    assert out["arr_at_risk"].is_monotonic_increasing
    # counterfactual value does not depend on current risk - that is the point of it
    assert out["clv_if_retained"].nunique() == 1


def test_retention_value_rises_with_risk():
    """A save on a high-risk customer must be worth more than on a low-risk one."""
    scored = PROCESSED / "scored_customers.parquet"
    if not scored.exists():
        pytest.skip("run the pipeline first")
    df = pd.read_parquet(scored)
    means = df.groupby("risk_band")["action_expected_value"].mean()
    assert means.get("High", 0) > means.get("Medium", 0) > means.get("Low", -1)


def test_every_high_risk_customer_has_an_action():
    scored = PROCESSED / "scored_customers.parquet"
    if not scored.exists():
        pytest.skip("run the pipeline first")
    df = pd.read_parquet(scored)
    high = df[df["risk_band"] == "High"]
    assert high["recommended_action"].notna().all()
    assert (high["play"] != "monitor").all(), "a high-risk customer should never be left alone"


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def test_saved_model_beats_random_on_held_out_data():
    from sklearn.metrics import roc_auc_score
    path = ROOT / "models" / "test_predictions.npy"
    if not path.exists():
        pytest.skip("train the model first")
    y, p = np.load(path)
    assert roc_auc_score(y.astype(int), p) > 0.75


def test_scores_are_valid_probabilities():
    scored = PROCESSED / "scored_customers.parquet"
    if not scored.exists():
        pytest.skip("run the pipeline first")
    p = pd.read_parquet(scored)["churn_probability"]
    assert p.between(0, 1).all()
    assert p.notna().all()
