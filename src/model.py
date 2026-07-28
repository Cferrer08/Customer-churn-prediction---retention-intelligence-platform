"""
Model training and evaluation.

Design decisions worth defending in an interview:

* Time-aware split. Features stop at the cutoff month, the label looks forward. Within
  that frame the split is stratified random, because every row is already anchored to the
  same point in time - a further chronological split would leave no positives.
* Imbalance is handled with class weights, not resampling. SMOTE on a 3.7% positive rate
  tends to inflate validation scores without improving the ranking that actually matters.
* The decision threshold is a business choice, not 0.5. It is chosen on the validation
  set by maximising expected retained margin, and reported alongside a recall-first
  alternative, because a missed churner costs far more than a wasted retention call.
* Everything is selected on validation; the test set is touched once, at the end.
"""

from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from config import (
    ASSUMED_SAVE_RATE,
    GROSS_MARGIN,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    RETENTION_OFFER_COST,
    TEST_SIZE,
    VALID_SIZE,
)
from preprocessing import FEATURE_COLUMNS_CATEGORICAL, FEATURE_COLUMNS_NUMERIC

warnings.filterwarnings("ignore")

CONTACT_COST_MONTHS = 3          # a save offer is honoured for three months
EXPECTED_LIFETIME_MONTHS = 18    # horizon used to value a retained customer


# --------------------------------------------------------------------------------------
# Pipeline plumbing
# --------------------------------------------------------------------------------------
def build_preprocessor(scale: bool = False) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), FEATURE_COLUMNS_NUMERIC),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary")),
                    ]
                ),
                FEATURE_COLUMNS_CATEGORICAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_models(pos_weight: float) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("prep", build_preprocessor(scale=True)),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=12,
                        min_samples_leaf=25,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=250, learning_rate=0.05, max_depth=3,
                        subsample=0.85, random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=500,
                        learning_rate=0.045,
                        max_depth=4,
                        min_child_weight=8,
                        subsample=0.85,
                        colsample_bytree=0.8,
                        reg_lambda=2.0,
                        gamma=0.2,
                        scale_pos_weight=pos_weight,
                        eval_metric="aucpr",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def lift_at_k(y_true, proba, k: float = 0.10) -> float:
    """Share of all churners captured in the top-k% highest-risk customers."""
    n = max(int(len(proba) * k), 1)
    top = np.argsort(-proba)[:n]
    return float(np.asarray(y_true)[top].sum() / max(np.asarray(y_true).sum(), 1))


def business_threshold(y_true, proba, monthly_revenue: np.ndarray) -> tuple[float, pd.DataFrame]:
    """
    Pick the operating threshold that maximises expected retained margin.

    Value of flagging a customer who really would churn:
        monthly_revenue * GROSS_MARGIN * EXPECTED_LIFETIME_MONTHS * ASSUMED_SAVE_RATE
    Cost of flagging anyone (churner or not):
        monthly_revenue * RETENTION_OFFER_COST * CONTACT_COST_MONTHS
    """
    y_true = np.asarray(y_true)
    save_value = monthly_revenue * GROSS_MARGIN * EXPECTED_LIFETIME_MONTHS * ASSUMED_SAVE_RATE
    offer_cost = monthly_revenue * RETENTION_OFFER_COST * CONTACT_COST_MONTHS

    rows = []
    for thr in np.round(np.arange(0.02, 0.81, 0.01), 3):
        flagged = proba >= thr
        value = float((save_value * flagged * y_true).sum() - (offer_cost * flagged).sum())
        rows.append(
            {
                "threshold": thr,
                "flagged": int(flagged.sum()),
                "recall": recall_score(y_true, flagged, zero_division=0),
                "precision": precision_score(y_true, flagged, zero_division=0),
                "expected_value": value,
            }
        )
    curve = pd.DataFrame(rows)
    best = float(curve.loc[curve["expected_value"].idxmax(), "threshold"])
    return best, curve


def recall_first_threshold(y_true, proba, target_recall: float = 0.70) -> float:
    """Cheapest threshold (highest precision) that still reaches the target recall."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    ok = np.where(rec[:-1] >= target_recall)[0]
    return float(thr[ok[-1]]) if len(ok) else 0.05


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "features_train.parquet")
    feature_cols = FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL
    X, y = df[feature_cols], df["churned"].astype(int)

    X_tmp, X_test, y_tmp, y_test, rev_tmp, rev_test = train_test_split(
        X, y, df["avg_monthly_spend"].to_numpy(),
        test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val, _, rev_val = train_test_split(
        X_tmp, y_tmp, rev_tmp, test_size=VALID_SIZE, stratify=y_tmp, random_state=RANDOM_STATE
    )
    print(f"train={len(X_train):,}  valid={len(X_val):,}  test={len(X_test):,}  "
          f"positive rate={y.mean():.2%}")

    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    models = get_models(pos_weight)

    rows, fitted, val_probas = [], {}, {}
    for name, pipe in models.items():
        print(f"  training {name} ...", flush=True)
        if name == "Gradient Boosting":
            sw = np.where(y_train == 1, pos_weight, 1.0)
            pipe.fit(X_train, y_train, clf__sample_weight=sw)
        else:
            pipe.fit(X_train, y_train)
        p_val = pipe.predict_proba(X_val)[:, 1]
        fitted[name], val_probas[name] = pipe, p_val
        m = evaluate(y_val, p_val, threshold=0.5)
        m["model"] = name
        m["lift_top10pct"] = lift_at_k(y_val, p_val, 0.10)
        rows.append(m)

    comparison = pd.DataFrame(rows).set_index("model")[
        ["roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy", "brier", "lift_top10pct"]
    ].sort_values("pr_auc", ascending=False)
    print("\nValidation comparison (threshold 0.5):")
    print(comparison.round(4).to_string())

    best_name = comparison.index[0]
    best_model = fitted[best_name]
    print(f"\nBest model by PR-AUC: {best_name}")

    # 5-fold CV on the training portion, for a stability statement in the README
    cv = cross_val_score(
        get_models(pos_weight)[best_name], X_tmp, y_tmp,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc", n_jobs=1,
    )
    print(f"5-fold CV ROC-AUC: {cv.mean():.4f} +/- {cv.std():.4f}")

    # ---- thresholds chosen on validation, applied once to test
    p_val = val_probas[best_name]
    thr_business, ev_curve = business_threshold(y_val, p_val, rev_val)
    thr_recall = recall_first_threshold(y_val, p_val, target_recall=0.85)
    print(f"Business-optimal threshold: {thr_business:.2f} | recall-first (85%) threshold: {thr_recall:.3f}")

    p_test = best_model.predict_proba(X_test)[:, 1]
    test_metrics = {
        "default_0.5": evaluate(y_test, p_test, 0.5),
        "business_optimal": evaluate(y_test, p_test, thr_business),
        "recall_first": evaluate(y_test, p_test, thr_recall),
    }
    test_metrics["lift_top10pct"] = lift_at_k(y_test, p_test, 0.10)
    test_metrics["lift_top20pct"] = lift_at_k(y_test, p_test, 0.20)
    test_metrics["best_model"] = best_name
    test_metrics["cv_roc_auc_mean"] = float(cv.mean())
    test_metrics["cv_roc_auc_std"] = float(cv.std())
    test_metrics["n_test"] = int(len(y_test))
    test_metrics["test_positive_rate"] = float(y_test.mean())

    print("\nTest-set performance:")
    for k in ["default_0.5", "business_optimal", "recall_first"]:
        m = test_metrics[k]
        print(f"  {k:<18} thr={m['threshold']:.3f}  recall={m['recall']:.3f}  "
              f"precision={m['precision']:.3f}  f1={m['f1']:.3f}  auc={m['roc_auc']:.3f}")

    # ---- persist artefacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "churn_model.pkl")
    joblib.dump(
        {"model_name": best_name, "threshold_business": thr_business, "threshold_recall": thr_recall,
         "feature_columns": feature_cols},
        MODELS_DIR / "model_metadata.pkl",
    )
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv")
    ev_curve.to_csv(REPORTS_DIR / "threshold_expected_value.csv", index=False)
    with open(REPORTS_DIR / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2, default=float)

    # curve data for the dashboard's model page
    fpr, tpr, _ = roc_curve(y_test, p_test)
    prec, rec, _ = precision_recall_curve(y_test, p_test)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(REPORTS_DIR / "roc_curve.csv", index=False)
    pd.DataFrame({"precision": prec, "recall": rec}).to_csv(REPORTS_DIR / "pr_curve.csv", index=False)
    np.save(MODELS_DIR / "test_predictions.npy", np.vstack([y_test.to_numpy(), p_test]))

    # per-model ROC for the comparison chart
    roc_all = []
    for name, p in val_probas.items():
        f, t, _ = roc_curve(y_val, p)
        roc_all.append(pd.DataFrame({"model": name, "fpr": f, "tpr": t}))
    pd.concat(roc_all).to_csv(REPORTS_DIR / "roc_curves_by_model.csv", index=False)

    print(f"\nSaved model to {MODELS_DIR / 'churn_model.pkl'}")


if __name__ == "__main__":
    main()
