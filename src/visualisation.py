"""
All figures for the EDA notebook, the model report and the README.

Charts are written to reports/figures/ and the underlying aggregates to reports/ as CSVs,
so every number quoted in the README can be traced back to a table rather than an image.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

from config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR  # noqa: E402

# --------------------------------------------------------------------------------------
# House style: a restrained slate/amber/clay palette rather than matplotlib defaults
# --------------------------------------------------------------------------------------
SLATE = "#335C67"
CLAY = "#9E2A2B"
AMBER = "#E09F3E"
SAND = "#E7DDC6"
INK = "#22333B"
CYCLE = [SLATE, CLAY, AMBER, "#6B9080", "#8A6552", "#B0A084"]

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#C9C4BA",
        "axes.labelcolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlepad": 12,
        "axes.grid": True,
        "grid.color": "#EAE6DF",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "font.size": 10,
        "legend.frameon": False,
        "figure.dpi": 130,
    }
)
MONEY = FuncFormatter(lambda v, _: f"${v/1e6:.1f}M" if abs(v) >= 1e6 else f"${v/1e3:.0f}K")


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def _bar_labels(ax, fmt="{:.1%}", offset=0.002):
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h):
            ax.annotate(fmt.format(h), (p.get_x() + p.get_width() / 2, h + offset),
                        ha="center", va="bottom", fontsize=9, color=INK)


# --------------------------------------------------------------------------------------
# EDA
# --------------------------------------------------------------------------------------
def eda_figures(clean: dict, panel: pd.DataFrame, feats: pd.DataFrame) -> None:
    customers, status, plans, dates = clean["customers"], clean["status"], clean["plans"], clean["dim_date"]
    cust = customers.merge(status, on="customer_id").merge(plans, on="plan_id")

    # ---- 1. Portfolio overview: who are these customers
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].hist(cust["age"], bins=30, color=SLATE, edgecolor="white")
    axes[0].set_title("Age distribution")
    axes[0].set_xlabel("Age")

    seg = cust["customer_segment"].value_counts()
    axes[1].bar(seg.index, seg.values, color=[SLATE, AMBER, CLAY])
    axes[1].set_title("Customers by segment")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}K"))

    reg = cust["region"].value_counts()
    axes[2].barh(reg.index[::-1], reg.values[::-1], color=SLATE)
    axes[2].set_title("Customers by region")
    axes[2].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}K"))
    _save(fig, "eda_customer_distribution.png")

    # ---- 2. Churn by contract and plan tier
    by_contract = cust.groupby("contract_type")["churn_flag"].agg(["mean", "size"])
    by_tier = cust.groupby("plan_tier")["churn_flag"].agg(["mean", "size"]).reindex(
        ["Starter", "Basic", "Standard", "Plus", "Pro", "Enterprise"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].bar(by_contract.index, by_contract["mean"], color=[CLAY, AMBER, SLATE])
    axes[0].set_title("Churn rate by contract type")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    _bar_labels(axes[0])
    axes[1].bar(by_tier.index, by_tier["mean"], color=SLATE)
    axes[1].set_title("Churn rate by plan tier")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].tick_params(axis="x", rotation=20)
    _bar_labels(axes[1])
    _save(fig, "eda_churn_by_contract_and_plan.png")
    by_contract.to_csv(REPORTS_DIR / "eda_churn_by_contract.csv")
    by_tier.to_csv(REPORTS_DIR / "eda_churn_by_plan_tier.csv")

    # ---- 3. Churn by segment, age band and price band
    cust["age_band"] = pd.cut(cust["age"], [17, 25, 35, 45, 55, 65, 100],
                              labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"])
    cust["price_band"] = pd.qcut(cust["monthly_price"], 5,
                                 labels=["Cheapest", "Low", "Mid", "High", "Premium"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, col, title in zip(
        axes,
        ["customer_segment", "age_band", "price_band"],
        ["Churn by segment", "Churn by age band", "Churn by price band"],
    ):
        s = cust.groupby(col, observed=True)["churn_flag"].mean()
        ax.bar(s.index.astype(str), s.values, color=SLATE)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.tick_params(axis="x", rotation=20)
        _bar_labels(ax)
    _save(fig, "eda_churn_by_demographics.png")

    # ---- 4. Monthly churn trend and revenue lost
    st = status[status["churn_flag"] == 1].copy()
    st["churn_month"] = st["churn_date"].dt.to_period("M").dt.to_timestamp()
    monthly_churn = st.groupby("churn_month").size()

    spend = panel.groupby("customer_id")["payment_amount"].mean()
    st["monthly_value"] = st["customer_id"].map(spend)
    revenue_lost = st.groupby("churn_month")["monthly_value"].sum() * 12

    active_by_month = panel.groupby("date")["customer_id"].nunique()
    active_by_month.index = pd.to_datetime(active_by_month.index)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    rate = (monthly_churn / active_by_month.reindex(monthly_churn.index)).dropna()
    axes[0].plot(rate.index, rate.values, color=CLAY, lw=2.2, marker="o", ms=4)
    axes[0].set_title("Monthly churn rate")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(revenue_lost.index, revenue_lost.values, width=22, color=CLAY, alpha=0.85)
    axes[1].set_title("Annualised revenue lost to churn, by month")
    axes[1].yaxis.set_major_formatter(MONEY)
    axes[1].tick_params(axis="x", rotation=30)
    _save(fig, "eda_churn_trend_and_revenue.png")
    pd.DataFrame({"churned": monthly_churn, "churn_rate": rate,
                  "annualised_revenue_lost": revenue_lost}).to_csv(REPORTS_DIR / "eda_monthly_churn.csv")

    # ---- 5. Cancellation reasons and their revenue weight
    reasons = st.groupby("cancellation_reason").agg(
        customers=("customer_id", "size"), annual_revenue=("monthly_value", lambda s: s.sum() * 12)
    ).sort_values("customers", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    axes[0].barh(reasons.index, reasons["customers"], color=SLATE)
    axes[0].set_title("Why customers said they left")
    axes[1].barh(reasons.index, reasons["annual_revenue"], color=AMBER)
    axes[1].set_title("Annualised revenue lost by reason")
    axes[1].xaxis.set_major_formatter(MONEY)
    _save(fig, "eda_cancellation_reasons.png")
    reasons.sort_values("customers", ascending=False).to_csv(REPORTS_DIR / "eda_cancellation_reasons.csv")

    # ---- 6. Customer lifetime distribution
    life = customers.merge(status, on="customer_id")
    life["lifetime_months"] = np.where(
        life["churn_flag"] == 1,
        (life["churn_date"] - life["signup_date"]).dt.days / 30.44,
        (pd.to_datetime(dates["date"].max()) - life["signup_date"]).dt.days / 30.44,
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    axes[0].hist(life.loc[life["churn_flag"] == 1, "lifetime_months"], bins=40, color=CLAY,
                 alpha=0.85, edgecolor="white", label="Churned")
    axes[0].hist(life.loc[life["churn_flag"] == 0, "lifetime_months"], bins=40, color=SLATE,
                 alpha=0.6, edgecolor="white", label="Still active")
    axes[0].set_title("Customer lifetime (months)")
    axes[0].legend()

    tenure_band = pd.cut(feats["tenure_months"], [-1, 6, 12, 24, 36, 60, 200],
                         labels=["0-6", "7-12", "13-24", "25-36", "37-60", "60+"])
    s = feats.groupby(tenure_band, observed=True)["churned"].mean()
    axes[1].plot(s.index.astype(str), s.values, color=CLAY, marker="o", lw=2.2)
    axes[1].set_title("Churn risk falls with tenure")
    axes[1].set_xlabel("Tenure at scoring date (months)")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    _save(fig, "eda_customer_lifetime.png")

    # ---- 7. Behaviour in the months before cancellation (the signal that makes prediction possible)
    p = panel.merge(status[["customer_id", "churn_flag", "churn_date"]], on="customer_id", how="left")
    p["date"] = pd.to_datetime(p["date"])
    churned_p = p[p["churn_flag"] == 1].copy()
    churned_p["months_to_churn"] = (
        (churned_p["churn_date"].dt.to_period("M") - churned_p["date"].dt.to_period("M"))
        .apply(lambda x: x.n)
    )
    win = churned_p[(churned_p["months_to_churn"] >= 0) & (churned_p["months_to_churn"] <= 8)]
    traj = win.groupby("months_to_churn").agg(
        usage_hours=("usage_hours", "mean"),
        support_tickets=("support_tickets", "mean"),
        satisfaction=("satisfaction_score", "mean"),
        logins=("login_frequency", "mean"),
    ).sort_index(ascending=False)
    stayed = p[p["churn_flag"] == 0].agg(
        {"usage_hours": "mean", "support_tickets": "mean", "satisfaction_score": "mean",
         "login_frequency": "mean"}
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    x = traj.index.astype(int)
    for ax, col, base, title in [
        (axes[0], "usage_hours", stayed["usage_hours"], "Usage hours"),
        (axes[1], "support_tickets", stayed["support_tickets"], "Support tickets"),
        (axes[2], "satisfaction", stayed["satisfaction_score"], "Satisfaction score"),
    ]:
        ax.plot(x, traj[col], color=CLAY, lw=2.4, marker="o", ms=5, label="Customers who left")
        ax.axhline(base, color=SLATE, ls="--", lw=1.6, label="Customers who stayed")
        ax.invert_xaxis()
        ax.set_title(title)
        ax.set_xlabel("Months before cancellation")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Churn is visible months before it happens", fontsize=14, fontweight="semibold", y=1.03)
    _save(fig, "eda_pre_churn_behaviour.png")
    traj.to_csv(REPORTS_DIR / "eda_pre_churn_trajectory.csv")

    # ---- 8. Usage / value-gap relationship with churn
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, col, title, label in [
        (axes[0], "usage_trend_3m", "Churn by usage trend", "Usage change vs prior quarter"),
        (axes[1], "revenue_per_usage_hour", "Churn by price paid per hour used", "$ per usage hour"),
        (axes[2], "satisfaction_l3", "Churn by recent satisfaction", "Satisfaction (1-5)"),
    ]:
        q = pd.qcut(feats[col], 5, duplicates="drop")
        s = feats.groupby(q, observed=True)["churned"].mean()
        ax.bar(range(len(s)), s.values, color=SLATE)
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels(["Q1\n(low)", "Q2", "Q3", "Q4", "Q5\n(high)"][: len(s)])
        ax.set_title(title)
        ax.set_xlabel(label)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        _bar_labels(ax)
    _save(fig, "eda_churn_drivers.png")

    # ---- 9. Correlation heatmap of the strongest numeric features
    cols = ["churned", "tenure_months", "usage_hours_l3", "usage_trend_3m", "tickets_l6",
            "satisfaction_l3", "late_payment_rate", "monthly_price", "revenue_per_usage_hour",
            "feature_usage_l3", "is_month_to_month"]
    corr = feats[cols].corr()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else INK)
    ax.grid(False)
    ax.set_title("Correlation between churn and engineered features")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, "eda_correlation_heatmap.png")


# --------------------------------------------------------------------------------------
# Model evaluation
# --------------------------------------------------------------------------------------
def model_figures() -> None:
    comparison = pd.read_csv(REPORTS_DIR / "model_comparison.csv", index_col=0)
    roc_by_model = pd.read_csv(REPORTS_DIR / "roc_curves_by_model.csv")
    pr = pd.read_csv(REPORTS_DIR / "pr_curve.csv")
    ev = pd.read_csv(REPORTS_DIR / "threshold_expected_value.csv")
    metrics = json.load(open(REPORTS_DIR / "test_metrics.json"))
    y_test, p_test = np.load(MODELS_DIR / "test_predictions.npy")

    # ---- ROC comparison + PR curve
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for (name, g), c in zip(roc_by_model.groupby("model"), CYCLE):
        auc = comparison.loc[name, "roc_auc"]
        axes[0].plot(g["fpr"], g["tpr"], color=c, lw=2, label=f"{name} (AUC {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], ls="--", color="#B0A084", lw=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curves, validation set")
    axes[0].legend(fontsize=9, loc="lower right")

    axes[1].plot(pr["recall"], pr["precision"], color=CLAY, lw=2.2)
    axes[1].axhline(metrics["test_positive_rate"], ls="--", color=SLATE, lw=1.4,
                    label=f"Base rate ({metrics['test_positive_rate']:.1%})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-recall, test set (PR-AUC {metrics['default_0.5']['pr_auc']:.3f})")
    axes[1].legend(fontsize=9)
    _save(fig, "model_roc_and_pr.png")

    # ---- Model comparison bars
    fig, ax = plt.subplots(figsize=(10, 4.6))
    m = comparison[["roc_auc", "pr_auc", "recall", "precision"]].sort_values("pr_auc")
    x = np.arange(len(m))
    w = 0.2
    for i, (col, c) in enumerate(zip(m.columns, CYCLE)):
        ax.bar(x + i * w, m[col], w, label=col.replace("_", " ").upper(), color=c)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(m.index, rotation=10)
    ax.set_title("Model comparison on the validation set")
    ax.legend(ncol=4, fontsize=9)
    _save(fig, "model_comparison.png")

    # ---- Confusion matrices at both operating points
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, key, title in [
        (axes[0], "business_optimal", "Business-optimal threshold"),
        (axes[1], "recall_first", "Recall-first threshold"),
    ]:
        d = metrics[key]
        cm = np.array([[d["tn"], d["fp"]], [d["fn"], d["tp"]]])
        ax.imshow(cm, cmap="Blues")
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() / 2 else INK)
        ax.set_xticks([0, 1], ["Predicted stay", "Predicted churn"])
        ax.set_yticks([0, 1], ["Actually stayed", "Actually churned"])
        ax.set_title(f"{title}\nthr={d['threshold']:.2f} | recall {d['recall']:.0%} | precision {d['precision']:.0%}",
                     fontsize=11)
    _save(fig, "model_confusion_matrices.png")

    # ---- Expected value vs threshold
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(ev["threshold"], ev["expected_value"], color=SLATE, lw=2.2)
    best = ev.loc[ev["expected_value"].idxmax()]
    ax.axvline(best["threshold"], color=CLAY, ls="--", lw=1.6)
    ax.annotate(
        f"Optimal threshold {best['threshold']:.2f}\n{int(best['flagged']):,} customers contacted\n"
        f"recall {best['recall']:.0%}, precision {best['precision']:.0%}",
        xy=(best["threshold"], best["expected_value"]),
        xytext=(best["threshold"] + 0.08, best["expected_value"] * 0.72),
        fontsize=9, color=INK, arrowprops=dict(arrowstyle="->", color=CLAY),
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected retained margin")
    ax.yaxis.set_major_formatter(MONEY)
    ax.set_title("The threshold is a business decision, not 0.5")
    _save(fig, "model_threshold_expected_value.png")

    # ---- Calibration and score distribution
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p_test, bins) - 1
    obs = [y_test[idx == b].mean() if (idx == b).sum() > 20 else np.nan for b in range(10)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].plot([0, 1], [0, 1], ls="--", color="#B0A084")
    axes[0].plot(bins[:-1] + 0.05, obs, marker="o", color=CLAY, lw=2)
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Observed churn rate")
    axes[0].set_title(f"Calibration (Brier {metrics['default_0.5']['brier']:.3f})")

    axes[1].hist(p_test[y_test == 0], bins=40, color=SLATE, alpha=0.75, label="Stayed")
    axes[1].hist(p_test[y_test == 1], bins=40, color=CLAY, alpha=0.75, label="Churned")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Predicted churn probability")
    axes[1].set_title("Score separation on the test set")
    axes[1].legend()
    _save(fig, "model_calibration.png")


def portfolio_figures() -> None:
    scored = pd.read_parquet(PROCESSED_DIR / "scored_customers.parquet")
    seg = pd.read_csv(REPORTS_DIR / "risk_segment_summary.csv", index_col=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = {"Low": SLATE, "Medium": AMBER, "High": CLAY}
    order = [b for b in ["Low", "Medium", "High"] if b in seg.index]
    c = [colors[b] for b in order]

    axes[0].bar(order, seg.loc[order, "customers"], color=c)
    axes[0].set_title("Customers by risk band")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}K"))
    for i, v in enumerate(seg.loc[order, "customers"]):
        axes[0].annotate(f"{int(v):,}", (i, v), ha="center", va="bottom", fontsize=9)

    axes[1].bar(order, seg.loc[order, "arr_at_risk"], color=c)
    axes[1].set_title("Revenue at risk by band")
    axes[1].yaxis.set_major_formatter(MONEY)

    top = (scored.groupby("recommended_action")["action_expected_value"].sum()
           .sort_values().tail(6))
    axes[2].barh([t[:38] + "..." if len(t) > 38 else t for t in top.index], top.values, color=SLATE)
    axes[2].set_title("Expected value by retention play")
    axes[2].xaxis.set_major_formatter(MONEY)
    axes[2].tick_params(axis="y", labelsize=8)
    _save(fig, "business_risk_segments.png")


def use_headless_backend() -> None:
    """Switch to a file-only backend when running as a script.

    Deliberately not done at import time: these modules are also imported by the
    notebooks, where forcing Agg would silently stop every inline chart from rendering.
    """
    matplotlib.use("Agg")


def main() -> None:
    use_headless_backend()
    print("Loading processed data ...")
    clean = {n: pd.read_parquet(PROCESSED_DIR / f"clean_{n}.parquet")
             for n in ["customers", "plans", "dim_date", "status"]}
    panel = pd.read_parquet(PROCESSED_DIR / "panel_customer_month.parquet")
    feats = pd.read_parquet(PROCESSED_DIR / "features_train.parquet")

    print("Building EDA figures ...")
    eda_figures(clean, panel, feats)
    print("Building model figures ...")
    model_figures()
    print("Building business figures ...")
    portfolio_figures()
    print(f"All figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
