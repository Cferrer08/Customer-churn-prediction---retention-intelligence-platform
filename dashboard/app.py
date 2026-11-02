"""
Retention Intelligence dashboard.

    streamlit run dashboard/app.py

Three surfaces, each aimed at a different reader:
    Executive overview  - what is the exposure and what is the plan worth?
    Risk explorer       - one customer, one score, one reason, one action.
    Model performance   - can we trust the number, and where should the cut-off sit?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

# --------------------------------------------------------------------------------------
# Look and feel: slate / clay / amber, hairline rules, numerals in a monospace face.
# --------------------------------------------------------------------------------------
SLATE, CLAY, AMBER, INK, SAND = "#335C67", "#9E2A2B", "#E09F3E", "#22333B", "#E7DDC6"
BAND_COLOUR = {"Low": SLATE, "Medium": AMBER, "High": CLAY}

st.set_page_config(page_title="Retention Intelligence", page_icon="📉", layout="wide")

st.markdown(
    f"""
    <style>
      .stApp {{ background: #FCFBF9; }}
      h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
      .eyebrow {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.72rem;
                  text-transform: uppercase; letter-spacing: 0.14em; color: #7A736A; }}
      .kpi {{ border: 1px solid #E4DFD6; border-top: 3px solid {SLATE};
              background: #FFFFFF; padding: 14px 16px; height: 100%; }}
      .kpi .label {{ font-family: ui-monospace, Menlo, monospace; font-size: 0.68rem;
                     text-transform: uppercase; letter-spacing: 0.1em; color: #7A736A; }}
      .kpi .value {{ font-size: 1.85rem; font-weight: 620; color: {INK}; line-height: 1.25; }}
      .kpi .sub {{ font-size: 0.78rem; color: #8A8279; }}
      .kpi.warn {{ border-top-color: {CLAY}; }}
      .kpi.mid  {{ border-top-color: {AMBER}; }}
      .card {{ border: 1px solid #E4DFD6; background: #FFFFFF; padding: 18px 20px; }}
      .rule {{ border-top: 1px solid #E4DFD6; margin: 1.4rem 0 1.1rem; }}
      .driver {{ font-size: 0.92rem; padding: 3px 0; }}
      .pill {{ display:inline-block; padding: 2px 10px; border-radius: 999px;
               font-family: ui-monospace, Menlo, monospace; font-size: 0.72rem; color: #FFF; }}
      div[data-testid="stMetricValue"] {{ color: {INK}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_scored() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "scored_customers.parquet")


@st.cache_data(show_spinner=False)
def load_drivers() -> pd.DataFrame:
    path = PROCESSED / "customer_drivers.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    path = REPORTS / name
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_csv(name: str, **kw) -> pd.DataFrame:
    path = REPORTS / name
    return pd.read_csv(path, **kw) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_test_predictions():
    path = MODELS / "test_predictions.npy"
    if not path.exists():
        return None, None
    y, p = np.load(path)
    return y.astype(int), p


def money(v: float) -> str:
    if abs(v) >= 1e6:
        return f"${v/1e6:,.2f}M"
    if abs(v) >= 1e3:
        return f"${v/1e3:,.0f}K"
    return f"${v:,.0f}"


def kpi(col, label: str, value: str, sub: str = "", tone: str = "") -> None:
    col.markdown(
        f'<div class="kpi {tone}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def base_layout(fig, height: int = 340, legend: bool = False):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=36, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=12),
        showlegend=legend,
        xaxis=dict(gridcolor="#EFEBE4", zerolinecolor="#E4DFD6"),
        yaxis=dict(gridcolor="#EFEBE4", zerolinecolor="#E4DFD6"),
    )
    return fig


# --------------------------------------------------------------------------------------
# Page 1: Executive overview
# --------------------------------------------------------------------------------------
def page_overview(scored: pd.DataFrame, impact: dict) -> None:
    st.markdown('<div class="eyebrow">Executive overview</div>', unsafe_allow_html=True)
    st.title("Where the revenue is walking out")

    high = scored[scored["risk_band"] == "High"]
    st.markdown(
        f"**{len(high):,} customers** are more likely than not to cancel within the next quarter, "
        f"carrying **{money(high['annual_revenue'].sum())}** of annual recurring revenue. "
        f"Working the {impact.get('customers_targeted', 0):,} customers above 30% risk, those included, "
        f"is worth an expected **{money(impact.get('net_benefit', 0))}** net of programme cost."
    )
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

    c = st.columns(5)
    kpi(c[0], "Active customers", f"{len(scored):,}",
        f"{money(scored['annual_revenue'].sum())} ARR")
    kpi(c[1], "Predicted churn rate", f"{scored['churn_probability'].mean():.1%}",
        "next 3 months, probability-weighted", "mid")
    kpi(c[2], "Revenue at risk", money(scored["arr_at_risk"].sum()),
        "ARR weighted by churn probability", "warn")
    kpi(c[3], "High-risk customers", f"{len(high):,}",
        f"{len(high)/len(scored):.1%} of the book", "warn")
    kpi(c[4], "Avg expected lifetime", f"{scored['expected_remaining_months'].mean():.0f} mo",
        f"avg bill {money(scored['avg_monthly_spend'].mean())}")

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Churn and the revenue it takes with it")
        monthly = load_csv("eda_monthly_churn.csv")
        if not monthly.empty:
            monthly.columns = ["month", *monthly.columns[1:]]
            fig = go.Figure()
            fig.add_bar(x=monthly["month"], y=monthly["annualised_revenue_lost"],
                        name="Annualised revenue lost", marker_color=SAND, yaxis="y2")
            fig.add_scatter(x=monthly["month"], y=monthly["churn_rate"], name="Churn rate",
                            mode="lines+markers", line=dict(color=CLAY, width=3))
            fig.update_layout(
                yaxis=dict(title="Monthly churn rate", tickformat=".1%"),
                yaxis2=dict(title="Revenue lost", overlaying="y", side="right", showgrid=False),
            )
            st.plotly_chart(base_layout(fig, 380, legend=True), width="stretch")

    with right:
        st.subheader("Risk distribution")
        band = scored["risk_band"].value_counts().reindex(["Low", "Medium", "High"]).dropna()
        fig = go.Figure(go.Pie(
            labels=band.index, values=band.values, hole=0.62, sort=False,
            marker=dict(colors=[BAND_COLOUR[b] for b in band.index], line=dict(color="#FFF", width=2)),
            textinfo="label+percent",
        ))
        fig.add_annotation(text=f"<b>{len(scored):,}</b><br>customers", showarrow=False,
                           font=dict(size=15, color=INK))
        st.plotly_chart(base_layout(fig, 380), width="stretch")

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    a, b = st.columns(2)

    with a:
        st.subheader("Revenue at risk by segment")
        seg = (scored.groupby(["customer_segment", "risk_band"])["arr_at_risk"].sum()
               .unstack(fill_value=0).reindex(columns=["Low", "Medium", "High"], fill_value=0))
        fig = go.Figure()
        for band_name in ["Low", "Medium", "High"]:
            fig.add_bar(x=seg.index, y=seg[band_name], name=band_name,
                        marker_color=BAND_COLOUR[band_name])
        fig.update_layout(barmode="stack", yaxis=dict(title="ARR at risk", tickprefix="$"))
        st.plotly_chart(base_layout(fig, 340, legend=True), width="stretch")

    with b:
        st.subheader("Churn risk by contract type")
        con = scored.groupby("contract_type")["churn_probability"].mean().sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=con.index, y=con.values,
                               marker_color=[CLAY, AMBER, SLATE][: len(con)],
                               text=[f"{v:.1%}" for v in con.values], textposition="outside"))
        fig.update_layout(yaxis=dict(title="Average churn probability", tickformat=".0%"))
        st.plotly_chart(base_layout(fig, 340), width="stretch")

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.subheader("The retention plan, ranked by expected value")
    plays = (scored[scored["action_cost"] > 0]
             .groupby(["recommended_action", "action_owner"])
             .agg(customers=("customer_id", "count"),
                  arr_at_risk=("arr_at_risk", "sum"),
                  programme_cost=("action_cost", "sum"),
                  expected_value=("action_expected_value", "sum"))
             .sort_values("expected_value", ascending=False)
             .reset_index())
    st.dataframe(
        plays.style.format({"customers": "{:,.0f}", "arr_at_risk": "${:,.0f}",
                            "programme_cost": "${:,.0f}", "expected_value": "${:,.0f}"}),
        width="stretch", hide_index=True,
    )
    st.caption(
        f"Assumes a {impact.get('assumptions', {}).get('save_rate', 0.3):.0%} save rate on contacted "
        f"customers and a {impact.get('assumptions', {}).get('gross_margin', 0.72):.0%} gross margin. "
        "Expected value is the retained margin net of the cost of the play."
    )


# --------------------------------------------------------------------------------------
# Page 2: Customer risk explorer
# --------------------------------------------------------------------------------------
def page_explorer(scored: pd.DataFrame, drivers: pd.DataFrame) -> None:
    st.markdown('<div class="eyebrow">Customer risk explorer</div>', unsafe_allow_html=True)
    st.title("One customer, one reason, one action")

    with st.sidebar:
        st.markdown("### Narrow the list")
        segs = st.multiselect("Segment", sorted(scored["customer_segment"].unique()))
        bands = st.multiselect("Risk band", ["High", "Medium", "Low"], default=["High"])
        min_spend = st.slider("Minimum monthly bill", 0, 260, 0, 10)

    view = scored.copy()
    if segs:
        view = view[view["customer_segment"].isin(segs)]
    if bands:
        view = view[view["risk_band"].isin(bands)]
    view = view[view["avg_monthly_spend"] >= min_spend]

    if view.empty:
        st.warning("No customers match those filters. Widen the selection in the sidebar.")
        return

    options = view.head(400)
    labels = [f"{r.customer_id} — {r.customer_name} ({r.churn_probability:.0%})"
              for r in options.itertuples()]
    typed = st.text_input("Search by customer ID", placeholder="e.g. CUS027204").strip().upper()

    if typed:
        match = scored[scored["customer_id"] == typed]
        if match.empty:
            st.error(f"No active customer with ID {typed}. The scoring table only holds customers "
                     "who have not already cancelled.")
            return
        cust = match.iloc[0]
    else:
        pick = st.selectbox("Or choose from the highest-risk accounts", labels)
        cust = options.iloc[labels.index(pick)]

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    head, gauge = st.columns([1.5, 1])

    with head:
        st.markdown(f"### {cust['customer_name']}  \n`{cust['customer_id']}`")
        f1, f2 = st.columns(2)
        f1.markdown(
            f"**Segment** {cust['customer_segment']}  \n"
            f"**Plan** {cust['plan_name']}  \n"
            f"**Contract** {cust['contract_type']}  \n"
            f"**Location** {cust['city']}, {cust['state']}"
        )
        f2.markdown(
            f"**Monthly bill** {money(cust['avg_monthly_spend'])}  \n"
            f"**Tenure** {cust['tenure_months']:.0f} months  \n"
            f"**Revenue to date** {money(cust['total_revenue'])}  \n"
            f"**Expected remaining life** {cust['expected_remaining_months']:.0f} months"
        )
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Recent usage", f"{cust['usage_hours_l3']:.1f} h/mo",
                  f"{cust['usage_trend_3m']:+.0%} vs prior quarter")
        b2.metric("Support tickets", f"{cust['tickets_l3']:.1f}/mo", f"{cust['tickets_total']:.0f} lifetime")
        b3.metric("Satisfaction", f"{cust['satisfaction_l3']:.1f}/5")
        b4.metric("ARR at risk", money(cust["arr_at_risk"]))

    with gauge:
        p = float(cust["churn_probability"])
        colour = BAND_COLOUR.get(cust["risk_band"], SLATE)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number={"suffix": "%", "font": {"size": 40, "color": INK}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#C9C4BA"},
                "bar": {"color": colour, "thickness": 0.7},
                "bgcolor": "#F3EFE8",
                "borderwidth": 0,
                "steps": [{"range": [0, 30], "color": "#EFF2F1"},
                          {"range": [30, 70], "color": "#F7EEDD"},
                          {"range": [70, 100], "color": "#F6E4E2"}],
            },
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=10, b=0),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            f'<div style="text-align:center"><span class="pill" style="background:{colour}">'
            f'{cust["risk_band"].upper()} RISK</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    why, action = st.columns([1.4, 1])

    with why:
        st.subheader("Why the model thinks so")
        d = drivers[drivers["customer_id"] == cust["customer_id"]] if not drivers.empty else pd.DataFrame()
        if d.empty:
            st.info("Run `python src/explain.py` to generate per-customer SHAP explanations.")
        else:
            d = d.sort_values("shap_value")
            fig = go.Figure(go.Bar(
                x=d["shap_value"], y=d["explanation"], orientation="h",
                marker_color=[CLAY if v > 0 else SLATE for v in d["shap_value"]],
            ))
            fig.update_layout(
                xaxis=dict(title="Effect on churn risk (log-odds)"),
                yaxis=dict(title=""), height=max(320, 26 * len(d)),
            )
            st.plotly_chart(base_layout(fig, max(320, 26 * len(d))), width="stretch")
            st.caption("Red pushes risk up, slate pulls it down. Contributions are SHAP values, "
                       "so they add up to this customer's score.")

    with action:
        st.subheader("What to do about it")
        st.markdown(
            f'<div class="card">'
            f'<div class="eyebrow">Recommended play</div>'
            f'<div style="font-size:1.1rem;font-weight:600;margin:6px 0 12px">'
            f'{cust["recommended_action"]}</div>'
            f'<div class="driver"><b>Owner</b> — {cust["action_owner"]}</div>'
            f'<div class="driver"><b>Cost of the play</b> — {money(cust["action_cost"])}</div>'
            f'<div class="driver"><b>Expected retained margin</b> — {money(cust["expected_saved_margin"])}</div>'
            f'<div class="driver"><b>Net expected value</b> — {money(cust["action_expected_value"])}</div>'
            f'</div>', unsafe_allow_html=True)
        if isinstance(cust.get("top_protective_factors"), str):
            st.markdown("**Working in our favour**")
            for line in str(cust["top_protective_factors"]).split(" | "):
                st.markdown(f'<div class="driver">• {line}</div>', unsafe_allow_html=True)

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.subheader("Today's call list")
    cols = ["customer_id", "customer_name", "customer_segment", "plan_name", "contract_type",
            "churn_probability", "avg_monthly_spend", "arr_at_risk", "recommended_action",
            "action_owner", "action_expected_value"]
    call_list = view.sort_values("action_expected_value", ascending=False)[cols].head(100)
    st.dataframe(
        call_list.style.format({
            "churn_probability": "{:.1%}", "avg_monthly_spend": "${:,.2f}",
            "arr_at_risk": "${:,.0f}", "action_expected_value": "${:,.0f}"}),
        width="stretch", hide_index=True, height=420,
    )
    st.download_button("Download this call list as CSV", call_list.to_csv(index=False),
                       "retention_call_list.csv", "text/csv")


# --------------------------------------------------------------------------------------
# Page 3: Model performance
# --------------------------------------------------------------------------------------
def page_model() -> None:
    st.markdown('<div class="eyebrow">Model performance</div>', unsafe_allow_html=True)
    st.title("Can we trust the number?")

    metrics = load_json("test_metrics.json")
    comparison = load_csv("model_comparison.csv", index_col=0)
    if not metrics:
        st.info("Run `python src/model.py` first to generate evaluation artefacts.")
        return

    biz = metrics.get("business_optimal", {})
    c = st.columns(5)
    kpi(c[0], "Chosen model", metrics.get("best_model", "-"), "selected on validation PR-AUC")
    kpi(c[1], "ROC-AUC", f"{biz.get('roc_auc', 0):.3f}",
        f"5-fold CV {metrics.get('cv_roc_auc_mean', 0):.3f} ± {metrics.get('cv_roc_auc_std', 0):.3f}")
    kpi(c[2], "Recall", f"{biz.get('recall', 0):.0%}", "churners caught at the chosen cut-off", "mid")
    kpi(c[3], "Precision", f"{biz.get('precision', 0):.0%}", "of flagged customers really leave")
    kpi(c[4], "Top-decile capture", f"{metrics.get('lift_top10pct', 0):.0%}",
        "of all churners sit in the top 10% of scores")

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    left, right = st.columns(2)

    with left:
        st.subheader("Model comparison")
        roc = load_csv("roc_curves_by_model.csv")
        if not roc.empty:
            fig = go.Figure()
            for (name, g), colour in zip(roc.groupby("model"), [SLATE, CLAY, AMBER, "#6B9080"]):
                auc = comparison.loc[name, "roc_auc"] if name in comparison.index else np.nan
                fig.add_scatter(x=g["fpr"], y=g["tpr"], mode="lines", name=f"{name} ({auc:.3f})",
                                line=dict(color=colour, width=2.4))
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                            line=dict(color="#C9C4BA", dash="dash"))
            fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate",
                              legend=dict(x=0.45, y=0.08))
            st.plotly_chart(base_layout(fig, 400, legend=True), width="stretch")
        if not comparison.empty:
            st.dataframe(comparison.round(4), width="stretch")

    with right:
        st.subheader("Where should the cut-off sit?")
        y_test, p_test = load_test_predictions()
        default_thr = float(biz.get("threshold", 0.5))
        thr = st.slider("Decision threshold", 0.01, 0.95, default_thr, 0.01)
        if y_test is not None:
            pred = (p_test >= thr).astype(int)
            tp = int(((pred == 1) & (y_test == 1)).sum())
            fp = int(((pred == 1) & (y_test == 0)).sum())
            fn = int(((pred == 0) & (y_test == 1)).sum())
            tn = int(((pred == 0) & (y_test == 0)).sum())
            recall = tp / max(tp + fn, 1)
            precision = tp / max(tp + fp, 1)

            m = st.columns(3)
            m[0].metric("Recall", f"{recall:.1%}")
            m[1].metric("Precision", f"{precision:.1%}")
            m[2].metric("Customers flagged", f"{tp + fp:,}")

            z = [[tn, fp], [fn, tp]]
            fig = go.Figure(go.Heatmap(
                z=z, x=["Predicted stay", "Predicted churn"],
                y=["Actually stayed", "Actually churned"],
                colorscale=[[0, "#F3EFE8"], [1, SLATE]], showscale=False,
                text=[[f"{v:,}" for v in row] for row in z],
                texttemplate="%{text}", textfont={"size": 18},
            ))
            st.plotly_chart(base_layout(fig, 300), width="stretch")
            st.caption(
                f"At this threshold the desk contacts {tp + fp:,} of {len(y_test):,} test customers "
                f"and reaches {recall:.0%} of everyone who actually left. Lowering it catches more "
                "churners but wastes more offers on customers who were never going to leave."
            )

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    a, b = st.columns(2)

    with a:
        st.subheader("What drives the predictions")
        imp = load_csv("shap_feature_importance.csv")
        if not imp.empty:
            top = imp.head(15).iloc[::-1]
            fig = go.Figure(go.Bar(x=top["mean_abs_shap"], y=top["feature"],
                                   orientation="h", marker_color=SLATE))
            fig.update_layout(xaxis_title="Mean |SHAP value|", yaxis_title="")
            st.plotly_chart(base_layout(fig, 460), width="stretch")

    with b:
        st.subheader("Expected value by threshold")
        ev = load_csv("threshold_expected_value.csv")
        if not ev.empty:
            best = ev.loc[ev["expected_value"].idxmax()]
            fig = go.Figure()
            fig.add_scatter(x=ev["threshold"], y=ev["expected_value"], mode="lines",
                            line=dict(color=SLATE, width=2.6))
            fig.add_vline(x=best["threshold"], line=dict(color=CLAY, dash="dash"))
            fig.update_layout(xaxis_title="Decision threshold",
                              yaxis_title="Expected retained margin", yaxis_tickprefix="$")
            st.plotly_chart(base_layout(fig, 400), width="stretch")
            st.caption(
                f"Optimum at {best['threshold']:.2f}: {int(best['flagged']):,} customers contacted, "
                f"{best['recall']:.0%} recall at {best['precision']:.0%} precision. "
                "The peak is where an extra retention call stops paying for itself."
            )


# --------------------------------------------------------------------------------------
def main() -> None:
    if not (PROCESSED / "scored_customers.parquet").exists():
        st.error("No scored customers found. Run `python src/run_pipeline.py` first.")
        st.stop()

    scored = load_scored()
    drivers = load_drivers()
    impact = load_json("business_impact.json")

    with st.sidebar:
        st.markdown('<div class="eyebrow">Retention intelligence</div>', unsafe_allow_html=True)
        page = st.radio("View", ["Executive overview", "Customer risk explorer", "Model performance"],
                        label_visibility="collapsed")
        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        st.caption(f"Scored {len(scored):,} active customers  \n"
                   f"Model: {impact.get('model', 'XGBoost')}  \n"
                   "Horizon: next 3 months")

    if page == "Executive overview":
        page_overview(scored, impact)
    elif page == "Customer risk explorer":
        page_explorer(scored, drivers)
    else:
        page_model()


main()
