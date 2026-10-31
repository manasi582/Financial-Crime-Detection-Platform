"""
app/pages/01_executive_overview.py
Executive Overview page: KPI cards + daily fraud trend.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json

st.set_page_config(page_title="Executive Overview — FraudLens", page_icon="🏠", layout="wide")


@st.cache_resource
def get_engine():
    try:
        from src.config import get_config
        from src.db import get_engine as _ge
        cfg = get_config()
        return _ge(cfg)
    except Exception as e:
        return None


@st.cache_data(ttl=300)
def load_kpis():
    engine = get_engine()
    if engine is None:
        return None
    try:
        from src.db import query_kpis
        return query_kpis(engine)
    except Exception as e:
        return None


@st.cache_data(ttl=300)
def load_daily_trend():
    engine = get_engine()
    if engine is None:
        return None
    try:
        from src.db import query_view
        return query_view("daily_fraud_trend", engine)
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_metrics():
    metrics_file = _ROOT / "reports" / "metrics.json"
    if not metrics_file.exists():
        return None
    with open(metrics_file) as f:
        return json.load(f)


# ── Page ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.kpi-card{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);border:1px solid #0f3460;border-radius:12px;padding:20px 24px;text-align:center;box-shadow:0 4px 20px rgba(0,0,0,.3);}
.kpi-title{color:#8892b0;font-size:.75rem;font-weight:500;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;}
.kpi-value{color:#e2e8f0;font-size:2rem;font-weight:700;line-height:1.2;}
.kpi-value.danger{color:#fc5c7d;}.kpi-value.success{color:#43d9a2;}.kpi-value.warning{color:#ffd166;}.kpi-value.info{color:#74b9ff;}
.page-header{background:linear-gradient(135deg,#0f3460 0%,#16213e 100%);border-radius:12px;padding:20px 28px;margin-bottom:24px;border-left:4px solid #e94560;}
.page-header h1{color:#e2e8f0;font-size:1.6rem;margin:0;font-weight:700;}
.page-header p{color:#8892b0;margin:4px 0 0;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h1>🏠 Executive Overview</h1>
    <p>High-level fraud KPIs and daily trend for leadership reporting</p>
</div>
""", unsafe_allow_html=True)

engine = get_engine()
if engine is None:
    st.error("⚠️ Database not found. Run `make all` to generate data and train models first.")
    st.stop()

kpis = load_kpis()
metrics = load_metrics()

if kpis is None:
    st.error("⚠️ Could not load KPIs. Run `make all` first.")
    st.stop()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
model_name = metrics.get("final_model", "N/A") if metrics else "N/A"
test_m = {}
if metrics and "models" in metrics:
    test_m = metrics["models"].get(model_name, {})

recall_val  = test_m.get("recall", 0)
prec_val    = test_m.get("precision", 0)
net_savings = metrics.get("business", {}).get("net_savings_at_opt_threshold", 0) if metrics else 0

col1, col2, col3, col4, col5, col6 = st.columns(6)

cards = [
    (col1, "Total Transactions", f"{kpis['total_txns']:,.0f}", "info", ""),
    (col2, "Fraud Cases",        f"{kpis['fraud_count']:,.0f}", "danger", ""),
    (col3, "Fraud Rate",         f"{kpis['fraud_rate_pct']:.3f}%", "warning", "% of transactions"),
    (col4, "Fraud Amount (Rs.)", f"₹{kpis['fraud_amount']:,.0f}", "danger", ""),
    (col5, "Model Recall",       f"{recall_val*100:.1f}%", "success", f"Precision: {prec_val*100:.1f}%"),
    (col6, "Net Savings (Rs.)",  f"₹{net_savings:,.0f}", "success", "vs. no-model baseline"),
]

for col, title, value, cls, delta in cards:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value {cls}">{value}</div>
            <div style="color:#8892b0;font-size:.78rem;margin-top:4px;">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Daily Fraud Trend ─────────────────────────────────────────────────────────
daily = load_daily_trend()

if daily is not None and not daily.empty:
    st.markdown("### 📈 Daily Fraud Trend (30-day view)")

    col_a, col_b = st.columns([3, 1])

    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily["day"], y=daily["fraud_amount"],
            name="Fraud Amount (Rs.)",
            marker_color="#e94560",
            opacity=0.8,
            yaxis="y2",
        ))
        fig.add_trace(go.Scatter(
            x=daily["day"], y=daily["fraud_count"],
            name="Fraud Count",
            line=dict(color="#ffd166", width=2.5),
            mode="lines+markers",
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=daily["day"], y=daily["fraud_rate_pct"],
            name="Fraud Rate (%)",
            line=dict(color="#74b9ff", width=2, dash="dot"),
            mode="lines",
            yaxis="y3",
            visible="legendonly",
        ))
        fig.update_layout(
            paper_bgcolor="#0a0a1a",
            plot_bgcolor="#0a0a1a",
            font=dict(color="#e2e8f0"),
            legend=dict(bgcolor="#16213e", bordercolor="#0f3460", borderwidth=1),
            yaxis=dict(title="Fraud Count", gridcolor="#1a2744"),
            yaxis2=dict(title="Fraud Amount (Rs.)", overlaying="y", side="right", gridcolor="#1a2744"),
            yaxis3=dict(overlaying="y", side="right", showgrid=False),
            xaxis=dict(title="Day", gridcolor="#1a2744"),
            height=400,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("##### Summary Statistics")
        st.metric("Total Days", int(daily["day"].max() + 1))
        st.metric("Avg Fraud/Day", f"{daily['fraud_count'].mean():.1f}")
        st.metric("Peak Fraud Day", f"Day {daily.loc[daily['fraud_count'].idxmax(), 'day']}")
        st.metric("Total Fraud Amount", f"₹{daily['fraud_amount'].sum():,.0f}")

# ── Model Summary ─────────────────────────────────────────────────────────────
if metrics and "models" in metrics:
    st.markdown("---")
    st.markdown("### 🤖 Model Performance Summary (Test Set)")
    rows = []
    for mname, mdict in metrics["models"].items():
        if not isinstance(mdict, dict):
            continue
        rows.append({
            "Model": mname.replace("_", " ").title(),
            "PR-AUC": f"{mdict.get('pr_auc', 0):.4f}",
            "ROC-AUC": f"{mdict.get('roc_auc', 0):.4f}",
            "Recall": f"{mdict.get('recall', 0)*100:.1f}%",
            "Precision": f"{mdict.get('precision', 0)*100:.1f}%",
            "F1": f"{mdict.get('f1', 0):.4f}",
            "Alert Rate": f"{mdict.get('alert_rate_pct', 0):.2f}%",
            "Final?": "✅" if mname == model_name else "",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
