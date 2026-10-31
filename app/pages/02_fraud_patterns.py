"""
app/pages/02_fraud_patterns.py
Fraud pattern analysis from SQL views with sidebar filters.
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

st.set_page_config(page_title="Fraud Patterns — FraudLens", page_icon="🕵️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.page-header{background:linear-gradient(135deg,#0f3460 0%,#16213e 100%);border-radius:12px;padding:20px 28px;margin-bottom:24px;border-left:4px solid #ffd166;}
.page-header h1{color:#e2e8f0;font-size:1.6rem;margin:0;font-weight:700;}
.page-header p{color:#8892b0;margin:4px 0 0;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h1>🕵️ Fraud Patterns Analysis</h1>
    <p>Fraud by transaction type, hour, amount bucket, and destination velocity — powered by SQL views</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_engine():
    try:
        from src.config import get_config
        from src.db import get_engine as _ge
        return _ge(get_config())
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_views():
    engine = get_engine()
    if engine is None:
        return {}, None
    from src.db import query_view, query_top_dest_accounts
    views = {}
    for v in ["fraud_by_type", "fraud_by_hour_of_day", "fraud_by_amount_bucket", "daily_fraud_trend"]:
        try:
            views[v] = query_view(v, engine)
        except Exception as e:
            views[v] = pd.DataFrame()
    try:
        top_dest = query_top_dest_accounts(engine, limit=15)
    except Exception:
        top_dest = pd.DataFrame()
    return views, top_dest


engine = get_engine()
if engine is None:
    st.error("⚠️ Database not found. Run `make all` first.")
    st.stop()

views, top_dest = load_views()

COLORS = {
    "CASH_OUT": "#e94560", "PAYMENT": "#74b9ff",
    "CASH_IN": "#43d9a2", "TRANSFER": "#ffd166", "DEBIT": "#a29bfe",
}

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Filters")
    show_rate = st.toggle("Show fraud rate (%)", value=False)
    show_all_types = st.toggle("Show all transaction types", value=True)

# ── Row 1: By Type + By Hour ──────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📌 Fraud by Transaction Type")
    df_type = views.get("fraud_by_type", pd.DataFrame())
    if not df_type.empty:
        metric = "fraud_rate_pct" if show_rate else "fraud_amount"
        label = "Fraud Rate (%)" if show_rate else "Fraud Amount (Rs.)"
        fig = px.bar(
            df_type, x="type", y=metric, text=metric,
            color="type",
            color_discrete_map=COLORS,
            labels={"type": "Transaction Type", metric: label},
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(
            paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
            font=dict(color="#e2e8f0"), showlegend=False,
            height=360, xaxis=dict(gridcolor="#1a2744"),
            yaxis=dict(gridcolor="#1a2744"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_type, use_container_width=True, hide_index=True)

with col2:
    st.markdown("#### 🕐 Fraud Rate by Hour of Day")
    df_hour = views.get("fraud_by_hour_of_day", pd.DataFrame())
    if not df_hour.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_hour["hour_of_day"], y=df_hour["fraud_count"],
            marker_color=["#e94560" if h < 6 else "#74b9ff" for h in df_hour["hour_of_day"]],
            name="Fraud Count",
        ))
        fig.add_trace(go.Scatter(
            x=df_hour["hour_of_day"], y=df_hour["fraud_rate_pct"],
            name="Fraud Rate (%)", yaxis="y2",
            line=dict(color="#ffd166", width=2.5),
            mode="lines+markers", marker=dict(size=5),
        ))
        fig.update_layout(
            paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
            font=dict(color="#e2e8f0"),
            xaxis=dict(title="Hour of Day", gridcolor="#1a2744", tickvals=list(range(24))),
            yaxis=dict(title="Fraud Count", gridcolor="#1a2744"),
            yaxis2=dict(title="Fraud Rate (%)", overlaying="y", side="right"),
            height=360,
            shapes=[dict(
                type="rect", x0=-0.5, x1=5.5,
                y0=0, y1=1, yref="paper",
                fillcolor="rgba(233,69,96,0.1)", line_width=0,
            )],
            annotations=[dict(
                x=2.5, y=0.95, xref="x", yref="paper",
                text="🌙 Night hours", showarrow=False,
                font=dict(color="#e94560", size=10),
            )],
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Row 2: By Amount Bucket + Top Destinations ────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.markdown("#### 💰 Fraud by Amount Bucket")
    df_amt = views.get("fraud_by_amount_bucket", pd.DataFrame())
    if not df_amt.empty:
        metric = "fraud_rate_pct" if show_rate else "fraud_amount"
        label = "Fraud Rate (%)" if show_rate else "Fraud Amount (Rs.)"
        fig = px.bar(
            df_amt, x="amount_bucket", y=metric,
            color="fraud_rate_pct",
            color_continuous_scale="RdYlGn_r",
            labels={"amount_bucket": "Amount Range", metric: label},
            text=metric,
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(
            paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
            font=dict(color="#e2e8f0"), coloraxis_showscale=False,
            height=360, xaxis=dict(gridcolor="#1a2744"),
            yaxis=dict(gridcolor="#1a2744"),
        )
        st.plotly_chart(fig, use_container_width=True)

with col4:
    st.markdown("#### 🎯 Top Destination Accounts by Fraud Amount")
    if top_dest is not None and not top_dest.empty:
        fig = px.bar(
            top_dest.head(10), x="total_fraud_amount", y="nameDest",
            orientation="h",
            color="total_fraud_amount",
            color_continuous_scale="Reds",
            labels={"nameDest": "Destination Account", "total_fraud_amount": "Fraud Amount (Rs.)"},
            text="fraud_txn_count",
        )
        fig.update_traces(texttemplate='%{text} txns', textposition='outside')
        fig.update_layout(
            paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
            font=dict(color="#e2e8f0"), coloraxis_showscale=False,
            yaxis=dict(autorange="reversed", gridcolor="#1a2744"),
            xaxis=dict(gridcolor="#1a2744"),
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Night vs Day comparison ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🌙 Night vs. Day Fraud Comparison")
df_hour = views.get("fraud_by_hour_of_day", pd.DataFrame())
if not df_hour.empty:
    night = df_hour[df_hour["hour_of_day"] < 6]
    day   = df_hour[df_hour["hour_of_day"] >= 6]
    col_n, col_d, col_insight = st.columns([1, 1, 2])
    with col_n:
        st.metric("🌙 Night Avg Fraud Rate", f"{night['fraud_rate_pct'].mean():.3f}%",
                  help="Hours 00:00-05:59")
    with col_d:
        st.metric("☀️ Day Avg Fraud Rate", f"{day['fraud_rate_pct'].mean():.3f}%",
                  help="Hours 06:00-23:59")
    with col_insight:
        ratio = night["fraud_rate_pct"].mean() / (day["fraud_rate_pct"].mean() + 1e-9)
        st.info(f"💡 **Night-time fraud rate is {ratio:.1f}x higher** than daytime. "
                f"This pattern is intentionally embedded in the data generator and is "
                f"a known PaySim characteristic.")
