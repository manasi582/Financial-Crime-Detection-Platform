"""
app/pages/04_threshold_cost.py
Interactive threshold and cost simulator.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Threshold & Cost Sim — FraudLens", page_icon="💰", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.page-header{background:linear-gradient(135deg,#0f3460 0%,#16213e 100%);border-radius:12px;padding:20px 28px;margin-bottom:24px;border-left:4px solid #ffd166;}
.page-header h1{color:#e2e8f0;font-size:1.6rem;margin:0;font-weight:700;}
.page-header p{color:#8892b0;margin:4px 0 0;font-size:.9rem;}
.sim-card{background:linear-gradient(135deg,#1a1a2e,#16213e);border:1px solid #0f3460;border-radius:12px;padding:16px 20px;text-align:center;}
.sim-label{color:#8892b0;font-size:.75rem;text-transform:uppercase;letter-spacing:1px;}
.sim-value{color:#e2e8f0;font-size:1.8rem;font-weight:700;}
.sim-value.green{color:#43d9a2;}.sim-value.red{color:#e94560;}.sim-value.yellow{color:#ffd166;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h1>💰 Threshold & Cost Simulator</h1>
    <p>Adjust the decision threshold and investigation cost to optimize net savings in real-time</p>
</div>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_cost_curve():
    """Load pre-computed cost curve if available, else rebuild from test predictions."""
    # Try to load from DB
    try:
        from src.config import get_config
        from src.db import get_engine, query_test_predictions
        cfg = get_config()
        engine = get_engine(cfg)
        pred_df = query_test_predictions(engine)
        return pred_df
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_metrics():
    p = _ROOT / "reports" / "metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


metrics = load_metrics()
pred_df = load_cost_curve()

if pred_df is None or pred_df.empty:
    st.error("⚠️ No predictions found. Run `make all` first.")
    st.stop()

# Filter to final model predictions
final_model = metrics.get("final_model", "") if metrics else ""
if "model_name" in pred_df.columns and final_model:
    pred_df = pred_df[pred_df["model_name"] == final_model]

scores = pred_df["score"].values
y_true = pred_df["isFraud"].values
amounts = pred_df["amount"].values

opt_threshold = metrics.get("business", {}).get("opt_threshold", 0.5) if metrics else 0.5
baseline_loss = float(amounts[y_true == 1].sum())

# ── Sidebar sliders ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Simulator Controls")
    threshold = st.slider(
        "Decision Threshold",
        min_value=float(np.percentile(scores, 10)),
        max_value=float(np.percentile(scores, 99)),
        value=float(np.clip(opt_threshold, np.percentile(scores, 10), np.percentile(scores, 99))),
        step=0.005,
        format="%.3f",
        help="Transactions with score ≥ threshold are flagged as fraud",
    )
    inv_cost = st.slider(
        "Investigation Cost per Alert (Rs.)",
        min_value=100, max_value=5000, value=500, step=100,
        help="Cost to investigate one flagged transaction",
    )
    recovery_rate = st.slider(
        "Fraud Recovery Rate",
        min_value=0.0, max_value=1.0, value=1.0, step=0.05,
        format="%.2f",
        help="Fraction of flagged fraud amount that is recovered",
    )
    st.markdown("---")
    st.markdown(f"**Optimal threshold** (from validation): `{opt_threshold:.4f}`")
    if st.button("🎯 Reset to Optimal"):
        threshold = opt_threshold

# ── Compute live metrics ──────────────────────────────────────────────────────
from src.cost_analysis import compute_cost_curve

y_pred = (scores >= threshold).astype(int)
n_alerts = int(y_pred.sum())
tp_mask = (y_pred == 1) & (y_true == 1)
fn_mask = (y_pred == 0) & (y_true == 1)

prevented = float(amounts[tp_mask].sum() * recovery_rate)
missed = float(amounts[fn_mask].sum())
invest = n_alerts * inv_cost
net = prevented - invest

tp = int(tp_mask.sum())
fn = int(fn_mask.sum())
fp = int(((y_pred == 1) & (y_true == 0)).sum())

recall    = tp / (tp + fn + 1e-9)
precision = tp / (tp + fp + 1e-9) if n_alerts > 0 else 0
alert_rate = 100 * n_alerts / len(scores)

# ── Live KPI cards ────────────────────────────────────────────────────────────
st.markdown("### 📊 Live Metrics at Current Threshold")
c1, c2, c3, c4, c5, c6 = st.columns(6)

card_data = [
    (c1, "Recall", f"{recall*100:.1f}%", "green" if recall > 0.7 else "yellow"),
    (c2, "Precision", f"{precision*100:.1f}%", "green" if precision > 0.5 else "yellow"),
    (c3, "Alerts Raised", f"{n_alerts:,}", "yellow"),
    (c4, "Alert Rate", f"{alert_rate:.2f}%", "yellow"),
    (c5, "Prevented Loss", f"₹{prevented:,.0f}", "green"),
    (c6, "Net Savings", f"₹{net:,.0f}", "green" if net > 0 else "red"),
]

for col, label, value, cls in card_data:
    with col:
        st.markdown(f"""
        <div class="sim-card">
            <div class="sim-label">{label}</div>
            <div class="sim-value {cls}">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Cost curve chart ──────────────────────────────────────────────────────────
cost_curve = compute_cost_curve(
    y_true, scores, amounts,
    investigation_cost=inv_cost,
    recovery_rate=recovery_rate,
    n_thresholds=150,
)

col_chart, col_table = st.columns([3, 1])

with col_chart:
    st.markdown("### 📈 Net Savings vs. Threshold")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cost_curve["threshold"], y=cost_curve["net_savings"],
        name="Net Savings (Rs.)", fill="tozeroy",
        fillcolor="rgba(67, 217, 162, 0.15)",
        line=dict(color="#43d9a2", width=2.5),
        mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=cost_curve["threshold"], y=cost_curve["prevented_loss"],
        name="Prevented Loss", line=dict(color="#74b9ff", width=2, dash="dot"),
        mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=cost_curve["threshold"], y=cost_curve["investigation_cost_total"],
        name="Investigation Cost", line=dict(color="#e94560", width=2, dash="dot"),
        mode="lines",
    ))
    # Mark current threshold
    closest_idx = (cost_curve["threshold"] - threshold).abs().idxmin()
    crow = cost_curve.loc[closest_idx]
    fig.add_vline(
        x=threshold, line_dash="dash", line_color="#ffd166",
        annotation_text=f"Current: {threshold:.3f}",
        annotation_font_color="#ffd166",
    )
    fig.add_scatter(
        x=[crow["threshold"]], y=[crow["net_savings"]],
        mode="markers", marker=dict(color="#ffd166", size=12, symbol="diamond"),
        name=f"Net Savings: ₹{crow['net_savings']:,.0f}",
    )
    fig.update_layout(
        paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
        font=dict(color="#e2e8f0"),
        xaxis=dict(title="Threshold", gridcolor="#1a2744"),
        yaxis=dict(title="Amount (Rs.)", gridcolor="#1a2744"),
        legend=dict(bgcolor="#16213e", bordercolor="#0f3460"),
        height=420, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

with col_table:
    st.markdown("### 📋 Cost Sensitivity")
    sens_rows = []
    for cost in [100, 250, 500, 750, 1000, 2000, 5000]:
        invest_c = n_alerts * cost
        net_c = prevented - invest_c
        sens_rows.append({
            "Cost/Alert": f"₹{cost}",
            "Net Savings": f"₹{net_c:,.0f}",
            "ROI": f"{net_c/(invest_c+1e-9):.2f}x",
        })
    st.dataframe(pd.DataFrame(sens_rows), use_container_width=True, hide_index=True)

    st.markdown(f"""
    **Baseline (no model):**  
    ₹{baseline_loss:,.0f} lost to fraud

    **Current savings:**  
    ₹{net:,.0f} net

    **Fraud missed:**  
    ₹{missed:,.0f}
    """)

# ── Precision-Recall tradeoff ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ⚖️ Precision vs. Recall Tradeoff")
fig2 = px.line(
    cost_curve, x="recall", y="precision",
    color_discrete_sequence=["#74b9ff"],
    labels={"recall": "Recall", "precision": "Precision"},
)
fig2.add_scatter(
    x=[recall], y=[precision],
    mode="markers", marker=dict(color="#ffd166", size=12, symbol="diamond"),
    name=f"Current (F1={2*precision*recall/(precision+recall+1e-9):.3f})",
)
fig2.update_layout(
    paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
    font=dict(color="#e2e8f0"),
    xaxis=dict(gridcolor="#1a2744"), yaxis=dict(gridcolor="#1a2744"),
    height=350,
)
st.plotly_chart(fig2, use_container_width=True)
