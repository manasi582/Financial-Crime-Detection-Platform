"""
app/pages/06_live_scorer.py
Live transaction scorer: enter transaction details, get fraud score and reason codes.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Live Scorer — FraudLens", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.page-header{background:linear-gradient(135deg,#0f3460 0%,#16213e 100%);border-radius:12px;padding:20px 28px;margin-bottom:24px;border-left:4px solid #a29bfe;}
.page-header h1{color:#e2e8f0;font-size:1.6rem;margin:0;font-weight:700;}
.page-header p{color:#8892b0;margin:4px 0 0;font-size:.9rem;}
.result-card{border-radius:14px;padding:24px 28px;margin:16px 0;text-align:center;}
.result-card.fraud{background:linear-gradient(135deg,#e94560,#c0392b);color:white;}
.result-card.legit{background:linear-gradient(135deg,#43d9a2,#27ae60);color:#0a0a0a;}
.result-card h2{font-size:2.5rem;margin:0;}
.result-card p{font-size:1.1rem;opacity:0.9;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h1>⚡ Live Transaction Scorer</h1>
    <p>Enter transaction details to get an instant fraud risk score and reason codes</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_config():
    try:
        from src.config import get_config
        from src.train import load_model
        import json
        cfg = get_config()
        final_model = cfg["modeling"]["final_model"]
        model_path = _ROOT / cfg["paths"]["model_dir"] / f"{final_model}.joblib"
        if not model_path.exists():
            return None, None, None, None

        model = load_model(model_path)

        metrics_path = _ROOT / "reports" / "metrics.json"
        threshold = 0.5
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            threshold = m.get("business", {}).get("opt_threshold", 0.5)

        return model, cfg, final_model, threshold
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None, None, None


model, cfg, final_model, threshold = load_model_and_config()

if model is None:
    st.error("⚠️ Model not found. Run `make all` to train the model first.")
    st.stop()

st.success(f"✅ Model loaded: **{final_model.replace('_', ' ').title()}** | Decision threshold: **{threshold:.4f}**")

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("### 📝 Enter Transaction Details")

with st.form("transaction_form", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        txn_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        amount = st.number_input("Amount (Rs.)", min_value=1.0, max_value=1_000_000.0,
                                  value=50000.0, step=1000.0)
        step = st.number_input("Step (hour since start)", min_value=0, max_value=720,
                                value=3, help="0-719, where 0-5 = first night")

    with col2:
        old_bal_orig = st.number_input("Origin Old Balance", min_value=0.0, max_value=10_000_000.0,
                                        value=50000.0, step=1000.0)
        new_bal_orig = st.number_input("Origin New Balance", min_value=0.0, max_value=10_000_000.0,
                                        value=0.0, step=1000.0,
                                        help="Should be oldBalance - amount for legit transactions")

    with col3:
        old_bal_dest = st.number_input("Dest Old Balance", min_value=0.0, max_value=10_000_000.0,
                                        value=1000.0, step=1000.0)
        new_bal_dest = st.number_input("Dest New Balance", min_value=0.0, max_value=10_000_000.0,
                                        value=1000.0, step=1000.0,
                                        help="If left unchanged after credit = mule pattern")
        dest_velocity = st.number_input("Dest velocity (last 24h)", min_value=0, max_value=100,
                                         value=0, help="# of transactions received by destination in last 24 steps")

    submitted = st.form_submit_button("🔍 Score Transaction", use_container_width=True)

if submitted:
    # Build feature dict
    hour_of_day = int(step) % 24
    is_night = 1 if hour_of_day < 6 else 0

    error_orig = round(new_bal_orig + amount - old_bal_orig, 4)
    error_dest = round(old_bal_dest + amount - new_bal_dest, 4)
    amt_ratio  = round(amount / (old_bal_orig + 1.0), 6)
    orig_zeroed = 1 if new_bal_orig == 0.0 else 0
    dest_zero   = 1 if old_bal_dest == 0.0 else 0

    feature_row = {
        "hour_of_day": hour_of_day,
        "day": int(step) // 24,
        "is_night": is_night,
        "errorBalanceOrig": error_orig,
        "errorBalanceDest": error_dest,
        "amount_to_balance_ratio": amt_ratio,
        "orig_balance_zeroed": orig_zeroed,
        "dest_balance_was_zero": dest_zero,
        "dest_velocity_24h": float(dest_velocity),
        "type_CASH_IN":   1 if txn_type == "CASH_IN"   else 0,
        "type_CASH_OUT":  1 if txn_type == "CASH_OUT"  else 0,
        "type_DEBIT":     1 if txn_type == "DEBIT"     else 0,
        "type_PAYMENT":   1 if txn_type == "PAYMENT"   else 0,
        "type_TRANSFER":  1 if txn_type == "TRANSFER"  else 0,
        "amount": float(amount),
    }

    from src.features import FEATURE_COLS
    X = np.array([[feature_row[c] for c in FEATURE_COLS]])

    try:
        score = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        st.error(f"Scoring error: {e}")
        st.stop()

    decision = "FRAUD" if score >= threshold else "LEGITIMATE"
    is_fraud_pred = score >= threshold

    # ── Result display ────────────────────────────────────────────────────────
    st.markdown("---")
    col_res, col_gauge, col_codes = st.columns([2, 1, 2])

    with col_res:
        if is_fraud_pred:
            st.markdown(f"""
            <div class="result-card fraud">
                <h2>🚨 FRAUD ALERT</h2>
                <p>Score: <strong>{score:.4f}</strong> ≥ threshold {threshold:.4f}</p>
                <p>This transaction would be flagged for investigation.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card legit">
                <h2>✅ LEGITIMATE</h2>
                <p>Score: <strong>{score:.4f}</strong> < threshold {threshold:.4f}</p>
                <p>This transaction passes the fraud check.</p>
            </div>
            """, unsafe_allow_html=True)

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Fraud Risk Score", "font": {"color": "#e2e8f0", "size": 14}},
            number={"suffix": "%", "font": {"color": "#e2e8f0", "size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8892b0"},
                "bar": {"color": "#e94560" if is_fraud_pred else "#43d9a2", "thickness": 0.25},
                "bgcolor": "#16213e",
                "bordercolor": "#0f3460",
                "steps": [
                    {"range": [0, 30], "color": "#1a2a44"},
                    {"range": [30, 60], "color": "#1a3a44"},
                    {"range": [60, 100], "color": "#2a1a2a"},
                ],
                "threshold": {
                    "line": {"color": "#ffd166", "width": 3},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="#0a0a1a",
            font=dict(color="#e2e8f0"),
            height=280,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_codes:
        from src.explain import get_reason_codes
        reasons = get_reason_codes(feature_row)
        st.markdown("**🚩 Risk Signals Detected:**")
        if reasons:
            for r in reasons:
                st.markdown(f"  🔴 {r}")
        else:
            st.markdown("  ✅ No risk signals detected")

        st.markdown("---")
        st.markdown("**📊 Feature Values:**")
        feat_display = {
            "Error Bal Orig": f"{error_orig:.2f}",
            "Error Bal Dest": f"{error_dest:.2f}",
            "Amount/Balance": f"{amt_ratio:.3f}",
            "Orig Zeroed": "Yes" if orig_zeroed else "No",
            "Night Time": "Yes" if is_night else "No",
            "Dest Velocity": str(dest_velocity),
        }
        for k, v in feat_display.items():
            st.markdown(f"  - **{k}**: {v}")

# ── Example transactions ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 💡 Try These Examples")
examples = {
    "Account Takeover (Full drain)": {
        "type": "TRANSFER", "amount": 45000, "step": 2,
        "old_orig": 45000, "new_orig": 0,
        "old_dest": 500, "new_dest": 500, "velocity": 3
    },
    "Normal Payment": {
        "type": "PAYMENT", "amount": 1500, "step": 14,
        "old_orig": 25000, "new_orig": 23500,
        "old_dest": 0, "new_dest": 0, "velocity": 0
    },
    "High-Value CASH_OUT at Night": {
        "type": "CASH_OUT", "amount": 200000, "step": 3,
        "old_orig": 200000, "new_orig": 0,
        "old_dest": 1000, "new_dest": 1000, "velocity": 8
    },
}

cols = st.columns(len(examples))
for col, (name, ex) in zip(cols, examples.items()):
    with col:
        st.info(f"**{name}**\n\n"
                f"Type: {ex['type']}\n\n"
                f"Amount: ₹{ex['amount']:,}\n\n"
                f"Hour: {ex['step'] % 24:02d}:00")
