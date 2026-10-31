"""
app/pages/05_investigator_queue.py
Investigator queue: alerts with scores and reason codes.
Status updates persist to the database.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Investigator Queue — FraudLens", page_icon="🚨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.page-header{background:linear-gradient(135deg,#0f3460 0%,#16213e 100%);border-radius:12px;padding:20px 28px;margin-bottom:24px;border-left:4px solid #e94560;}
.page-header h1{color:#e2e8f0;font-size:1.6rem;margin:0;font-weight:700;}
.page-header p{color:#8892b0;margin:4px 0 0;font-size:.9rem;}
.alert-card{background:#16213e;border:1px solid #0f3460;border-radius:10px;padding:14px 18px;margin:8px 0;}
.score-bar{height:6px;border-radius:3px;margin-top:8px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h1>🚨 Investigator Queue</h1>
    <p>Top fraud alerts with risk scores and reason codes — mark outcomes to update the database</p>
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


@st.cache_data(ttl=60)
def load_alerts(limit=100):
    engine = get_engine()
    if engine is None:
        return None
    from src.db import query_top_alerts
    return query_top_alerts(engine, limit=limit)


engine = get_engine()
if engine is None:
    st.error("⚠️ Database not found. Run `make all` first.")
    st.stop()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Queue Filters")
    status_filter = st.multiselect(
        "Show status",
        ["Open", "Confirmed Fraud", "False Positive"],
        default=["Open"],
    )
    score_min = st.slider("Minimum score", 0.0, 1.0, 0.5, 0.01)
    n_display = st.slider("Alerts to display", 10, 100, 25)

alerts_df = load_alerts(limit=200)
if alerts_df is None or alerts_df.empty:
    st.warning("No alerts found. Run `make all` to generate alerts.")
    st.stop()

# Apply filters
filtered = alerts_df[
    (alerts_df["status"].isin(status_filter)) &
    (alerts_df["score"] >= score_min)
].head(n_display)

# ── Summary stats ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Alerts", len(alerts_df))
col2.metric("Open", int((alerts_df["status"] == "Open").sum()))
col3.metric("Confirmed Fraud", int((alerts_df["status"] == "Confirmed Fraud").sum()))
col4.metric("False Positive", int((alerts_df["status"] == "False Positive").sum()))

st.markdown("---")
st.markdown(f"### 📋 Alert Queue ({len(filtered)} shown)")

if filtered.empty:
    st.info("No alerts match current filters.")
    st.stop()

# ── Alert cards ───────────────────────────────────────────────────────────────
from src.db import update_alert_status

for _, row in filtered.iterrows():
    alert_id = int(row["alert_id"])
    txn_id   = int(row["txn_id"])
    score    = float(row["score"])
    status   = str(row["status"])
    txn_type = str(row.get("type", "N/A"))
    amount   = float(row.get("amount", 0))
    step     = int(row.get("step", 0))
    hour     = int(row.get("hour_of_day", 0))
    actual_fraud = int(row.get("isFraud", 0))

    try:
        reasons = json.loads(row.get("reason_codes", "[]"))
    except Exception:
        reasons = []

    color = "#e94560" if score > 0.8 else "#ffd166" if score > 0.5 else "#43d9a2"
    status_icon = {"Open": "🔴", "Confirmed Fraud": "✅", "False Positive": "⚫"}.get(status, "❓")

    with st.expander(
        f"{status_icon} Alert #{alert_id} | Score: {score:.3f} | ₹{amount:,.0f} | {txn_type} | Step {step} (h{hour:02d})",
        expanded=score > 0.85,
    ):
        col_detail, col_action = st.columns([3, 1])
        with col_detail:
            st.markdown(f"""
            | Field | Value |
            |-------|-------|
            | Transaction ID | `{txn_id}` |
            | Type | **{txn_type}** |
            | Amount | **₹{amount:,.2f}** |
            | Fraud Score | **{score:.4f}** |
            | Step / Hour | {step} / {hour:02d}:00 |
            | Actual Label | {"🚨 FRAUD" if actual_fraud else "✅ LEGIT"} |
            | Current Status | **{status}** |
            """)
            if reasons:
                st.markdown("**🚩 Risk Signals:**")
                for r in reasons:
                    st.markdown(f"  - {r}")
            else:
                st.markdown("*No specific risk signals detected.*")

        with col_action:
            st.markdown("**Update Status**")
            new_status = st.selectbox(
                "Decision",
                ["Open", "Confirmed Fraud", "False Positive"],
                index=["Open", "Confirmed Fraud", "False Positive"].index(status),
                key=f"status_{alert_id}",
            )
            if st.button("💾 Save", key=f"save_{alert_id}"):
                try:
                    update_alert_status(alert_id, new_status, engine)
                    st.success(f"Updated → {new_status}")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Outcome summary ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Investigation Outcome Summary")
try:
    from sqlalchemy import text
    with engine.connect() as conn:
        outcome_df = pd.read_sql("""
        SELECT a.status,
               COUNT(*) AS alert_count,
               ROUND(AVG(a.score), 4) AS avg_score,
               ROUND(SUM(t.amount), 2) AS total_amount
        FROM alerts a
        JOIN transactions t ON a.txn_id = t.txn_id
        GROUP BY a.status
        """, conn)
    st.dataframe(outcome_df, use_container_width=True, hide_index=True)
except Exception as e:
    st.warning(f"Could not load outcome summary: {e}")
