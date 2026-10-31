"""
app/main.py — FraudLens Streamlit Dashboard Entry Point

Multi-page Streamlit application for fraud detection analytics.
Pages are defined in app/pages/.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

st.set_page_config(
    page_title="FraudLens — Fraud Detection Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}
.kpi-title { color: #8892b0; font-size: 0.75rem; font-weight: 500; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
.kpi-value { color: #e2e8f0; font-size: 2rem; font-weight: 700; line-height: 1.2; }
.kpi-value.danger  { color: #fc5c7d; }
.kpi-value.success { color: #43d9a2; }
.kpi-value.warning { color: #ffd166; }
.kpi-value.info    { color: #74b9ff; }
.kpi-delta { color: #8892b0; font-size: 0.8rem; margin-top: 4px; }

/* Page header */
.page-header {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 24px;
    border-left: 4px solid #e94560;
}
.page-header h1 { color: #e2e8f0; font-size: 1.6rem; margin: 0; font-weight: 700; }
.page-header p  { color: #8892b0; margin: 4px 0 0; font-size: 0.9rem; }

/* Alert badge */
.badge-open     { background:#e94560; color:white; padding:2px 8px; border-radius:99px; font-size:0.75rem; font-weight:600; }
.badge-confirmed{ background:#43d9a2; color:#0a0a0a; padding:2px 8px; border-radius:99px; font-size:0.75rem; font-weight:600; }
.badge-fp       { background:#636e72; color:white; padding:2px 8px; border-radius:99px; font-size:0.75rem; font-weight:600; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 FraudLens")
    st.markdown("*Transaction Fraud Detection*")
    st.markdown("---")
    st.markdown("### Navigation")

    pages = {
        "🏠 Executive Overview":      "pages/01_executive_overview.py",
        "🕵️ Fraud Patterns":          "pages/02_fraud_patterns.py",
        "📊 Model Performance":        "pages/03_model_performance.py",
        "💰 Threshold & Cost Sim":    "pages/04_threshold_cost.py",
        "🚨 Investigator Queue":       "pages/05_investigator_queue.py",
        "⚡ Live Transaction Scorer":  "pages/06_live_scorer.py",
        "📖 About & Methodology":     "pages/07_about.py",
    }

    st.markdown("""
    > Navigate using the **Pages** sidebar below ↓
    
    ---
    **Data**: Synthetic PaySim  
    **Model**: HistGradientBoosting  
    **Metric**: PR-AUC  
    ---
    *Financial Crime & Risk Analytics Platform*
    """)

# ── Landing page content ──────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>🔍 FraudLens — Transaction Fraud Detection Platform</h1>
    <p>Real-time fraud intelligence for payments risk and forensic analytics teams</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### Welcome to FraudLens

Use the **sidebar navigation** (Pages menu) to explore:

| Page | Description |
|------|-------------|
| 🏠 Executive Overview | KPI cards, daily fraud trend |
| 🕵️ Fraud Patterns | Fraud by type, hour, amount bucket |
| 📊 Model Performance | PR/ROC curves, feature importance |
| 💰 Threshold & Cost Simulator | Interactive business optimization |
| 🚨 Investigator Queue | Alert triage with reason codes |
| ⚡ Live Transaction Scorer | Real-time fraud scoring |
| 📖 About & Methodology | Design decisions, limitations |

---

> **First time?** Run `make all` to generate data, train models, and populate the database.  
> Then run `make app` to launch this dashboard.
""")
