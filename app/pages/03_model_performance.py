"""
app/pages/03_model_performance.py
Model performance: comparison table, curves, feature importance, ablation.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(page_title="Model Performance — FraudLens", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.page-header{background:linear-gradient(135deg,#0f3460 0%,#16213e 100%);border-radius:12px;padding:20px 28px;margin-bottom:24px;border-left:4px solid #43d9a2;}
.page-header h1{color:#e2e8f0;font-size:1.6rem;margin:0;font-weight:700;}
.page-header p{color:#8892b0;margin:4px 0 0;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h1>📊 Model Performance</h1>
    <p>Comparison table, PR/ROC curves, feature importance, imbalance handling, and ablation study</p>
</div>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_metrics():
    p = _ROOT / "reports" / "metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_feature_importance():
    p = _ROOT / "reports" / "feature_importance.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


metrics = load_metrics()
if metrics is None:
    st.error("⚠️ No metrics found. Run `make all` first.")
    st.stop()

final_model = metrics.get("final_model", "N/A")

# ── Model Comparison Table ────────────────────────────────────────────────────
st.markdown("### 🤖 Model Comparison (Test Set)")
rows = []
for mname, mdict in metrics.get("models", {}).items():
    if not isinstance(mdict, dict):
        continue
    rows.append({
        "Model": mname.replace("_", " ").title(),
        "PR-AUC ↑": round(mdict.get("pr_auc", 0), 4),
        "ROC-AUC ↑": round(mdict.get("roc_auc", 0), 4),
        "F1 ↑": round(mdict.get("f1", 0), 4),
        "Recall ↑": f"{mdict.get('recall', 0)*100:.1f}%",
        "Precision ↑": f"{mdict.get('precision', 0)*100:.1f}%",
        "Alert Rate": f"{mdict.get('alert_rate_pct', 0):.2f}%",
        "Recall@0.5%": f"{mdict.get('recall_at_5bp', 0)*100:.1f}%" if mdict.get('recall_at_5bp') is not None else "N/A",
        "Recall@1%": f"{mdict.get('recall_at_10bp', 0)*100:.1f}%" if mdict.get('recall_at_10bp') is not None else "N/A",
        "Final": "✅" if mname == final_model else "",
    })

if rows:
    df_comp = pd.DataFrame(rows)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

# ── PR / ROC Curve Images ─────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)

pr_roc_path = _ROOT / "reports" / "figures" / "pr_roc_test.png"
with col1:
    st.markdown("### 📈 PR & ROC Curves (Test Set)")
    if pr_roc_path.exists():
        st.image(str(pr_roc_path), use_container_width=True)
    else:
        st.info("Run `make all` to generate figures.")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm_path = _ROOT / "reports" / "figures" / f"confusion_matrix_{final_model}.png"
with col2:
    st.markdown(f"### 🔢 Confusion Matrix — {final_model.replace('_',' ').title()}")
    if cm_path.exists():
        st.image(str(cm_path), use_container_width=True)
    else:
        # Render from metrics
        mdict = metrics.get("models", {}).get(final_model, {})
        cm = mdict.get("confusion_matrix", [[0,0],[0,0]])
        cm_arr = np.array(cm)
        labels = [["TN", "FP"], ["FN", "TP"]]
        fig = go.Figure(data=go.Heatmap(
            z=cm_arr, colorscale="Blues",
            text=[[f"{labels[i][j]}: {cm_arr[i,j]:,}" for j in range(2)] for i in range(2)],
            texttemplate="%{text}",
        ))
        fig.update_layout(
            paper_bgcolor="#0a0a1a", font=dict(color="#e2e8f0"),
            xaxis=dict(ticktext=["Predicted Legit", "Predicted Fraud"], tickvals=[0, 1]),
            yaxis=dict(ticktext=["Actual Legit", "Actual Fraud"], tickvals=[0, 1]),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Feature Importance ────────────────────────────────────────────────────────
st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    st.markdown(f"### 🎯 Feature Importance — {final_model.replace('_',' ').title()}")
    fi_path = _ROOT / "reports" / "figures" / f"feature_importance_{final_model}.png"
    if fi_path.exists():
        st.image(str(fi_path), use_container_width=True)
    else:
        fi_df = load_feature_importance()
        if fi_df is not None:
            fig = px.bar(
                fi_df.head(15), x="importance", y="feature",
                orientation="h", color="importance",
                color_continuous_scale="RdYlGn",
                labels={"importance": fi_df["method"].iloc[0] if not fi_df.empty else "Importance"},
            )
            fig.update_layout(
                paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
                font=dict(color="#e2e8f0"), coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

with col4:
    st.markdown("### ⚖️ Imbalance Handling Comparison")
    imb = metrics.get("imbalance_comparison", {})
    if imb:
        imb_df = pd.DataFrame([
            {"Method": k.replace("_", " ").title(), "Val PR-AUC": round(v, 4) if v else "N/A"}
            for k, v in imb.items()
        ])
        st.dataframe(imb_df, use_container_width=True, hide_index=True)

        # Visualize
        valid_imb = {k: v for k, v in imb.items() if v is not None}
        if valid_imb:
            fig = px.bar(
                x=list(valid_imb.keys()), y=list(valid_imb.values()),
                labels={"x": "Imbalance Method", "y": "Validation PR-AUC"},
                color=list(valid_imb.values()),
                color_continuous_scale="Viridis",
            )
            fig.update_layout(
                paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
                font=dict(color="#e2e8f0"), coloraxis_showscale=False,
                xaxis=dict(gridcolor="#1a2744"), yaxis=dict(gridcolor="#1a2744"),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧪 Ablation: Balance-Error Features Impact")
    ablation = metrics.get("ablation", {})
    if ablation:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("With Balance Error", f"{ablation.get('with_balance_error', 0):.4f}")
        col_b.metric("Without Balance Error", f"{ablation.get('without_balance_error', 0):.4f}")
        col_c.metric("Delta (PR-AUC)", f"{ablation.get('delta', 0):+.4f}")
        st.info(
            "💡 **Note**: Balance-error features are highly predictive because PaySim/synthetic "
            "data is generated with deterministic balance rules. In real production data, "
            "balance errors occur due to timing, fees, rounding — making these features less "
            "discriminative. The ablation delta should be interpreted with this caveat."
        )

# ── Calibration ───────────────────────────────────────────────────────────────
cal_path = _ROOT / "reports" / "figures" / f"calibration_{final_model}.png"
if cal_path.exists():
    st.markdown("---")
    st.markdown("### 📐 Probability Calibration Check")
    col_cal, col_cal_info = st.columns([2, 1])
    with col_cal:
        st.image(str(cal_path), use_container_width=True)
    with col_cal_info:
        st.markdown("""
        **What this shows:**  
        How well the model's predicted probabilities match actual fraud rates.
        
        - Points on the diagonal = perfectly calibrated
        - Points above diagonal = under-confident
        - Points below diagonal = over-confident
        
        For fraud detection, mild over-confidence (scores too high) is preferable 
        to under-confidence — it catches more fraud at the cost of more false positives.
        """)
