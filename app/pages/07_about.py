"""
app/pages/07_about.py
About & Methodology page.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

st.set_page_config(page_title="About & Methodology — FraudLens", page_icon="📖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.page-header{background:linear-gradient(135deg,#0f3460 0%,#16213e 100%);border-radius:12px;padding:20px 28px;margin-bottom:24px;border-left:4px solid #74b9ff;}
.page-header h1{color:#e2e8f0;font-size:1.6rem;margin:0;font-weight:700;}
.page-header p{color:#8892b0;margin:4px 0 0;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h1>📖 About & Methodology</h1>
    <p>Problem framing, data description, design decisions, leakage safeguards, and limitations</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
## 🏦 Business Problem

A payments company wants to catch fraudulent transactions without overwhelming its investigators.  
The core question is: **"How much fraud loss can we prevent per unit of investigation cost, and at what alert volume?"**

Standard accuracy (~99.8% on this dataset) is **meaningless** — a model that predicts all transactions as 
legitimate achieves this. We use:
- **PR-AUC** (Precision-Recall area): the primary metric for imbalanced classification
- **Recall at fixed alert rates**: reflects operational constraints
- **Net Monetary Savings**: the business bottom line

---

## 📊 Data Description

### Synthetic Mode (default)
- **500,000 transactions** over 720 steps (30 days × 24 hours)
- **PaySim schema**: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud
- **Type mix**: CASH_OUT (35%), PAYMENT (34%), CASH_IN (22%), TRANSFER (8%), DEBIT (1%)
- **Fraud rate**: ~0.15–0.3%, **only in TRANSFER and CASH_OUT**

#### Fraud Patterns Embedded:
| Pattern | Description |
|---------|-------------|
| Account Takeover | Amount ≈ full origin balance; balance drained to zero |
| Mule Chain | Transfer to mule; destination balance left unchanged |
| Night Activity | Elevated fraud probability in hours 00–05 |
| Large Amounts | Top 5% of amounts in eligible types have higher fraud probability |

### Real PaySim Mode
Download: https://www.kaggle.com/datasets/ntnu-testimon/paysim1  
Place at `data/raw/PS_20174392719_1491204439457_log.csv`  
Then set `data.mode: REAL_PAYSIM` in `config.yaml`

---

## 🔧 Architecture

```
config.yaml
    │
    ├── data_generator.py ──► SQLite DB (via SQLAlchemy)
    │                              │
    ├── features.py ───────────────┤
    │                              │
    ├── split.py (time-based) ─────┤
    │                              │
    ├── train.py ──────────────────┤──► models/*.joblib
    │    └── imbalance handling    │
    │    └── hyperparameter tuning │
    │                              │
    ├── evaluate.py ───────────────┤──► reports/metrics.json
    │    └── PR/ROC/CM/calibration │    reports/figures/
    │                              │
    ├── cost_analysis.py ──────────┤──► Net savings analysis
    │                              │
    ├── explain.py ────────────────┤──► Feature importance + reason codes
    │                              │
    └── export_powerbi.py ─────────┴──► exports/powerbi/*.csv
```

---

## ⚠️ Leakage Safeguards

| Safeguard | Implementation |
|-----------|---------------|
| Time-based split | Train = first 70% of steps; never shuffle |
| No future data in velocity | Computed using strict past-step window |
| No isFlaggedFraud | Dropped before any feature computation |
| No identifiers in model | nameOrig, nameDest dropped from model inputs |
| Validation-only tuning | Threshold chosen on validation, reported once on test |
| No test resampling | SMOTE/undersampling applied only to training data |

---

## ⚖️ Feature Engineering Notes

**Balance-error features** (`errorBalanceOrig`, `errorBalanceDest`) are very predictive in this dataset 
because PaySim/synthetic data generates fraud with **deterministic balance rules**.

In real production data:
- Balances change for many reasons (fees, timing lag, float, reversals)
- These features would be far less discriminative
- The ablation study quantifies this dependency

See the ablation results on the **Model Performance** page.

---

## 🚧 Known Limitations

1. **Synthetic data** — PaySim is a simulation. Real payment fraud is more complex.
2. **No concept drift** — The model does not adapt to evolving fraud patterns.
3. **No real-time streaming** — This is a batch pipeline; production would use Kafka/Flink.
4. **Balance features are optimistic** — They inflate performance metrics vs. real data.
5. **Single payment rail** — Real systems have multiple payment types and processors.
6. **No graph features** — Social network / graph-based features (who sends to whom) are highly effective but not included here.
7. **Calibration** — Scores are not perfectly calibrated; apply isotonic regression before using as probabilities in production.

---

## 🔮 Future Work

- **Graph neural networks** for transaction network analysis
- **Online learning** for concept drift adaptation
- **Kafka + Flink** for real-time streaming inference
- **Model monitoring** (PSI, KS-test, feature drift)
- **Federated learning** across banks without sharing raw data
- **Explainable AI** with LIME/SHAP for regulatory compliance (SR 11-7)
- **Multi-model ensemble** with dynamic weighting

---

## 👤 Author

Final-year Computer Engineering student  
Applying for KPMG Analyst — Risk, Forensic & Financial Crime Analytics

*This project demonstrates: business framing, SQL analytics, ML engineering, 
cost-based optimization, and production-ready code organization.*
""")
