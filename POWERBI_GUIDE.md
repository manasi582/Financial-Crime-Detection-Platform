# Power BI Report Guide — FraudLens

## Overview
This guide helps you build a professional Power BI report from the exported CSVs.
All CSVs are in `exports/powerbi/`. Run `make all` to generate them.

---

## 1. Data Model

### Tables and Relationships

```
fact_transactions (txn_id [PK], step, type, amount, isFraud, score, flagged, split, ...)
       │
       ├──[type]──────────── dim_type (type [PK], risk_level, description)
       │
       └──[step]──────────── dim_time (step [PK], hour_of_day, day, is_night, day_label)

model_metrics    (model_name, pr_auc, roc_auc, f1, precision, recall, ...)
threshold_cost_curve (threshold, n_alerts, alert_rate_pct, recall, precision, net_savings, ...)
```

### Relationships (set in Power BI Model view)
| From Table | From Column | To Table | To Column | Cardinality |
|-----------|------------|---------|----------|-------------|
| fact_transactions | type | dim_type | type | Many-to-One |
| fact_transactions | step | dim_time | step | Many-to-One |

---

## 2. DAX Measures

Copy-paste these DAX measures into your Power BI report:

```dax
// 1. Total Transactions
Total Transactions = COUNTROWS(fact_transactions)

// 2. Total Fraud Count
Fraud Count = CALCULATE(COUNTROWS(fact_transactions), fact_transactions[isFraud] = 1)

// 3. Fraud Rate (%)
Fraud Rate % = 
DIVIDE(
    CALCULATE(COUNTROWS(fact_transactions), fact_transactions[isFraud] = 1),
    COUNTROWS(fact_transactions),
    0
) * 100

// 4. Total Fraud Amount
Fraud Amount = 
CALCULATE(SUM(fact_transactions[amount]), fact_transactions[isFraud] = 1)

// 5. Model Recall (True Positive Rate)
Model Recall = 
VAR TruePositives = CALCULATE(COUNTROWS(fact_transactions), fact_transactions[flagged] = 1, fact_transactions[isFraud] = 1)
VAR FalseNegatives = CALCULATE(COUNTROWS(fact_transactions), fact_transactions[flagged] = 0, fact_transactions[isFraud] = 1)
RETURN DIVIDE(TruePositives, TruePositives + FalseNegatives, 0)

// 6. Model Precision
Model Precision = 
VAR TruePositives = CALCULATE(COUNTROWS(fact_transactions), fact_transactions[flagged] = 1, fact_transactions[isFraud] = 1)
VAR FalsePositives = CALCULATE(COUNTROWS(fact_transactions), fact_transactions[flagged] = 1, fact_transactions[isFraud] = 0)
RETURN DIVIDE(TruePositives, TruePositives + FalsePositives, 0)

// 7. Net Savings (using investigation cost = 500)
Net Savings = 
VAR PreventedLoss = CALCULATE(SUM(fact_transactions[amount]), fact_transactions[flagged] = 1, fact_transactions[isFraud] = 1)
VAR InvestigationCost = CALCULATE(COUNTROWS(fact_transactions), fact_transactions[flagged] = 1) * 500
RETURN PreventedLoss - InvestigationCost

// 8. Alert Rate (%)
Alert Rate % = 
DIVIDE(
    CALCULATE(COUNTROWS(fact_transactions), fact_transactions[flagged] = 1),
    COUNTROWS(fact_transactions),
    0
) * 100

// 9. Night Fraud Rate (%)
Night Fraud Rate % = 
CALCULATE(
    DIVIDE(
        CALCULATE(COUNTROWS(fact_transactions), fact_transactions[isFraud] = 1),
        COUNTROWS(fact_transactions),
        0
    ) * 100,
    dim_time[is_night] = 1
)

// 10. F1 Score
F1 Score = 
VAR P = [Model Precision]
VAR R = [Model Recall]
RETURN IF(P + R > 0, 2 * P * R / (P + R), 0)
```

---

## 3. Suggested 3-Page Report Layout

### Page 1: Executive Dashboard
**Purpose**: Leadership overview — one glance to understand fraud impact.

**Visuals**:
- **Row 1 (KPI Cards)**: Total Transactions | Fraud Count | Fraud Rate % | Fraud Amount | Net Savings | Model Recall
- **Row 2 (Main chart)**: Clustered bar + line chart — `daily_fraud_trend` by day (bars = fraud amount, line = fraud count)
- **Row 3 (Two cards)**: Donut chart of fraud by type (from `fraud_by_type`) | Gauge visual for Model Precision

**Filters**: Slicer for `split` (train/val/test) and `type`

---

### Page 2: Fraud Pattern Analytics
**Purpose**: Fraud intelligence for risk analysts.

**Visuals**:
- **Top-left**: Horizontal bar chart — Fraud amount by transaction type (from `dim_type` linked)
- **Top-right**: Line chart — Fraud rate by hour of day
- **Bottom-left**: Clustered bar — Fraud count by amount bucket (from `fact_transactions` bins)
- **Bottom-right**: Matrix/table — Top 10 destination accounts by fraud amount
- **Background annotation**: Highlight risk levels (High = red, Low = green) from `dim_type[risk_level]`

**Filters**: Date slicer on `day` | Type filter | Night/Day toggle

---

### Page 3: Model Analytics & Cost Optimization
**Purpose**: Model performance and business ROI for data science and risk teams.

**Visuals**:
- **Top row (Table)**: Model comparison from `model_metrics` — sortable by PR-AUC
- **Middle-left**: Line chart — `threshold_cost_curve` (X=threshold, Y=net_savings) with target line
- **Middle-right**: Scatter plot — Precision vs Recall from `threshold_cost_curve`
- **Bottom-left**: KPI cards — F1 Score | PR-AUC | ROC-AUC (from `model_metrics` for final model)
- **Bottom-right**: Sensitivity table — Net savings at different investigation costs

**Interactive element**: What-if parameter for `investigation_cost_per_alert` (100–5000)
linked to a DAX measure:
```dax
// Dynamic Net Savings (requires What-If parameter named "Investigation Cost")
Dynamic Net Savings = 
VAR CostPerAlert = 'Investigation Cost'[Investigation Cost Value]
VAR PreventedLoss = CALCULATE(SUM(fact_transactions[amount]), fact_transactions[flagged] = 1, fact_transactions[isFraud] = 1)
VAR TotalAlerts = CALCULATE(COUNTROWS(fact_transactions), fact_transactions[flagged] = 1)
RETURN PreventedLoss - TotalAlerts * CostPerAlert
```

---

## 4. Color Theme Recommendations

| Element | Hex Color |
|---------|-----------|
| Fraud (danger) | `#E94560` |
| Legitimate (safe) | `#43D9A2` |
| Warning/neutral | `#FFD166` |
| Info/model | `#74B9FF` |
| Background | `#0A0A1A` |
| Card background | `#16213E` |
| Text primary | `#E2E8F0` |
| Text secondary | `#8892B0` |

---

## 5. Tips for Publishing

1. Set the report theme to match the dark color palette above
2. Enable Row-Level Security if restricting investigators to their own alerts
3. Use Power BI Service scheduled refresh if the SQLite DB is replaced with SQL Server
4. Add a "Last Refreshed" card using `NOW()` DAX
5. For production: connect directly to PostgreSQL (change one line in `config.yaml`)
