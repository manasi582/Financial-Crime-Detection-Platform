# FraudLens: Transaction Fraud Detection and Risk Analytics Platform

> **Portfolio project** by a final-year Computer Engineering student applying for  
> KPMG Analyst — Risk, Forensic & Financial Crime Analytics

---

## Business Problem

A payments company wants to catch fraudulent transactions without overwhelming its investigators.
The central question is **not** "What is the accuracy?" but:

> *"How much fraud loss can we prevent per unit of investigation cost, and at what alert volume?"*

A model that flags every transaction achieves 100% recall — but also generates 500,000 alerts per day,
which no team can investigate. FraudLens frames the problem as a **cost-benefit optimization**:

```
Net Savings = (Prevented Fraud Amount × Recovery Rate) − (N Alerts × Investigation Cost)
```

All modeling decisions flow from this objective.

---

## Quick Start

```bash
git clone <repo-url>
cd fraud_detection

# 1. Create virtual environment
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline (generates data, trains models, exports)
make all           # ~5 minutes on a laptop

# 4. Launch dashboard
make app           # Opens at http://localhost:8501

# 5. Run tests
make test
```

---

## Results (from `reports/metrics.json` — all numbers from actual pipeline runs)

| Metric | Value |
|--------|-------|
| **Final Model** | HistGradientBoosting |
| **Test PR-AUC** | **0.8197** |
| **Test ROC-AUC** | **0.9957** |
| **Optimal Threshold** | **0.9545** (cost-maximising, chosen on validation) |
| **Recall at optimal** | **79.6%** (74 of 93 fraud caught on test set) |
| **Alert Rate at optimal** | **0.13%** (101 of 76,955 transactions flagged) |
| **Precision at optimal** | **73.3%** (27 false positives out of 101 alerts) |
| **Net Savings (test set)** | **Rs. 8,949,748** (baseline loss Rs. 9,170,032) |
| **LR val PR-AUC** | 0.7656 |
| **RF val PR-AUC** | 0.8772 |
| **HGB val PR-AUC** | 0.8804 ← selected |
| **Ablation: balance-error features** | **+0.1473 PR-AUC** (0.8591 → 0.7118 without them) |
| **Best imbalance method** | random_undersample (val 0.8875 > SMOTE 0.8701 > class_weight 0.8591) |

> **Business result (verbatim from pipeline):**  
> *"At threshold 0.9545, the model catches 79.6% of fraud while flagging only 0.13% of  
> transactions, preventing Rs.9,000,248 in losses at an investigation cost of Rs.50,500  
> (net saving Rs.8,949,748)."*

### Why PR-AUC, not Accuracy?

Fraud rate ≈ 0.13%. A model that labels every transaction "legitimate" achieves 99.87% accuracy
— and catches zero fraud. PR-AUC measures model quality on the minority (fraud) class across
all decision thresholds. It answers: *"Among the top N% I flag, what fraction are actually fraud?"*
This is operationally meaningful for investigators.

---

## Architecture

```
config.yaml
    │
    ├── src/data_generator.py   ──► 500K synthetic PaySim-schema rows
    │                                  Fraud only in TRANSFER & CASH_OUT
    │                                  4 embedded fraud patterns
    │
    ├── src/features.py ─────────► Balance-error features, velocity (anti-leakage)
    │
    ├── src/split.py ────────────► Strict time-based split (70/15/15)
    │                                  No shuffle, train < val < test guaranteed
    │
    ├── src/train.py ────────────► LR, RF, HGB, Isolation Forest
    │                                  Imbalance: class_weight vs SMOTE vs undersample
    │                                  Hyperparameter tuning on validation PR-AUC
    │
    ├── src/evaluate.py ─────────► PR/ROC curves, calibration, ablation study
    │
    ├── src/cost_analysis.py ────► Net savings curve, sensitivity table
    │
    ├── src/explain.py ──────────► Permutation importance, SHAP-style reason codes
    │
    ├── src/export_powerbi.py ───► Dimensional CSV exports (fact + dims)
    │
    ├── src/db.py ───────────────► SQLite via SQLAlchemy
    │                                  4 tables, 4 analytical views
    │
    └── app/                     ──► 7-page Streamlit dashboard
        ├── 01_executive_overview.py
        ├── 02_fraud_patterns.py
        ├── 03_model_performance.py
        ├── 04_threshold_cost.py
        ├── 05_investigator_queue.py
        ├── 06_live_scorer.py
        └── 07_about.py
```

---

## Data

### Synthetic Mode (default — works offline)

- **500,000 transactions** over 720 time steps (30 simulated days)
- **PaySim schema**: `step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud`
- **Type mix**: CASH_OUT 35%, PAYMENT 34%, CASH_IN 22%, TRANSFER 8%, DEBIT 1%
- **Fraud rate**: ~0.13%, only in TRANSFER and CASH_OUT

#### Embedded Fraud Patterns

| Pattern | Signal | Feature |
|---------|--------|---------|
| Account Takeover (full drain) | Amount ≈ origin balance → origin zeroed | `orig_balance_zeroed`, `errorBalanceOrig` |
| Mule Chain | Destination balance unchanged after receiving funds | `errorBalanceDest` |
| Night-time Activity | Hours 00–05 have elevated fraud probability | `is_night`, `hour_of_day` |
| Large Amounts | Top 10% of eligible-type amounts flagged with 8× prob | `amount`, `amount_to_balance_ratio` |

Realistic 8% balance noise + 40% of fraud uses partial drains (not full) to prevent trivial separation.

### Real PaySim Mode

1. Download: https://www.kaggle.com/datasets/ntnu-testimon/paysim1
2. Place at `data/raw/PS_20174392719_1491204439457_log.csv`
3. Set `data.mode: REAL_PAYSIM` in `config.yaml`

---

## Feature Engineering

| Feature | Description | Leakage Risk | Mitigation |
|---------|------------|-------------|------------|
| `errorBalanceOrig` | `newBal + amount - oldBal` for origin | None | — |
| `errorBalanceDest` | `oldBal + amount - newBal` for dest | None | — |
| `orig_balance_zeroed` | 1 if origin new balance = 0 | None | — |
| `dest_balance_was_zero` | 1 if destination had zero balance | None | — |
| `amount_to_balance_ratio` | `amount / (oldBal + 1)` | None | — |
| `is_night` | Hour 00–05 | None | — |
| `hour_of_day` | 0–23 | None | — |
| `dest_velocity_24h` | # times this dest received $ in past 24 steps | **HIGH** | Strict past-only sliding window |
| `type_*` | One-hot of transaction type | None | — |

### Anti-Leakage Measures

1. **No future data in velocity**: Uses a sorted scan that only counts steps strictly before the current row's step
2. **No `isFlaggedFraud`**: Dropped before any computation (label-derived feature in PaySim)
3. **No account identifiers** in model features: `nameOrig`, `nameDest` dropped from model inputs
4. **No test-set resampling**: SMOTE and undersampling applied only to training data
5. **Threshold selected on validation, reported once on test**: Prevents overfitting to test set
6. **Time-based split enforced**: `assert train.step.max() < val.step.min() < test.step.min()`

---

## Models Compared

| Model | Type | Imbalance Handling |
|-------|------|--------------------|
| Logistic Regression | Supervised, linear | `class_weight='balanced'` |
| Random Forest | Supervised, ensemble | `class_weight='balanced'` |
| HistGradientBoosting | Supervised, boosting | `class_weight='balanced'` |
| Isolation Forest | Unsupervised, anomaly | N/A (no labels used) |

Imbalance methods also compared: `class_weight` vs SMOTE vs random undersampling (on training data only).

---

## Dashboard Pages

| Page | Purpose |
|------|---------|
| **Executive Overview** | KPI cards, daily trend, fraud by type — for leadership |
| **Fraud Patterns** | Hour-of-day heatmap, amount distribution, SQL analytics |
| **Model Performance** | PR/ROC curves, calibration, confusion matrices, ablation |
| **Threshold & Cost Sim** | Interactive net savings curve, sensitivity table |
| **Investigator Queue** | Alert triage with DB-persisted status updates |
| **Live Scorer** | Enter any transaction → instant fraud score + reason codes |
| **About & Methodology** | Problem framing, leakage safeguards, limitations |

---

## Database

SQLite (zero-config). Schema: 4 tables, 4 analytical views, indexes.

```sql
-- Tables
transactions    -- 500K rows + all features + split label
predictions     -- model scores on test set
alerts          -- top alerts with reason codes + mutable status
model_runs      -- experiment tracking

-- Views
fraud_by_type        -- fraud count/rate/amount by transaction type
fraud_by_hour_of_day -- hourly fraud pattern (24 rows)
fraud_by_amount_bucket -- fraud by amount quartile
daily_fraud_trend    -- day-by-day counts
```

Change to PostgreSQL by editing one line in `config.yaml`:
```yaml
database:
  url: "postgresql://user:pass@localhost:5432/fraudlens"
```

---

## Project Structure

```
fraud_detection/
├── app/                    # Streamlit dashboard (7 pages)
├── data/
│   ├── raw/                # Place real PaySim CSV here (optional)
│   └── processed/          # Auto-generated
├── db/                     # SQLite database (auto-generated)
├── exports/
│   └── powerbi/            # Dimensional CSVs for Power BI
├── models/                 # Trained model joblib files
├── reports/
│   ├── figures/            # PR/ROC/confusion matrix PNGs
│   └── metrics.json        # All evaluation metrics (real numbers)
├── sql/                    # schema.sql, views.sql, analysis_queries.sql
├── src/                    # Core Python modules
├── tests/                  # pytest test suite
├── config.yaml             # Single source of truth for all parameters
├── Makefile                # make all | make app | make test | make clean
├── requirements.txt
├── README.md               # This file
├── INTERVIEW_NOTES.md      # 15 KPMG interview Q&As + resume bullets
└── POWERBI_GUIDE.md        # Data model, DAX measures, 3-page report layout
```

---

## Make Targets

```bash
make all        # Generate data → train models → evaluate → export Power BI CSVs
make app        # Launch Streamlit dashboard
make test       # Run pytest (6 test classes, 18 test cases)
make clean      # Remove DB, models, reports (keeps code)
make lint       # Run flake8 (requires flake8)
```

---

## Design Decisions (noted here per brief)

1. **SQLite not PostgreSQL**: Zero-config for portfolio reproducibility. One-line migration to Postgres.
2. **Synthetic data by default**: Works offline without Kaggle credentials. Real PaySim supported via config switch.
3. **PR-AUC as primary metric**: Accuracy is misleading at 0.13% fraud rate. PR-AUC directly measures quality on the minority class.
4. **Cost-based threshold**: Threshold chosen to maximize `prevented_loss - investigation_cost`, not F1. This aligns with business reality.
5. **`class_weight='balanced'` as baseline**: Simpler and more interpretable than SMOTE; SMOTE comparison included in output.
6. **SQLite bulk insert via raw `sqlite3.executemany`**: Bypasses SQLAlchemy's named-parameter limit (SQLITE_MAX_VARIABLE_NUMBER), enabling fast 500K-row loads.
7. **Destination velocity uses past-only sliding window**: Critical to prevent future data leakage in velocity feature.

---

## Known Limitations

1. **Synthetic data is optimistic**: PaySim fraud has deterministic balance patterns. Real payment fraud has more noise.
2. **No concept drift**: Model trained once; production needs continuous retraining.
3. **No streaming inference**: Batch pipeline; production would use Kafka/Flink.
4. **No graph features**: Transaction network topology is a strong real-world signal, not modeled here.
5. **8% noise may not fully represent real data complexity**: Real balance inconsistencies arise from fees, FX, reversals.

---

## Running Tests

```bash
make test
# or
pytest tests/ -v
```

Test suite covers:
- Generator schema, fraud-type constraint, fraud rate, unique IDs
- Time split — no overlap, strict temporal ordering, all splits have fraud
- Velocity anti-leakage (past-only window)
- Cost function hand-computed correctness
- DB round trip (load → query view → update alert status)
- Model load and score (if trained model exists)
- Reason code generation for known fraud patterns

---

## Power BI

See [POWERBI_GUIDE.md](POWERBI_GUIDE.md) for:
- Data model and relationships
- 10 DAX measures (including dynamic net savings with What-If parameter)
- 3-page report layout with visual specifications
- Color theme

CSVs are in `exports/powerbi/` after `make all`.

---

## Interview Preparation

See [INTERVIEW_NOTES.md](INTERVIEW_NOTES.md) for:
- 15 likely KPMG interview questions with detailed answers
- 3 resume bullet templates (fill in real numbers from `reports/metrics.json`)

---

*Built end-to-end from scratch. All numbers in `reports/metrics.json` come from real pipeline runs.*
