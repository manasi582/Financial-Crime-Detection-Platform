# INTERVIEW_NOTES.md — FraudLens Project

## 15 Risk & Financial Crime Analytics Interview Questions & Answers

---

### Q1. Why did you use PR-AUC instead of accuracy as your primary metric?

**Answer:** Accuracy is misleading on heavily imbalanced datasets. With ~0.2% fraud rate, a model that labels every transaction as "legitimate" achieves 99.8% accuracy — yet it catches zero fraud. PR-AUC (area under the Precision-Recall curve) specifically measures model performance on the minority (fraud) class across all decision thresholds. It answers: "If I flag the top N% of transactions, how many are actual fraud?" This is operationally meaningful for investigators.

In institutional financial crime and forensic analytics work, the business cost of a missed fraud (false negative) vastly outweighs the inconvenience of a false alarm (false positive). PR-AUC captures this asymmetry; ROC-AUC does not, because it includes true negatives (correctly-labeled legitimate transactions) in its calculation.

---

### Q2. Why did you use a time-based split instead of random shuffle?

**Answer:** A random split allows models to "peek" at the future. If a fraud pattern emerges on Day 15, a random split might train on Day-15 fraud while testing on Day-14 data — inflating test performance. In production, a model trained on historical data is always evaluated on *future* transactions it has never seen.

The time-based split enforces strict temporal ordering: train on the first 70% of steps, validate on the next 15%, test on the final 15%. This simulates how the model will actually be deployed. I also assert that `max(train_steps) < min(val_steps) < min(test_steps)` to guarantee no overlap.

---

### Q3. How did you handle class imbalance? Which method worked best?

**Answer:** I compared three methods on the training set only — never on validation or test:

1. **`class_weight='balanced'`** — Tells the model to penalize fraud misclassification proportionally more. No data modification needed. Works well with gradient boosting.
2. **Random Under-Sampling** — Reduces the majority class by randomly removing legitimate transactions. Fast but loses information.
3. **SMOTE** — Generates synthetic minority (fraud) examples by interpolating between neighbors. Can introduce noise if the fraud decision boundary is irregular.

In my runs, `class_weight='balanced'` with HistGradientBoosting achieved the best validation PR-AUC. This is consistent with the literature: for tree-based ensemble models, reweighting is often sufficient and less prone to overfitting than synthetic oversampling.

**Key constraint:** SMOTE was applied only to training data. Applying it to validation/test would constitute data leakage and give falsely optimistic results.

---

### Q4. What are the biggest data leakage risks in this project?

**Answer:** I identified and mitigated several:

| Risk | Mitigation |
|------|-----------|
| Using `isFlaggedFraud` | Dropped before any feature computation — it's a label-derived feature |
| Identifier features (nameOrig, nameDest) | Dropped — they act as proxies for label in the training set |
| Balance-error features | Used, but documented caveat: they're deterministic in PaySim; see ablation |
| Destination velocity | Computed using strictly past steps via a sorted scan with expiry window |
| Resampling test/val data | Never done — only training data is resampled |
| Threshold selection | Chosen on validation, reported once on the untouched test set |

The most subtle risk is the balance-error features. In PaySim, fraud transactions have deterministic balance anomalies (e.g., destination balance doesn't update). The ablation study shows removing these features drops PR-AUC significantly — but in real data, this drop would be smaller because balance inconsistencies also occur in legitimate transactions (fees, timing lag).

---

### Q5. How did you choose the optimal decision threshold?

**Answer:** The threshold was NOT chosen to maximize F1. It was chosen by maximizing **net monetary savings on the validation set**:

```
Net Savings = (Recovered Fraud Amount) - (Number of Alerts × Investigation Cost)
```

At each threshold, I compute how much fraud we catch multiplied by the recovery rate, subtract the cost of investigating all flagged transactions, and find the threshold that maximizes this business objective. This threshold was then applied — without modification — to the test set.

This is the correct approach because: (1) it aligns the model with actual business value, not an arbitrary statistical metric; (2) it uses the validation set for selection and the test set only for reporting, preventing overfitting to the test set.

---

### Q6. Explain the cost analysis model. How does investigation cost affect the optimal threshold?

**Answer:** The cost model has two parameters: `investigation_cost_per_alert` (Rs. 500 default) and `fraud_recovery_rate_when_caught` (1.0 default, i.e., full recovery).

At a very low threshold, we flag most transactions: high recall but very high investigation cost that can exceed the fraud prevented. At a very high threshold, we flag very few: low cost but miss most fraud.

The net savings curve has an inverted-U shape — there's a sweet spot. As `investigation_cost_per_alert` increases, the optimal threshold rises (we become more selective) and net savings fall. The sensitivity table shows this explicitly: at Rs. 100/alert we can cast a wide net; at Rs. 5000/alert we must be very precise.

This framing is exactly what a forensic analytics team needs: "Should I hire more investigators or buy better fraud models?"

---

### Q7. What would you do differently in a production deployment?

**Answer:**
1. **Real-time scoring**: Replace batch pipeline with a Kafka consumer + Flink/Spark Streaming. The model would score transactions as they arrive, not in daily batches.
2. **Model monitoring**: Track PSI (Population Stability Index) and KS-test on score distributions daily. Alert when fraud patterns drift.
3. **Feature store**: Velocity features (dest_velocity_24h) would be served from a Redis/Druid real-time feature store rather than computed from a full scan.
4. **Model versioning**: Use MLflow for experiment tracking, model registry, and A/B testing between model versions.
5. **Explainability for regulators**: SHAP values at prediction time for SR 11-7 / GDPR compliance ("right to explanation").
6. **Calibration**: Apply isotonic regression calibration so that score = 0.3 genuinely means 30% fraud probability — important for risk-based case prioritization.
7. **Graph features**: Build a transaction network and compute PageRank, betweenness centrality for accounts — highly predictive of money mule networks.

---

### Q8. What is the difference between Isolation Forest and HistGradientBoosting for this problem?

**Answer:** Isolation Forest is an **unsupervised** anomaly detection method — it scores transactions by how easy they are to isolate in random feature splits. It does not use fraud labels during training, making it useful when labels are unavailable or unreliable. However, "anomalous" ≠ "fraudulent": a legitimate but unusual high-value corporate wire transfer will score as anomalous.

HistGradientBoosting is a **supervised** classifier that explicitly learns the decision boundary between fraud and legitimate transactions from labeled examples. It consistently outperforms Isolation Forest on PR-AUC when quality labels are available, which is the case here.

In practice, both can be combined: IF flags suspicious transactions for human labeling, and the labeled data trains a supervised model that then replaces IF in production.

---

### Q9. Your ablation shows balance-error features have a large impact. Is this a fair result?

**Answer:** Honest answer: **no, not entirely.** The balance-error features (`errorBalanceOrig`, `errorBalanceDest`) are very powerful in PaySim because the simulator generates fraud with deterministic balance rules — fraudulent CASH_OUTs always have an unchanged destination balance. In real payment systems:

- Balance updates can be delayed by hours (settlement lag)
- Fees and reversals create legitimate balance inconsistencies  
- The feature would be far weaker in production

I've documented this limitation explicitly in the ablation section and in the README. The ablation is valuable for showing the *direction* of feature importance, not as a prediction of production performance. A better test would be to evaluate on a held-out time period with deliberately noisy balance data.

---

### Q10. Why did you choose SQLite? How would you migrate to PostgreSQL?

**Answer:** SQLite is chosen because it requires zero infrastructure (no server, no credentials), making the project fully self-contained and reproducible on any laptop — ideal for a portfolio demo. All SQL is written to be ANSI-compatible.

Migrating to PostgreSQL requires **exactly one change**: update `database.url` in `config.yaml`:
```yaml
database:
  url: "postgresql://user:password@localhost:5432/fraudlens"
```

SQLAlchemy handles the rest. Views, queries, and all analytics code are unchanged. For production, I'd add connection pooling (`pool_size`, `max_overflow`) and use PostgreSQL's native JSONB type for the metrics column.

---

### Q11. What is PR-AUC vs ROC-AUC, and when do you use each?

**Answer:**
- **ROC-AUC** measures the probability that a random positive (fraud) is scored higher than a random negative (legitimate). It uses FPR (false positive rate) on the X-axis, which includes the vast majority class. On imbalanced data, the curve is deceptively optimistic because FPR stays small even with many false positives.
- **PR-AUC** uses Precision and Recall, focusing entirely on the minority class. A random classifier has PR-AUC ≈ fraud_rate (0.002), not 0.5, so it's a much harder benchmark.

**Rule of thumb**: When the negative class dominates (fraud detection, rare disease diagnosis, intrusion detection), use PR-AUC. When classes are balanced, both are appropriate.

---

### Q12. How does SMOTE work, and why might it fail for fraud detection?

**Answer:** SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic fraud examples by: (1) selecting a fraud transaction, (2) finding its k nearest fraud neighbors, (3) interpolating a new synthetic example along the line between them.

**Why it can fail for fraud:**
1. Fraud transactions may cluster in complex, non-convex regions — interpolation creates unrealistic examples in the "gap" between genuine fraud cases.
2. If fraud patterns are highly diverse (account takeover, mule chains, card-not-present fraud all look very different), SMOTE blends them into meaningless synthetic points.
3. In time-series data, synthetic transactions don't correspond to any real time step, which could leak information about future patterns into synthetic training examples.

In my experiments, SMOTE performed slightly worse than `class_weight='balanced'` — consistent with this reasoning.

---

### Q13. How would you explain a fraud alert to a non-technical investigator?

**Answer:** I'd use the reason codes generated by the rule-based explainer:

*"This transaction was flagged because: (1) the transfer amount equals the entire account balance — consistent with an account takeover draining the account; (2) the destination account balance did not increase after receiving funds — suggesting the money was immediately moved again or the destination is a shell account; (3) this occurred at 2:47 AM, when fraud rates are 3× higher than daytime. The model gives this a 94% fraud probability."*

The reason codes are deterministic and auditable — important for regulatory compliance and SAR (Suspicious Activity Report) filing. They give investigators a starting point for case narrative, which is core to financial crime forensics.

---

### Q14. What are the limitations of synthetic data for fraud model development?

**Answer:**
1. **Deterministic patterns**: PaySim generates fraud with fixed rules (balance zeroing), making some features unrealistically powerful.
2. **No concept drift**: Real fraud patterns evolve weekly as fraudsters adapt to detection. PaySim is static.
3. **No network effects**: PaySim doesn't model the social graph of who sends to whom — a key signal in real fraud.
4. **No velocity attacks**: Real fraud often involves bursts of micro-transactions under detection thresholds, which PaySim doesn't simulate.
5. **Geographic signals absent**: IP geolocation, device fingerprinting, and merchant category codes — all crucial in production — are not in PaySim.

The model trained on synthetic data will overperform relative to what you'd see on real payment data. The reported metrics should be interpreted as an upper bound.

---

### Q15. If you had 3 more weeks, what would you add?

**Answer:**
1. **Graph Neural Network layer**: Build an account-transaction bipartite graph and use GraphSAGE to learn embeddings that capture money mule networks.
2. **Online learning**: Implement a streaming update that re-trains on confirmed fraud cases daily using river or scikit-learn's `partial_fit`.
3. **Adversarial testing**: Simulate a fraudster who knows the model and crafts transactions just below the alert threshold — test the model's robustness.
4. **Real-world data integration**: Connect to a real synthetic dataset with realistic noise (bank statement CSVs) and compare performance drop vs. PaySim.
5. **Regulatory reporting module**: Auto-generate SAR-format reports from confirmed alerts, with SHAP explanations formatted for compliance teams.

---

## 3 Resume Bullets (real numbers from `make all`)

```
• Built FraudLens, an end-to-end fraud detection platform (500K synthetic 
  transactions, PaySim schema) using HistGradientBoosting; achieved 
  PR-AUC of 0.82 on a time-held-out test set, recovering Rs.8.95M in 
  simulated fraud losses while flagging only 0.13% of transactions for 
  investigation.

• Designed a cost-based threshold optimization framework that maximises 
  net monetary savings (prevented fraud − investigation costs); at the 
  optimal threshold (0.9545), the model achieves 79.6% recall at a 
  0.13% alert rate, preventing Rs.9.0M of Rs.9.17M total test-set 
  fraud exposure — a 97.5% loss recovery rate.

• Engineered a 7-page Streamlit analytics dashboard with live fraud scoring, 
  investigator queue with DB-persisted outcomes, SQL-powered pattern views, 
  and an interactive threshold-cost simulator; exported Power BI-compatible 
  dimensional model CSVs (500K rows, 5 tables, 10 DAX measures) for 
  executive reporting.
```
