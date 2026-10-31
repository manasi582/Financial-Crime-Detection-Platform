-- ─────────────────────────────────────────────────────────────
-- FraudLens Analytical Queries
-- These are written in plain SQL to demonstrate SQL proficiency.
-- Run these against the fraudlens.db database.
-- ─────────────────────────────────────────────────────────────

-- Q1: Overall fraud summary
SELECT
    COUNT(*)                                    AS total_transactions,
    SUM(isFraud)                                AS total_fraud,
    ROUND(100.0 * SUM(isFraud) / COUNT(*), 4)   AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN isFraud=1 THEN amount ELSE 0 END), 2) AS total_fraud_amount,
    ROUND(AVG(CASE WHEN isFraud=1 THEN amount END), 2)         AS avg_fraud_amount
FROM transactions;

-- Q2: Top 10 destination accounts by fraud amount received
SELECT
    nameDest,
    COUNT(*)                                    AS fraud_txn_count,
    ROUND(SUM(amount), 2)                       AS total_fraud_amount,
    ROUND(AVG(amount), 2)                       AS avg_fraud_amount
FROM transactions
WHERE isFraud = 1
GROUP BY nameDest
ORDER BY total_fraud_amount DESC
LIMIT 10;

-- Q3: Model performance summary by model
SELECT
    m.model_name,
    json_extract(m.metrics, '$.pr_auc')         AS pr_auc,
    json_extract(m.metrics, '$.roc_auc')        AS roc_auc,
    json_extract(m.metrics, '$.f1')             AS f1,
    json_extract(m.metrics, '$.precision')      AS precision_score,
    json_extract(m.metrics, '$.recall')         AS recall_score,
    m.created_at
FROM model_runs m
ORDER BY CAST(json_extract(m.metrics, '$.pr_auc') AS REAL) DESC;

-- Q4: Alert queue summary – open alerts with transaction details
SELECT
    a.alert_id,
    a.txn_id,
    t.type,
    t.amount,
    t.step,
    a.score,
    a.reason_codes,
    a.status,
    a.updated_at
FROM alerts a
JOIN transactions t ON a.txn_id = t.txn_id
WHERE a.status = 'Open'
ORDER BY a.score DESC
LIMIT 50;

-- Q5: Fraud detected vs missed by current model (test split)
SELECT
    p.flagged                                   AS model_flagged,
    t.isFraud                                   AS actual_fraud,
    COUNT(*)                                    AS count,
    ROUND(SUM(t.amount), 2)                     AS total_amount
FROM predictions p
JOIN transactions t ON p.txn_id = t.txn_id
WHERE p.split = 'test'
GROUP BY p.flagged, t.isFraud
ORDER BY p.flagged, t.isFraud;

-- Q6: Fraud by night vs day
SELECT
    CASE WHEN is_night = 1 THEN 'Night (00-05)' ELSE 'Day (06-23)' END AS time_period,
    COUNT(*)                                    AS total_txns,
    SUM(isFraud)                                AS fraud_count,
    ROUND(100.0 * SUM(isFraud) / COUNT(*), 4)   AS fraud_rate_pct
FROM transactions
GROUP BY is_night;

-- Q7: Daily net savings simulation (using default threshold predictions)
SELECT
    t.day,
    SUM(CASE WHEN t.isFraud=1 AND p.flagged=1 THEN t.amount ELSE 0 END) AS prevented_loss,
    COUNT(CASE WHEN p.flagged=1 THEN 1 END) * 500                        AS investigation_cost,
    SUM(CASE WHEN t.isFraud=1 AND p.flagged=1 THEN t.amount ELSE 0 END)
        - COUNT(CASE WHEN p.flagged=1 THEN 1 END) * 500                  AS net_savings
FROM transactions t
JOIN predictions p ON t.txn_id = p.txn_id
WHERE p.split = 'test'
GROUP BY t.day
ORDER BY t.day;

-- Q8: Confirmed vs False Positive investigation outcomes
SELECT
    a.status,
    COUNT(*)                                    AS alert_count,
    ROUND(AVG(a.score), 4)                      AS avg_score,
    ROUND(SUM(t.amount), 2)                     AS total_amount
FROM alerts a
JOIN transactions t ON a.txn_id = t.txn_id
GROUP BY a.status;
