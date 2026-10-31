-- ─────────────────────────────────────────────────────────────
-- FraudLens SQL Views (analytical layer used by the dashboard)
-- ─────────────────────────────────────────────────────────────

-- 1. Fraud counts and amounts grouped by transaction type
DROP VIEW IF EXISTS fraud_by_type;
CREATE VIEW fraud_by_type AS
SELECT
    type,
    COUNT(*)                                    AS total_txns,
    SUM(isFraud)                                AS fraud_count,
    ROUND(100.0 * SUM(isFraud) / COUNT(*), 4)   AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN isFraud=1 THEN amount ELSE 0 END), 2) AS fraud_amount
FROM transactions
GROUP BY type
ORDER BY fraud_count DESC;

-- 2. Fraud pattern by hour of day
DROP VIEW IF EXISTS fraud_by_hour_of_day;
CREATE VIEW fraud_by_hour_of_day AS
SELECT
    hour_of_day,
    COUNT(*)                                    AS total_txns,
    SUM(isFraud)                                AS fraud_count,
    ROUND(100.0 * SUM(isFraud) / COUNT(*), 4)   AS fraud_rate_pct,
    ROUND(AVG(CASE WHEN isFraud=1 THEN amount END), 2) AS avg_fraud_amount
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- 3. Fraud distribution across amount buckets
DROP VIEW IF EXISTS fraud_by_amount_bucket;
CREATE VIEW fraud_by_amount_bucket AS
SELECT
    CASE
        WHEN amount < 1000      THEN '< 1K'
        WHEN amount < 10000     THEN '1K - 10K'
        WHEN amount < 50000     THEN '10K - 50K'
        WHEN amount < 100000    THEN '50K - 100K'
        WHEN amount < 500000    THEN '100K - 500K'
        ELSE                         '500K+'
    END                                         AS amount_bucket,
    COUNT(*)                                    AS total_txns,
    SUM(isFraud)                                AS fraud_count,
    ROUND(100.0 * SUM(isFraud) / COUNT(*), 4)   AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN isFraud=1 THEN amount ELSE 0 END), 2) AS fraud_amount
FROM transactions
GROUP BY amount_bucket
ORDER BY
    CASE amount_bucket
        WHEN '< 1K'        THEN 1
        WHEN '1K - 10K'    THEN 2
        WHEN '10K - 50K'   THEN 3
        WHEN '50K - 100K'  THEN 4
        WHEN '100K - 500K' THEN 5
        ELSE 6
    END;

-- 4. Daily fraud trend (day = step // 24)
DROP VIEW IF EXISTS daily_fraud_trend;
CREATE VIEW daily_fraud_trend AS
SELECT
    day,
    COUNT(*)                                    AS total_txns,
    SUM(isFraud)                                AS fraud_count,
    ROUND(100.0 * SUM(isFraud) / COUNT(*), 4)   AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN isFraud=1 THEN amount ELSE 0 END), 2) AS fraud_amount,
    ROUND(SUM(amount), 2)                       AS total_amount
FROM transactions
GROUP BY day
ORDER BY day;
