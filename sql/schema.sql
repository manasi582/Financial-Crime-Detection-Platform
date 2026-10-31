-- ─────────────────────────────────────────────────────────────
-- FraudLens Schema (SQLite / PostgreSQL compatible)
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS transactions (
    txn_id          INTEGER PRIMARY KEY,
    step            INTEGER NOT NULL,
    type            TEXT    NOT NULL,
    amount          REAL    NOT NULL,
    nameOrig        TEXT,
    oldbalanceOrg   REAL,
    newbalanceOrig  REAL,
    nameDest        TEXT,
    oldbalanceDest  REAL,
    newbalanceDest  REAL,
    isFraud         INTEGER NOT NULL DEFAULT 0,
    hour_of_day     INTEGER,
    day             INTEGER,
    is_night        INTEGER,
    errorBalanceOrig  REAL,
    errorBalanceDest  REAL,
    amount_to_balance_ratio REAL,
    orig_balance_zeroed     INTEGER,
    dest_balance_was_zero   INTEGER,
    dest_velocity_24h       REAL,
    type_CASH_IN    INTEGER DEFAULT 0,
    type_CASH_OUT   INTEGER DEFAULT 0,
    type_DEBIT      INTEGER DEFAULT 0,
    type_PAYMENT    INTEGER DEFAULT 0,
    type_TRANSFER   INTEGER DEFAULT 0,
    split           TEXT    -- train / val / test
);

CREATE TABLE IF NOT EXISTS predictions (
    pred_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id      INTEGER NOT NULL REFERENCES transactions(txn_id),
    model_name  TEXT    NOT NULL,
    score       REAL    NOT NULL,
    flagged     INTEGER NOT NULL DEFAULT 0,
    split       TEXT,
    created_at  TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id      INTEGER NOT NULL REFERENCES transactions(txn_id),
    score       REAL    NOT NULL,
    reason_codes TEXT,   -- JSON array of strings
    status      TEXT    NOT NULL DEFAULT 'Open',  -- Open | Confirmed Fraud | False Positive
    updated_at  TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name  TEXT NOT NULL,
    params      TEXT,   -- JSON
    metrics     TEXT,   -- JSON
    created_at  TEXT DEFAULT (datetime('now'))
);

-- ── Indexes ───────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_txn_step    ON transactions(step);
CREATE INDEX IF NOT EXISTS idx_txn_type    ON transactions(type);
CREATE INDEX IF NOT EXISTS idx_txn_fraud   ON transactions(isFraud);
CREATE INDEX IF NOT EXISTS idx_txn_split   ON transactions(split);
CREATE INDEX IF NOT EXISTS idx_pred_txn    ON predictions(txn_id);
CREATE INDEX IF NOT EXISTS idx_pred_model  ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_alert_txn   ON alerts(txn_id);
CREATE INDEX IF NOT EXISTS idx_alert_status ON alerts(status);
