"""
src/db.py

Database layer: SQLite via SQLAlchemy.
Handles schema creation, data loading (in chunks), view creation,
and all read queries used by the dashboard.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import get_config, root as project_root


def _resolve_db_url(cfg: dict) -> str:
    url = cfg["database"]["url"]
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        # Relative path — make absolute
        rel = url[len("sqlite:///"):]
        abs_path = project_root() / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{abs_path}"
    return url


def get_engine(cfg: dict | None = None):
    if cfg is None:
        cfg = get_config()
    url = _resolve_db_url(cfg)
    return create_engine(url, echo=False)


def create_schema(engine) -> None:
    """Execute schema.sql and views.sql against the database."""
    sql_dir = project_root() / "sql"
    with engine.begin() as conn:
        for sql_file in ["schema.sql", "views.sql"]:
            sql = (sql_dir / sql_file).read_text()
            # SQLAlchemy executes statement by statement
            for stmt in sql.split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(text(stmt))
    print("[db] Schema and views created.")


def _sqlite_bulk_insert(df: pd.DataFrame, table_name: str, engine,
                        if_exists: str = "append", chunk_size: int = 10_000) -> None:
    """
    Fast bulk insert using raw sqlite3 executemany with positional parameters.
    Avoids SQLAlchemy's named-parameter limit (SQLITE_MAX_VARIABLE_NUMBER).
    Uses PRAGMA synchronous=OFF for speed during initial load.
    """
    import sqlite3

    # Get raw sqlite3 connection from SQLAlchemy engine
    raw_conn = engine.raw_connection()
    cursor = raw_conn.cursor()

    # Speed pragmas (safe for bulk load)
    cursor.execute("PRAGMA synchronous=OFF")
    cursor.execute("PRAGMA journal_mode=MEMORY")

    cols = list(df.columns)
    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join([f'"{c}"' for c in cols])
    insert_sql = f'INSERT OR REPLACE INTO "{table_name}" ({col_names}) VALUES ({placeholders})'

    if if_exists == "replace":
        cursor.execute(f'DELETE FROM "{table_name}"')

    total = len(df)
    for start in range(0, total, chunk_size):
        batch = df.iloc[start : start + chunk_size]
        records = [tuple(row) for row in batch.itertuples(index=False, name=None)]
        cursor.executemany(insert_sql, records)
        raw_conn.commit()
        print(f"[db] Inserted rows {start} – {start + len(batch)} / {total}")

    # Restore safe pragmas
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA journal_mode=WAL")
    raw_conn.commit()
    raw_conn.close()


def load_transactions(df: pd.DataFrame, engine, chunk_size: int = 10_000) -> None:
    """Load transactions DataFrame into the DB using fast sqlite3 bulk insert."""
    total = len(df)
    _sqlite_bulk_insert(df, "transactions", engine, if_exists="replace", chunk_size=chunk_size)
    print(f"[db] transactions table loaded: {total:,} rows.")


def save_predictions(df: pd.DataFrame, engine) -> None:
    """Save predictions to the predictions table."""
    _sqlite_bulk_insert(df, "predictions", engine, if_exists="append", chunk_size=10_000)
    print(f"[db] {len(df):,} predictions saved.")


def save_alerts(alerts_df: pd.DataFrame, engine) -> None:
    """Save top-scored test alerts to the alerts table."""
    _sqlite_bulk_insert(alerts_df, "alerts", engine, if_exists="append", chunk_size=5_000)
    print(f"[db] {len(alerts_df):,} alerts saved.")


def save_model_run(model_name: str, params: dict, metrics: dict, engine) -> int:
    with engine.begin() as conn:
        result = conn.execute(
            text(
                "INSERT INTO model_runs (model_name, params, metrics, created_at) "
                "VALUES (:name, :params, :metrics, :ts)"
            ),
            {
                "name": model_name,
                "params": json.dumps(params),
                "metrics": json.dumps(metrics),
                "ts": datetime.utcnow().isoformat(),
            },
        )
        return result.lastrowid


# ── Dashboard read queries (plain SQL) ───────────────────────────────────────

def query_view(view_name: str, engine) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(f"SELECT * FROM {view_name}", conn)


def query_kpis(engine) -> dict:
    sql = """
    SELECT
        COUNT(*)                                          AS total_txns,
        SUM(isFraud)                                      AS fraud_count,
        ROUND(100.0 * SUM(isFraud) / COUNT(*), 4)         AS fraud_rate_pct,
        ROUND(SUM(CASE WHEN isFraud=1 THEN amount END), 2) AS fraud_amount
    FROM transactions
    """
    with engine.connect() as conn:
        row = conn.execute(text(sql)).fetchone()
    return dict(row._mapping)


def query_model_runs(engine) -> pd.DataFrame:
    sql = """
    SELECT model_name,
           json_extract(metrics, '$.pr_auc')    AS pr_auc,
           json_extract(metrics, '$.roc_auc')   AS roc_auc,
           json_extract(metrics, '$.f1')        AS f1,
           json_extract(metrics, '$.precision') AS precision_score,
           json_extract(metrics, '$.recall')    AS recall_score,
           json_extract(metrics, '$.net_savings') AS net_savings,
           created_at
    FROM model_runs
    ORDER BY CAST(json_extract(metrics, '$.pr_auc') AS REAL) DESC
    """
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def query_top_alerts(engine, limit: int = 100) -> pd.DataFrame:
    sql = """
    SELECT a.alert_id, a.txn_id, a.score, a.reason_codes, a.status, a.updated_at,
           t.type, t.amount, t.step, t.hour_of_day, t.isFraud
    FROM alerts a
    JOIN transactions t ON a.txn_id = t.txn_id
    ORDER BY a.score DESC
    LIMIT :limit
    """
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params={"limit": limit})


def update_alert_status(alert_id: int, new_status: str, engine) -> None:
    """Update an alert's status in the database (persisted)."""
    assert new_status in ("Open", "Confirmed Fraud", "False Positive")
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE alerts SET status=:status, updated_at=:ts WHERE alert_id=:aid"
            ),
            {"status": new_status, "ts": datetime.utcnow().isoformat(), "aid": alert_id},
        )


def query_test_predictions(engine) -> pd.DataFrame:
    sql = """
    SELECT p.txn_id, p.model_name, p.score, p.flagged, p.split,
           t.isFraud, t.amount, t.type, t.step, t.hour_of_day,
           t.errorBalanceOrig, t.errorBalanceDest, t.orig_balance_zeroed
    FROM predictions p
    JOIN transactions t ON p.txn_id = t.txn_id
    WHERE p.split = 'test'
    """
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def query_top_dest_accounts(engine, limit: int = 10) -> pd.DataFrame:
    sql = """
    SELECT nameDest,
           COUNT(*) AS fraud_txn_count,
           ROUND(SUM(amount), 2) AS total_fraud_amount
    FROM transactions
    WHERE isFraud = 1
    GROUP BY nameDest
    ORDER BY total_fraud_amount DESC
    LIMIT :limit
    """
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params={"limit": limit})
