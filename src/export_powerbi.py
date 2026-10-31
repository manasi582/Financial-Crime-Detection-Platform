"""
src/export_powerbi.py

Exports clean CSVs for Power BI consumption.

Files generated in exports/powerbi/:
- fact_transactions.csv    : transactions with model scores
- dim_type.csv             : transaction type dimension
- dim_time.csv             : time dimension (step, hour, day)
- model_metrics.csv        : model comparison table
- threshold_cost_curve.csv : net savings across thresholds
"""
from __future__ import annotations

import json
import pandas as pd
from pathlib import Path


EXPORT_DIR = Path("exports/powerbi")


def export_all(
    df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    model_metrics: dict,
    cost_curve_df: pd.DataFrame,
) -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── fact_transactions ─────────────────────────────────────────────────────
    fact = df[[
        "txn_id", "step", "type", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "isFraud", "hour_of_day", "day", "is_night",
        "errorBalanceOrig", "errorBalanceDest",
        "amount_to_balance_ratio", "orig_balance_zeroed", "dest_balance_was_zero",
        "dest_velocity_24h", "split",
    ]].copy()

    # Merge in model scores
    score_cols = predictions_df[["txn_id", "score", "flagged", "model_name"]].copy()
    fact = fact.merge(score_cols, on="txn_id", how="left")
    fact.to_csv(EXPORT_DIR / "fact_transactions.csv", index=False)
    print(f"[export] fact_transactions.csv ({len(fact):,} rows)")

    # ── dim_type ─────────────────────────────────────────────────────────────
    dim_type = pd.DataFrame({
        "type":        ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"],
        "risk_level":  ["High",     "Low",     "Low",     "High",     "Low"],
        "description": [
            "Cash withdrawal from an account",
            "Payment to a merchant",
            "Cash deposit into an account",
            "Fund transfer between accounts",
            "Debit card transaction",
        ],
    })
    dim_type.to_csv(EXPORT_DIR / "dim_type.csv", index=False)
    print(f"[export] dim_type.csv")

    # ── dim_time ─────────────────────────────────────────────────────────────
    steps = sorted(df["step"].unique())
    dim_time = pd.DataFrame({
        "step":       steps,
        "hour_of_day": [s % 24 for s in steps],
        "day":         [s // 24 for s in steps],
        "is_night":    [1 if s % 24 < 6 else 0 for s in steps],
        "day_label":   [f"Day {s // 24 + 1}" for s in steps],
    })
    dim_time.to_csv(EXPORT_DIR / "dim_time.csv", index=False)
    print(f"[export] dim_time.csv ({len(dim_time)} rows)")

    # ── model_metrics ─────────────────────────────────────────────────────────
    rows = []
    for model_name, metrics in model_metrics.items():
        if not isinstance(metrics, dict) or "_scores" in metrics:
            continue
        row = {"model_name": model_name}
        row.update({k: v for k, v in metrics.items()
                    if isinstance(v, (int, float, str)) and not k.startswith("_")})
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(EXPORT_DIR / "model_metrics.csv", index=False)
        print(f"[export] model_metrics.csv")

    # ── threshold_cost_curve ─────────────────────────────────────────────────
    cost_curve_df.drop(columns=["_scores", "_y_true"], errors="ignore").to_csv(
        EXPORT_DIR / "threshold_cost_curve.csv", index=False
    )
    print(f"[export] threshold_cost_curve.csv ({len(cost_curve_df)} rows)")

    print(f"\n[export] All Power BI CSVs saved to {EXPORT_DIR}/")
