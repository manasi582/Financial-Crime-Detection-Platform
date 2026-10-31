"""
src/cost_analysis.py

Cost-based business analysis for FraudLens.

Business parameters (from config.yaml):
- investigation_cost_per_alert: Rs. 500 (cost per alert investigated)
- fraud_recovery_rate_when_caught: 1.0 (100% recovery)
- currency_symbol: "Rs."

Key outputs:
- Loss with no model (all fraud amounts lost)
- Net savings = (recovered fraud amount) - (n_alerts * cost_per_alert)
  for each threshold on a sweep
- Optimal threshold: chosen on VALIDATION data by maximizing net savings,
  then applied to test
- Sensitivity table: net savings as investigation_cost varies
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path


def compute_cost_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    investigation_cost: float = 500.0,
    recovery_rate: float = 1.0,
    n_thresholds: int = 200,
) -> pd.DataFrame:
    """
    Sweep thresholds and compute business metrics at each.

    Returns DataFrame with columns:
        threshold, n_alerts, alert_rate_pct, recall, precision,
        prevented_loss, investigation_cost_total, net_savings
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    rows = []
    total_fraud_amount = amounts[y_true == 1].sum()

    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        n_alerts = int(y_pred.sum())
        # True positives: fraud caught
        tp_mask = (y_pred == 1) & (y_true == 1)
        prevented = float(amounts[tp_mask].sum() * recovery_rate)
        invest_cost = n_alerts * investigation_cost
        net = prevented - invest_cost

        tp = tp_mask.sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()

        recall = float(tp / (tp + fn + 1e-9))
        precision = float(tp / (tp + fp + 1e-9)) if n_alerts > 0 else 0.0

        rows.append({
            "threshold":              round(float(t), 6),
            "n_alerts":               n_alerts,
            "alert_rate_pct":         round(100.0 * n_alerts / len(y_true), 4),
            "recall":                 round(recall, 6),
            "precision":              round(precision, 6),
            "prevented_loss":         round(prevented, 2),
            "investigation_cost_total": round(invest_cost, 2),
            "net_savings":            round(net, 2),
        })

    df = pd.DataFrame(rows)
    df["baseline_loss"] = round(float(total_fraud_amount), 2)
    return df


def find_optimal_threshold(cost_curve_df: pd.DataFrame) -> float:
    """Return the threshold that maximizes net savings."""
    idx = cost_curve_df["net_savings"].idxmax()
    return float(cost_curve_df.loc[idx, "threshold"])


def sensitivity_table(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    optimal_threshold: float,
    cost_range: list[float] | None = None,
    recovery_rate: float = 1.0,
) -> pd.DataFrame:
    """
    Show how net savings change as investigation_cost varies,
    at the fixed optimal threshold.
    """
    if cost_range is None:
        cost_range = [100, 250, 500, 750, 1000, 2000, 5000]

    y_pred = (scores >= optimal_threshold).astype(int)
    n_alerts = y_pred.sum()
    tp_mask = (y_pred == 1) & (y_true == 1)
    prevented = float(amounts[tp_mask].sum() * recovery_rate)

    rows = []
    for cost in cost_range:
        invest = n_alerts * cost
        net = prevented - invest
        rows.append({
            "investigation_cost_per_alert": cost,
            "n_alerts": int(n_alerts),
            "prevented_loss": round(prevented, 2),
            "investigation_cost_total": round(float(invest), 2),
            "net_savings": round(net, 2),
            "roi": round(net / (invest + 1e-9), 4),
        })

    return pd.DataFrame(rows)


def generate_result_statement(
    cost_curve_df: pd.DataFrame,
    optimal_threshold: float,
    total_txns: int,
    currency: str = "Rs.",
) -> str:
    """Generate the human-readable result statement from real numbers."""
    row = cost_curve_df[
        cost_curve_df["threshold"] == cost_curve_df["threshold"].iloc[
            (cost_curve_df["threshold"] - optimal_threshold).abs().argmin()
        ]
    ].iloc[0]

    return (
        f"At threshold {optimal_threshold:.4f}, the model catches "
        f"{row['recall']*100:.1f}% of fraud while flagging only "
        f"{row['alert_rate_pct']:.2f}% of transactions, preventing "
        f"{currency}{row['prevented_loss']:,.0f} in losses at an "
        f"investigation cost of {currency}{row['investigation_cost_total']:,.0f} "
        f"(net saving {currency}{row['net_savings']:,.0f})."
    )
