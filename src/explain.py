"""
src/explain.py

Explainability for FraudLens.

Provides:
1. Global feature importance: SHAP (if available) or permutation importance
2. Per-transaction rule-based reason codes for investigators
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.inspection import permutation_importance

from src.features import FEATURE_COLS, get_feature_matrix


FIGURES_DIR = Path("reports/figures")


def compute_feature_importance(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Compute global feature importance using SHAP (preferred) or permutation
    importance (fallback). Returns DataFrame with feature and importance columns.
    """
    shap_available = False
    try:
        import shap
        shap_available = True
    except ImportError:
        pass

    if shap_available:
        try:
            print("[explain] Computing SHAP values...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val[:2000])  # sample for speed
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # class 1 (fraud)
            importance = np.abs(shap_values).mean(axis=0)
            method = "SHAP"
        except Exception as e:
            print(f"[explain] SHAP failed ({e}); falling back to permutation importance.")
            shap_available = False

    if not shap_available:
        print("[explain] Computing permutation importance...")
        from sklearn.metrics import average_precision_score
        result = permutation_importance(
            model, X_val, y_val,
            n_repeats=10,
            scoring="average_precision",
            random_state=42,
            n_jobs=-1,
        )
        importance = result.importances_mean
        method = "Permutation (PR-AUC)"

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
        "method": method,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return df


def plot_feature_importance(fi_df: pd.DataFrame, model_name: str, top_n: int = 15,
                             save_path: Path | None = None):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top = fi_df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top)))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1])
    ax.set_xlabel(f"Importance ({top['method'].iloc[0]})")
    ax.set_title(f"Top {top_n} Features — {model_name}", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out = save_path or FIGURES_DIR / f"feature_importance_{model_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] Feature importance plot saved → {out}")


# ── Rule-based reason codes ───────────────────────────────────────────────────

REASON_RULES = [
    {
        "code": "FULL_BALANCE_DRAIN",
        "label": "Amount equals full origin balance (account takeover pattern)",
        "fn": lambda row: abs(row.get("errorBalanceOrig", 0)) < 1.0
                          and row.get("orig_balance_zeroed", 0) == 1,
    },
    {
        "code": "DEST_BALANCE_UNCHANGED",
        "label": "Destination balance unchanged after credit (mule account pattern)",
        "fn": lambda row: abs(row.get("errorBalanceDest", 0)) > 10.0,
    },
    {
        "code": "NIGHT_TIME",
        "label": "Night-time transaction (00:00-05:59)",
        "fn": lambda row: row.get("is_night", 0) == 1,
    },
    {
        "code": "HIGH_RISK_TYPE_TRANSFER",
        "label": "High-risk type: TRANSFER",
        "fn": lambda row: row.get("type_TRANSFER", 0) == 1,
    },
    {
        "code": "HIGH_RISK_TYPE_CASHOUT",
        "label": "High-risk type: CASH_OUT",
        "fn": lambda row: row.get("type_CASH_OUT", 0) == 1,
    },
    {
        "code": "LARGE_AMOUNT_RATIO",
        "label": "Amount is large relative to origin balance (>90%)",
        "fn": lambda row: row.get("amount_to_balance_ratio", 0) > 0.90,
    },
    {
        "code": "HIGH_DEST_VELOCITY",
        "label": "Destination account received many transactions recently (velocity risk)",
        "fn": lambda row: row.get("dest_velocity_24h", 0) > 5,
    },
    {
        "code": "DEST_WAS_EMPTY",
        "label": "Destination account had zero balance before transaction",
        "fn": lambda row: row.get("dest_balance_was_zero", 0) == 1,
    },
]


def get_reason_codes(row: dict) -> list[str]:
    """Return list of reason code labels that apply to a transaction."""
    codes = []
    for rule in REASON_RULES:
        try:
            if rule["fn"](row):
                codes.append(rule["label"])
        except Exception:
            pass
    return codes


def add_reason_codes_to_df(df: pd.DataFrame) -> pd.Series:
    """Apply reason code logic to every row; return a Series of JSON strings."""
    import json

    def _apply(row):
        codes = get_reason_codes(row.to_dict())
        return json.dumps(codes)

    return df.apply(_apply, axis=1)
