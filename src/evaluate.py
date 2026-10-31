"""
src/evaluate.py

Model evaluation for FraudLens.

Computes and saves:
- PR curve, ROC curve, confusion matrix
- PR-AUC, ROC-AUC, F1, precision, recall
- Recall at fixed alert rates (0.5% and 1% of transactions)
- Probability calibration plot
- Ablation: with vs without balance-error features

All figures saved to reports/figures/.
Metrics saved to reports/metrics.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.features import get_feature_matrix, FEATURE_COLS


FIGURES_DIR = Path("reports/figures")
METRICS_FILE = Path("reports/metrics.json")


def _ensure_dirs():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)


def _get_scores(model, X, is_iforest=False) -> np.ndarray:
    if is_iforest:
        return -model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def recall_at_alert_rate(y_true, scores, alert_rate: float) -> float:
    """Recall when only the top `alert_rate` fraction of transactions are flagged."""
    n_alerts = max(1, int(len(scores) * alert_rate))
    threshold = np.sort(scores)[::-1][n_alerts - 1]
    y_pred = (scores >= threshold).astype(int)
    return recall_score(y_true, y_pred, zero_division=0)


def find_optimal_threshold_by_prauc(y_true, scores) -> float:
    """Return the threshold that maximises F1 on the given set."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1s = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.argmax(f1s[:-1])  # thresholds has len = len(precision)-1
    return float(thresholds[best_idx])


def evaluate_model(
    model,
    split_df: pd.DataFrame,
    split_name: str,
    threshold: float | None = None,
    is_iforest: bool = False,
    use_balance_error: bool = True,
    alert_rate_benchmarks: list[float] | None = None,
) -> dict:
    """Compute all metrics for a model on a given split."""
    X = get_feature_matrix(split_df, use_balance_error).values
    y = split_df["isFraud"].values
    scores = _get_scores(model, X, is_iforest)

    pr_auc = average_precision_score(y, scores)
    roc_auc = roc_auc_score(y, scores) if len(np.unique(y)) > 1 else 0.0

    if threshold is None:
        threshold = find_optimal_threshold_by_prauc(y, scores)

    y_pred = (scores >= threshold).astype(int)
    precision = precision_score(y, y_pred, zero_division=0)
    recall    = recall_score(y, y_pred, zero_division=0)
    f1        = f1_score(y, y_pred, zero_division=0)
    cm        = confusion_matrix(y, y_pred).tolist()

    metrics = {
        "split": split_name,
        "pr_auc":    round(float(pr_auc), 6),
        "roc_auc":   round(float(roc_auc), 6),
        "f1":        round(float(f1), 6),
        "precision": round(float(precision), 6),
        "recall":    round(float(recall), 6),
        "threshold": round(float(threshold), 6),
        "confusion_matrix": cm,
        "n_flagged": int(y_pred.sum()),
        "n_total":   int(len(y)),
        "n_fraud":   int(y.sum()),
        "alert_rate_pct": round(100.0 * y_pred.sum() / len(y), 4),
    }

    if alert_rate_benchmarks:
        for ar in alert_rate_benchmarks:
            key = f"recall_at_{int(ar*1000)}bp"
            metrics[key] = round(recall_at_alert_rate(y, scores, ar), 6)

    metrics["_scores"] = scores   # kept for plotting
    metrics["_y_true"] = y

    return metrics


def plot_pr_roc_curves(all_results: dict, split_name: str, save_path: Path | None = None):
    """Plot PR and ROC curves for all models."""
    _ensure_dirs()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Model Comparison — {split_name} split", fontsize=14, fontweight="bold")

    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6"]
    for i, (name, res) in enumerate(all_results.items()):
        scores = res.get("_scores")
        y_true = res.get("_y_true")
        if scores is None:
            continue
        color = colors[i % len(colors)]
        label = f"{name} (PR-AUC={res['pr_auc']:.3f})"

        # PR curve
        p, r, _ = precision_recall_curve(y_true, scores)
        axes[0].plot(r, p, label=label, color=color, linewidth=2)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, scores)
        axes[1].plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})", color=color, linewidth=2)

    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_path or FIGURES_DIR / f"pr_roc_{split_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] Saved PR/ROC curves → {out}")


def plot_confusion_matrix(metrics: dict, model_name: str, save_path: Path | None = None):
    _ensure_dirs()
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Legit", "Predicted Fraud"])
    ax.set_yticklabels(["Actual Legit", "Actual Fraud"])
    ax.set_title(f"Confusion Matrix — {model_name}\n(threshold={metrics['threshold']:.3f})", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    plt.tight_layout()
    out = save_path or FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] Saved confusion matrix → {out}")


def plot_calibration(model, split_df: pd.DataFrame, model_name: str,
                     use_balance_error: bool = True, save_path: Path | None = None):
    _ensure_dirs()
    X = get_feature_matrix(split_df, use_balance_error).values
    y = split_df["isFraud"].values
    scores = model.predict_proba(X)[:, 1]

    prob_true, prob_pred = calibration_curve(y, scores, n_bins=20, strategy="quantile")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(prob_pred, prob_true, "s-", label="Model", color="#3498db", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {model_name}", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = save_path or FIGURES_DIR / f"calibration_{model_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] Saved calibration curve → {out}")


def ablation_comparison(
    model_class_fn,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict:
    """
    Train the same model with and without balance-error features.
    Returns dict with val PR-AUC for both configurations.
    """
    from src.train import _make_hgb

    results = {}
    for use_be in [True, False]:
        X_train = get_feature_matrix(train_df, use_be).values
        y_train = train_df["isFraud"].values
        X_val   = get_feature_matrix(val_df, use_be).values
        y_val   = val_df["isFraud"].values

        m = _make_hgb(class_weight="balanced")
        m.fit(X_train, y_train)
        prauc = average_precision_score(y_val, m.predict_proba(X_val)[:, 1])
        key = "with_balance_error" if use_be else "without_balance_error"
        results[key] = round(float(prauc), 6)
        print(f"[ablation] {key}: val PR-AUC={prauc:.4f}")

    results["delta"] = round(results["with_balance_error"] - results["without_balance_error"], 6)
    print(f"[ablation] Delta (balance-error impact): {results['delta']:+.4f}")
    return results


def save_metrics(all_metrics: dict, path: Path | None = None):
    _ensure_dirs()
    path = path or METRICS_FILE
    # Remove non-serialisable numpy arrays
    clean = {}
    for k, v in all_metrics.items():
        if isinstance(v, dict):
            clean[k] = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        else:
            clean[k] = v

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, "w") as f:
        json.dump(clean, f, indent=2, default=_convert)
    print(f"[eval] Metrics saved → {path}")
