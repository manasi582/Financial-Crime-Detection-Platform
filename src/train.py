"""
src/train.py

Model training for FraudLens.

Models trained:
1. Logistic Regression (baseline)
2. Random Forest
3. HistGradientBoostingClassifier (main)
4. Isolation Forest (unsupervised comparison)

Imbalance handling compared (training set only):
- class_weight='balanced'
- RandomUnderSampler
- SMOTE

Final model selected on validation PR-AUC.
Light hyperparameter tuning on validation set (< 10 min on laptop).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    IsolationForest,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("[train] imbalanced-learn not available; skipping SMOTE/undersampling.")

from src.features import get_feature_matrix, FEATURE_COLS


def _make_logistic(class_weight=None) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            solver="lbfgs",
            C=0.1,
            random_state=42,
        )),
    ])


def _make_rf(class_weight=None, n_estimators=200) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_leaf=4,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=42,
    )


def _make_hgb(class_weight=None) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        class_weight=class_weight,
        random_state=42,
    )


def _make_isolation_forest() -> IsolationForest:
    return IsolationForest(
        n_estimators=200,
        contamination=0.002,
        random_state=42,
        n_jobs=-1,
    )


def _pr_auc(model, X_val, y_val, is_iforest=False) -> float:
    if is_iforest:
        scores = -model.decision_function(X_val)  # higher = more anomalous
    else:
        scores = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, scores)


def compare_imbalance_methods(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
) -> dict:
    """
    Train the main model (HGB) with different imbalance handling strategies
    and return validation PR-AUCs for comparison.
    Only the training set is resampled.
    """
    results = {}

    # class_weight='balanced'
    m = _make_hgb(class_weight="balanced")
    m.fit(X_train, y_train)
    results["class_weight"] = _pr_auc(m, X_val, y_val)
    print(f"[train] Imbalance: class_weight PR-AUC = {results['class_weight']:.4f}")

    if HAS_IMBLEARN:
        # Random undersampling
        rus = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        m = _make_hgb()
        m.fit(X_res, y_res)
        results["random_undersample"] = _pr_auc(m, X_val, y_val)
        print(f"[train] Imbalance: random_undersample PR-AUC = {results['random_undersample']:.4f}")

        # SMOTE
        try:
            smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            m = _make_hgb()
            m.fit(X_res, y_res)
            results["smote"] = _pr_auc(m, X_val, y_val)
            print(f"[train] Imbalance: SMOTE PR-AUC = {results['smote']:.4f}")
        except Exception as e:
            print(f"[train] SMOTE failed: {e}; skipping.")
    else:
        results["random_undersample"] = None
        results["smote"] = None

    return results


def tune_hgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Light grid search on validation PR-AUC (no CV needed — time split)."""
    param_grid = [
        {"max_iter": 200, "learning_rate": 0.05, "max_depth": 5},
        {"max_iter": 300, "learning_rate": 0.05, "max_depth": 6},
        {"max_iter": 300, "learning_rate": 0.03, "max_depth": 7},
        {"max_iter": 400, "learning_rate": 0.03, "max_depth": 6},
    ]
    best_score, best_params = -1, param_grid[0]
    for params in param_grid:
        m = HistGradientBoostingClassifier(
            **params,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
        )
        m.fit(X_train, y_train)
        score = _pr_auc(m, X_val, y_val)
        print(f"[tune] params={params} val_PR-AUC={score:.4f}")
        if score > best_score:
            best_score = score
            best_params = params
    print(f"[tune] Best HGB params: {best_params} (val PR-AUC={best_score:.4f})")
    return best_params


def train_all(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
    use_balance_error: bool = True,
) -> dict:
    """
    Train all models and return a dict with trained model objects and metrics.
    Final model selected on validation PR-AUC.
    """
    X_train = get_feature_matrix(train_df, use_balance_error).values
    y_train = train_df["isFraud"].values
    X_val   = get_feature_matrix(val_df, use_balance_error).values
    y_val   = val_df["isFraud"].values

    print(f"[train] Train: {X_train.shape}, fraud={y_train.sum()} ({100*y_train.mean():.3f}%)")
    print(f"[train] Val:   {X_val.shape},   fraud={y_val.sum()} ({100*y_val.mean():.3f}%)")

    trained = {}

    # ── 1. Logistic Regression baseline ──────────────────────────────────────
    print("\n[train] Logistic Regression...")
    t0 = time.time()
    lr = _make_logistic(class_weight="balanced")
    lr.fit(X_train, y_train)
    lr_prauc = _pr_auc(lr, X_val, y_val)
    print(f"[train] LR val PR-AUC={lr_prauc:.4f} ({time.time()-t0:.1f}s)")
    trained["logistic_regression"] = {
        "model": lr, "val_pr_auc": lr_prauc, "params": {}
    }

    # ── 2. Random Forest ──────────────────────────────────────────────────────
    print("\n[train] Random Forest...")
    t0 = time.time()
    rf = _make_rf(class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_prauc = _pr_auc(rf, X_val, y_val)
    print(f"[train] RF val PR-AUC={rf_prauc:.4f} ({time.time()-t0:.1f}s)")
    trained["random_forest"] = {
        "model": rf, "val_pr_auc": rf_prauc,
        "params": {"n_estimators": 200, "max_depth": 12}
    }

    # ── 3. HistGradientBoosting (tuned) ──────────────────────────────────────
    print("\n[train] HGB hyperparameter tuning...")
    t0 = time.time()
    best_params = tune_hgb(X_train, y_train, X_val, y_val)
    hgb = HistGradientBoostingClassifier(
        **best_params,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    )
    hgb.fit(X_train, y_train)
    hgb_prauc = _pr_auc(hgb, X_val, y_val)
    print(f"[train] HGB val PR-AUC={hgb_prauc:.4f} ({time.time()-t0:.1f}s)")
    trained["hist_gradient_boosting"] = {
        "model": hgb, "val_pr_auc": hgb_prauc, "params": best_params
    }

    # ── 4. Isolation Forest (unsupervised) ────────────────────────────────────
    print("\n[train] Isolation Forest (unsupervised)...")
    t0 = time.time()
    iforest = _make_isolation_forest()
    iforest.fit(X_train)
    iforest_prauc = average_precision_score(
        y_val, -iforest.decision_function(X_val)
    )
    print(f"[train] IF val PR-AUC={iforest_prauc:.4f} ({time.time()-t0:.1f}s)")
    trained["isolation_forest"] = {
        "model": iforest, "val_pr_auc": iforest_prauc, "params": {"contamination": 0.002}
    }

    # ── 5. Imbalance comparison ──────────────────────────────────────────────
    print("\n[train] Imbalance handling comparison...")
    imbalance_results = compare_imbalance_methods(X_train, y_train, X_val, y_val, cfg)

    # ── 6. Select final model on validation PR-AUC ───────────────────────────
    supervised = {k: v for k, v in trained.items() if k != "isolation_forest"}
    final_name = max(supervised, key=lambda k: supervised[k]["val_pr_auc"])
    print(f"\n[train] Final model: {final_name} (val PR-AUC={trained[final_name]['val_pr_auc']:.4f})")

    return {
        "models": trained,
        "final_model_name": final_name,
        "imbalance_comparison": imbalance_results,
    }


def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[train] Model saved → {path}")


def load_model(path: str | Path):
    return joblib.load(path)
