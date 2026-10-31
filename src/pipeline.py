"""
src/pipeline.py

End-to-end FraudLens pipeline.

Usage:
    python -m src.pipeline              # run full pipeline
    python -m src.pipeline --stage data
    python -m src.pipeline --stage train
    python -m src.pipeline --stage exports
    python -m src.pipeline --stage all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Make project root importable when run as module ──────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import get_config
from src.data_generator import load_or_generate
from src.db import (
    create_schema,
    get_engine,
    load_transactions,
    save_alerts,
    save_model_run,
    save_predictions,
)
from src.features import engineer, get_feature_matrix, FEATURE_COLS
from src.split import time_split
from src.train import load_model, save_model, train_all
from src.evaluate import (
    ablation_comparison,
    evaluate_model,
    find_optimal_threshold_by_prauc,
    plot_calibration,
    plot_confusion_matrix,
    plot_pr_roc_curves,
    save_metrics,
)
from src.cost_analysis import (
    compute_cost_curve,
    find_optimal_threshold,
    generate_result_statement,
    sensitivity_table,
)
from src.explain import (
    add_reason_codes_to_df,
    compute_feature_importance,
    plot_feature_importance,
)
from src.export_powerbi import export_all


def run_data_stage(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate data, engineer features, split, load to DB."""
    print("\n" + "="*60)
    print("STAGE 1: Data Generation & Loading")
    print("="*60)

    # 1. Load / generate raw data
    raw_df = load_or_generate(cfg)
    print(f"[pipeline] Raw data: {raw_df.shape}, fraud rate={raw_df['isFraud'].mean()*100:.3f}%")

    # 2. Feature engineering
    print("[pipeline] Engineering features...")
    feat_df = engineer(raw_df, velocity_window=cfg["features"]["velocity_window_steps"])
    print(f"[pipeline] Feature engineering done. Columns: {len(feat_df.columns)}")

    # 3. Time-based split
    split_cfg = cfg["split"]
    train_df, val_df, test_df = time_split(
        feat_df,
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
    )

    # 4. Tag rows with split label in full df
    split_map = {}
    for s_name, s_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for tid in s_df["txn_id"]:
            split_map[tid] = s_name
    feat_df["split"] = feat_df["txn_id"].map(split_map)

    # 5. Load to database
    engine = get_engine(cfg)
    create_schema(engine)
    load_transactions(feat_df, engine, chunk_size=cfg["database"]["chunk_size"])

    return feat_df, train_df, val_df, test_df


def run_train_stage(
    cfg: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Train models, evaluate, save artifacts."""
    print("\n" + "="*60)
    print("STAGE 2: Training & Evaluation")
    print("="*60)

    engine = get_engine(cfg)
    model_dir = _ROOT / cfg["paths"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Train all models ──────────────────────────────────────────────────────
    train_result = train_all(train_df, val_df, cfg)
    models = train_result["models"]
    final_name = train_result["final_model_name"]
    imbalance_cmp = train_result["imbalance_comparison"]

    # ── Ablation (with/without balance error features) ────────────────────────
    print("\n[pipeline] Running ablation study...")
    ablation = ablation_comparison(None, train_df, val_df)

    # ── Evaluate all supervised models on test set ────────────────────────────
    alert_rates = cfg["business"]["alert_rate_benchmarks"]
    all_test_metrics = {}
    val_scores_for_cost = {}
    final_opt_threshold = 0.5   # default; overwritten by the final model's cost analysis

    for model_name, info in models.items():
        model = info["model"]
        is_iforest = model_name == "isolation_forest"

        # Val metrics (for threshold selection)
        val_m = evaluate_model(
            model, val_df, "val",
            is_iforest=is_iforest,
            alert_rate_benchmarks=alert_rates,
        )
        val_scores_for_cost[model_name] = val_m.get("_scores")

        # For the final model: pick optimal threshold on VAL via cost analysis
        if model_name == final_name:
            val_amounts = val_df["amount"].values
            val_y = val_df["isFraud"].values
            inv_cost = cfg["business"]["investigation_cost_per_alert"]
            rec_rate = cfg["business"]["fraud_recovery_rate_when_caught"]

            val_cost_curve = compute_cost_curve(
                val_y, val_m["_scores"], val_amounts,
                investigation_cost=inv_cost,
                recovery_rate=rec_rate,
            )
            final_opt_threshold = find_optimal_threshold(val_cost_curve)
            print(f"[pipeline] Optimal threshold (val cost analysis): {final_opt_threshold:.4f}")

        # Test evaluation (final model uses cost-optimal threshold)
        test_m = evaluate_model(
            model, test_df, "test",
            threshold=final_opt_threshold if model_name == final_name else None,
            is_iforest=is_iforest,
            alert_rate_benchmarks=alert_rates,
        )
        test_m["model_name"] = model_name
        test_m["val_pr_auc"] = info["val_pr_auc"]
        all_test_metrics[model_name] = test_m

        # Save model
        save_model(model, model_dir / f"{model_name}.joblib")

        # Record in DB
        save_model_run(
            model_name=model_name,
            params=info.get("params", {}),
            metrics={k: v for k, v in test_m.items() if not k.startswith("_")},
            engine=engine,
        )

    # ── Plots (all models, test split) ───────────────────────────────────────
    plot_pr_roc_curves(all_test_metrics, "test")
    for mname, minfo in models.items():
        if mname != "isolation_forest":
            plot_confusion_matrix(all_test_metrics[mname], mname)
            plot_calibration(minfo["model"], test_df, mname)

    # ── Feature importance for final model ───────────────────────────────────
    final_model = models[final_name]["model"]
    X_val = get_feature_matrix(val_df).values
    y_val = val_df["isFraud"].values
    fi_df = compute_feature_importance(final_model, X_val, y_val, FEATURE_COLS)
    plot_feature_importance(fi_df, final_name)
    fi_df.to_csv(_ROOT / "reports" / "feature_importance.csv", index=False)

    # ── Cost analysis on TEST ────────────────────────────────────────────────
    test_scores = all_test_metrics[final_name]["_scores"]
    test_y      = all_test_metrics[final_name]["_y_true"]
    test_amounts = test_df["amount"].values

    inv_cost = cfg["business"]["investigation_cost_per_alert"]
    rec_rate = cfg["business"]["fraud_recovery_rate_when_caught"]

    test_cost_curve = compute_cost_curve(
        test_y, test_scores, test_amounts,
        investigation_cost=inv_cost,
        recovery_rate=rec_rate,
    )

    # Use the threshold from validation
    opt_row = test_cost_curve.loc[
        (test_cost_curve["threshold"] - final_opt_threshold).abs().idxmin()
    ]
    net_savings = float(opt_row["net_savings"])
    all_test_metrics[final_name]["net_savings"] = net_savings
    all_test_metrics[final_name]["cost_optimal_threshold"] = final_opt_threshold

    result_stmt = generate_result_statement(
        test_cost_curve,
        final_opt_threshold,
        total_txns=len(test_df),
        currency=cfg["business"]["currency_symbol"],
    )
    print(f"\n[pipeline] Business result:\n  {result_stmt}")

    sens_df = sensitivity_table(test_y, test_scores, test_amounts, final_opt_threshold)

    # ── Save predictions to DB ───────────────────────────────────────────────
    final_test_flagged = (test_scores >= final_opt_threshold).astype(int)
    pred_df = pd.DataFrame({
        "txn_id":     test_df["txn_id"].values,
        "model_name": final_name,
        "score":      test_scores,
        "flagged":    final_test_flagged,
        "split":      "test",
    })
    save_predictions(pred_df, engine)

    # ── Save top alerts to DB ────────────────────────────────────────────────
    flagged_mask = final_test_flagged == 1
    flagged_df = test_df[flagged_mask].copy()
    flagged_df["score"] = test_scores[flagged_mask]
    flagged_df["reason_codes"] = add_reason_codes_to_df(flagged_df)

    alerts_df = flagged_df[["txn_id", "score", "reason_codes"]].copy()
    alerts_df["status"] = "Open"
    save_alerts(alerts_df.head(500), engine)  # top 500 for investigator queue

    # ── Compile and save metrics ─────────────────────────────────────────────
    all_metrics = {
        "final_model": final_name,
        "models": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                   for k, v in all_test_metrics.items()},
        "imbalance_comparison": imbalance_cmp,
        "ablation": ablation,
        "business": {
            "result_statement": result_stmt,
            "opt_threshold": final_opt_threshold,
            "net_savings_at_opt_threshold": net_savings,
            "baseline_loss": float(test_amounts[test_y == 1].sum()),
            "investigation_cost_per_alert": inv_cost,
        },
        "sensitivity": sens_df.to_dict(orient="records"),
    }
    save_metrics(all_metrics)

    return {
        "models": models,
        "final_name": final_name,
        "all_metrics": all_metrics,
        "test_df": test_df,
        "pred_df": pred_df,
        "cost_curve": test_cost_curve,
    }


def run_exports_stage(
    cfg: dict,
    feat_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    all_metrics: dict,
    cost_curve_df: pd.DataFrame,
) -> None:
    """Export Power BI CSVs."""
    print("\n" + "="*60)
    print("STAGE 3: Power BI Exports")
    print("="*60)
    export_all(feat_df, pred_df, all_metrics.get("models", {}), cost_curve_df)


def main():
    parser = argparse.ArgumentParser(description="FraudLens Pipeline")
    parser.add_argument(
        "--stage",
        choices=["data", "train", "exports", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    args = parser.parse_args()

    cfg = get_config()

    print("\n" + "="*60)
    print("  FraudLens: Transaction Fraud Detection Platform")
    print(f"  Stage: {args.stage.upper()}")
    print("="*60)

    if args.stage in ("data", "all"):
        feat_df, train_df, val_df, test_df = run_data_stage(cfg)

    if args.stage in ("train", "all"):
        if args.stage == "train":
            # Reload from DB if only running train
            engine = get_engine(cfg)
            from sqlalchemy import text
            with engine.connect() as conn:
                feat_df = pd.read_sql("SELECT * FROM transactions", conn)
            train_df = feat_df[feat_df["split"] == "train"]
            val_df   = feat_df[feat_df["split"] == "val"]
            test_df  = feat_df[feat_df["split"] == "test"]

        train_out = run_train_stage(cfg, train_df, val_df, test_df)
        feat_df_for_export = feat_df if args.stage == "all" else train_out["test_df"]

    if args.stage in ("exports", "all"):
        if args.stage == "exports":
            # Reload from DB
            engine = get_engine(cfg)
            with engine.connect() as conn:
                feat_df = pd.read_sql("SELECT * FROM transactions", conn)
                pred_df = pd.read_sql("SELECT * FROM predictions", conn)
            with open(_ROOT / "reports/metrics.json") as f:
                all_metrics = json.load(f)
            # Rebuild cost curve from saved scores would require re-running model
            # Use a dummy empty df in this case
            cost_curve_df = pd.DataFrame()
        else:
            pred_df = train_out["pred_df"]
            all_metrics = train_out["all_metrics"]
            cost_curve_df = train_out["cost_curve"]

        run_exports_stage(cfg, feat_df, pred_df, all_metrics, cost_curve_df)

    print("\n" + "="*60)
    print("  Pipeline complete!")
    print("="*60)
    print("\nNext steps:")
    print("  make app   → Launch Streamlit dashboard")
    print("  make test  → Run pytest")


if __name__ == "__main__":
    main()
