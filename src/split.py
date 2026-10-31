"""
src/split.py

Time-based train/val/test split for FraudLens.

Design decisions:
- Split is based on transaction step (time), NOT random shuffling.
  This prevents future leakage and better simulates production deployment.
- Train = first 70% of unique steps
- Val   = next 15% of unique steps
- Test  = final 15% of unique steps
- Assertions verify no temporal overlap and that every split has fraud cases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into train/val/test DataFrames based on transaction step.

    Returns
    -------
    train_df, val_df, test_df
    """
    assert 0 < train_frac < 1, "train_frac must be in (0,1)"
    assert 0 < val_frac < 1, "val_frac must be in (0,1)"
    assert train_frac + val_frac < 1, "train + val must be < 1"

    unique_steps = np.sort(df["step"].unique())
    n_steps = len(unique_steps)

    train_end = int(n_steps * train_frac)
    val_end   = int(n_steps * (train_frac + val_frac))

    train_steps = set(unique_steps[:train_end])
    val_steps   = set(unique_steps[train_end:val_end])
    test_steps  = set(unique_steps[val_end:])

    train_df = df[df["step"].isin(train_steps)].copy()
    val_df   = df[df["step"].isin(val_steps)].copy()
    test_df  = df[df["step"].isin(test_steps)].copy()

    train_df["split"] = "train"
    val_df["split"]   = "val"
    test_df["split"]  = "test"

    # ── Assertions ────────────────────────────────────────────────────────────
    # 1. No time overlap
    assert train_steps.isdisjoint(val_steps),  "Train/Val steps overlap!"
    assert train_steps.isdisjoint(test_steps), "Train/Test steps overlap!"
    assert val_steps.isdisjoint(test_steps),   "Val/Test steps overlap!"

    # 2. Temporal ordering: all train steps < all val steps < all test steps
    assert max(train_steps) < min(val_steps),  "Train bleeds into Val!"
    assert max(val_steps)   < min(test_steps), "Val bleeds into Test!"

    # 3. Every split must contain fraud cases
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        fraud_count = split_df["isFraud"].sum()
        assert fraud_count > 0, (
            f"Split '{name}' contains NO fraud cases! "
            "Consider increasing fraud_rate_target in config.yaml."
        )
        print(
            f"[split] {name:5s}: {len(split_df):>7,} rows | "
            f"fraud={fraud_count:>4} ({100.*fraud_count/len(split_df):.3f}%)"
        )

    return train_df, val_df, test_df
