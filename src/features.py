"""
src/features.py

Feature engineering pipeline for FraudLens.

Each feature is documented with:
    - Description: what it captures
    - Leakage risk: can it leak future/label information?

Features:
─────────────────────────────────────────────────────────────────────
Feature                  | Description                  | Leakage?
─────────────────────────────────────────────────────────────────────
hour_of_day              | step % 24                    | None
day                      | step // 24                   | None
is_night                 | 1 if hour in 0-5             | None
type_*                   | one-hot of transaction type  | None
errorBalanceOrig         | newOrig + amount - oldOrig   | Moderate — very
                         | (balance inconsistency)      | predictive in
                         |                              | synthetic data;
                         |                              | weaker on real data
errorBalanceDest         | oldDest + amount - newDest   | Same as above
amount_to_balance_ratio  | amount / (oldbalanceOrg+1)   | Moderate
orig_balance_zeroed      | 1 if newbalanceOrig == 0     | Moderate — linked
                         | after transaction            | to pattern 1 fraud
dest_balance_was_zero    | 1 if oldbalanceDest == 0     | Low
dest_velocity_24h        | # inbound txns to nameDest   | LOW — computed
                         | in previous 24 steps         | from past data only
─────────────────────────────────────────────────────────────────────
Dropped: nameOrig, nameDest (identifiers), isFlaggedFraud (label leakage)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "hour_of_day",
    "day",
    "is_night",
    "errorBalanceOrig",
    "errorBalanceDest",
    "amount_to_balance_ratio",
    "orig_balance_zeroed",
    "dest_balance_was_zero",
    "dest_velocity_24h",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    "amount",
]

FEATURE_COLS_NO_BALANCE_ERROR = [
    c for c in FEATURE_COLS
    if c not in ("errorBalanceOrig", "errorBalanceDest")
]


def engineer(df: pd.DataFrame, velocity_window: int = 24) -> pd.DataFrame:
    """
    Apply all feature engineering transformations to df (in-place copy).

    Parameters
    ----------
    df : DataFrame with raw PaySim columns + txn_id
    velocity_window : number of steps to look back for destination velocity

    Returns
    -------
    DataFrame with new feature columns appended.
    No future data is used (velocity is computed using only past steps).
    """
    df = df.copy()

    # ── Temporal ──────────────────────────────────────────────────────────────
    df["hour_of_day"] = df["step"] % 24
    df["day"]         = df["step"] // 24
    df["is_night"]    = (df["hour_of_day"].between(0, 5)).astype(int)

    # ── One-hot transaction type ──────────────────────────────────────────────
    for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        df[f"type_{t}"] = (df["type"] == t).astype(int)

    # ── Balance error features ────────────────────────────────────────────────
    # errorBalanceOrig: if legitimate, newbalanceOrig = oldbalanceOrg - amount
    # So error should be 0 for honest transactions.
    df["errorBalanceOrig"] = (
        df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    ).round(4)

    # errorBalanceDest: if legitimate, newbalanceDest = oldbalanceDest + amount
    df["errorBalanceDest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    ).round(4)

    # ── Amount-derived features ───────────────────────────────────────────────
    df["amount_to_balance_ratio"] = (
        df["amount"] / (df["oldbalanceOrg"] + 1.0)
    ).round(6)

    df["orig_balance_zeroed"] = (df["newbalanceOrig"] == 0.0).astype(int)
    df["dest_balance_was_zero"] = (df["oldbalanceDest"] == 0.0).astype(int)

    # ── Destination velocity (past 24 steps, NO future leakage) ──────────────
    # Sort by step to ensure temporal ordering
    df = df.sort_values("step").reset_index(drop=True)
    df["dest_velocity_24h"] = _compute_dest_velocity(df, window=velocity_window)

    return df


def _compute_dest_velocity(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """
    For each transaction at step t, count how many times nameDest appeared
    as a destination in the interval [t - window, t) — strictly past data.

    Implementation uses a sorted scan to guarantee no future leakage:
    the velocity for row i is computed from rows with step < df.step[i]
    within the last `window` steps.
    """
    velocity = np.zeros(len(df), dtype=float)
    steps = df["step"].values
    dests = df["nameDest"].values

    # Build a step-indexed list of destinations
    # We group by step for efficiency
    from collections import defaultdict, deque

    step_dest_counts: dict[str, deque] = defaultdict(deque)
    step_groups: dict[int, list] = defaultdict(list)

    for i, (s, d) in enumerate(zip(steps, dests)):
        step_groups[int(s)].append((i, d))

    sorted_steps = sorted(step_groups.keys())
    # Running window: tracks (step, dest) pairs
    window_queue: deque = deque()  # each element: (step_value, dest)
    dest_counter: dict[str, int] = defaultdict(int)

    for s in sorted_steps:
        # Expire entries older than window
        while window_queue and window_queue[0][0] < s - window:
            old_step, old_dest = window_queue.popleft()
            dest_counter[old_dest] -= 1
            if dest_counter[old_dest] == 0:
                del dest_counter[old_dest]

        # Assign velocity to all transactions at step s
        for i, d in step_groups[s]:
            velocity[i] = dest_counter.get(d, 0)

        # Now add current step's destinations to the window (AFTER assigning)
        for i, d in step_groups[s]:
            window_queue.append((s, d))
            dest_counter[d] += 1

    return pd.Series(velocity, index=df.index)


def get_feature_matrix(df: pd.DataFrame, use_balance_error: bool = True) -> pd.DataFrame:
    """Return the model input matrix (only feature columns)."""
    cols = FEATURE_COLS if use_balance_error else FEATURE_COLS_NO_BALANCE_ERROR
    # Ensure all columns exist
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[cols].copy()
