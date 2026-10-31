"""
src/data_generator.py

Generates a synthetic PaySim-schema dataset.

Schema columns:
    step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
    nameDest, oldbalanceDest, newbalanceDest, isFraud

Design decisions:
- Fraud occurs ONLY in TRANSFER and CASH_OUT transactions.
- Three fraud patterns are embedded:
    1. Account takeover: amount == oldbalanceOrg (drains full balance)
       → errorBalanceOrig is large; newbalanceOrig = 0
    2. Mule chain: destination balance unchanged after receiving funds
       → errorBalanceDest is large (amount was supposed to add to dest)
    3. Night-time (hour 0-5) elevated fraud probability.
    4. Larger-than-typical amounts.
- Legitimate transactions have consistent balances (small rounding noise only)
- Fraud transactions preserve the anomaly AFTER noise is applied
- Target PR-AUC ~ 0.75-0.93 (not 1.0) due to overlapping distributions
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ── Type distribution (matches PaySim statistics) ─────────────────────────────
TYPE_PROBS = {
    "CASH_OUT": 0.35,
    "PAYMENT":  0.34,
    "CASH_IN":  0.22,
    "TRANSFER": 0.08,
    "DEBIT":    0.01,
}

# Amount distributions per type (log-normal params: mu, sigma of log(amount))
AMOUNT_PARAMS = {
    "CASH_OUT": (8.5, 1.5),
    "PAYMENT":  (7.0, 1.2),
    "CASH_IN":  (8.0, 1.3),
    "TRANSFER": (9.5, 1.8),
    "DEBIT":    (6.0, 1.0),
}


def generate(
    n_rows: int = 500_000,
    n_steps: int = 720,
    random_seed: int = 42,
    fraud_rate_target: float = 0.002,
) -> pd.DataFrame:
    """Return a DataFrame with the PaySim schema."""
    rng = np.random.default_rng(random_seed)

    # ── 1. Transaction type and step ──────────────────────────────────────────
    types = rng.choice(
        list(TYPE_PROBS.keys()),
        size=n_rows,
        p=list(TYPE_PROBS.values()),
    )
    # Steps follow a realistic daily rhythm (more transactions during business hours)
    hour_weights = np.array([
        0.5, 0.4, 0.3, 0.3, 0.35, 0.5,   # 00-05  night
        0.8, 1.2, 1.5, 1.6, 1.7, 1.7,    # 06-11  morning
        1.6, 1.5, 1.5, 1.6, 1.7, 1.8,    # 12-17  afternoon
        1.6, 1.4, 1.2, 1.0, 0.8, 0.6,    # 18-23  evening
    ])
    hour_weights = hour_weights / hour_weights.sum()
    hour_weights_tiled = np.tile(hour_weights, n_steps // 24 + 1)[:n_steps]
    hour_weights_tiled = hour_weights_tiled / hour_weights_tiled.sum()
    steps = rng.choice(np.arange(n_steps), size=n_rows, p=hour_weights_tiled)

    # ── 2. Amounts ─────────────────────────────────────────────────────────────
    amounts = np.zeros(n_rows)
    for t, (mu, sig) in AMOUNT_PARAMS.items():
        mask = types == t
        amounts[mask] = np.exp(rng.normal(mu, sig, mask.sum()))
    amounts = np.clip(amounts, 1.0, 1_000_000.0).round(2)

    # ── 3. Account names ──────────────────────────────────────────────────────
    n_orig_accounts = max(100_000, n_rows // 3)
    n_dest_accounts = max(150_000, n_rows // 2)
    orig_ids = rng.integers(1, n_orig_accounts, size=n_rows)
    dest_ids = rng.integers(1, n_dest_accounts, size=n_rows)
    name_orig = np.array([f"C{i}" for i in orig_ids])
    name_dest_c = np.array([f"C{i}" for i in dest_ids])
    name_dest_m = np.array([f"M{i}" for i in rng.integers(1, 10_000, size=n_rows)])
    name_dest = np.where(
        np.isin(types, ["PAYMENT", "DEBIT"]), name_dest_m, name_dest_c
    )

    # ── 4. Origin balances ────────────────────────────────────────────────────
    # Origin starting balance ~ log-normal, must be >= amount for legit txns
    old_bal_orig = np.exp(rng.normal(9.5, 1.5, n_rows)).round(2)
    # Ensure legitimate transactions can afford the amount
    old_bal_orig = np.maximum(old_bal_orig, amounts + 1.0)

    old_bal_dest = np.exp(rng.normal(8.5, 1.6, n_rows)).round(2)
    merchant_mask = np.isin(types, ["PAYMENT", "DEBIT"])
    old_bal_dest = np.where(merchant_mask, 0.0, old_bal_dest)

    # ── 5. Fraud labeling (before balance computation) ────────────────────────
    fraud_eligible_mask = np.isin(types, ["TRANSFER", "CASH_OUT"])
    hour_of_day = steps % 24
    night_mask = np.isin(hour_of_day, list(range(0, 6)))

    # Base fraud probability
    base_prob = np.where(
        fraud_eligible_mask,
        np.where(night_mask, 0.004, 0.0015),
        0.0
    )

    # Pattern 4: Large amounts relative to typical for the type
    eligible_idx = np.where(fraud_eligible_mask)[0]
    large_thresh = np.percentile(amounts[eligible_idx], 90) if len(eligible_idx) > 0 else 1e9
    large_mask = amounts > large_thresh
    base_prob = np.where(
        fraud_eligible_mask & large_mask,
        np.minimum(base_prob * 8, 0.40),
        base_prob,
    )

    # Draw initial fraud labels
    is_fraud = (rng.uniform(0, 1, n_rows) < base_prob).astype(int)

    # Rescale to hit target fraud rate
    current_rate = is_fraud.mean()
    if current_rate > fraud_rate_target and current_rate > 0:
        scale = fraud_rate_target / current_rate
        fraud_idx = np.where(is_fraud == 1)[0]
        n_keep = int(len(fraud_idx) * scale)
        keep = rng.choice(fraud_idx, size=n_keep, replace=False)
        is_fraud[:] = 0
        is_fraud[keep] = 1

    fraud_mask = is_fraud == 1

    # ── 6. Compute balances ───────────────────────────────────────────────────
    # For LEGITIMATE transactions: correct balance updates
    new_bal_orig = np.maximum(0.0, old_bal_orig - amounts).round(2)
    new_bal_dest = (old_bal_dest + amounts).round(2)
    new_bal_dest = np.where(merchant_mask, 0.0, new_bal_dest)

    # Pattern 1 & 2: Account takeover + mule chain
    # Only 60% of fraud does a full drain (rest does partial drain — overlap with legit)
    takeover_fraud = fraud_mask & (types == "TRANSFER")
    cashout_fraud  = fraud_mask & (types == "CASH_OUT")

    # 60% of fraud = full drain, 40% = partial (more realistic, causes overlap)
    full_drain = rng.uniform(0, 1, n_rows) < 0.60
    drain_fraction = np.where(full_drain, rng.uniform(0.90, 1.0, n_rows),
                               rng.uniform(0.40, 0.89, n_rows))
    fraud_amount_override = (old_bal_orig * drain_fraction).round(2)

    amounts = np.where(
        (takeover_fraud | cashout_fraud) & full_drain,
        np.clip(fraud_amount_override, 1.0, 1_000_000.0),
        amounts,
    )

    # Only full-drain fraud completely zeros the origin balance
    new_bal_orig = np.where(
        (takeover_fraud | cashout_fraud) & full_drain,
        0.0,
        np.maximum(0.0, old_bal_orig - amounts).round(2),
    )

    # Mule pattern: dest balance unchanged only for 70% of TRANSFER fraud
    mule_mask = takeover_fraud & (rng.uniform(0, 1, n_rows) < 0.70)
    new_bal_dest = np.where(
        mule_mask,
        old_bal_dest,          # dest unchanged = mule signal
        new_bal_dest,
    )

    # CASH_OUT fraud: dest is cash agent (balance stays 0)
    new_bal_dest = np.where(
        cashout_fraud,
        0.0,
        new_bal_dest,
    )

    # Add moderate noise to ALL transactions (legitimate AND fraud)
    # This creates realistic overlap and prevents trivial separation
    noise_scale = 0.08   # 8% noise
    old_bal_orig = (old_bal_orig * (1 + rng.normal(0, noise_scale, n_rows))).clip(1.0).round(2)
    old_bal_dest = (old_bal_dest * (1 + rng.normal(0, noise_scale, n_rows))).clip(0.0).round(2)
    old_bal_dest = np.where(merchant_mask, 0.0, old_bal_dest)

    # Add smaller noise to new balances (legitimate ones) — fraud patterns preserved
    legit_mask = is_fraud == 0
    new_bal_orig = np.where(
        legit_mask,
        (new_bal_orig * (1 + rng.normal(0, 0.03, n_rows))).clip(0.0).round(2),
        new_bal_orig,
    )
    new_bal_dest = np.where(
        legit_mask & ~merchant_mask,
        (new_bal_dest * (1 + rng.normal(0, 0.03, n_rows))).clip(0.0).round(2),
        new_bal_dest,
    )


    # ── 7. Assemble DataFrame ─────────────────────────────────────────────────
    df = pd.DataFrame({
        "step":            steps,
        "type":            types,
        "amount":          amounts,
        "nameOrig":        name_orig,
        "oldbalanceOrg":   old_bal_orig,
        "newbalanceOrig":  new_bal_orig,
        "nameDest":        name_dest,
        "oldbalanceDest":  old_bal_dest,
        "newbalanceDest":  new_bal_dest,
        "isFraud":         is_fraud,
    })

    # Sort by step so time-based split is meaningful
    df = df.sort_values("step").reset_index(drop=True)
    df["txn_id"] = np.arange(len(df))

    return df


def load_or_generate(cfg: dict) -> pd.DataFrame:
    """Load real PaySim CSV or generate synthetic data based on config."""
    mode = cfg["data"]["mode"]
    root = Path(cfg["_root"])

    if mode == "REAL_PAYSIM":
        csv_path = root / cfg["data"]["real_paysim"]["path"]
        if not csv_path.exists():
            raise FileNotFoundError(
                f"PaySim CSV not found at {csv_path}.\n"
                "Download from: https://www.kaggle.com/datasets/ntnu-testimon/paysim1\n"
                "Place it at data/raw/PS_20174392719_1491204439457_log.csv\n"
                "Then set data.mode: REAL_PAYSIM in config.yaml"
            )
        sample = cfg["data"]["real_paysim"]["sample_rows"]
        print(f"[data] Loading real PaySim CSV from {csv_path}")
        df = pd.read_csv(csv_path, nrows=sample)
        if "isFlaggedFraud" in df.columns:
            df = df.rename(columns={"isFlaggedFraud": "_isFlaggedFraud"})
        df["txn_id"] = np.arange(len(df))
        return df
    else:
        syn_cfg = cfg["data"]["synthetic"]
        print(
            f"[data] Generating {syn_cfg['n_rows']:,} synthetic rows "
            f"over {syn_cfg['n_steps']} steps (seed={syn_cfg['random_seed']})"
        )
        df = generate(
            n_rows=syn_cfg["n_rows"],
            n_steps=syn_cfg["n_steps"],
            random_seed=syn_cfg["random_seed"],
            fraud_rate_target=syn_cfg["fraud_rate_target"],
        )
        return df


if __name__ == "__main__":
    from src.config import get_config
    cfg = get_config()
    df = load_or_generate(cfg)
    print(df.shape)
    print(df["isFraud"].value_counts())
    print("\nFraud by type:")
    print(df.groupby("type")["isFraud"].sum())
    print(df.dtypes)
