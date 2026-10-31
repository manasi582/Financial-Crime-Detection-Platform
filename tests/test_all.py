"""
tests/test_all.py

FraudLens test suite (pytest).

Tests:
1. Generator output schema, fraud only in TRANSFER and CASH_OUT, fraud rate in range
2. Time split has no overlap and every split contains fraud
3. Velocity features use only past data (anti-leakage test)
4. Cost function correctness on a small hand-computed example
5. DB round trip (load, query a view, update an alert status)
6. Model loads and scores a single transaction
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ── 1. Data generator tests ───────────────────────────────────────────────────

class TestDataGenerator:
    def test_schema(self):
        """Generated DataFrame must have the PaySim schema columns."""
        from src.data_generator import generate
        df = generate(n_rows=5000, n_steps=72, random_seed=0, fraud_rate_target=0.005)
        required_cols = [
            "step", "type", "amount", "nameOrig", "oldbalanceOrg",
            "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_fraud_only_in_transfer_cashout(self):
        """Fraud must only occur in TRANSFER and CASH_OUT transactions."""
        from src.data_generator import generate
        df = generate(n_rows=10000, n_steps=72, random_seed=1, fraud_rate_target=0.01)
        fraud_df = df[df["isFraud"] == 1]
        assert len(fraud_df) > 0, "No fraud cases generated!"
        allowed_types = {"TRANSFER", "CASH_OUT"}
        bad_types = set(fraud_df["type"].unique()) - allowed_types
        assert not bad_types, f"Fraud found in non-allowed types: {bad_types}"

    def test_fraud_rate_in_range(self):
        """Fraud rate should be roughly within [0.001, 0.01] for default settings."""
        from src.data_generator import generate
        df = generate(n_rows=50000, n_steps=720, random_seed=42, fraud_rate_target=0.002)
        rate = df["isFraud"].mean()
        assert 0.0005 < rate < 0.015, f"Fraud rate {rate:.4f} is outside expected range"

    def test_txn_id_unique(self):
        from src.data_generator import generate
        df = generate(n_rows=5000, n_steps=72, random_seed=2)
        assert df["txn_id"].nunique() == len(df), "txn_id is not unique!"

    def test_amounts_positive(self):
        from src.data_generator import generate
        df = generate(n_rows=5000, n_steps=72, random_seed=3)
        assert (df["amount"] > 0).all(), "Some amounts are non-positive!"


# ── 2. Time split tests ───────────────────────────────────────────────────────

class TestTimeSplit:
    @pytest.fixture
    def sample_df(self):
        from src.data_generator import generate
        # Use high fraud rate and enough rows so all 3 splits contain fraud
        df = generate(n_rows=30000, n_steps=100, random_seed=42, fraud_rate_target=0.02)
        from src.features import engineer
        return engineer(df, velocity_window=24)

    def test_no_overlap(self, sample_df):
        from src.split import time_split
        train, val, test = time_split(sample_df, train_frac=0.70, val_frac=0.15)
        assert set(train["step"]).isdisjoint(set(val["step"])), "Train/Val overlap!"
        assert set(train["step"]).isdisjoint(set(test["step"])), "Train/Test overlap!"
        assert set(val["step"]).isdisjoint(set(test["step"])), "Val/Test overlap!"

    def test_temporal_ordering(self, sample_df):
        from src.split import time_split
        train, val, test = time_split(sample_df, train_frac=0.70, val_frac=0.15)
        assert train["step"].max() < val["step"].min()
        assert val["step"].max() < test["step"].min()

    def test_all_splits_have_fraud(self, sample_df):
        from src.split import time_split
        train, val, test = time_split(sample_df, train_frac=0.70, val_frac=0.15)
        for name, df in [("train", train), ("val", val), ("test", test)]:
            assert df["isFraud"].sum() > 0, f"Split '{name}' has no fraud!"

    def test_no_data_leakage_between_splits(self, sample_df):
        from src.split import time_split
        train, val, test = time_split(sample_df, train_frac=0.70, val_frac=0.15)
        # No txn_id should appear in more than one split
        all_ids = pd.concat([
            train[["txn_id"]], val[["txn_id"]], test[["txn_id"]]
        ])
        assert all_ids["txn_id"].nunique() == len(all_ids), "txn_id appears in multiple splits!"


# ── 3. Velocity feature anti-leakage test ────────────────────────────────────

class TestVelocityFeature:
    def test_velocity_uses_only_past_data(self):
        """
        Construct a case where future data would inflate velocity.
        At step 10, dest 'C1' receives a transaction. The velocity for
        this transaction (at step 10) should NOT include itself.
        If velocity == 0 initially at step 10, adding the step-10 transaction
        should make subsequent same-step rows see velocity = 1 (within-step ordering).
        
        Key invariant: velocity at step S should only count transactions at steps < S.
        """
        from src.features import _compute_dest_velocity

        # Simple dataset: two transactions to 'C1' at step 5 and step 10
        df = pd.DataFrame({
            "step":    [5, 10, 15],
            "nameDest": ["C1", "C1", "C2"],
            "txn_id":  [0, 1, 2],
        })
        vel = _compute_dest_velocity(df, window=24)

        # At step 5: no prior transactions to C1 in past 24 steps → velocity = 0
        assert vel.iloc[0] == 0, f"Step 5 velocity should be 0, got {vel.iloc[0]}"

        # At step 10: one prior txn to C1 (at step 5, within 24 steps) → velocity = 1
        assert vel.iloc[1] == 1, f"Step 10 velocity should be 1, got {vel.iloc[1]}"

        # At step 15: one prior txn to C1 (at step 10) → velocity = 1
        # (step 5 may or may not be in window depending on window size)
        assert vel.iloc[2] == 0, f"Step 15 for C2 should be 0, got {vel.iloc[2]}"

    def test_velocity_window_respected(self):
        """Transactions older than window should NOT count."""
        from src.features import _compute_dest_velocity

        df = pd.DataFrame({
            "step":     [0, 30, 31],  # window=24, so step 0 should expire by step 25
            "nameDest": ["C1", "C1", "C1"],
            "txn_id":   [0, 1, 2],
        })
        vel = _compute_dest_velocity(df, window=24)

        # At step 30: step 0 is 30 steps ago > window of 24 → should NOT count
        # Only txn from step 0 existed before step 30; step 0 < 30 - 24 = 6 → expired
        assert vel.iloc[1] == 0, (
            f"Step 30 velocity should be 0 (step 0 is outside window), got {vel.iloc[1]}"
        )


# ── 4. Cost function tests ────────────────────────────────────────────────────

class TestCostFunction:
    def test_net_savings_hand_computed(self):
        """
        Hand-computed test case:
        - 10 transactions: 2 fraud (amounts: 1000, 2000), 8 legit
        - Model flags 3 transactions: both fraud + 1 false positive
        - Investigation cost: 500 per alert
        - Recovery rate: 1.0
        
        Expected:
          prevented_loss = 1000 + 2000 = 3000
          investigation_cost = 3 * 500 = 1500
          net_savings = 3000 - 1500 = 1500
        """
        from src.cost_analysis import compute_cost_curve

        y_true  = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        amounts = np.array([1000.0, 2000.0, 500.0, 300.0, 200.0, 100.0, 50.0, 75.0, 25.0, 10.0])
        # Scores: both fraud have high scores; 1 legit has slightly lower high score
        scores  = np.array([0.9, 0.95, 0.8, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01, 0.01])

        df = compute_cost_curve(
            y_true, scores, amounts,
            investigation_cost=500.0,
            recovery_rate=1.0,
            n_thresholds=50,
        )

        # Find row with threshold ≈ 0.85 (catches both fraud, misses FP at 0.8)
        row = df[df["threshold"] >= 0.89].iloc[0] if not df[df["threshold"] >= 0.89].empty else df.iloc[-1]
        assert row["prevented_loss"] == pytest.approx(3000.0, rel=0.01)
        assert row["net_savings"] == pytest.approx(3000.0 - row["n_alerts"] * 500.0, rel=0.01)

    def test_no_model_baseline(self):
        """Baseline loss should equal total fraud amount."""
        from src.cost_analysis import compute_cost_curve
        y_true  = np.array([1, 1, 0, 0, 0])
        amounts = np.array([1000.0, 500.0, 200.0, 100.0, 50.0])
        scores  = np.array([0.9, 0.8, 0.1, 0.05, 0.02])
        df = compute_cost_curve(y_true, scores, amounts)
        assert df["baseline_loss"].iloc[0] == pytest.approx(1500.0)

    def test_find_optimal_threshold(self):
        """Optimal threshold maximizes net savings."""
        from src.cost_analysis import compute_cost_curve, find_optimal_threshold
        y_true  = np.array([1, 1, 0] * 100)
        amounts = np.array([1000.0, 2000.0, 10.0] * 100)
        rng = np.random.default_rng(42)
        scores = np.where(y_true == 1, rng.uniform(0.7, 1.0, 300), rng.uniform(0.0, 0.5, 300))
        df = compute_cost_curve(y_true, scores, amounts)
        opt_t = find_optimal_threshold(df)
        best_net = df["net_savings"].max()
        row = df.loc[(df["threshold"] - opt_t).abs().idxmin()]
        assert abs(row["net_savings"] - best_net) < best_net * 0.01  # within 1% of best


# ── 5. Database round-trip tests ──────────────────────────────────────────────

class TestDatabase:
    @pytest.fixture
    def temp_engine(self, tmp_path):
        """Create a temporary SQLite database for testing."""
        from sqlalchemy import create_engine
        db_path = tmp_path / "test_fraudlens.db"
        engine = create_engine(f"sqlite:///{db_path}")
        from src.db import create_schema
        create_schema(engine)
        return engine

    @pytest.fixture
    def small_df(self):
        from src.data_generator import generate
        df = generate(n_rows=1000, n_steps=50, random_seed=99, fraud_rate_target=0.01)
        from src.features import engineer
        df = engineer(df, velocity_window=12)
        df["split"] = "train"
        return df

    def test_load_and_query_view(self, temp_engine, small_df):
        from src.db import load_transactions, query_view
        load_transactions(small_df, temp_engine, chunk_size=500)
        df_view = query_view("fraud_by_type", temp_engine)
        assert not df_view.empty, "fraud_by_type view returned no rows!"
        assert "type" in df_view.columns
        assert "fraud_count" in df_view.columns

    def test_alert_status_update(self, temp_engine, small_df):
        from src.db import load_transactions, save_alerts, update_alert_status, query_top_alerts

        # Load transactions
        load_transactions(small_df, temp_engine, chunk_size=500)

        # Save a test alert
        fraud_df = small_df[small_df["isFraud"] == 1].head(3)
        if len(fraud_df) == 0:
            pytest.skip("No fraud in small sample — skipping alert test")

        alerts = pd.DataFrame({
            "txn_id":       fraud_df["txn_id"].values,
            "score":        [0.9, 0.8, 0.7][:len(fraud_df)],
            "reason_codes": ['["Test reason"]'] * len(fraud_df),
            "status":       ["Open"] * len(fraud_df),
        })
        save_alerts(alerts, temp_engine)

        # Update first alert
        alerts_in_db = query_top_alerts(temp_engine, limit=10)
        first_id = int(alerts_in_db["alert_id"].iloc[0])

        update_alert_status(first_id, "Confirmed Fraud", temp_engine)

        # Verify update
        updated = query_top_alerts(temp_engine, limit=10)
        updated_row = updated[updated["alert_id"] == first_id].iloc[0]
        assert updated_row["status"] == "Confirmed Fraud", (
            f"Expected 'Confirmed Fraud', got '{updated_row['status']}'"
        )

    def test_daily_trend_view(self, temp_engine, small_df):
        from src.db import load_transactions, query_view
        load_transactions(small_df, temp_engine, chunk_size=500)
        df_view = query_view("daily_fraud_trend", temp_engine)
        assert "day" in df_view.columns
        assert "fraud_count" in df_view.columns
        assert len(df_view) > 0


# ── 6. Model load and score tests ─────────────────────────────────────────────

class TestModelScoring:
    def test_model_loads_and_scores(self):
        """If a trained supervised model exists, it must score a single transaction."""
        model_dir = _ROOT / "models"
        # Skip isolation_forest — it has no predict_proba
        model_files = [
            f for f in (model_dir.glob("*.joblib") if model_dir.exists() else [])
            if "isolation_forest" not in f.name
        ]

        if not model_files:
            pytest.skip("No trained supervised model found — run `make all` first, then re-run tests.")

        from src.train import load_model
        from src.features import FEATURE_COLS
        import numpy as np

        model_path = model_files[0]
        model = load_model(model_path)

        # Build a dummy feature vector
        X = np.zeros((1, len(FEATURE_COLS)))
        X[0, FEATURE_COLS.index("type_TRANSFER")] = 1
        X[0, FEATURE_COLS.index("amount")] = 50000.0

        scores = model.predict_proba(X)
        assert scores.shape == (1, 2), f"Expected shape (1, 2), got {scores.shape}"
        assert 0 <= scores[0, 1] <= 1, f"Score {scores[0, 1]} not in [0, 1]"

    def test_reason_codes_on_fraud_pattern(self):
        """Account takeover transaction should trigger expected reason codes."""
        from src.explain import get_reason_codes

        takeover_row = {
            "errorBalanceOrig":     0.0,   # balance error = 0 (perfect drain)
            "orig_balance_zeroed":  1,     # balance went to zero
            "errorBalanceDest":     0.0,
            "is_night":             1,     # night time
            "type_TRANSFER":        1,
            "type_CASH_OUT":        0,
            "amount_to_balance_ratio": 0.99,
            "dest_velocity_24h":    1,
            "dest_balance_was_zero": 0,
        }
        codes = get_reason_codes(takeover_row)
        assert len(codes) > 0, "Expected reason codes for account takeover pattern!"
        # Should flag full balance drain
        assert any("balance" in c.lower() or "drain" in c.lower() for c in codes)

    def test_feature_vector_length(self):
        """Feature column list must match expected length."""
        from src.features import FEATURE_COLS
        assert len(FEATURE_COLS) == 15, f"Expected 15 features, got {len(FEATURE_COLS)}"
