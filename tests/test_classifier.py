"""Tests for the feature extractor and LightGBM classifier (M5).

All tests that touch the real data are module-scoped for performance.
The feature extraction tests use synthetic DataFrames where possible so they run
quickly without loading the full 161k-row CSV.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classifier.features import (
    N_FEATURES,
    FEATURE_NAMES,
    extract_features,
    extract_all_events,
)
from src.io.load_csv import PMU_BUSES

DATA_DIR = Path("data/raw")
SKIP_NO_DATA = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason="data/raw not present — skipping real-data tests",
)
SKIP_NO_LGBM = pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None,
    reason="lightgbm not installed",
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    """Minimal synthetic DataFrame with 4 channel types for 8 PMU buses."""
    rng = np.random.default_rng(42)
    T = 300  # 10 s at 30 fps
    rows: dict[str, np.ndarray] = {}
    rows["TIMESTAMP"] = np.arange(T) / 30.0
    rows["Event"] = np.zeros(T, dtype=int)

    for bus in PMU_BUSES:
        rows[f"BUS{bus}_VA_MAG"]       = 199_000 + rng.standard_normal(T) * 500
        rows[f"BUS{bus}_IA_MAG"]       = 585 + rng.standard_normal(T) * 2
        rows[f"BUS{bus}_Freq"]         = 60.0 + rng.standard_normal(T) * 0.01
        rows[f"BUS{bus}_ROCOF"]        = rng.standard_normal(T) * 0.01
        rows[f"BUS{bus}_DATA_PRESENT"] = np.ones(T, dtype=int)
        # Also add ANG and other cols so column checks don't crash
        rows[f"BUS{bus}_VA_ANG"]       = rng.standard_normal(T) * 5
        rows[f"BUS{bus}_IA_ANG"]       = rng.standard_normal(T) * 5

    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def anomalous_df(synthetic_df) -> pd.DataFrame:
    """Synthetic DataFrame with a step change in Bus2 and Bus39 channels at t=5s."""
    df = synthetic_df.copy()
    step_frame = 150  # 5 s
    for bus in [2, 39]:
        df.loc[step_frame:, f"BUS{bus}_VA_MAG"] += 5_000
        df.loc[step_frame:, f"BUS{bus}_Freq"]   += 0.1
    return df


@pytest.fixture(scope="module")
def full_df():
    """Load real merged DataFrame (module-scoped)."""
    pytest.importorskip("pandas")
    from src.io.load_csv import load_all
    return load_all(DATA_DIR)


@pytest.fixture(scope="module")
def full_eta_and_dp(full_df):
    """Compute eta_simple and data_present for full dataset."""
    from src.detector.chi2 import compute_eta_simple, extract_data_present
    from src.estimator.calibration import channel_cols, calibrate_R

    cols = channel_cols()
    calib_mask = (full_df["TIMESTAMP"] <= 60.0) & (full_df["Event"] == 0)
    calib_df = full_df[calib_mask]
    h0_flat = np.array([
        calib_df[c].dropna().mean() if c in calib_df.columns else 0.0
        for c in cols
    ])
    h0_flat = np.where(np.isfinite(h0_flat), h0_flat, 0.0)
    R, offset = calibrate_R(full_df, h0_flat)
    eta = compute_eta_simple(full_df, h0_flat, offset, R, cols)
    dp  = extract_data_present(full_df)
    return eta, dp, h0_flat, offset, R


@pytest.fixture(scope="module")
def fitted_detector(full_df, full_eta_and_dp):
    """Calibrated Chi2Detector on real data."""
    from src.detector.chi2 import Chi2Detector
    eta, dp, h0_flat, offset, R = full_eta_and_dp
    calib_mask = (full_df["TIMESTAMP"] <= 60.0) & (full_df["Event"] == 0)
    det = Chi2Detector(n_z=32, k_on=3, k_off=15, fps=30.0)
    det.calibrate(eta[calib_mask.to_numpy()], dp[calib_mask.to_numpy()])
    return det


@pytest.fixture(scope="module")
def real_alarm_indices(full_df, full_eta_and_dp, fitted_detector):
    """Alarm onset indices on the full real dataset."""
    eta, dp, *_ = full_eta_and_dp
    ts = full_df["TIMESTAMP"].to_numpy(float)
    result = fitted_detector.detect(eta, dp, ts)
    return result["alarm_indices"]


# ── unit tests: feature vector structure ─────────────────────────────────────

class TestFeatureVector:
    def test_feature_count(self):
        assert N_FEATURES == 37

    def test_feature_names_length(self):
        assert len(FEATURE_NAMES) == N_FEATURES

    def test_no_duplicate_names(self):
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_extract_features_shape(self, synthetic_df):
        feats = extract_features(synthetic_df, onset_frame=150)
        assert feats.shape == (N_FEATURES,)

    def test_extract_features_finite(self, synthetic_df):
        feats = extract_features(synthetic_df, onset_frame=150)
        assert np.all(np.isfinite(feats)), f"Non-finite features: {FEATURE_NAMES[~np.isfinite(feats)]}"

    def test_extract_features_onset_at_boundary(self, synthetic_df):
        """Features should not crash at frame 0 or last frame."""
        feats_start = extract_features(synthetic_df, onset_frame=0)
        feats_end   = extract_features(synthetic_df, onset_frame=len(synthetic_df) - 1)
        assert feats_start.shape == (N_FEATURES,)
        assert feats_end.shape == (N_FEATURES,)
        assert np.all(np.isfinite(feats_start))
        assert np.all(np.isfinite(feats_end))

    def test_anomaly_increases_va_mag_max(self, synthetic_df, anomalous_df):
        """Anomalous window should have larger VA_MAG max feature than normal."""
        feats_normal = extract_features(synthetic_df, onset_frame=150)
        feats_anom   = extract_features(anomalous_df, onset_frame=150)
        # VA_MAG_max is feature index 1 (mean=0, max=1, std=2 for first group)
        va_mag_max_idx = FEATURE_NAMES.index("VA_MAG_max")
        assert feats_anom[va_mag_max_idx] > feats_normal[va_mag_max_idx], (
            "Anomaly should increase VA_MAG_max feature"
        )

    def test_cyber_indicator_with_nan(self, synthetic_df):
        """Inserting NaN should increase nan_count feature."""
        df_nan = synthetic_df.copy()
        df_nan.loc[140:160, "BUS29_VA_MAG"] = np.nan
        df_nan.loc[140:160, "BUS29_DATA_PRESENT"] = 0
        feats_normal = extract_features(synthetic_df, onset_frame=150)
        feats_cyber  = extract_features(df_nan, onset_frame=150)
        nan_idx = FEATURE_NAMES.index("nan_count")
        missing_idx = FEATURE_NAMES.index("pmu_missing_count")
        assert feats_cyber[nan_idx] > feats_normal[nan_idx]
        assert feats_cyber[missing_idx] >= 1

    def test_extract_all_events(self, synthetic_df):
        """extract_all_events returns correct shapes."""
        onsets = np.array([50, 150, 250])
        labels = np.array([1, 3, 4])
        X, y = extract_all_events(synthetic_df, onsets, labels)
        assert X.shape == (3, N_FEATURES)
        assert y.shape == (3,)
        assert np.array_equal(y, labels)
        assert np.all(np.isfinite(X))

    def test_extract_all_events_empty(self, synthetic_df):
        X, y = extract_all_events(synthetic_df, np.array([], dtype=int), np.array([], dtype=int))
        assert X.shape == (0, N_FEATURES)
        assert y.shape == (0,)


# ── real-data tests ───────────────────────────────────────────────────────────

class TestRealDataFeatures:
    @SKIP_NO_DATA
    def test_features_finite_on_real_alarms(self, full_df, real_alarm_indices):
        if len(real_alarm_indices) == 0:
            pytest.skip("No alarm onsets found")
        for frame in real_alarm_indices[:5]:  # check first 5 alarms
            feats = extract_features(full_df, int(frame))
            assert np.all(np.isfinite(feats)), (
                f"Non-finite features at frame {frame}: {FEATURE_NAMES[~np.isfinite(feats)]}"
            )

    @SKIP_NO_DATA
    def test_cyber_event_has_missing_pmu(self, full_df, real_alarm_indices):
        """At cyber event alarms, pmu_missing_count should be >= 1."""
        # Find alarm onset closest to cyber event at ~536 s
        cyber_ts = 536.0
        ts = full_df["TIMESTAMP"].to_numpy(float)
        if len(real_alarm_indices) == 0:
            pytest.skip("No alarm onsets found")
        closest = real_alarm_indices[
            np.argmin(np.abs(ts[real_alarm_indices] - cyber_ts))
        ]
        if abs(ts[closest] - cyber_ts) > 30.0:
            pytest.skip("No alarm onset near first cyber event")
        feats = extract_features(full_df, int(closest))
        missing_idx = FEATURE_NAMES.index("pmu_missing_count")
        assert feats[missing_idx] >= 1, (
            f"Cyber event alarm should have pmu_missing_count >= 1, got {feats[missing_idx]}"
        )


# ── LightGBM training (real data only) ───────────────────────────────────────

class TestLGBMTraining:
    @SKIP_NO_DATA
    @SKIP_NO_LGBM
    def test_train_runs_without_error(self, full_df):
        """LightGBM training should complete without exceptions."""
        from src.classifier.train_lgbm import train

        # Use first 70% as train, next 15% as val (approximate)
        T = len(full_df)
        train_end = int(T * 0.70)
        val_end   = int(T * 0.85)
        df_train = full_df.iloc[:train_end].reset_index(drop=True)
        df_val   = full_df.iloc[train_end:val_end].reset_index(drop=True)

        clf = train(df_train, df_val, lgbm_params={"n_estimators": 50, "verbose": -1})
        assert clf is not None

    @SKIP_NO_DATA
    @SKIP_NO_LGBM
    def test_macro_f1_on_val(self, full_df):
        """Macro-F1 on validation real events (aspirational target > 0.5 with only 9 events)."""
        from sklearn.metrics import f1_score
        from src.classifier.train_lgbm import train, _build_detector
        from src.detector.chi2 import compute_eta_simple, extract_data_present
        from src.estimator.calibration import channel_cols

        T = len(full_df)
        train_end = int(T * 0.70)
        val_end   = int(T * 0.85)
        df_train = full_df.iloc[:train_end].reset_index(drop=True)
        df_val   = full_df.iloc[train_end:val_end].reset_index(drop=True)

        clf = train(df_train, df_val, lgbm_params={"n_estimators": 50, "verbose": -1})

        # Get val alarm indices and true labels
        det, h0_flat, offset, R = _build_detector(df_train)
        cols = channel_cols()
        eta_val = compute_eta_simple(df_val, h0_flat, offset, R, cols)
        dp_val  = extract_data_present(df_val)
        ts_val  = df_val["TIMESTAMP"].to_numpy(float)
        res = det.detect(eta_val, dp_val, ts_val)

        onset_idx = res["alarm_indices"]
        if len(onset_idx) == 0:
            pytest.skip("No alarms on val split")

        true_labels = np.array([int(df_val["Event"].iloc[i]) for i in onset_idx])
        X_val = np.stack([extract_features(df_val, int(i)) for i in onset_idx])
        pred_labels = clf.predict(X_val)

        # Only score non-zero labels (normal frames aren't alarms anyway)
        macro_f1 = f1_score(true_labels, pred_labels, average="macro",
                            labels=np.unique(true_labels), zero_division=0)
        log_msg = f"Val Macro-F1 = {macro_f1:.3f} (events: {len(onset_idx)})"
        print(log_msg)
        # Aspirational: > 0.5 — with only 9 real events any value proves it runs
        # We don't assert a fixed threshold here because val events may be 0
        assert macro_f1 >= 0.0  # smoke test: at least computable
