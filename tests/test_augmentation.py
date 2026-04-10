"""Tests for synthetic event generators (src/augmentation/andes_sim.py).

All tests use synthetic DataFrames constructed in-memory — no real data required.
The only real-data-dependent test (generate_all) is gated behind SKIP_NO_DATA.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.io.load_csv import PMU_BUSES
from src.augmentation.andes_sim import (
    FPS,
    _SUFFIXES,
    SyntheticWindow,
    extract_normal_baseline,
    generate_fault,
    generate_gen_change,
    generate_line_outage,
    generate_load_change,
    generate_pmu_dropout,
    load_synthetic,
    _write_synthetic_csvs,
)

DATA_DIR = Path("data/raw")
SKIP_NO_DATA = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason="data/raw not present — skipping real-data tests",
)

# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def minimal_stats() -> dict:
    """Minimal baseline stats dict sufficient for all generators."""
    stats: dict = {}
    rng = np.random.default_rng(0)
    for bus in PMU_BUSES:
        stats[f"BUS{bus}_VA_MAG"] = (200_000.0, 500.0)
        stats[f"BUS{bus}_VB_MAG"] = (200_000.0, 500.0)
        stats[f"BUS{bus}_VC_MAG"] = (200_000.0, 500.0)
        stats[f"BUS{bus}_VA_ANG"] = (-5.0, 0.1)
        stats[f"BUS{bus}_VB_ANG"] = (-125.0, 0.1)
        stats[f"BUS{bus}_VC_ANG"] = (115.0, 0.1)
        stats[f"BUS{bus}_IA_MAG"] = (585.0, 2.0)
        stats[f"BUS{bus}_IB_MAG"] = (585.0, 2.0)
        stats[f"BUS{bus}_IC_MAG"] = (585.0, 2.0)
        stats[f"BUS{bus}_IA_ANG"] = (-30.0, 0.5)
        stats[f"BUS{bus}_IB_ANG"] = (-150.0, 0.5)
        stats[f"BUS{bus}_IC_ANG"] = (90.0, 0.5)
        stats[f"BUS{bus}_Freq"]   = (60.0, 0.005)
        stats[f"BUS{bus}_ROCOF"]  = (0.0, 0.002)
    return stats


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ── SyntheticWindow ────────────────────────────────────────────────────────────

class TestSyntheticWindow:
    def test_all_channels_present(self, minimal_stats, rng):
        win = SyntheticWindow(bus=2, T=30, stats=minimal_stats, rng=rng)
        for suf in _SUFFIXES:
            assert f"BUS2_{suf}" in win.data

    def test_shape(self, minimal_stats, rng):
        T = 45
        win = SyntheticWindow(bus=39, T=T, stats=minimal_stats, rng=rng)
        for arr in win.data.values():
            assert arr.shape == (T,), f"Expected ({T},), got {arr.shape}"

    def test_set_and_col(self, minimal_stats, rng):
        win = SyntheticWindow(bus=5, T=20, stats=minimal_stats, rng=rng)
        custom = np.arange(20, dtype=float)
        win.set("Freq", custom)
        np.testing.assert_array_equal(win.col("Freq"), custom)


# ── generate_fault ─────────────────────────────────────────────────────────────

class TestGenerateFault:
    @pytest.mark.parametrize("event_bus", [2, 39])
    def test_output_shape(self, minimal_stats, rng, event_bus):
        bus_data, ts, labels = generate_fault(event_bus, minimal_stats, rng, window_sec=6.0)
        T = int(round(6.0 * FPS))
        assert ts.shape == (T,)
        assert labels.shape == (T,)
        for bus in PMU_BUSES:
            for suf in _SUFFIXES:
                col = f"BUS{bus}_{suf}"
                assert col in bus_data, f"Missing column {col}"
                assert bus_data[col].shape == (T,)

    def test_event_label_1_during_fault(self, minimal_stats, rng):
        _, ts, labels = generate_fault(2, minimal_stats, rng, t_event=2.0, clear_cycles=5)
        assert np.any(labels == 1), "Expected label=1 during fault"
        assert labels[0] == 0, "Pre-event should be label=0"

    def test_voltage_drops_at_fault_bus(self, minimal_stats, rng):
        """Faulted bus should have lower VA_MAG during fault than before."""
        bus_data, ts, labels = generate_fault(39, minimal_stats, rng, t_event=2.0,
                                              fault_impedance=0.0, clear_cycles=5)
        fault_mask = labels == 1
        pre_mask = ts < 2.0
        v_during = bus_data["BUS39_VA_MAG"][fault_mask].mean()
        v_before = bus_data["BUS39_VA_MAG"][pre_mask].mean()
        assert v_during < v_before, (
            f"Fault bus voltage should drop during fault: {v_during:.1f} vs {v_before:.1f}"
        )

    def test_finite_outputs(self, minimal_stats, rng):
        bus_data, ts, labels = generate_fault(2, minimal_stats, rng)
        for col, arr in bus_data.items():
            assert np.all(np.isfinite(arr)), f"Non-finite values in {col}"

    def test_timestamps_monotone(self, minimal_stats, rng):
        _, ts, _ = generate_fault(5, minimal_stats, rng)
        assert np.all(np.diff(ts) > 0), "Timestamps should be strictly increasing"


# ── generate_gen_change ────────────────────────────────────────────────────────

class TestGenerateGenChange:
    def test_output_shape(self, minimal_stats, rng):
        bus_data, ts, labels = generate_gen_change(39, minimal_stats, rng)
        T = int(round(6.0 * FPS))
        assert ts.shape == (T,)
        assert labels.shape == (T,)

    def test_event_label_3(self, minimal_stats, rng):
        _, _, labels = generate_gen_change(2, minimal_stats, rng, t_event=2.0)
        assert np.any(labels == 3), "Expected label=3 during gen change"

    def test_rocof_nonzero_after_event(self, minimal_stats, rng):
        """ROCOF should be nonzero after the event starts."""
        bus_data, ts, labels = generate_gen_change(39, minimal_stats, rng,
                                                    t_event=2.0, delta_mw=50.0)
        post_mask = ts > 2.1
        rocof_post = bus_data["BUS39_ROCOF"][post_mask]
        assert np.any(np.abs(rocof_post) > 1e-3), "ROCOF should be nonzero after gen step"

    def test_finite_outputs(self, minimal_stats, rng):
        bus_data, _, _ = generate_gen_change(30, minimal_stats, rng)
        for col, arr in bus_data.items():
            assert np.all(np.isfinite(arr))


# ── generate_line_outage ───────────────────────────────────────────────────────

class TestGenerateLineOutage:
    def test_output_shape(self, minimal_stats, rng):
        bus_data, ts, labels = generate_line_outage(16, 19, minimal_stats, rng)
        T = int(round(6.0 * FPS))
        assert ts.shape == (T,)
        assert labels.shape == (T,)

    def test_event_label_2(self, minimal_stats, rng):
        _, _, labels = generate_line_outage(1, 2, minimal_stats, rng, t_event=2.0)
        assert np.any(labels == 2), "Expected label=2 during line outage"
        assert labels[0] == 0

    def test_finite_outputs(self, minimal_stats, rng):
        bus_data, _, _ = generate_line_outage(22, 21, minimal_stats, rng)
        for col, arr in bus_data.items():
            assert np.all(np.isfinite(arr))


# ── generate_load_change ───────────────────────────────────────────────────────

class TestGenerateLoadChange:
    def test_output_shape(self, minimal_stats, rng):
        bus_data, ts, labels = generate_load_change(7, minimal_stats, rng)
        T = int(round(6.0 * FPS))
        assert ts.shape == (T,)
        assert labels.shape == (T,)

    def test_event_label_4(self, minimal_stats, rng):
        _, _, labels = generate_load_change(15, minimal_stats, rng, t_event=2.0)
        assert np.any(labels == 4), "Expected label=4 during load change"

    def test_finite_outputs(self, minimal_stats, rng):
        bus_data, _, _ = generate_load_change(20, minimal_stats, rng)
        for col, arr in bus_data.items():
            assert np.all(np.isfinite(arr))


# ── generate_pmu_dropout ───────────────────────────────────────────────────────

class TestGeneratePmuDropout:
    @pytest.mark.parametrize("dropout_bus", [29, 2])
    def test_output_shape(self, minimal_stats, rng, dropout_bus):
        bus_data, ts, labels = generate_pmu_dropout(dropout_bus, minimal_stats, rng)
        T = int(round(6.0 * FPS))
        assert ts.shape == (T,)
        assert labels.shape == (T,)

    def test_event_label_5(self, minimal_stats, rng):
        _, _, labels = generate_pmu_dropout(29, minimal_stats, rng, t_event=2.0,
                                            dropout_sec=1.5)
        assert np.any(labels == 5), "Expected label=5 during dropout"

    def test_dropout_bus_has_nan(self, minimal_stats, rng):
        """dropout_bus channels should be NaN during the dropout window."""
        bus_data, ts, labels = generate_pmu_dropout(29, minimal_stats, rng,
                                                     t_event=2.0, dropout_sec=1.5)
        dropout_mask = labels == 5
        assert np.any(dropout_mask), "No dropout frames found"
        va_mag = bus_data["BUS29_VA_MAG"]
        assert np.all(np.isnan(va_mag[dropout_mask])), (
            "dropout_bus VA_MAG should be NaN during dropout"
        )

    def test_other_buses_not_nan(self, minimal_stats, rng):
        """Non-dropout buses should remain valid during the dropout."""
        bus_data, ts, labels = generate_pmu_dropout(29, minimal_stats, rng,
                                                     t_event=2.0, dropout_sec=1.5)
        dropout_mask = labels == 5
        for bus in PMU_BUSES:
            if bus == 29:
                continue
            va = bus_data[f"BUS{bus}_VA_MAG"]
            assert np.all(np.isfinite(va[dropout_mask])), (
                f"BUS{bus}_VA_MAG should be finite during Bus29 dropout"
            )

    def test_data_present_flag(self, minimal_stats, rng):
        """BUSk_DATA_PRESENT should be 0 for dropout_bus during dropout."""
        bus_data, ts, labels = generate_pmu_dropout(29, minimal_stats, rng,
                                                     t_event=2.0, dropout_sec=1.5)
        dp = bus_data["BUS29_DATA_PRESENT"]
        dropout_mask = labels == 5
        assert np.all(dp[dropout_mask] == 0), "DATA_PRESENT should be 0 during dropout"
        assert np.all(dp[~dropout_mask] == 1), "DATA_PRESENT should be 1 outside dropout"


# ── CSV writer and loader ──────────────────────────────────────────────────────

class TestCsvWriteLoad:
    def test_write_and_load_roundtrip(self, minimal_stats, rng):
        """Written CSVs can be loaded back with load_synthetic."""
        bus_data, ts, labels = generate_fault(2, minimal_stats, rng)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            _write_synthetic_csvs(0, bus_data, ts, labels, out_dir)

            # Check files exist
            for bus in PMU_BUSES:
                fname = out_dir / f"syn0000_Bus{bus}_Competition_Data_nanmask.csv"
                assert fname.exists(), f"Missing file {fname}"

            # Load and verify
            df = load_synthetic(out_dir, 0)
            assert "TIMESTAMP" in df.columns
            assert "Event" in df.columns
            assert len(df) == len(ts)
            assert np.all(np.isin(df["Event"].unique(), [0, 1]))

    def test_loaded_df_has_all_measurement_cols(self, minimal_stats, rng):
        bus_data, ts, labels = generate_gen_change(39, minimal_stats, rng)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            _write_synthetic_csvs(0, bus_data, ts, labels, out_dir)
            df = load_synthetic(out_dir, 0)
            for bus in PMU_BUSES:
                for suf in ("VA_MAG", "IA_MAG", "Freq", "ROCOF"):
                    assert f"BUS{bus}_{suf}" in df.columns

    def test_dropout_nan_survives_roundtrip(self, minimal_stats, rng):
        """NaN values in dropout windows should survive CSV write/read."""
        bus_data, ts, labels = generate_pmu_dropout(29, minimal_stats, rng,
                                                     t_event=2.0, dropout_sec=1.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            _write_synthetic_csvs(0, bus_data, ts, labels, out_dir)
            df = load_synthetic(out_dir, 0)
            dropout_mask = df["Event"] == 5
            if dropout_mask.any():
                assert df.loc[dropout_mask, "BUS29_VA_MAG"].isna().any(), (
                    "NaN values should survive CSV roundtrip"
                )


# ── extract_normal_baseline ────────────────────────────────────────────────────

class TestExtractNormalBaseline:
    def test_returns_dict_with_all_cols(self):
        """extract_normal_baseline should return stats for all PMU channel columns."""
        rng = np.random.default_rng(0)
        T = 1800  # 60 s at 30 fps
        rows: dict = {"TIMESTAMP": np.arange(T) / 30.0, "Event": np.zeros(T, dtype=int)}
        for bus in PMU_BUSES:
            for suf in _SUFFIXES:
                rows[f"BUS{bus}_{suf}"] = rng.standard_normal(T)
            rows[f"BUS{bus}_DATA_PRESENT"] = np.ones(T, dtype=int)
        df = pd.DataFrame(rows)

        stats = extract_normal_baseline(df)
        for bus in PMU_BUSES:
            for suf in _SUFFIXES:
                col = f"BUS{bus}_{suf}"
                assert col in stats, f"Missing stats for {col}"
                mean, std = stats[col]
                assert np.isfinite(mean)
                assert np.isfinite(std)


# ── real-data integration test ─────────────────────────────────────────────────

class TestRealDataGeneration:
    @SKIP_NO_DATA
    def test_generate_small_batch(self, tmp_path):
        """generate_all with small counts produces expected CSVs."""
        from src.augmentation.andes_sim import generate_all

        result = generate_all(
            n_faults=2, n_line_outages=2, n_gen_changes=2,
            n_load_changes=2, n_dropouts=1,
            data_dir=DATA_DIR, out_dir=tmp_path, seed=0,
        )
        assert result["n_generated"] == 9
        assert result["event_counts"][1] == 2
        assert result["event_counts"][2] == 2
        assert result["event_counts"][3] == 2
        assert result["event_counts"][4] == 2
        assert result["event_counts"][5] == 1

        # Verify structure of one event
        df = load_synthetic(tmp_path, 0)
        assert "TIMESTAMP" in df.columns
        assert "Event" in df.columns
        assert len(df) > 0
        for bus in PMU_BUSES:
            assert f"BUS{bus}_VA_MAG" in df.columns

    @SKIP_NO_DATA
    def test_synthetic_features_finite(self, tmp_path):
        """Feature extraction on synthetic data should produce finite vectors."""
        from src.augmentation.andes_sim import generate_all
        from src.classifier.features import extract_features, N_FEATURES

        generate_all(
            n_faults=1, n_line_outages=0, n_gen_changes=1,
            n_load_changes=0, n_dropouts=0,
            data_dir=DATA_DIR, out_dir=tmp_path, seed=0,
        )
        for run_id in [0, 1]:
            df = load_synthetic(tmp_path, run_id)
            ev_frames = np.where(df["Event"].to_numpy() != 0)[0]
            if len(ev_frames) == 0:
                continue
            onset = int(ev_frames[len(ev_frames) // 2])
            feats = extract_features(df, onset)
            assert feats.shape == (N_FEATURES,)
            assert np.all(np.isfinite(feats)), f"Non-finite features for run {run_id}"
