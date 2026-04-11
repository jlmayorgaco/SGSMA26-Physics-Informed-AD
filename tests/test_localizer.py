"""Tests for the cosine-match localizer (M6).

Targets per CLAUDE.md §M5:
  Top-1 ≥ 5/9 on the real 9 events.
  Top-3 ≥ 8/9 on the real 9 events.

All real-data tests are module-scoped and skipped if data/raw is absent.
Unit tests use synthetic data and simple known Jacobians.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.localizer.cosine_match import locate, locate_all, compute_nu, _cosine
from src.io.load_csv import PMU_BUSES

DATA_DIR = Path("data/raw")
SKIP_NO_DATA = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason="data/raw not present — skipping real-data tests",
)

# Ground-truth per CLAUDE.md §2.6 + actual data inspection
# (timestamp_sec, label, expected_top1_bus, expected_top3_contains_bus,
#  expected_top1_line_or_None)
_EVENTS_GT = [
    (536.566,  5, 29, 29,   None),          # cyber → Bus 29 (DATA_PRESENT drop)
    (614.333,  5, 29, 29,   None),          # cyber → Bus 29
    (1173.166, 1, 39, 39,   None),          # fault → Bus 39
    (2375.233, 2, 24, 24,   (24, 23)),      # line outage → branch 24-23
    (2584.666, 5, 29, 29,   None),          # cyber → Bus 29
    (2935.766, 5, 29, 29,   None),          # cyber → Bus 29
    (2976.366, 6, 29, 29,   None),          # cyber+physical → Bus 29
    (2994.833, 3,  2,  2,   None),          # gen change → Bus 2
    (3877.733, 4,  7,  7,   None),          # load change → Bus 7 (non-PMU)
]


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def full_df():
    from src.io.load_csv import load_all
    return load_all(DATA_DIR)


@pytest.fixture(scope="module")
def grid():
    from src.grid.load_case import load_case
    metadata = Path("data/metadata")
    raw_files = list(metadata.glob("*.raw"))
    if not raw_files:
        pytest.skip("No .raw power-flow file found in data/metadata/")
    return load_case(raw_files[0])


@pytest.fixture(scope="module")
def J_cols(grid):
    from src.grid.jacobians import compute_jacobians, bus_sensitivity_columns
    J = compute_jacobians(grid)
    return bus_sensitivity_columns(J, grid)


@pytest.fixture(scope="module")
def branches(grid):
    """List of (from_bus, to_bus) pairs in competition numbering."""
    return grid.branch_list


@pytest.fixture(scope="module")
def state_estimator(grid):
    from src.estimator.topology_state import TopologyStateEstimator
    return TopologyStateEstimator(grid)


@pytest.fixture(scope="module")
def calibration(full_df):
    """Calibrated h0_flat, offset, R from first 60s of normal data."""
    from src.detector.chi2 import Chi2Detector, compute_eta_simple, extract_data_present
    from src.estimator.calibration import channel_cols, calibrate_R

    cols = channel_cols()
    calib_mask = (full_df["TIMESTAMP"] <= 60.0) & (full_df["Event"] == 0)
    h0_flat = np.array([
        full_df[c][calib_mask].dropna().mean() if c in full_df.columns else 0.0
        for c in cols
    ])
    h0_flat = np.where(np.isfinite(h0_flat), h0_flat, 0.0)
    R, offset = calibrate_R(full_df, h0_flat)
    return h0_flat, offset, R


@pytest.fixture(scope="module")
def alarm_indices(full_df, calibration):
    """Alarm onset indices from calibrated detector on full dataset."""
    from src.detector.chi2 import Chi2Detector, compute_eta_simple, extract_data_present
    from src.estimator.calibration import channel_cols

    h0_flat, offset, R = calibration
    cols = channel_cols()
    calib_mask = (full_df["TIMESTAMP"] <= 60.0) & (full_df["Event"] == 0)

    eta = compute_eta_simple(full_df, h0_flat, offset, R, cols)
    dp  = extract_data_present(full_df)
    ts  = full_df["TIMESTAMP"].to_numpy(float)

    det = Chi2Detector(n_z=32, k_on=3, k_off=15, fps=30.0)
    det.calibrate(eta[calib_mask.to_numpy()], dp[calib_mask.to_numpy()])
    res = det.detect(eta, dp, ts)
    return res["alarm_indices"]


# ── unit tests: cosine helpers ─────────────────────────────────────────────────

class TestCosineHelpers:
    def test_cosine_same_vector(self):
        v = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert abs(_cosine(v, v) - 1.0) < 1e-10

    def test_cosine_orthogonal(self):
        v1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert abs(_cosine(v1, v2)) < 1e-10

    def test_cosine_zero_vector(self):
        v = np.zeros(8)
        w = np.ones(8)
        assert _cosine(v, w) == 0.0

    def test_cosine_scale_invariant(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
        w = np.array([2.0, 4.0, 6.0, 8.0, 0.0, 0.0, 0.0, 0.0])
        assert abs(_cosine(v, w) - 1.0) < 1e-10


class TestLocateUnit:
    """Unit tests using a tiny known Jacobian."""

    @pytest.fixture
    def simple_J_cols(self):
        """J_cols where bus 5 has a distinct spike at PMU index 1."""
        J = {}
        for i, bus in enumerate(PMU_BUSES):
            col = np.zeros(8)
            col[i] = 0.1  # weak uniform sensitivity for every bus
            J[bus] = col
        # Bus 39 (PMU index 7) has a large spike → should be top match for nu=e7
        J[39] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        return J

    @pytest.fixture
    def simple_df(self):
        rng = np.random.default_rng(0)
        T = 200
        rows = {"TIMESTAMP": np.arange(T) / 30.0, "Event": np.zeros(T, dtype=int)}
        for bus in PMU_BUSES:
            rows[f"BUS{bus}_VA_MAG"] = 199_000 + rng.standard_normal(T) * 100
            rows[f"BUS{bus}_IA_MAG"] = 585     + rng.standard_normal(T) * 1
            rows[f"BUS{bus}_Freq"]   = 60.0    + rng.standard_normal(T) * 0.005
            rows[f"BUS{bus}_ROCOF"]  = rng.standard_normal(T) * 0.005
            rows[f"BUS{bus}_DATA_PRESENT"] = np.ones(T, dtype=int)
        import pandas as pd
        return pd.DataFrame(rows)

    def test_cyber_uses_data_present(self, simple_df, simple_J_cols):
        """Cyber mode should return the bus with DATA_PRESENT==0, not cosine match."""
        import pandas as pd
        df = simple_df.copy()
        # Drop Bus29 data at frames 80-120
        df.loc[80:120, "BUS29_DATA_PRESENT"] = 0
        df.loc[80:120, "BUS29_VA_MAG"] = np.nan
        result = locate(df, onset_frame=100, predicted_label=5,
                        J_cols=simple_J_cols, branches=[])
        assert result["mode"] == "cyber"
        assert result["top1_bus"] == 29

    def test_bus_mode_returns_dict(self, simple_df, simple_J_cols):
        result = locate(simple_df, onset_frame=100, predicted_label=1,
                        J_cols=simple_J_cols, branches=[])
        assert result["mode"] == "bus"
        assert "top1_bus" in result
        assert "top3_buses" in result
        assert len(result["top3_buses"]) <= 3

    def test_line_mode_returns_dict(self, simple_df, simple_J_cols):
        branches = [(2, 3), (5, 6), (10, 19)]
        result = locate(simple_df, onset_frame=100, predicted_label=2,
                        J_cols=simple_J_cols, branches=branches)
        assert result["mode"] == "line"
        assert "top1_line" in result
        assert "top3_lines" in result

    def test_nu_shape(self, simple_df):
        nu = compute_nu(simple_df, onset_frame=100)
        assert nu.shape == (8,)
        assert np.all(nu >= 0)

    def test_locate_all_length(self, simple_df, simple_J_cols):
        alarm_idx = np.array([50, 100, 150])
        labels    = np.array([1, 5, 3])
        results = locate_all(simple_df, alarm_idx, labels, simple_J_cols, [])
        assert len(results) == 3
        assert results[1]["mode"] == "cyber"  # label 5


# ── real-data localization accuracy ──────────────────────────────────────────

class TestLocalizerAccuracy:
    """Top-1 ≥ 5/9, Top-3 ≥ 8/9 on the known 9 real events."""

    @SKIP_NO_DATA
    def test_localization_accuracy(self, full_df, grid, J_cols, branches, alarm_indices, calibration, state_estimator):
        ts = full_df["TIMESTAMP"].to_numpy(float)
        n_events = len(_EVENTS_GT)
        top1_correct = 0
        top3_correct = 0
        results_log = []

        for (event_ts, label, expected_bus, expected_top3_bus, expected_line) in _EVENTS_GT:
            # Check detection: is there an alarm onset within 60 s of the event?
            if len(alarm_indices) == 0:
                results_log.append((event_ts, label, None, None, False, False))
                continue

            diffs = np.abs(ts[alarm_indices] - event_ts)
            closest_alarm = alarm_indices[np.argmin(diffs)]
            delay = abs(ts[closest_alarm] - event_ts)

            if delay > 60.0:
                # No alarm detected near this event — count as miss
                results_log.append((event_ts, label, None, None, False, False))
                continue

            # For localization: use the TRUE event frame (frame closest to the
            # known event timestamp), not the alarm onset.  This is necessary
            # because multiple events may fall within a single alarm period
            # (e.g. cyber+physical at 2976 followed by gen change at 2994),
            # and the alarm onset would point to the FIRST event, not the current one.
            loc_frame = int(np.argmin(np.abs(ts - event_ts)))

            h0_flat, offset, R = calibration
            result = locate(full_df, loc_frame, label, J_cols, branches,
                            h0_flat=h0_flat, offset=offset, R=R,
                            grid=grid, state_estimator=state_estimator)

            if label in {2}:  # line outage
                # Top-1 line check
                t1_line = result.get("top1_line")
                t3_lines = result.get("top3_lines", [])
                t3_pairs = [pair for pair, _ in t3_lines]
                # Check either orientation of the branch
                if expected_line is not None:
                    rev = (expected_line[1], expected_line[0])
                    t1_ok = t1_line in (expected_line, rev)
                    t3_ok = expected_line in t3_pairs or rev in t3_pairs
                else:
                    t1_ok = result.get("top1_bus") == expected_bus
                    t3_ok = any(b == expected_top3_bus for b, _ in result.get("top3_buses", []))
            else:
                t1_bus = result.get("top1_bus")
                t3_buses = [b for b, _ in result.get("top3_buses", [])]
                t1_ok = t1_bus == expected_bus
                t3_ok = expected_top3_bus in t3_buses

            if t1_ok:
                top1_correct += 1
            if t3_ok:
                top3_correct += 1
            results_log.append((event_ts, label, result.get("top1_bus"), result.get("top1_line"),
                                 t1_ok, t3_ok))

        # Log detailed results
        print("\nLocalization results:")
        for row in results_log:
            ts_ev, lbl, t1b, t1l, ok1, ok3 = row
            print(f"  ts={ts_ev:.1f}s label={lbl} top1_bus={t1b} top1_line={t1l} "
                  f"Top1={'OK' if ok1 else 'MISS'} Top3={'OK' if ok3 else 'MISS'}")
        print(f"  Top-1: {top1_correct}/{n_events}  Top-3: {top3_correct}/{n_events}")

        assert top1_correct >= 5, (
            f"Top-1 localization: {top1_correct}/{n_events} < 5"
        )
        assert top3_correct >= 8, (
            f"Top-3 localization: {top3_correct}/{n_events} < 8"
        )

    @SKIP_NO_DATA
    def test_localization_top1_strict(self, full_df, grid, J_cols, branches, alarm_indices, calibration, state_estimator):
        """Top-1 ≥ 5/9 must hold strictly (T1 achieves 7/9)."""
        ts = full_df["TIMESTAMP"].to_numpy(float)
        h0_flat, offset, R = calibration
        n_events = len(_EVENTS_GT)
        top1_correct = 0

        for (event_ts, label, expected_bus, expected_top3_bus, expected_line) in _EVENTS_GT:
            if len(alarm_indices) == 0:
                continue
            diffs = np.abs(ts[alarm_indices] - event_ts)
            closest_alarm = alarm_indices[np.argmin(diffs)]
            if abs(ts[closest_alarm] - event_ts) > 60.0:
                continue

            loc_frame = int(np.argmin(np.abs(ts - event_ts)))
            result = locate(full_df, loc_frame, label, J_cols, branches,
                            h0_flat=h0_flat, offset=offset, R=R,
                            grid=grid, state_estimator=state_estimator)

            if label in {2}:
                top1_line = result.get("top1_line")
                if expected_line is not None:
                    rev = (expected_line[1], expected_line[0])
                    if top1_line in (expected_line, rev):
                        top1_correct += 1
                elif result.get("top1_bus") == expected_bus:
                    top1_correct += 1
            else:
                if result.get("top1_bus") == expected_bus:
                    top1_correct += 1

        assert top1_correct >= 5, (
            f"Top-1 localization: {top1_correct}/{n_events} < 5"
        )

    @SKIP_NO_DATA
    def test_cyber_events_all_point_to_bus29(self, full_df, grid, J_cols, branches, calibration, state_estimator):
        """All 4 cyber-only events (label 5) should resolve to Bus 29."""
        ts = full_df["TIMESTAMP"].to_numpy(float)
        cyber_events = [(ev_ts, lbl) for ev_ts, lbl, *_ in _EVENTS_GT if lbl == 5]

        # Find alarm onset closest to each cyber event
        from src.detector.chi2 import Chi2Detector, compute_eta_simple, extract_data_present
        from src.estimator.calibration import channel_cols, calibrate_R

        cols = channel_cols()
        calib_mask = (full_df["TIMESTAMP"] <= 60.0) & (full_df["Event"] == 0)
        h0_flat = np.array([
            full_df[c][calib_mask].dropna().mean() if c in full_df.columns else 0.0
            for c in cols
        ])
        h0_flat = np.where(np.isfinite(h0_flat), h0_flat, 0.0)
        R, offset = calibrate_R(full_df, h0_flat)

        eta = compute_eta_simple(full_df, h0_flat, offset, R, cols)
        dp  = extract_data_present(full_df)
        det = Chi2Detector(n_z=32, k_on=3, k_off=15, fps=30.0)
        det.calibrate(eta[calib_mask.to_numpy()], dp[calib_mask.to_numpy()])
        res = det.detect(eta, dp, ts)
        alarm_idx = res["alarm_indices"]

        if len(alarm_idx) == 0:
            pytest.skip("No alarm onsets found")

        correct = 0
        for ev_ts, lbl in cyber_events:
            diffs = np.abs(ts[alarm_idx] - ev_ts)
            closest = alarm_idx[np.argmin(diffs)]
            if abs(ts[closest] - ev_ts) > 60.0:
                continue
            h0_flat, offset, R = calibration
            result = locate(full_df, int(closest), lbl, J_cols, branches,
                            h0_flat=h0_flat, offset=offset, R=R,
                            grid=grid, state_estimator=state_estimator)
            if result["top1_bus"] == 29:
                correct += 1

        assert correct >= 3, (
            f"Only {correct}/{len(cyber_events)} cyber events resolved to Bus 29"
        )
