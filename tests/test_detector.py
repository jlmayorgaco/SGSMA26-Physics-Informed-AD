"""M4 tests: chi-squared detector and debouncing.

Tests:
  1. debounce() state-machine correctness.
  2. compute_eta_simple() shape, non-negativity, calibration (chi2 mean under H0).
  3. Chi2Detector calibration: threshold is set, FP rate < 0.1/min on calib window.
  4. Chi2Detector detection: all 9 known events fired within 1 s detection delay.
  5. Chi2Detector FP rate: < 0.1/min on the first 8.9 min of normal data.

Detection uses compute_eta_simple (fast, calibrated by construction) instead of
the UKF eta_t so the tests run in seconds rather than minutes.  The 9 events are
identified from Event column transitions (0 -> non-zero, excluding Event=7 startup
glitches).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.detector.debounce import debounce, alarm_onsets, alarm_offsets
from src.detector.chi2 import (
    Chi2Detector,
    compute_eta_simple,
    extract_data_present,
    refine_alarm_onsets,
)
from src.estimator.calibration import calibrate_R, channel_cols, N_Z, MEAS_CHANNELS
from src.io.load_csv import PMU_BUSES


# ── Module-scoped helpers ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gen_params():
    from src.dynamics.parameters import load_params
    return load_params()


@pytest.fixture(scope="module")
def swing(grid_case, gen_params):
    from src.dynamics.parameters import GEN_PSSE_BUSES
    from src.dynamics.swing import build_model
    from src.grid.kron_reduce import kron_reduce
    Y_red = kron_reduce(grid_case.Ybus, grid_case.gen_indices)
    angles = [
        grid_case.buses[grid_case.psse_to_idx[b]].va_deg
        if grid_case.psse_to_idx.get(b) is not None else 0.0
        for b in GEN_PSSE_BUSES
    ]
    vmags = [
        grid_case.buses[grid_case.psse_to_idx[b]].vm_pu
        if grid_case.psse_to_idx.get(b) is not None else 1.0
        for b in GEN_PSSE_BUSES
    ]
    return build_model(gen_params, Y_red, angles, vmags)


@pytest.fixture(scope="module")
def meas_model(swing, grid_case):
    from src.dynamics.measurement import build_measurement_model
    bus_th = np.deg2rad(np.array([b.va_deg for b in grid_case.buses]))
    bus_vm = np.array([b.vm_pu for b in grid_case.buses])
    return build_measurement_model(
        swing_model=swing,
        Ybus=grid_case.Ybus,
        pmu_indices=grid_case.pmu_bus_indices,
        gen_indices=grid_case.gen_indices,
        bus_theta0_rad=bus_th,
        bus_vmag0_pu=bus_vm,
    )


@pytest.fixture(scope="module")
def h0_flat(swing, meas_model):
    x0 = swing.x0()
    y = np.array(meas_model.h(x0))          # (8, 14)
    return y[:, MEAS_CHANNELS].ravel()       # (32,)


@pytest.fixture(scope="module")
def calibration(merged_df, h0_flat):
    R, offset = calibrate_R(merged_df, h0_flat)
    return R, offset


@pytest.fixture(scope="module")
def full_eta(merged_df, h0_flat, calibration):
    """Compute eta_simple for the full dataset — fast (no UKF)."""
    R, offset = calibration
    cols = channel_cols()
    return compute_eta_simple(merged_df, h0_flat, offset, R, cols)


@pytest.fixture(scope="module")
def full_dp(merged_df):
    """(T, 8) DATA_PRESENT matrix for the full dataset."""
    return extract_data_present(merged_df)


@pytest.fixture(scope="module")
def event_starts(merged_df):
    """List of (timestamp, label) tuples for the 9 real events.

    Excludes Event=7 (brief bad-data startup artifacts in the simulation).
    Captures ALL transitions to a new non-zero, non-7 label — including
    direct transitions between events (e.g., Event=5 → Event=6 without
    returning to Event=0 first).
    """
    ts = merged_df["TIMESTAMP"].to_numpy(dtype=float)
    ev = merged_df["Event"].to_numpy(dtype=int)

    starts = []
    for i in range(1, len(ev)):
        prev, curr = ev[i - 1], ev[i]
        # New non-zero, non-7 label that differs from the previous label
        if curr != 0 and curr != 7 and curr != prev:
            starts.append((float(ts[i]), int(curr)))
    return starts


@pytest.fixture(scope="module")
def calib_mask(merged_df):
    """Boolean mask: Event==0 and TIMESTAMP <= 60 s (calibration window)."""
    ts = merged_df["TIMESTAMP"].to_numpy(dtype=float)
    ev = merged_df["Event"].to_numpy(dtype=int)
    return (ev == 0) & (ts <= 60.0)


@pytest.fixture(scope="module")
def normal_mask(merged_df, event_starts):
    """Boolean mask: first normal region before any real event (Event=0)."""
    if not event_starts:
        pytest.skip("No event starts found")
    first_event_ts = event_starts[0][0]
    ts = merged_df["TIMESTAMP"].to_numpy(dtype=float)
    ev = merged_df["Event"].to_numpy(dtype=int)
    return (ev == 0) & (ts < first_event_ts)


@pytest.fixture(scope="module")
def fitted_detector(merged_df, full_eta, full_dp, calib_mask):
    """Chi2Detector calibrated from the first 60 s of Event=0 data."""
    eta_calib = full_eta[calib_mask]
    dp_calib  = full_dp[calib_mask]

    det = Chi2Detector(n_z=N_Z, k_on=3, k_off=15, fps=30.0)
    det.calibrate(eta_calib, dp_calib, fp_per_min_target=0.1)
    return det


@pytest.fixture(scope="module")
def detection_result(fitted_detector, merged_df, full_eta, full_dp):
    """Run the fitted detector on the full dataset."""
    ts = merged_df["TIMESTAMP"].to_numpy(dtype=float)
    return fitted_detector.detect(full_eta, full_dp, ts)


# ── 1. Debounce unit tests ────────────────────────────────────────────────────

class TestDebounce:
    def test_all_zeros_no_alarm(self):
        sig = np.zeros(100, dtype=bool)
        assert not debounce(sig, k_on=3).any()

    def test_all_ones_alarm_after_k(self):
        sig   = np.ones(100, dtype=bool)
        alarm = debounce(sig, k_on=3)
        # Alarm should be off for first k_on-1 frames, then on
        assert not alarm[:2].any()
        assert alarm[2:].all()

    def test_short_pulse_no_alarm(self):
        """A pulse shorter than k_on must not trigger alarm."""
        sig   = np.zeros(20, dtype=bool)
        sig[5:7] = True   # 2 frames — below k_on=3
        alarm = debounce(sig, k_on=3)
        assert not alarm.any(), "Short pulse should not trigger alarm"

    def test_exact_k_on_triggers(self):
        sig   = np.zeros(20, dtype=bool)
        sig[5:8] = True   # exactly k_on=3 frames
        alarm = debounce(sig, k_on=3)
        assert alarm[7], "Exactly k_on consecutive frames must trigger alarm"

    def test_alarm_stays_on_until_k_off(self):
        sig   = np.zeros(30, dtype=bool)
        sig[0:3] = True   # triggers alarm
        # frames 3..4 are False (< k_off=5) → alarm should stay on
        alarm = debounce(sig, k_on=3, k_off=5)
        assert alarm[3], "Alarm must stay on for fewer than k_off low frames"
        assert alarm[4], "Alarm must stay on for fewer than k_off low frames"

    def test_alarm_clears_after_k_off(self):
        sig   = np.zeros(30, dtype=bool)
        sig[0:3] = True   # triggers alarm
        # frames 3..7 are False → k_off=5 consecutive lows → alarm clears
        alarm = debounce(sig, k_on=3, k_off=5)
        assert not alarm[7], "Alarm must clear after k_off consecutive low frames"

    def test_alarm_onsets_count(self):
        sig   = np.zeros(50, dtype=bool)
        sig[5:8]   = True
        sig[20:23] = True
        alarm  = debounce(sig, k_on=3, k_off=3)
        onsets = alarm_onsets(alarm)
        assert len(onsets) == 2, f"Expected 2 onsets, got {len(onsets)}"

    def test_alarm_offsets_count(self):
        sig   = np.zeros(50, dtype=bool)
        sig[5:8]   = True
        sig[20:23] = True
        alarm   = debounce(sig, k_on=3, k_off=3)
        offsets = alarm_offsets(alarm)
        assert len(offsets) == 2, f"Expected 2 offsets, got {len(offsets)}"

    def test_single_alarm_starts_before_offsets(self):
        sig   = np.zeros(50, dtype=bool)
        sig[5:8] = True
        alarm  = debounce(sig, k_on=3, k_off=3)
        onsets  = alarm_onsets(alarm)
        offsets = alarm_offsets(alarm)
        assert onsets[0] < offsets[0], "Alarm onset must precede offset"

    def test_output_dtype_bool(self):
        sig = np.random.rand(50) > 0.5
        alarm = debounce(sig, k_on=3)
        assert alarm.dtype == bool


# ── 2. compute_eta_simple ─────────────────────────────────────────────────────

class TestEtaSimple:
    def test_shape(self, merged_df, h0_flat, calibration):
        R, offset = calibration
        eta = compute_eta_simple(merged_df, h0_flat, offset, R, channel_cols())
        assert eta.shape == (len(merged_df),)

    def test_non_negative(self, full_eta):
        assert np.all(full_eta >= 0.0), "eta_simple must be non-negative"

    def test_finite(self, full_eta):
        assert np.all(np.isfinite(full_eta)), "eta_simple must be finite"

    def test_chi2_mean_normal(self, full_eta, merged_df):
        """Under normal (Event=0) operation, mean(eta) should be near N_Z=32."""
        ts = merged_df["TIMESTAMP"].to_numpy()
        ev = merged_df["Event"].to_numpy()
        # First 60 s, Event==0 only
        mask = (ev == 0) & (ts <= 60.0)
        eta_normal = full_eta[mask]
        if len(eta_normal) < 10:
            pytest.skip("Too few normal rows")
        mean_eta = float(eta_normal.mean())
        # The chi2 approximation gives mean = N_Z = 32; allow generous 3× range
        assert mean_eta < 3 * N_Z, (
            f"eta mean too large: {mean_eta:.1f} > 3*N_Z={3*N_Z} during normal"
        )
        assert mean_eta > N_Z / 3, (
            f"eta mean too small: {mean_eta:.1f} < N_Z/3={N_Z/3:.1f}"
        )

    def test_events_higher_than_normal(self, full_eta, merged_df):
        """Physical events must have higher mean eta than normal operation."""
        ev = merged_df["Event"].to_numpy()
        ts = merged_df["TIMESTAMP"].to_numpy()
        normal_mask  = (ev == 0) & (ts <= 60.0)
        fault_mask   = (ev == 1)
        if not fault_mask.any():
            pytest.skip("No fault rows in data")
        mean_normal = full_eta[normal_mask].mean()
        mean_fault  = full_eta[fault_mask].mean()
        assert mean_fault > 10 * mean_normal, (
            f"Fault eta ({mean_fault:.0f}) should be >> 10x normal ({mean_normal:.1f})"
        )

    def test_extract_data_present_shape(self, merged_df):
        dp = extract_data_present(merged_df)
        assert dp.shape == (len(merged_df), len(PMU_BUSES))

    def test_extract_data_present_binary(self, merged_df):
        dp = extract_data_present(merged_df)
        assert dp.min() >= 0.0
        assert dp.max() <= 1.0


# ── 3. Chi2Detector calibration ───────────────────────────────────────────────

class TestChi2Calibration:
    def test_threshold_is_set(self, fitted_detector):
        assert fitted_detector.threshold is not None
        assert fitted_detector.threshold > 0

    def test_threshold_exceeds_normal_median(self, fitted_detector, full_eta, calib_mask):
        """Threshold must be above the median of normal eta (median ≈ 32)."""
        median_normal = float(np.median(full_eta[calib_mask]))
        assert fitted_detector.threshold > median_normal, (
            f"Threshold {fitted_detector.threshold:.1f} <= median {median_normal:.1f}"
        )

    def test_theoretical_threshold_positive(self, fitted_detector):
        th = fitted_detector.theoretical_threshold(alpha=0.001)
        assert th > 0

    def test_fp_rate_on_calibration_window(self, fitted_detector, full_eta, full_dp, calib_mask):
        """FP rate on the 60 s calibration window must be < 0.1/min."""
        eta_cal = full_eta[calib_mask]
        dp_cal  = full_dp[calib_mask]
        raw     = (eta_cal > fitted_detector.threshold) | (dp_cal < 1).any(axis=1)
        alarm   = debounce(raw, fitted_detector.k_on, fitted_detector.k_off)
        n_fps   = len(alarm_onsets(alarm))
        dur_min = len(eta_cal) / (fitted_detector.fps * 60.0)
        fp_rate = n_fps / max(dur_min, 1e-9)
        assert fp_rate < 0.1, (
            f"FP rate on calibration window: {fp_rate:.4f}/min > 0.1/min"
        )

    def test_detect_raises_without_calibration(self):
        det = Chi2Detector(n_z=N_Z)
        with pytest.raises(RuntimeError):
            det.detect(np.zeros(10))


class TestRefineAlarmOnsets:
    def test_splits_dropout_then_physical_then_recovery(self):
        fps = 30.0
        n = 240
        alarm = np.zeros(n, dtype=bool)
        alarm[30:210] = True
        eta = np.full(n, 20.0)
        eta[90:140] = 6000.0
        eta[150:190] = 5500.0
        dp = np.ones((n, 8), dtype=float)
        dp[30:150, 0] = 0.0
        ts = np.arange(n) / fps

        onsets = refine_alarm_onsets(
            alarm,
            eta,
            dp,
            threshold=100.0,
            timestamps=ts,
            fps=fps,
            min_separation_sec=1.0,
            confirm_sec=0.5,
        )
        assert onsets.tolist() == [30, 90, 150]


# ── 4. False-alarm rate on normal operation ───────────────────────────────────

class TestFalseAlarmRate:
    """FP rate on the first normal region (before any event) must be < 0.1/min."""

    def test_fp_rate_first_normal_region(
        self, fitted_detector, merged_df, full_eta, full_dp, normal_mask, event_starts
    ):
        if not event_starts:
            pytest.skip("No event starts found")

        eta_n = full_eta[normal_mask]
        dp_n  = full_dp[normal_mask]
        ts_n  = merged_df["TIMESTAMP"].to_numpy()[normal_mask]

        raw   = (eta_n > fitted_detector.threshold) | (dp_n < 1).any(axis=1)
        alarm = debounce(raw, fitted_detector.k_on, fitted_detector.k_off)
        n_fps = len(alarm_onsets(alarm))
        dur_min = (ts_n[-1] - ts_n[0]) / 60.0
        fp_rate = n_fps / max(dur_min, 1e-9)

        assert fp_rate < 0.1, (
            f"FP rate on first normal region: {fp_rate:.4f}/min > 0.1/min. "
            f"({n_fps} alarms in {dur_min:.1f} min, threshold={fitted_detector.threshold:.1f})"
        )


# ── 5. Event detection (all 9 events, delay < 1 s) ───────────────────────────

class TestEventDetection:
    """Detector must fire within 1 s of each of the 9 known event starts."""

    MAX_DELAY_S = 1.0

    def _detection_delay(self, alarm, timestamps, event_ts, window_s=2.0):
        """Return detection delay for one event, or inf if not detected.

        The alarm may already be active from a prior event; we check that
        it is True within window_s of event_ts (in addition to checking
        for new onsets).
        """
        idx_event = np.searchsorted(timestamps, event_ts)
        idx_end   = np.searchsorted(timestamps, event_ts + window_s)

        # If already in alarm state at event onset → delay = 0
        if idx_event < len(alarm) and alarm[idx_event]:
            return 0.0

        # Otherwise look for first onset within window
        for i in range(idx_event, min(idx_end, len(alarm))):
            if i > 0 and alarm[i] and not alarm[i - 1]:
                return float(timestamps[i] - event_ts)

        return float("inf")

    def test_all_9_events_detected(
        self, detection_result, merged_df, event_starts
    ):
        """Every known non-bad-data event must trigger alarm within 1 s."""
        if not event_starts:
            pytest.skip("No event starts found in data")
        if len(event_starts) < 9:
            pytest.skip(f"Expected 9 events, found {len(event_starts)}")

        alarm = detection_result["alarm"]
        ts    = merged_df["TIMESTAMP"].to_numpy(dtype=float)

        failures = []
        for t_event, label in event_starts:
            delay = self._detection_delay(alarm, ts, t_event)
            if delay > self.MAX_DELAY_S:
                failures.append(
                    f"Event={label} at {t_event/60:.2f} min: delay={delay:.3f} s"
                )

        assert not failures, (
            f"{len(failures)} event(s) not detected within {self.MAX_DELAY_S} s:\n"
            + "\n".join(failures)
        )

    @pytest.mark.parametrize("event_label,min_delay_s", [
        (1, 0.0),   # 3LG fault: massive eta spike
        (2, 0.0),   # line outage: large eta spike
        (3, 0.0),   # gen change: large eta spike
        (4, 0.0),   # load change: large eta spike at onset
    ])
    def test_physical_event_detected_fast(
        self, detection_result, merged_df, event_starts, event_label, min_delay_s
    ):
        """Physical events must trigger alarm within 1 s via eta_t."""
        ts = merged_df["TIMESTAMP"].to_numpy(dtype=float)
        alarm = detection_result["alarm"]

        target_events = [(t, l) for t, l in event_starts if l == event_label]
        if not target_events:
            pytest.skip(f"Event={event_label} not found in data")

        for t_event, _ in target_events:
            delay = self._detection_delay(alarm, ts, t_event)
            assert delay <= self.MAX_DELAY_S, (
                f"Event={event_label} at {t_event/60:.2f} min: delay {delay:.3f} s > {self.MAX_DELAY_S} s"
            )

    @pytest.mark.parametrize("event_label", [5, 6])
    def test_cyber_event_detected_via_data_present(
        self, merged_df, full_eta, full_dp, fitted_detector, event_starts, event_label
    ):
        """Cyber events must be caught by the DATA_PRESENT monitor."""
        ts = merged_df["TIMESTAMP"].to_numpy(dtype=float)
        target_events = [(t, l) for t, l in event_starts if l == event_label]
        if not target_events:
            pytest.skip(f"Event={event_label} not found")

        for t_event, _ in target_events:
            idx = np.searchsorted(ts, t_event)
            window = slice(idx, min(idx + 30, len(ts)))  # 1 s window
            # DATA_PRESENT should drop within 1 s of event onset
            dp_window = full_dp[window]
            has_missing = (dp_window < 1).any()
            # OR: alarm fires within 1 s
            raw   = (full_eta > fitted_detector.threshold) | (full_dp < 1).any(axis=1)
            alarm = debounce(raw, fitted_detector.k_on, fitted_detector.k_off)
            delay = self._detection_delay(alarm, ts, t_event)
            assert has_missing or delay <= self.MAX_DELAY_S, (
                f"Event={event_label} at {t_event/60:.2f} min: "
                f"no DATA_PRESENT drop AND delay {delay:.3f} s > {self.MAX_DELAY_S} s"
            )

    def test_event_count_is_9(self, event_starts):
        """Exactly 9 non-bad-data event starts must be present."""
        assert len(event_starts) == 9, (
            f"Expected 9 event starts (excl. Event=7), found {len(event_starts)}:\n"
            + "\n".join(f"  {t/60:.2f} min  label={l}" for t, l in event_starts)
        )
