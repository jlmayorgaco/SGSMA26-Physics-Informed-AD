"""Chi-squared anomaly detector using the UKF innovation statistic eta_t.

Two-signal detection:
  1. eta_t > threshold  — model-data mismatch (physical events: fault, line
     outage, gen/load change, cyber+physical).
  2. any DATA_PRESENT == 0 — missing PMU data (pure cyber/data-quality events).

The combined raw signal is the OR of these two indicators, then debounced with
k_on consecutive high frames required to fire and k_off to clear.

Calibration:
  The detection threshold is found empirically from a calibration window of
  Event=0 data (first 60 s).  A binary search over percentiles of the
  calibration eta distribution finds the lowest threshold that keeps the
  debounced false-alarm rate below fp_per_min_target.  This automatically
  handles the heavy tails in the real measurement data without requiring the
  UKF eta_t to be exactly chi2-distributed.

Fast eta path:
  compute_eta_simple() bypasses the UKF entirely.  It computes
      eta_t = (z_t - h0 - offset)' diag(R)^{-1} (z_t - h0 - offset)
  where h0 = h(x0) at the base case, offset = mean(CSV) - h0.
  NaN entries (from DATA_PRESENT==0) are filled with the expected value so
  they contribute zero to eta_t; the DATA_PRESENT signal handles those events
  separately.  Under correct calibration the simple eta_t ~ chi2(n_z) during
  normal operation, giving clean threshold semantics.
"""
from __future__ import annotations

import logging

import numpy as np
import scipy.stats

from src.detector.debounce import debounce, alarm_onsets, alarm_offsets
from src.io.load_csv import PMU_BUSES

log = logging.getLogger(__name__)


# ── Fast base-case residual statistic ────────────────────────────────────────

def compute_eta_simple(
    df: "pd.DataFrame",
    h0_flat: np.ndarray,
    offset: np.ndarray,
    R: np.ndarray,
    cols: list[str],
) -> np.ndarray:
    """Compute eta_t = (z - expected)' diag(R)^{-1} (z - expected) per row.

    NaN entries (missing channels) are filled with the expected value, so they
    contribute zero to eta_t.  Use extract_data_present() to get a separate
    DATA_PRESENT anomaly signal for those rows.

    Args:
        df:      merged DataFrame with measurement columns and TIMESTAMP.
        h0_flat: (n_z,) base-case model prediction h(x0) for selected channels.
        offset:  (n_z,) per-channel bias = mean(CSV) - h0  (from calibrate_R).
        R:       (n_z, n_z) diagonal measurement noise covariance.
        cols:    ordered list of column names matching h0_flat / offset layout.

    Returns:
        eta: (T,) float array, one value per row.
    """
    z_all = df[cols].to_numpy(dtype=float)    # (T, n_z), NaN where missing
    expected = h0_flat + offset               # (n_z,)
    # NaN → expected → zero innovation for that channel
    z_filled = np.where(np.isnan(z_all), expected[None, :], z_all)
    residual  = z_filled - expected[None, :]  # (T, n_z)
    R_diag    = np.maximum(np.diag(R), 1e-30) # guard against zero variance
    eta = np.sum(residual ** 2 / R_diag[None, :], axis=1)  # (T,)
    return eta


def extract_data_present(df: "pd.DataFrame") -> np.ndarray:
    """Extract (T, 8) DATA_PRESENT matrix from the merged DataFrame.

    Returns:
        dp: (T, 8) float array — 1.0 for present, 0.0 for missing.
    """
    cols = [f"BUS{b}_DATA_PRESENT" for b in PMU_BUSES]
    available = [c for c in cols if c in df.columns]
    if not available:
        return np.ones((len(df), len(PMU_BUSES)), dtype=float)
    dp = df[available].to_numpy(dtype=float)
    # Pad missing bus columns with 1.0 (assume present)
    if dp.shape[1] < len(PMU_BUSES):
        full = np.ones((len(df), len(PMU_BUSES)), dtype=float)
        full[:, : dp.shape[1]] = dp
        dp = full
    return dp


def refine_alarm_onsets(
    alarm: np.ndarray,
    eta: np.ndarray,
    data_present: np.ndarray | None,
    threshold: float,
    timestamps: np.ndarray | None = None,
    *,
    fps: float = 30.0,
    min_separation_sec: float = 5.0,
    confirm_sec: float = 2.0,
    min_remaining_sec: float = 20.0,
) -> np.ndarray:
    """Split a long alarm block into sub-events when evidence supports it.

    The debounced alarm is excellent at suppressing flicker, but it can merge
    stacked events such as ``5 -> 6 -> 3`` into one long abnormal period. We
    recover additional onset candidates from:

    1. ``DATA_PRESENT`` transitions inside an already-active alarm block.
    2. Large physical jumps where ``eta`` crosses the detection threshold while
       the alarm is still active because of a concurrent cyber event.

    To avoid false sub-events on the recovery edge of a pure dropout, any
    internal candidate must be followed by sustained physical evidence
    (``eta > threshold``) within a short confirmation window.
    """
    alarm = np.asarray(alarm, dtype=bool)
    eta = np.asarray(eta, dtype=float)
    if len(alarm) == 0:
        return np.empty(0, dtype=int)

    onsets = set(alarm_onsets(alarm).tolist())
    offsets = alarm_offsets(alarm).tolist()
    if alarm[-1]:
        offsets.append(len(alarm))

    min_sep = max(1, int(round(min_separation_sec * fps)))
    confirm = max(1, int(round(confirm_sec * fps)))
    min_remaining = max(1, int(round(min_remaining_sec * fps)))
    dp_any = None
    if data_present is not None:
        dp_any = (np.asarray(data_present, dtype=float) < 1).any(axis=1).astype(np.int8)

    start_list = sorted(onsets)
    for seg_idx, seg_start in enumerate(start_list):
        seg_end = next((o for o in offsets if o > seg_start), len(alarm))
        if seg_end - seg_start <= 1:
            continue

        candidates: list[int] = []

        if dp_any is not None:
            dp_diff = np.diff(dp_any[seg_start:seg_end], prepend=dp_any[seg_start])
            dp_changes = np.where(dp_diff != 0)[0] + seg_start
            for idx in dp_changes:
                if idx <= seg_start:
                    continue
                if seg_end - idx < min_remaining:
                    continue
                hi = min(seg_end, idx + confirm)
                lo = max(seg_start, idx - confirm)
                prev_missing = bool(dp_any[idx - 1]) if idx > 0 else False
                curr_missing = bool(dp_any[idx])
                pre_support = np.nanmax(eta[lo:idx]) > threshold if idx > lo else False
                post_support = np.nanmax(eta[idx:hi]) > threshold
                if prev_missing and (not curr_missing) and not pre_support:
                    continue
                if post_support:
                    candidates.append(int(idx))

        eta_prev = eta[seg_start:seg_end]
        eta_cross = np.where(
            (eta_prev[1:] > threshold) & (eta_prev[:-1] <= threshold)
        )[0] + seg_start + 1
        for idx in eta_cross:
            if idx <= seg_start or seg_end - idx < min_remaining:
                continue
            if dp_any is not None and dp_any[max(seg_start, idx - 1)] == 0:
                continue
            hi = min(seg_end, idx + confirm)
            support = eta[idx:hi] > threshold
            if support.any() and support.mean() >= 0.5:
                candidates.append(int(idx))

        for idx in sorted(set(candidates)):
            if all(abs(idx - prev) >= min_sep for prev in onsets):
                onsets.add(int(idx))

    refined = np.array(sorted(onsets), dtype=int)
    if dp_any is not None and len(refined) > 0:
        filtered: list[int] = []
        for idx in refined.tolist():
            keep_idx = True
            if idx > 0 and not bool(dp_any[idx]):
                lo = max(1, idx - confirm)
                rel = np.where((dp_any[lo - 1:idx - 1] == 1) & (dp_any[lo:idx] == 0))[0]
                if len(rel) > 0:
                    change_idx = lo + int(rel[-1])
                    pre_lo = max(0, change_idx - confirm)
                    if np.nanmax(eta[pre_lo:change_idx]) <= threshold:
                        keep_idx = False
            if keep_idx:
                filtered.append(int(idx))
        refined = np.array(filtered, dtype=int)

    if timestamps is not None and len(refined) > 1:
        ts = np.asarray(timestamps, dtype=float)
        keep = [int(refined[0])]
        for idx in refined[1:]:
            if ts[idx] - ts[keep[-1]] >= min_separation_sec - 1e-9:
                keep.append(int(idx))
        refined = np.array(keep, dtype=int)
    return refined


# ── Chi-squared detector ──────────────────────────────────────────────────────

class Chi2Detector:
    """Chi-squared anomaly detector combining eta_t and DATA_PRESENT signals.

    Usage
    -----
    det = Chi2Detector(n_z=32, k_on=3, k_off=15)
    det.calibrate(eta_calib, data_present_calib)   # from first 60 s Event=0
    result = det.detect(eta_full, data_present_full, timestamps)
    """

    def __init__(
        self,
        n_z: int = 32,
        k_on: int = 3,
        k_off: int = 15,
        fps: float = 30.0,
    ):
        """
        Args:
            n_z:   measurement vector dimension (32 for 4 channels × 8 buses).
            k_on:  consecutive high frames to enter alarm (detection delay = k_on/fps).
            k_off: consecutive low frames to exit alarm.
            fps:   nominal sampling rate (frames per second).
        """
        self.n_z  = n_z
        self.k_on  = k_on
        self.k_off = k_off
        self.fps   = fps
        self.threshold: float | None = None

    # ── calibration ───────────────────────────────────────────────────────────

    def theoretical_threshold(self, alpha: float = 0.001) -> float:
        """Return the theoretical chi2(n_z, 1-alpha) threshold.

        For a perfectly calibrated UKF (eta_t ~ chi2(n_z) under H0) this is
        the optimal threshold.  When using compute_eta_simple the calibration
        is exact by construction and this threshold is directly applicable.
        """
        return float(scipy.stats.chi2.ppf(1.0 - alpha, df=self.n_z))

    def calibrate(
        self,
        eta_calib: np.ndarray,
        data_present_calib: np.ndarray | None = None,
        fp_per_min_target: float = 0.1,
    ) -> None:
        """Set detection threshold from calibration (Event=0) eta distribution.

        Strategy:
          1. Start from the 99th percentile and step up until the debounced
             false-alarm rate on the calibration data falls below fp_per_min_target.
          2. If no percentile in [99, 99.999] achieves the target, use 1.1x
             the maximum calibration value.

        Args:
            eta_calib:          (T_cal,) eta_t during normal (Event=0) operation.
            data_present_calib: (T_cal, 8) DATA_PRESENT flags for same window;
                                used to exclude rows where data was already missing.
            fp_per_min_target:  maximum allowed false-alarm rate (alarms / minute).
        """
        valid = eta_calib[np.isfinite(eta_calib) & (eta_calib >= 0)]
        if len(valid) < 10:
            log.warning("Too few calibration samples; using theoretical threshold")
            self.threshold = self.theoretical_threshold(alpha=0.001)
            return

        # DATA_PRESENT anomaly in calibration window (should be zero; included
        # for robustness in case calibration window has brief outages)
        dp_anom = np.zeros(len(eta_calib), dtype=bool)
        if data_present_calib is not None:
            dp_anom = (data_present_calib < 1).any(axis=1)

        dur_min = len(eta_calib) / (self.fps * 60.0)

        # Search from high percentile down: find smallest p such that FP < target
        search = np.concatenate([
            np.arange(99.0, 99.9, 0.1),
            np.arange(99.9, 99.99, 0.01),
            np.arange(99.99, 99.999, 0.001),
        ])
        for p in reversed(sorted(search)):   # highest first → most conservative
            thresh = float(np.percentile(valid, p))
            raw   = (eta_calib > thresh) | dp_anom
            alm   = debounce(raw, self.k_on, self.k_off)
            n_fps = len(alarm_onsets(alm))
            fp_rate = n_fps / max(dur_min, 1e-9)
            if fp_rate < fp_per_min_target:
                self.threshold = thresh
                log.info(
                    "Chi2 threshold calibrated: p=%.3f thresh=%.2f FP=%.4f/min",
                    p, thresh, fp_rate,
                )
                return

        # Fallback: use 1.1 × max of valid calibration values
        self.threshold = float(valid.max() * 1.1)
        log.warning(
            "Could not meet FP target %.2f/min; using fallback threshold %.2f",
            fp_per_min_target, self.threshold,
        )

    # ── detection ─────────────────────────────────────────────────────────────

    def detect(
        self,
        eta: np.ndarray,
        data_present: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
    ) -> dict:
        """Run detection on an eta_t time series.

        Args:
            eta:          (T,) innovation statistic per frame.
            data_present: (T, 8) DATA_PRESENT flags (0 = missing PMU).
            timestamps:   (T,) TIMESTAMP values (optional; used for alarm_times).

        Returns:
            dict with keys:
              alarm:        (T,) bool — debounced alarm state per frame.
              raw:          (T,) bool — un-debounced anomaly indicator.
              alarm_times:  list of TIMESTAMP values at alarm onsets (or frame
                            indices if timestamps is None).
              alarm_indices: (K,) int array of frame indices at alarm onsets.
        """
        if self.threshold is None:
            raise RuntimeError("Call calibrate() before detect()")

        dp_anom = np.zeros(len(eta), dtype=bool)
        if data_present is not None:
            dp_anom = (data_present < 1).any(axis=1)

        raw   = (eta > self.threshold) | dp_anom
        alarm = debounce(raw, self.k_on, self.k_off)

        onset_idx = alarm_onsets(alarm)
        onset_idx = refine_alarm_onsets(
            alarm,
            eta,
            data_present,
            float(self.threshold),
            timestamps=timestamps,
            fps=self.fps,
        )
        if timestamps is not None:
            alarm_times = list(timestamps[onset_idx])
        else:
            alarm_times = list(onset_idx.tolist())

        return {
            "alarm":         alarm,
            "raw":           raw,
            "alarm_times":   alarm_times,
            "alarm_indices": onset_idx,
        }

    # ── convenience: run directly on a merged DataFrame ───────────────────────

    def run_on_df(
        self,
        df: "pd.DataFrame",
        h0_flat: np.ndarray,
        offset: np.ndarray,
        R: np.ndarray,
        cols: list[str],
    ) -> dict:
        """Compute eta_simple and run detect() in one call.

        Args:
            df:      merged DataFrame with measurement columns.
            h0_flat: (n_z,) base-case model prediction.
            offset:  (n_z,) per-channel bias (from calibrate_R).
            R:       (n_z, n_z) diagonal measurement noise covariance.
            cols:    ordered measurement column names.

        Returns:
            Same dict as detect(), plus 'eta' key.
        """
        eta = compute_eta_simple(df, h0_flat, offset, R, cols)
        dp  = extract_data_present(df)
        ts  = df["TIMESTAMP"].to_numpy(dtype=float)
        result = self.detect(eta, dp, ts)
        result["eta"] = eta
        return result
