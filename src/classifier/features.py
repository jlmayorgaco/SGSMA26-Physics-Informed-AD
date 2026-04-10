"""Feature extractor for the T1 LightGBM event classifier.

For each detected event (alarm onset), extract a fixed-length feature vector from
a 3-second window centered on the onset frame.

Feature families
----------------
1. Per-channel residual statistics (mean, max, std) for 4 channel types:
   VA_MAG, IA_MAG, Freq, ROCOF  — shape 4 × 3 = 12 features
2. Spatial pattern ν̄ ∈ R^8 (per-PMU innovation norm) + entropy + argmax
   — shape 8 + 1 + 1 = 10 features
3. Spectral energy in 0.1–2 Hz band of the Freq residual (per bus) — shape 8
4. Cyber indicators: NaN count, PMU missing count, longest NaN run — shape 3
5. Cosine similarity of ν̄ to the top-K sensitivity columns J_k — shape 3 (top-3 values)
   + argmax bus (ordinal) — shape 1; total 4 features

Total: 12 + 10 + 8 + 3 + 4 = 37 features per event.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import scipy.signal
import scipy.stats

from src.io.load_csv import PMU_BUSES

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)

# Channel groups — suffix in merged DataFrame columns
_CHAN_GROUPS = {
    "VA_MAG": [f"BUS{b}_VA_MAG" for b in PMU_BUSES],
    "IA_MAG": [f"BUS{b}_IA_MAG" for b in PMU_BUSES],
    "Freq":   [f"BUS{b}_Freq"   for b in PMU_BUSES],
    "ROCOF":  [f"BUS{b}_ROCOF"  for b in PMU_BUSES],
}

FEATURE_NAMES: list[str] = []
for _grp in _CHAN_GROUPS:
    for _stat in ("mean", "max", "std"):
        FEATURE_NAMES.append(f"{_grp}_{_stat}")         # 12 features
for _b in PMU_BUSES:
    FEATURE_NAMES.append(f"nu_bus{_b}")                 # 8 features
FEATURE_NAMES += ["nu_entropy", "nu_argmax"]            # 2 features
for _b in PMU_BUSES:
    FEATURE_NAMES.append(f"freq_spectral_bus{_b}")      # 8 features
FEATURE_NAMES += [
    "nan_count", "pmu_missing_count", "longest_nan_run", # 3 features
    "cosine_top1", "cosine_top2", "cosine_top3",         # 3 features
    "cosine_argmax_bus",                                  # 1 feature
]
# Total = 37

N_FEATURES = len(FEATURE_NAMES)  # 37


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_std(arr: np.ndarray) -> float:
    return float(np.nanstd(arr)) if len(arr) > 1 else 0.0


def _spectral_band_energy(
    series: np.ndarray,
    fps: float,
    f_lo: float = 0.1,
    f_hi: float = 2.0,
) -> float:
    """Return fraction of signal energy in [f_lo, f_hi] Hz band.

    Uses Welch's method so short windows don't blow up.
    """
    n = len(series)
    if n < 4 or np.all(np.isnan(series)):
        return 0.0
    x = np.where(np.isnan(series), 0.0, series)
    nperseg = min(n, max(4, n // 2))
    try:
        freqs, psd = scipy.signal.welch(x, fs=fps, nperseg=nperseg)
    except Exception:
        return 0.0
    total = np.sum(psd)
    if total < 1e-30:
        return 0.0
    band = psd[(freqs >= f_lo) & (freqs <= f_hi)]
    return float(np.sum(band) / total)


def _longest_nan_run(arr: np.ndarray) -> int:
    """Length of the longest consecutive NaN run in a 1-D array."""
    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        return 0
    max_run = run = 0
    for v in nan_mask:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


# ── residuals ─────────────────────────────────────────────────────────────────

def _compute_residuals(
    window: "pd.DataFrame",
    baseline_mean: dict[str, float],
) -> dict[str, np.ndarray]:
    """Compute (window_len,) residual arrays for each channel group.

    The baseline is subtracted so the residuals represent deviations from normal.
    """
    residuals: dict[str, np.ndarray] = {}
    for grp, cols in _CHAN_GROUPS.items():
        available = [c for c in cols if c in window.columns]
        if not available:
            residuals[grp] = np.zeros((len(window), len(PMU_BUSES)))
            continue
        Z = window[available].to_numpy(dtype=float)  # (T, ≤8)
        # subtract per-channel baseline
        means = np.array([baseline_mean.get(c, 0.0) for c in available])
        residuals[grp] = Z - means[None, :]
    return residuals


def _nu_bar(residuals: dict[str, np.ndarray]) -> np.ndarray:
    """Compute spatial pattern ν̄ ∈ R^8 = per-PMU RMS across all channels."""
    # Sum squared innovations per bus across all channel groups, then RMS over time
    n_bus = len(PMU_BUSES)
    sq_sum = np.zeros(n_bus)
    for grp, res in residuals.items():
        # res is (T, ≤8); pad to 8 if needed
        T, ncols = res.shape
        if ncols < n_bus:
            pad = np.zeros((T, n_bus - ncols))
            res = np.concatenate([res, pad], axis=1)
        # Replace NaN with 0 (missing → no innovation)
        res = np.where(np.isnan(res), 0.0, res)
        sq_sum += np.mean(res**2, axis=0)   # mean over time dimension
    return np.sqrt(sq_sum / max(len(residuals), 1))


# ── main feature extractor ────────────────────────────────────────────────────

def extract_features(
    df: "pd.DataFrame",
    onset_frame: int,
    J_cols: dict[int, np.ndarray] | None = None,
    fps: float = 30.0,
    window_sec: float = 3.0,
) -> np.ndarray:
    """Extract a 37-element feature vector for a single alarm onset.

    Args:
        df:          Merged DataFrame (all buses, aligned by row).
        onset_frame: Row index of the alarm onset.
        J_cols:      {bus_number: sensitivity_column (len 8)} from compute_jacobians.
                     If None, cosine similarity features are 0.
        fps:         Sampling rate in frames per second.
        window_sec:  Window half-width on each side of onset (total = window_sec).

    Returns:
        feats: (N_FEATURES,) float array.
    """
    half = int(round(fps * window_sec / 2))
    t0 = max(0, onset_frame - half)
    t1 = min(len(df), onset_frame + half)

    window = df.iloc[t0:t1]

    # ── Baseline: mean of the 30-second block BEFORE the window ───────────────
    base_start = max(0, t0 - int(round(fps * 30.0)))
    baseline_df = df.iloc[base_start:t0] if base_start < t0 else df.iloc[:max(1, t0)]
    baseline_mean: dict[str, float] = {}
    all_cols = [c for grp in _CHAN_GROUPS.values() for c in grp if c in df.columns]
    for col in all_cols:
        vals = baseline_df[col].dropna().to_numpy(float)
        baseline_mean[col] = float(vals.mean()) if len(vals) > 0 else 0.0

    residuals = _compute_residuals(window, baseline_mean)

    feats: list[float] = []

    # ── Family 1: per-channel stats ───────────────────────────────────────────
    for grp in ("VA_MAG", "IA_MAG", "Freq", "ROCOF"):
        res = residuals[grp]  # (T, ≤8)
        flat = res.ravel()
        finite = flat[np.isfinite(flat)]
        if len(finite) == 0:
            feats.extend([0.0, 0.0, 0.0])
        else:
            feats.append(float(np.nanmean(np.abs(res))))
            feats.append(float(np.nanmax(np.abs(res))))
            feats.append(_safe_std(finite))

    # ── Family 2: spatial pattern ──────────────────────────────────────────────
    nu = _nu_bar(residuals)  # (8,)
    feats.extend(nu.tolist())
    nu_sum = nu.sum()
    if nu_sum > 1e-12:
        nu_prob = nu / nu_sum
        entropy = float(-np.sum(nu_prob * np.log(nu_prob + 1e-30)))
    else:
        entropy = 0.0
    feats.append(entropy)
    feats.append(float(np.argmax(nu)))  # 0-indexed into PMU_BUSES

    # ── Family 3: spectral energy of Freq channel ─────────────────────────────
    for col in _CHAN_GROUPS["Freq"]:
        if col in window.columns:
            raw_freq = window[col].to_numpy(dtype=float)
            base_val = baseline_mean.get(col, float(np.nanmean(raw_freq)))
            freq_res = raw_freq - base_val
            feats.append(_spectral_band_energy(freq_res, fps))
        else:
            feats.append(0.0)

    # ── Family 4: cyber indicators ────────────────────────────────────────────
    dp_cols = [f"BUS{b}_DATA_PRESENT" for b in PMU_BUSES]
    dp_available = [c for c in dp_cols if c in window.columns]

    meas_cols = [c for grp in _CHAN_GROUPS.values() for c in grp if c in window.columns]
    if meas_cols:
        meas_arr = window[meas_cols].to_numpy(dtype=float)
        nan_count = int(np.isnan(meas_arr).sum())
        longest_run = max((_longest_nan_run(meas_arr[:, j]) for j in range(meas_arr.shape[1])), default=0)
    else:
        nan_count = 0
        longest_run = 0

    if dp_available:
        dp_arr = window[dp_available].to_numpy(dtype=float)
        pmu_missing_count = int((dp_arr < 1).any(axis=0).sum())
    else:
        pmu_missing_count = 0

    feats.extend([float(nan_count), float(pmu_missing_count), float(longest_run)])

    # ── Family 5: cosine similarity to J columns ──────────────────────────────
    if J_cols and nu.sum() > 1e-12:
        nu_norm = nu / (np.linalg.norm(nu) + 1e-30)
        sims: list[tuple[float, int]] = []
        for bus_num, jcol in J_cols.items():
            jnorm = np.linalg.norm(jcol)
            if jnorm < 1e-30:
                continue
            cos_sim = float(np.dot(nu_norm, jcol / jnorm))
            sims.append((cos_sim, bus_num))
        sims.sort(key=lambda x: -x[0])
        top3 = sims[:3]
        while len(top3) < 3:
            top3.append((0.0, 0))
        feats.extend([top3[0][0], top3[1][0], top3[2][0]])
        feats.append(float(top3[0][1]))  # bus number of top-1
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    arr = np.array(feats, dtype=float)
    # Replace any remaining inf/nan with 0
    arr = np.where(np.isfinite(arr), arr, 0.0)
    assert len(arr) == N_FEATURES, f"Expected {N_FEATURES} features, got {len(arr)}"
    return arr


def extract_all_events(
    df: "pd.DataFrame",
    alarm_indices: np.ndarray,
    labels: np.ndarray,
    J_cols: dict[int, np.ndarray] | None = None,
    fps: float = 30.0,
    window_sec: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features for all alarm onsets and return (X, y).

    Args:
        df:            Merged DataFrame.
        alarm_indices: (K,) int array of onset frame indices.
        labels:        (K,) int array of event labels (0-8).
        J_cols:        Sensitivity columns from compute_jacobians.
        fps:           Sampling rate.
        window_sec:    Window size around each onset.

    Returns:
        X: (K, N_FEATURES) float array.
        y: (K,) int array.
    """
    X_rows = []
    y_rows = []
    for idx, label in zip(alarm_indices, labels):
        feats = extract_features(df, int(idx), J_cols=J_cols, fps=fps, window_sec=window_sec)
        X_rows.append(feats)
        y_rows.append(int(label))
    X = np.stack(X_rows, axis=0) if X_rows else np.zeros((0, N_FEATURES))
    y = np.array(y_rows, dtype=int)
    return X, y
