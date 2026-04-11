"""Residual-window feature extraction shared by POC classifiers."""

from __future__ import annotations

import numpy as np

from poc.schema import PMU_BUSES

GROUP_IDXS = {
    "v_ang": [0, 2, 4],
    "v_mag": [1, 3, 5],
    "i_ang": [6, 8, 10],
    "i_mag": [7, 9, 11],
    "freq": [12],
    "rocof": [13],
}

FEATURE_NAMES: list[str] = []
for _group in GROUP_IDXS:
    for _stat in ("mean_abs", "max_abs", "std", "p95_abs"):
        FEATURE_NAMES.append(f"{_group}_{_stat}")
for _bus in PMU_BUSES:
    FEATURE_NAMES.append(f"pmu_energy_bus{_bus}")
FEATURE_NAMES += [
    "pmu_entropy",
    "pmu_argmax",
    "nan_fraction",
    "pmu_missing_fraction",
    "longest_nan_run",
    "v_mag_step",
    "i_mag_step",
    "freq_step",
    "rocof_step",
    "spike_ratio",
]

FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def make_windows(series: np.ndarray, centers: np.ndarray, window_size: int = 90) -> np.ndarray:
    """Extract fixed-size windows centered on event frames without wrapping."""

    series = np.asarray(series, dtype=float)
    centers = np.asarray(centers, dtype=int)
    half = window_size // 2
    windows = np.empty((len(centers), window_size, series.shape[1]), dtype=float)
    for i, center in enumerate(centers):
        start = int(center) - half
        end = start + window_size
        src_start = max(0, start)
        src_end = min(len(series), end)
        dst_start = src_start - start
        fill = np.nanmedian(series[max(0, center - half) : min(len(series), center + half + 1)], axis=0)
        fill = np.where(np.isfinite(fill), fill, 0.0)
        windows[i] = fill[None, :]
        if src_end > src_start:
            windows[i, dst_start : dst_start + (src_end - src_start)] = series[src_start:src_end]
    return windows


def tabular_features_from_windows(windows: np.ndarray) -> np.ndarray:
    """Convert ``(N, W, 112)`` residual windows into hand-crafted features."""

    windows = np.asarray(windows, dtype=float)
    if windows.ndim == 2:
        windows = windows[:, None, :]
    if windows.shape[-1] != 112:
        raise ValueError(f"Expected 112 PMU channels, got {windows.shape[-1]}")

    grouped = windows.reshape(windows.shape[0], windows.shape[1], len(PMU_BUSES), 14)
    nan_mask = np.isnan(grouped)
    clean = np.where(nan_mask, 0.0, grouped)
    feats: list[np.ndarray] = []

    for idxs in GROUP_IDXS.values():
        vals = clean[:, :, :, idxs].reshape(len(windows), -1)
        feats.extend(
            [
                np.mean(np.abs(vals), axis=1),
                np.max(np.abs(vals), axis=1),
                np.std(vals, axis=1),
                np.percentile(np.abs(vals), 95, axis=1),
            ]
        )

    pmu_energy = np.sqrt(np.mean(clean * clean, axis=(1, 3)))
    feats.append(pmu_energy)
    energy_sum = np.maximum(pmu_energy.sum(axis=1), 1e-12)
    prob = pmu_energy / energy_sum[:, None]
    feats.append(-np.sum(prob * np.log(prob + 1e-12), axis=1)[:, None])
    feats.append(np.argmax(pmu_energy, axis=1).astype(float)[:, None])

    feats.append(nan_mask.mean(axis=(1, 2, 3))[:, None])
    pmu_missing = nan_mask.all(axis=3).any(axis=1).mean(axis=1)
    feats.append(pmu_missing[:, None])
    longest = np.array([_longest_nan_run(nan_mask[i].any(axis=(1, 2))) for i in range(len(windows))], dtype=float)
    feats.append(longest[:, None])

    third = max(1, windows.shape[1] // 3)
    first = clean[:, :third]
    last = clean[:, -third:]
    for idxs in (GROUP_IDXS["v_mag"], GROUP_IDXS["i_mag"], GROUP_IDXS["freq"], GROUP_IDXS["rocof"]):
        step = np.mean(np.abs(last[:, :, :, idxs]), axis=(1, 2, 3)) - np.mean(
            np.abs(first[:, :, :, idxs]),
            axis=(1, 2, 3),
        )
        feats.append(step[:, None])

    abs_clean = np.abs(clean.reshape(len(windows), -1))
    spike = np.max(abs_clean, axis=1) / (np.mean(abs_clean, axis=1) + 1e-9)
    feats.append(spike[:, None])

    return np.concatenate([f if f.ndim == 2 else f[:, None] for f in feats], axis=1)


def ensure_tabular(features) -> np.ndarray:
    """Accept tabular arrays or raw windows and return tabular features."""

    arr = np.asarray(features, dtype=float)
    if arr.ndim == 3:
        return tabular_features_from_windows(arr)
    return arr


def _longest_nan_run(mask: np.ndarray) -> int:
    best = run = 0
    for value in mask.astype(bool):
        if value:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best

