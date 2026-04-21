from __future__ import annotations

import numpy as np


def missing_data_evidence(
    windows: np.ndarray,
    *,
    feature_names: list[str],
    data_present_index: int | None,
) -> dict[str, np.ndarray]:
    n = windows.shape[0]
    if n == 0:
        zeros = np.zeros((0,), dtype=float)
        return {
            "missing_ratio": zeros,
            "all_missing_flag": zeros,
            "nan_ratio": zeros,
            "data_present_transition_rate": zeros,
            "nan_burst_count": zeros,
            "nan_burst_max_len": zeros,
            "missing_asymmetry": zeros,
            "partial_dropout_ratio": zeros,
            "score": zeros,
        }
    missing_ratio = np.zeros((n,), dtype=float)
    if data_present_index is not None and 0 <= data_present_index < windows.shape[2]:
        dp = np.clip(windows[:, :, data_present_index], 0.0, 1.0)
        missing_ratio = 1.0 - dp.mean(axis=1)
    nan_mask_cols = [i for i, name in enumerate(feature_names) if name.endswith("__is_nan")]
    if nan_mask_cols:
        nan_values = np.clip(windows[:, :, nan_mask_cols], 0.0, 1.0)
        nan_ratio = nan_values.mean(axis=(1, 2))
        # How uneven missingness is across channels (captures partial channel dropout).
        channel_missing = nan_values.mean(axis=1)
        missing_asymmetry = channel_missing.std(axis=1)
        partial_dropout_ratio = ((channel_missing > 0.2) & (channel_missing < 0.95)).mean(axis=1)
        burst_counts = np.zeros((n,), dtype=float)
        burst_max_len = np.zeros((n,), dtype=float)
        for i in range(n):
            frame_missing = (nan_values[i].mean(axis=1) > 0.5).astype(int)
            if frame_missing.sum() == 0:
                continue
            padded = np.r_[0, frame_missing, 0]
            starts = np.where(np.diff(padded) == 1)[0]
            ends = np.where(np.diff(padded) == -1)[0]
            lengths = ends - starts
            burst_counts[i] = float(len(lengths))
            burst_max_len[i] = float(lengths.max()) if len(lengths) else 0.0
    else:
        nan_ratio = np.zeros((n,), dtype=float)
        missing_asymmetry = np.zeros((n,), dtype=float)
        partial_dropout_ratio = np.zeros((n,), dtype=float)
        burst_counts = np.zeros((n,), dtype=float)
        burst_max_len = np.zeros((n,), dtype=float)
    if data_present_index is not None and 0 <= data_present_index < windows.shape[2]:
        dp = np.clip(windows[:, :, data_present_index], 0.0, 1.0)
        data_present_transition_rate = np.abs(np.diff(dp, axis=1)).mean(axis=1)
    else:
        data_present_transition_rate = np.zeros((n,), dtype=float)
    all_missing_flag = ((missing_ratio > 0.95) | (nan_ratio > 0.95)).astype(float)
    score = np.clip(
        0.45 * missing_ratio
        + 0.20 * nan_ratio
        + 0.15 * np.minimum(burst_max_len / max(windows.shape[1], 1), 1.0)
        + 0.10 * np.minimum(data_present_transition_rate / 0.3, 1.0)
        + 0.10 * partial_dropout_ratio
        + 0.15 * all_missing_flag,
        0.0,
        1.0,
    )
    return {
        "missing_ratio": missing_ratio,
        "all_missing_flag": all_missing_flag,
        "nan_ratio": nan_ratio,
        "data_present_transition_rate": data_present_transition_rate,
        "nan_burst_count": burst_counts,
        "nan_burst_max_len": burst_max_len,
        "missing_asymmetry": missing_asymmetry,
        "partial_dropout_ratio": partial_dropout_ratio,
        "score": score,
    }
