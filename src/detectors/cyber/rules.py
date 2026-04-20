from __future__ import annotations

import numpy as np

from src.detectors.configs import CyberRulesConfig


def _safe_zscores(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1e-6, std)
    return (x - mean) / std


def compute_cyber_rule_features(windows: np.ndarray, timestamps: np.ndarray, data_present_index: int | None = None) -> dict[str, np.ndarray]:
    n = windows.shape[0]
    flat = windows.reshape(n, -1)
    std_per_window = flat.std(axis=1)
    missing_ratio = np.zeros((n,), dtype=float)
    if data_present_index is not None and windows.shape[2] > data_present_index:
        dp = np.clip(windows[:, :, data_present_index], 0.0, 1.0)
        missing_ratio = 1.0 - dp.mean(axis=1)
    z = _safe_zscores(flat.reshape(n, -1))
    spike_ratio = (np.abs(z) > 5.0).mean(axis=1)
    # Timestamp jitter per window from center timestamps.
    if len(timestamps) > 1:
        diffs = np.diff(timestamps, prepend=timestamps[0])
        jitter = np.abs(diffs - np.median(diffs))
    else:
        jitter = np.zeros_like(timestamps)
    return {
        "missing_ratio": missing_ratio,
        "stuck_metric": 1.0 / (1.0 + std_per_window),
        "spike_ratio": spike_ratio,
        "timestamp_jitter": jitter,
    }


def cyber_rule_probabilities(features: dict[str, np.ndarray], config: CyberRulesConfig) -> np.ndarray:
    missing_term = np.clip(features["missing_ratio"], 0.0, 1.0)
    stuck_term = np.clip(features["stuck_metric"], 0.0, 1.0)
    spike_term = np.clip(features["spike_ratio"] / 0.2, 0.0, 1.0)
    jitter_term = np.clip(features["timestamp_jitter"] / max(config.timestamp_jitter_threshold, 1e-6), 0.0, 1.0)
    weighted = (
        config.missing_ratio_weight * missing_term
        + config.stuck_weight * stuck_term
        + config.spike_weight * spike_term
        + config.timestamp_weight * jitter_term
    )
    return np.clip(weighted, 0.0, 1.0)

