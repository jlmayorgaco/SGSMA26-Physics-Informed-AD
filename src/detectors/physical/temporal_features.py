from __future__ import annotations

import numpy as np


def _dilated_diff(x: np.ndarray, dilation: int) -> np.ndarray:
    if dilation <= 0:
        raise ValueError("dilation must be > 0")
    shifted = np.roll(x, shift=dilation, axis=1)
    shifted[:, :dilation, :] = x[:, :1, :]
    return x - shifted


def build_temporal_feature_matrix(windows: np.ndarray, dilations: tuple[int, ...]) -> np.ndarray:
    # windows: [n, t, f]
    n = windows.shape[0]
    feats: list[np.ndarray] = []
    feats.append(windows.mean(axis=1))
    feats.append(windows.std(axis=1))
    for d in dilations:
        diff = _dilated_diff(windows, d)
        # TCN-like temporal derivative activations with global pooling.
        feats.append(np.abs(diff).mean(axis=1))
        feats.append(np.abs(diff).max(axis=1))
    spectral_proxy = np.abs(np.fft.rfft(windows, axis=1))
    feats.append(spectral_proxy.mean(axis=1))
    return np.concatenate(feats, axis=1).reshape(n, -1)

