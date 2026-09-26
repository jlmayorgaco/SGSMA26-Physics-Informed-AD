from __future__ import annotations

import numpy as np

from src.signals.rocof import robust_rocof


def test_robust_rocof_preserves_shape() -> None:
    t = np.linspace(0.0, 1.0, 31)
    f = 60.0 + 0.01 * t
    out = robust_rocof(f, t)
    assert out.shape == f.shape


def test_robust_rocof_clips_values() -> None:
    t = np.linspace(0.0, 1.0, 31)
    f = np.zeros_like(t)
    f[15:] = 1000.0
    out = robust_rocof(f, t, clip_limits=(-5.0, 5.0))
    assert np.max(out) <= 5.0
    assert np.min(out) >= -5.0
