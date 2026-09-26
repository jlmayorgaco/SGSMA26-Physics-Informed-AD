from __future__ import annotations

import numpy as np

from src.estimation.ybus_estimator import positive_sequence_from_abc


def test_positive_sequence_from_abc_balanced_case() -> None:
    va = 1.0 + 0j
    vb = np.exp(-1j * 2 * np.pi / 3)
    vc = np.exp(1j * 2 * np.pi / 3)
    seq = positive_sequence_from_abc(va, vb, vc)
    assert np.isfinite(np.real(seq))
    assert np.isfinite(np.imag(seq))


def test_positive_sequence_vector_style_finite() -> None:
    va = 100.0 * np.exp(1j * np.deg2rad(10.0))
    vb = 100.0 * np.exp(1j * np.deg2rad(-110.0))
    vc = 100.0 * np.exp(1j * np.deg2rad(130.0))
    seq = positive_sequence_from_abc(va, vb, vc)
    assert np.isfinite(np.abs(seq))
