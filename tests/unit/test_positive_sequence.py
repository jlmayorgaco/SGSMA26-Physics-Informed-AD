from __future__ import annotations

import numpy as np

from src.physics.positive_sequence import positive_sequence_from_abc


def test_positive_sequence_balanced_three_phase() -> None:
    va = 1.0 + 0j
    vb = np.exp(-1j * 2.0 * np.pi / 3.0)
    vc = np.exp(1j * 2.0 * np.pi / 3.0)
    v1 = positive_sequence_from_abc(va, vb, vc)
    assert np.isclose(v1.real, 1.0, atol=1e-12)
    assert np.isclose(v1.imag, 0.0, atol=1e-12)
