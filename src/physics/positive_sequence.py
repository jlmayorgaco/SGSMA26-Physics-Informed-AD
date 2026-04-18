"""Three-phase phasor utilities."""

from __future__ import annotations

import numpy as np


def positive_sequence_from_abc(va: complex, vb: complex, vc: complex) -> complex:
    """Return positive-sequence phasor V1 from phase phasors Va, Vb, Vc."""
    a = np.exp(1j * 2.0 * np.pi / 3.0)
    return (va + a * vb + (a**2) * vc) / 3.0
