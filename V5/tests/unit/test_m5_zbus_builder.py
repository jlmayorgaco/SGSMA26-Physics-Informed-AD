from __future__ import annotations

import numpy as np

from src.metadata.zbus_builder import build_zbus


def test_build_zbus_returns_square_matrix_when_invertible() -> None:
    y = np.array([[2 + 1j, -1], [-1, 2 + 1j]], dtype=complex)
    z, meta = build_zbus(y)
    assert z.shape == y.shape
    assert "condition_number" in meta


def test_invertibility_status_present() -> None:
    y = np.array([[2 + 0j, 0], [0, 3 + 0j]], dtype=complex)
    _, meta = build_zbus(y)
    assert "invertible" in meta


def test_singular_case_handled_cleanly() -> None:
    y = np.array([[1 + 0j, 1 + 0j], [1 + 0j, 1 + 0j]], dtype=complex)
    z, meta = build_zbus(y)
    assert z.shape == y.shape
    assert not meta["invertible"]
