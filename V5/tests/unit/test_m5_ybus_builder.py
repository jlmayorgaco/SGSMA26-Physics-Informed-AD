from __future__ import annotations

from pathlib import Path

import numpy as np

from src.metadata.raw_parser import parse_raw_file
from src.metadata.ybus_builder import build_ybus


def test_build_ybus_returns_square_matrix() -> None:
    raw = parse_raw_file(Path("data/metadata/IEEE_39_Bus_Power_System.raw"))
    y, order, _ = build_ybus(raw)
    assert y.shape[0] == y.shape[1]
    assert y.shape[0] == len(order)


def test_matrix_is_complex() -> None:
    raw = parse_raw_file(Path("data/metadata/IEEE_39_Bus_Power_System.raw"))
    y, _, _ = build_ybus(raw)
    assert np.iscomplexobj(y)


def test_diagonal_entries_finite() -> None:
    raw = parse_raw_file(Path("data/metadata/IEEE_39_Bus_Power_System.raw"))
    y, _, _ = build_ybus(raw)
    assert np.all(np.isfinite(np.real(np.diag(y))))


def test_bus_order_explicit() -> None:
    raw = parse_raw_file(Path("data/metadata/IEEE_39_Bus_Power_System.raw"))
    _, order, _ = build_ybus(raw)
    assert len(order) == 39


def test_matrix_not_all_zeros() -> None:
    raw = parse_raw_file(Path("data/metadata/IEEE_39_Bus_Power_System.raw"))
    y, _, _ = build_ybus(raw)
    assert np.any(np.abs(y) > 0.0)
