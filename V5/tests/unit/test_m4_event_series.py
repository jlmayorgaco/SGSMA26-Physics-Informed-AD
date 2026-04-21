from __future__ import annotations

import numpy as np

from src.simulation.competition_export import event_series


def test_event_series_windowed_marks_fault_interval() -> None:
    t = np.array([4.9, 5.0, 5.05, 5.11])
    ev = event_series(t, mode="windowed")
    assert ev.tolist() == [0, 1, 1, 0]


def test_event_series_file_constant_mode_all_ones() -> None:
    t = np.array([0.0, 1.0, 2.0])
    ev = event_series(t, mode="file_constant")
    assert ev.tolist() == [1, 1, 1]


def test_event_series_preserves_length() -> None:
    t = np.linspace(0.0, 10.0, 21)
    ev = event_series(t)
    assert len(ev) == len(t)
