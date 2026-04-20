from __future__ import annotations

import numpy as np

from src.simulation.scenario_labels import event_series_type0


def test_type0_event_series_all_zeros() -> None:
    t = np.array([0.0, 1.0, 2.0, 3.0])
    ev = event_series_type0(t)
    assert ev.tolist() == [0, 0, 0, 0]


def test_type0_event_length_preserved() -> None:
    t = np.linspace(0.0, 10.0, 101)
    ev = event_series_type0(t)
    assert len(ev) == len(t)


def test_type0_no_fault_interval_assumption() -> None:
    t = np.array([4.9, 5.0, 5.1, 8.0])
    ev = event_series_type0(t)
    assert np.all(ev == 0)
