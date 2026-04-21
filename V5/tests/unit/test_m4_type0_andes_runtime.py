from __future__ import annotations

import numpy as np

from src.simulation.andes_normal_runtime import _is_usable_tds_result


class _FakeTs:
    def __init__(self, t):
        self.t = t


class _FakeDae:
    def __init__(self, t):
        self.ts = _FakeTs(t)


class _FakeSystem:
    def __init__(self, t):
        self.dae = _FakeDae(t)


def test_is_usable_tds_result_true_near_complete() -> None:
    sys = _FakeSystem(np.linspace(0.0, 10.08, 305))
    assert _is_usable_tds_result(sys, tf=10.1, tstep=1.0 / 30.0)


def test_is_usable_tds_result_false_short_series() -> None:
    sys = _FakeSystem(np.array([0.0, 0.03]))
    assert not _is_usable_tds_result(sys, tf=10.1, tstep=1.0 / 30.0)


def test_is_usable_tds_result_false_far_from_tf() -> None:
    sys = _FakeSystem(np.linspace(0.0, 8.0, 200))
    assert not _is_usable_tds_result(sys, tf=10.1, tstep=1.0 / 30.0)
