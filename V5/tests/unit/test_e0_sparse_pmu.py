from __future__ import annotations

import numpy as np

from src.estimation.e0_sparse_pmu.pipeline import (
    _observed_change_gate,
    build_ybus_from_audit,
    circular_error_deg,
    wrap_angle_deg,
)


def test_circular_angle_error_handles_wraparound() -> None:
    assert np.allclose(circular_error_deg(np.array([179.0]), np.array([-179.0])), np.array([-2.0]))
    assert np.allclose(wrap_angle_deg(np.array([180.0, -180.0, 540.0])), np.array([-180.0, -180.0, -180.0]))


def test_powerdynamics_tap_convention_is_used() -> None:
    audit = {
        "branches": [
            {
                "src_bus": 1,
                "dst_bus": 2,
                "R": 0.0,
                "X": 1.0,
                "G_src": 0.0,
                "B_src": 0.0,
                "G_dst": 0.0,
                "B_dst": 0.0,
                "r_src": 2.0,
                "r_dst": 1.0,
            }
        ]
    }
    ybus = build_ybus_from_audit(audit, bus_count=2)
    y = -1j
    assert np.isclose(ybus[0, 0], 4.0 * y)
    assert np.isclose(ybus[1, 1], y)
    assert np.isclose(ybus[0, 1], -2.0 * y)
    assert np.isclose(ybus[1, 0], -2.0 * y)


def test_observed_change_gate_uses_sparse_voltage_only() -> None:
    timestamps = np.arange(451, dtype=float) / 30.0
    voltage = np.ones((451, 2), dtype=complex)
    voltage[150:, :] *= 0.999
    voltage[300:, :] /= 0.999
    gate = _observed_change_gate(voltage, timestamps)
    assert gate[151]
    assert gate[299]
    assert not gate[100]
    assert not gate[350]
