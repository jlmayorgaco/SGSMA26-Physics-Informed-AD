from __future__ import annotations

import numpy as np

from src.estimation.ybus_estimator import current_base_amp_from_kv, voltage_base_phase_volts_from_kv


def test_current_base_amp_from_kv_finite_for_normal_kv() -> None:
    assert np.isfinite(current_base_amp_from_kv(345.0))


def test_current_base_amp_from_kv_fallback_for_zero_kv() -> None:
    assert np.isfinite(current_base_amp_from_kv(0.0))


def test_voltage_base_phase_volts_from_kv_finite() -> None:
    assert np.isfinite(voltage_base_phase_volts_from_kv(345.0))


def test_voltage_base_phase_volts_from_kv_fallback_for_zero() -> None:
    assert np.isfinite(voltage_base_phase_volts_from_kv(0.0))
