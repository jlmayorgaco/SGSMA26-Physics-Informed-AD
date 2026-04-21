from __future__ import annotations

import numpy as np

from src.estimation.state_estimation.per_unit import current_to_pu, voltage_to_pu


def test_per_unit_voltage_and_current() -> None:
    v = np.asarray([199185.0 + 0j], dtype=complex)
    i = np.asarray([167.35 + 0j], dtype=complex)
    v_pu = voltage_to_pu(v, kv_ll=345.0)
    i_pu = current_to_pu(i, kv_ll=345.0, base_mva=100.0)
    assert np.isclose(float(np.abs(v_pu[0])), 1.0, atol=1e-3)
    assert np.isclose(float(np.abs(i_pu[0])), 1.0, atol=1e-2)

