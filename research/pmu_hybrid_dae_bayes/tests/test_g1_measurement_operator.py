from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pmu_hybrid.constants import PMU_BUSES
from pmu_hybrid.physics.measurement import (
    PMUTerminal,
    build_measurement_batch,
    causal_frequency_rocof,
    ideal_state_frequency,
    select_controlled_terminal_map,
    terminal_current,
    voltage_operator,
)
from pmu_hybrid.physics.network import Branch, branch_terminal_currents, branch_terminal_powers
from pmu_hybrid.simulator.noise import MeasurementNoiseSpec, corrupt_measurements


def test_voltage_operator_selects_declared_bus_order() -> None:
    buses = (1, 2, 5, 6, 10, 19, 22, 29, 39)
    voltage = np.arange(len(buses), dtype=float) + 1j * np.arange(len(buses), dtype=float)
    observed = voltage_operator(buses, voltage)
    assert tuple(PMU_BUSES) == (2, 5, 6, 10, 19, 22, 29, 39)
    assert observed.tolist() == [voltage[buses.index(bus)] for bus in PMU_BUSES]


def test_current_from_and_to_terminals_are_exact_and_oriented() -> None:
    branch = Branch("L", 2, 7, 0.01, 0.1, charging_pu=0.02)
    v_from, v_to = 1.02 + 0.1j, 0.98 - 0.04j
    i_from, i_to = branch_terminal_currents(branch, v_from, v_to)
    assert terminal_current(PMUTerminal(2, "L", "from", 7), branch, v_from, v_to) == i_from
    assert terminal_current(PMUTerminal(7, "L", "to", 2), branch, v_from, v_to) == i_to
    with pytest.raises(ValueError):
        terminal_current(PMUTerminal(2, "L", "to", 7), branch, v_from, v_to)


def test_transformer_tap_and_complex_shift_are_in_primitive() -> None:
    branch = Branch("T", 1, 2, 0.01, 0.08, charging_pu=0.01, tap=1.05, shift_rad=0.03)
    untapped = Branch("U", 1, 2, 0.01, 0.08, charging_pu=0.01)
    current_tap = branch_terminal_currents(branch, 1 + 0j, 0.97 - 0.03j)[0]
    current_plain = branch_terminal_currents(untapped, 1 + 0j, 0.97 - 0.03j)[0]
    assert abs(current_tap - current_plain) > 1e-3


def test_orientation_and_power_use_current_leaving_each_terminal() -> None:
    branch = Branch("L", 1, 2, 0.02, 0.12, charging_pu=0.01)
    s_from, s_to = branch_terminal_powers(branch, 1.02 + 0j, 0.98 - 0.02j, 100.0)
    assert (s_from + s_to).real >= -1e-10
    assert np.isclose(s_from, (1.02 + 0j) * np.conj(branch_terminal_currents(branch, 1.02 + 0j, 0.98 - 0.02j)[0]) * 100.0)


def test_controlled_map_is_frozen_and_does_not_mutate() -> None:
    branches = (Branch("z", 2, 7, 0.01, 0.1), Branch("a", 1, 2, 0.01, 0.1))
    mapping = select_controlled_terminal_map(branches, (2,))
    assert mapping[2].branch_id == "a"
    with pytest.raises(dataclasses.FrozenInstanceError):
        mapping[2].branch_id = "z"  # type: ignore[misc]


def test_positive_sequence_row_has_seven_fields_and_explicit_mask() -> None:
    shape = (2, 2)
    batch = build_measurement_batch(
        np.ones(shape, dtype=complex), np.full(shape, 2 + 1j), np.full(shape, 60.0),
        np.zeros(shape), np.array([[True, False], [True, True]]), metadata={"mode": "test"},
    )
    assert batch.values.shape == (2, 2, 7)
    assert batch.values[0, 1, -1] == 0.0
    assert bool(batch.data_present[0, 1]) is False
    assert batch.metadata["operator"] == "clean_positive_sequence"


def test_noise_is_seeded_and_missingness_is_not_zero_filled() -> None:
    values = np.ones((4, 2), dtype=complex)
    mask = np.array([[True, False], [True, True], [False, True], [True, True]])
    spec = MeasurementNoiseSpec(voltage_std_pu=0.0, current_std_pu=0.0, frequency_std_hz=0.0, rocof_std_hz_s=0.0)
    a = corrupt_measurements(values, values, np.full((4, 2), 60.0), np.zeros((4, 2)), mask, spec, seed=7, namespace="g1")
    b = corrupt_measurements(values, values, np.full((4, 2), 60.0), np.zeros((4, 2)), mask, spec, seed=7, namespace="g1")
    assert np.array_equal(a.data_present, mask)
    assert np.array_equal(a.voltage_pu, b.voltage_pu, equal_nan=True)
    assert np.isnan(a.voltage_pu[0, 1])


def test_ideal_frequency_and_rocof_are_explicit_modes() -> None:
    t = np.arange(0.0, 1.0, 1.0 / 30.0)
    phase = 2 * np.pi * 0.2 * t + 0.5 * 0.4 * t * t
    frequency, rocof = ideal_state_frequency(phase, t)
    assert np.nanmean(frequency[2:-2]) == pytest.approx(60.2 + 0.4 * np.mean(t[2:-2]) / (2 * np.pi), abs=1e-3)
    assert np.nanmean(rocof[2:-2]) == pytest.approx(0.4 / (2 * np.pi), abs=1e-2)
    causal, _ = causal_frequency_rocof(phase, 30.0, window_frames=5)
    altered = phase.copy(); altered[15:] += 1.0
    changed, _ = causal_frequency_rocof(altered, 30.0, window_frames=5)
    assert np.allclose(causal[:15], changed[:15], equal_nan=True)


def test_static_multipoint_operator_parity_has_five_ac_points() -> None:
    pytest.importorskip("andes")
    pytest.importorskip("pandapower")
    from pmu_hybrid.cases.ieee39_andes import solve_static
    from pmu_hybrid.experiments.g1_measurement_audit import _audit_static
    from pathlib import Path

    static = solve_static()
    mapping = select_controlled_terminal_map(static.branches)
    test_root = Path(__file__).resolve().parents[1] / "output" / "g1_test_artifacts"
    summary = _audit_static(test_root, static, mapping)
    table = np.genfromtxt(test_root / "output" / "results" / "g1_static_multipoint.csv", delimiter=",", names=True)
    assert len(table) == 5
    assert np.all(table["converged"])
    assert summary["max_current_error_pu"] < 1e-10
