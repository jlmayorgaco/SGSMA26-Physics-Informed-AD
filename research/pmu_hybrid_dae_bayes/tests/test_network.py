from __future__ import annotations

import numpy as np

from pmu_hybrid.physics.network import (
    Branch,
    branch_terminal_currents,
    gauge_aligned_angles,
    net_injections_mva,
    polar_power_injections,
    ybus,
)


def test_terminal_currents_match_ybus_contributions_with_complex_tap() -> None:
    branch = Branch("t", 1, 2, 0.02, 0.20, charging_pu=0.03, tap=1.08, shift_rad=0.12)
    voltage = np.array([1.01 + 0.04j, 0.97 - 0.02j])
    i_from, i_to = branch_terminal_currents(branch, *voltage)
    current = ybus((1, 2), (branch,)) @ voltage
    assert np.allclose(current, [i_from, i_to])


def test_rectangular_and_polar_power_identities_agree() -> None:
    branches = (
        Branch("a", 1, 2, 0.01, 0.10, charging_pu=0.02),
        Branch("b", 2, 3, 0.02, 0.15, charging_pu=0.03, tap=1.03, shift_rad=-0.04),
    )
    magnitude = np.array([1.02, 0.99, 1.01])
    angle = np.array([0.03, -0.02, 0.01])
    voltage = magnitude * np.exp(1j * angle)
    rectangular = net_injections_mva((1, 2, 3), voltage, branches, 100.0)
    polar = polar_power_injections((1, 2, 3), magnitude, angle, ybus((1, 2, 3), branches), 100.0)
    assert all(np.isclose(rectangular[bus], polar[bus]) for bus in rectangular)


def test_angle_gauge_alignment_is_invariant_to_global_rotation() -> None:
    angles = np.array([-0.2, 0.1, 0.4])
    assert np.allclose(gauge_aligned_angles(angles, 1), gauge_aligned_angles(angles + 1.7, 1))
