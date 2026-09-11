from __future__ import annotations

import numpy as np

from pmu_hybrid.physics.measurement import causal_frequency_rocof, select_controlled_terminal_map, terminal_current
from pmu_hybrid.physics.network import Branch, branch_terminal_currents


def test_controlled_map_is_deterministic_and_oriented_at_pmu() -> None:
    branches = (
        Branch("b", 2, 7, 0.01, 0.1),
        Branch("a", 2, 1, 0.01, 0.1),
    )
    mapping = select_controlled_terminal_map(branches, (2,))
    assert mapping[2].branch_id == "a"
    assert mapping[2].terminal == "from"
    current = terminal_current(mapping[2], branches[1], 1 + 0j, 0.99 - 0.01j)
    assert current == branch_terminal_currents(branches[1], 1 + 0j, 0.99 - 0.01j)[0]


def test_causal_frequency_never_changes_past_estimates_when_future_changes() -> None:
    phase = 2.0 * np.pi * 0.2 * np.arange(12) / 30.0
    baseline, _ = causal_frequency_rocof(phase, 30.0, window_frames=4)
    altered = phase.copy()
    altered[8:] += 0.7
    changed, _ = causal_frequency_rocof(altered, 30.0, window_frames=4)
    assert np.allclose(baseline[:8], changed[:8], equal_nan=True)
    assert np.nanmean(baseline[3:]) > 60.19
