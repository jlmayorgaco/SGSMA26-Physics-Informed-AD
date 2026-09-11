from __future__ import annotations

import numpy as np

from pmu_hybrid.physics.configuration import ConfigurationJump, EventFamily
from pmu_hybrid.simulator.noise import MeasurementNoiseSpec, corrupt_measurements
from pmu_hybrid.simulator.scenario import ScenarioSpec, configuration_trajectory, data_present_mask


def test_line_jump_persists_and_fault_clears() -> None:
    spec = ScenarioSpec(
        scenario_id="test", seed=1, operating_point_id="op", model_seed=2, duration_s=1.0, sample_rate_hz=10.0,
        physical_jumps=(
            ConfigurationJump("line", EventFamily.LINE, "L1", 0.2, line_in_service=False),
            ConfigurationJump("fault", EventFamily.FAULT, "BUS3", 0.3, amplitude=2j, duration_s=0.2),
        ),
    )
    history = configuration_trajectory(spec)
    assert history[3].line_status["L1"] is False
    assert "fault" in history[3].fault_shunts_pu
    assert "fault" not in history[5].fault_shunts_pu
    assert history[-1].line_status["L1"] is False


def test_measurement_corruption_is_seeded_and_honors_dropouts() -> None:
    spec = ScenarioSpec(scenario_id="drop", seed=5, operating_point_id="op", model_seed=1, duration_s=1.0, sample_rate_hz=4.0)
    present = data_present_mask(spec, (2, 5))
    present[1, 0] = False
    shape = present.shape
    arguments = (np.ones(shape, complex), np.zeros(shape, complex), np.ones(shape) * 60, np.zeros(shape), present, MeasurementNoiseSpec(ar1=0.2))
    first = corrupt_measurements(*arguments, seed=9, namespace="test")
    second = corrupt_measurements(*arguments, seed=9, namespace="test")
    assert np.array_equal(first.voltage_pu, second.voltage_pu, equal_nan=True)
    assert np.isnan(first.voltage_pu[1, 0])
