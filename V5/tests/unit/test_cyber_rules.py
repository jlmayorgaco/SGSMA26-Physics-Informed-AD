from __future__ import annotations

import numpy as np

from src.detectors.cyber.heuristics.missing_data_rules import missing_data_evidence
from src.detectors.cyber.heuristics.spike_rules import spike_evidence
from src.detectors.cyber.heuristics.stuck_value_rules import stuck_value_evidence
from src.detectors.cyber.heuristics.timestamp_rules import timestamp_evidence


def test_cyber_rules_behave_sensibly() -> None:
    x = np.zeros((4, 8, 3), dtype=float)
    x[2:, :, 2] = 0.0
    x[:2, :, 2] = 1.0
    x[3, 4, 0] = 20.0
    missing = missing_data_evidence(x, feature_names=["A", "B", "DATA_PRESENT"], data_present_index=2)
    stuck = stuck_value_evidence(x)
    spike = spike_evidence(x)
    ts = timestamp_evidence(np.array([0.0, 0.1, 0.2, 0.7]))
    assert missing["score"].shape == (4,)
    assert float(missing["score"][3]) >= float(missing["score"][1])
    assert float(stuck["score"][0]) >= 0.0
    assert float(spike["score"][3]) > float(spike["score"][0])
    assert float(ts["score"][-1]) >= 0.0

