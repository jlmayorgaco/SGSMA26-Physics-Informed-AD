from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.physical.heuristics.current_surge_rules import current_surge_evidence
from src.detectors.physical.heuristics.estimator_innovation_rules import estimator_innovation_evidence
from src.detectors.physical.heuristics.freq_rocof_rules import freq_rocof_evidence
from src.detectors.physical.heuristics.voltage_sag_rules import voltage_sag_evidence


def test_physical_heuristics_basic_behavior() -> None:
    x = np.ones((5, 12, 6), dtype=float)
    # voltage sag in last sample
    x[4, :, 0] = np.linspace(1.0, -2.0, 12)
    # current surge in sample 3
    x[3, 8, 3] = 10.0
    # freq/rocof anomalies
    x[2, :, 4] = np.linspace(0.0, 2.0, 12)
    x[2, :, 5] = np.linspace(0.0, 3.0, 12)
    names = ["BUS10_VA_MAG", "BUS10_VB_MAG", "BUS10_VC_MAG", "BUS10_IA_MAG", "BUS10_Freq", "BUS10_ROCOF"]
    sag = voltage_sag_evidence(x, names)
    surge = current_surge_evidence(x, names)
    freq = freq_rocof_evidence(x, names)
    est = estimator_innovation_evidence(pd.DataFrame({"ESTIMATOR_DIFFICULTY_SCORE": [0.0, 0.2, 0.7, 0.1, 0.3]}))
    assert float(sag["score"][4]) >= float(sag["score"][0])
    assert float(surge["score"][3]) >= float(surge["score"][0])
    assert float(freq["score"][2]) >= float(freq["score"][0])
    assert float(est["score"][2]) == 0.7

