from __future__ import annotations

import numpy as np
import pandas as pd

from src.infrastructure.legacy.m1_adapter import normalize_bus_data


def test_magnitude_normalization_uses_baseline() -> None:
    idx = pd.Index([0.0, 0.1, 0.2, 0.3], name="TIMESTAMP")
    bus_df = pd.DataFrame(
        {
            "BUS10_VA_MAG": [100.0, 100.0, 120.0, 140.0],
            "BUS10_ROCOF": [0.1, 0.1, 0.4, 0.5],
            "DATA_PRESENT": [1, 1, 1, 1],
            "Event": [0, 0, 1, 1],
        },
        index=idx,
    )
    bus_data = {"Bus10": bus_df}
    event_df = pd.DataFrame({"Bus10": [0, 0, 1, 1]}, index=idx)

    normalized, baselines = normalize_bus_data(bus_data, event_df)

    va = normalized["Bus10"]["BUS10_VA_MAG"].to_numpy(dtype=float)
    assert np.isclose(va[0], 1.0)
    assert np.isclose(va[2], 1.2)

    rocof = normalized["Bus10"]["BUS10_ROCOF"].to_numpy(dtype=float)
    assert np.isclose(rocof[0], 0.0)
    assert np.isclose(rocof[2], 0.3)

    assert set(baselines["raw_signal"].tolist()) == {"BUS10_VA_MAG", "BUS10_ROCOF"}
