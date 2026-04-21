from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation.m9.cyber import apply_timing_attack


def test_timestamp_attack_operators() -> None:
    df = pd.DataFrame(
        {
            "TIMESTAMP": np.arange(0.0, 1.0, 0.1),
            "BUS10_VA_ANG": np.linspace(0.0, 1.0, 10),
            "BUS10_VA_MAG": np.linspace(100.0, 110.0, 10),
            "DATA_PRESENT": 1,
            "Event": 0,
        }
    )
    event = {
        "event_label": 7,
        "event_type": "bad_data",
        "subtype": "fixed_delay",
        "start_time_s": 0.2,
        "end_time_s": 0.7,
        "target_channels": ["VA_ANG", "VA_MAG"],
        "params": {"delay_frames": 2},
    }
    out = apply_timing_attack(df, "BUS10", event, seed=1)
    assert (out["Event"] == 7).any()
    assert out["DATA_PRESENT"].min() == 1
    assert not np.allclose(out["BUS10_VA_ANG"].to_numpy(), df["BUS10_VA_ANG"].to_numpy())
