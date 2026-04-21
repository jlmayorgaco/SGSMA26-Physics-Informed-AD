from __future__ import annotations

import numpy as np
import pandas as pd

from src.estimators.estimator_event_7 import Event7Estimator


def _frame(prefix: str, signal: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"{prefix}_VA_MAG": signal,
            f"{prefix}_IA_MAG": signal * 0.15 + 0.7,
            f"{prefix}_Freq": 60.0 + 0.01 * np.sin(np.arange(signal.size) * 0.1),
            "DATA_PRESENT": np.ones((signal.size,), dtype=float),
            "Event": event,
        }
    )


def test_event7_estimator_detects_single_bus_corruption() -> None:
    n = 140
    t = np.arange(n, dtype=float)
    base = 1.2 + 0.3 * np.sin(0.05 * t)
    event_bus2 = np.zeros((n,), dtype=int)
    event_bus39 = np.zeros((n,), dtype=int)

    signal_bus2 = base.copy()
    signal_bus2[70] += 9.0
    signal_bus2[71] -= 7.0
    signal_bus2[90:102] = signal_bus2[89]
    event_bus2[70:72] = 7

    frame_bus2 = _frame("BUS2", signal_bus2, event_bus2)
    frame_bus39 = _frame("BUS39", base + 0.01, event_bus39)
    timeline = pd.Index(np.arange(n, dtype=float) * 0.033, name="TIMESTAMP")
    frame_bus2.index = timeline
    frame_bus39.index = timeline

    estimator = Event7Estimator()
    result = estimator.estimate({"Bus2": frame_bus2, "Bus39": frame_bus39})

    pred = result.frame_prediction.to_numpy(dtype=int)
    assert int(pred[70]) == 1
    assert result.frame_top_bus.iloc[70] == "Bus2"

