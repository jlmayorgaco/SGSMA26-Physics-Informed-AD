from __future__ import annotations

import numpy as np
import pandas as pd

from src.estimators.estimator_event_5 import Event5Estimator


def _build_frame(prefix: str, values: np.ndarray, data_present: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"{prefix}_VA_MAG": values,
            f"{prefix}_IA_MAG": values * 0.1 + 0.5,
            "DATA_PRESENT": data_present,
            "Event": event,
        }
    )


def test_event5_estimator_detects_missing_from_nan_and_data_present() -> None:
    timeline = pd.Index(np.arange(6, dtype=float), name="TIMESTAMP")
    event = np.zeros((6,), dtype=int)
    frame_bus2 = _build_frame(
        prefix="BUS2",
        values=np.array([1.0, 1.1, np.nan, np.nan, 1.3, 1.4]),
        data_present=np.array([1, 1, 0, 0, 1, 1], dtype=float),
        event=event,
    )
    frame_bus39 = _build_frame(
        prefix="BUS39",
        values=np.array([1.0, 1.1, 1.2, 1.25, 1.3, 1.35]),
        data_present=np.ones((6,), dtype=float),
        event=event,
    )
    frame_bus2.index = timeline
    frame_bus39.index = timeline

    estimator = Event5Estimator()
    result = estimator.estimate({"Bus2": frame_bus2, "Bus39": frame_bus39})

    pred = result.frame_prediction.to_numpy(dtype=int)
    assert pred.tolist() == [0, 0, 1, 1, 0, 0]

