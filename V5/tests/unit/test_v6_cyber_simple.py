from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.v6.cyber_simple import CyberV6Detector


def _build_bus_frame(bus: str, values: np.ndarray, *, data_present: np.ndarray | None = None) -> pd.DataFrame:
    n = len(values)
    if data_present is None:
        data_present = np.ones((n,), dtype=float)
    upper = bus.upper()
    return pd.DataFrame(
        {
            "TIMESTAMP": np.arange(n, dtype=float) * 0.033,
            f"{upper}_VA_MAG": values,
            f"{upper}_IA_MAG": values * 0.12 + 0.2,
            f"{upper}_FREQ": 60.0 + 0.01 * np.sin(np.arange(n, dtype=float) * 0.07),
            "DATA_PRESENT": data_present,
            "Event": np.zeros((n,), dtype=int),
        }
    )


def _build_normal_chunk(seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    t = np.arange(150, dtype=float)
    buses = ["Bus2", "Bus10", "Bus39"]
    chunk: dict[str, pd.DataFrame] = {}
    for i, bus in enumerate(buses):
        signal = 1.0 + 0.4 * np.sin(0.06 * t + i) + 0.02 * rng.normal(size=t.shape[0])
        chunk[bus] = _build_bus_frame(bus, signal)
    return chunk


def test_v6_detect_event5_from_nan_and_data_present() -> None:
    detector = CyberV6Detector()
    chunk = _build_normal_chunk(seed=1)
    missing = chunk["Bus39"].copy()
    measurement_cols = [col for col in missing.columns if col not in {"TIMESTAMP", "DATA_PRESENT", "Event"}]
    missing.loc[:, measurement_cols] = np.nan
    missing.loc[:, "DATA_PRESENT"] = 0.0
    chunk["Bus39"] = missing

    decision = detector.detect_event5(chunk)

    assert decision.predicted
    assert decision.chunk_score >= detector.config.event5_missing_threshold
    assert "Bus39" in decision.missing_buses


def test_v6_detect_event7_from_value_corruption() -> None:
    detector = CyberV6Detector()
    normal_chunks = [_build_normal_chunk(seed=7), _build_normal_chunk(seed=17), _build_normal_chunk(seed=27)]
    detector.fit_event7_thresholds(normal_chunks)

    corrupted = _build_normal_chunk(seed=5)
    bus2 = corrupted["Bus2"].copy()
    col = "BUS2_VA_MAG"
    values = pd.to_numeric(bus2[col], errors="coerce").to_numpy(dtype=float)
    values[30:35] += 3.0
    values[58:78] = values[57]
    values[95:110] = values[70:85]
    bus2[col] = values
    corrupted["Bus2"] = bus2

    decision = detector.classify_chunk(corrupted)

    assert decision.predicted_label == 7
    assert decision.event7.predicted
    assert decision.event7.top_bus == "Bus2"


def test_v6_event5_gate_has_priority_over_event7() -> None:
    detector = CyberV6Detector()
    detector.fit_event7_thresholds([_build_normal_chunk(seed=11), _build_normal_chunk(seed=21)])
    chunk = _build_normal_chunk(seed=31)

    bus2 = chunk["Bus2"].copy()
    values = pd.to_numeric(bus2["BUS2_VA_MAG"], errors="coerce").to_numpy(dtype=float)
    values[15:30] += 4.0
    bus2["BUS2_VA_MAG"] = values
    measurement_cols = [col for col in bus2.columns if col not in {"TIMESTAMP", "DATA_PRESENT", "Event"}]
    bus2.loc[:, measurement_cols] = np.nan
    bus2.loc[:, "DATA_PRESENT"] = 0.0
    chunk["Bus2"] = bus2

    decision = detector.classify_chunk(chunk)

    assert decision.predicted_label == 5
    assert decision.event5.predicted
    assert not decision.event7.predicted

