from __future__ import annotations

import numpy as np
import pandas as pd

import src.estimation.temporal_regularized_estimator as tre


def test_build_estimated_voltage_matrix_from_pmuses_returns_expected_shape(monkeypatch) -> None:
    class _Stub:
        @staticmethod
        def build_estimated_voltage_matrix_from_pmuses(simulation_dfs, meta):
            _ = simulation_dfs
            return np.ones((len(meta["t"]), len(meta["bus_ids_common"])), dtype=complex)

    monkeypatch.setattr(tre, "_legacy_m4", lambda: _Stub())
    out = tre.build_estimated_voltage_matrix_from_pmuses({}, {"t": np.arange(5), "bus_ids_common": ["1", "2"]})
    assert out.shape == (5, 2)


def test_build_estimated_bus_dataframes_preserves_pmu_passthrough(monkeypatch) -> None:
    df = pd.DataFrame({"TIMESTAMP": [0.0], "DATA_PRESENT": [1], "Event": [0], "BUS39_VA_MAG": [1.0]})

    class _Stub:
        @staticmethod
        def build_estimated_bus_dataframes(simulation_dfs, meta, fault_bus):
            _ = (meta, fault_bus)
            return {"39": simulation_dfs["39"], "1": df.copy()}

    monkeypatch.setattr(tre, "_legacy_m4", lambda: _Stub())
    out = tre.build_estimated_bus_dataframes({"39": df.copy()}, {"t": np.array([0.0]), "bus_ids_common": ["39", "1"]}, "39")
    assert "39" in out and "1" in out
    assert out["39"]["BUS39_VA_MAG"].iloc[0] == 1.0


def test_estimated_dataframes_include_data_present_and_event(monkeypatch) -> None:
    df = pd.DataFrame({"TIMESTAMP": [0.0], "DATA_PRESENT": [1], "Event": [0]})

    class _Stub:
        @staticmethod
        def build_estimated_bus_dataframes(simulation_dfs, meta, fault_bus):
            _ = (simulation_dfs, meta, fault_bus)
            return {"1": df.copy()}

    monkeypatch.setattr(tre, "_legacy_m4", lambda: _Stub())
    out = tre.build_estimated_bus_dataframes({}, {"t": np.array([0.0]), "bus_ids_common": ["1"]}, "39")
    assert "DATA_PRESENT" in out["1"].columns
    assert "Event" in out["1"].columns
