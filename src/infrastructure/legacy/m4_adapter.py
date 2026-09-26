"""Thin adapter around legacy `m4_andes_faults_type1.py`."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, Iterator
import sys


@contextmanager
def _temporary_attr(obj: Any, attr: str, value: Any) -> Iterator[None]:
    had_attr = hasattr(obj, attr)
    old_value = getattr(obj, attr, None)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        if had_attr:
            setattr(obj, attr, old_value)
        else:
            delattr(obj, attr)


def _legacy_m4():
    workspace_root = Path(__file__).resolve().parents[3]
    root_str = str(workspace_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return import_module("m4_andes_faults_type1")


def run_type1_fault_simulation(
    fault_bus: str = "39",
    output_root: str | Path = "output",
    event0_profile_path: str | Path = "output/M2_RAW0001_NOISE_PROFILE_NEWARCH/dataset_profiles_raw.json",
    event0_current_mapping_csv: str | Path = "output/ANDES_CALIBRATION_RAW_NEWARCH/current_mapping_selection.csv",
    event0_support_matrix_csv: str | Path = "output/ANDES_CALIBRATION_RAW_NEWARCH/signal_support_matrix.csv",
    event0_calibration_json: str | Path = "output/ANDES_CALIBRATION_RAW_NEWARCH/metrics/calibration_results.json",
    generate_plots: bool = True,
) -> None:
    """Run legacy m4 pipeline with overrideable inputs/outputs."""
    m4 = _legacy_m4()
    with ExitStack() as stack:
        stack.enter_context(_temporary_attr(m4, "FAULT_BUS", str(fault_bus)))
        stack.enter_context(_temporary_attr(m4, "BASE_OUTPUT_DIR", str(Path(output_root))))
        stack.enter_context(_temporary_attr(m4, "EVENT0_PROFILE_PATH", str(Path(event0_profile_path))))
        stack.enter_context(_temporary_attr(m4, "EVENT0_CURRENT_MAPPING_CSV", str(Path(event0_current_mapping_csv))))
        stack.enter_context(_temporary_attr(m4, "EVENT0_SUPPORT_MATRIX_CSV", str(Path(event0_support_matrix_csv))))
        stack.enter_context(_temporary_attr(m4, "EVENT0_CALIBRATION_JSON", str(Path(event0_calibration_json))))
        stack.enter_context(_temporary_attr(m4, "GENERATE_PLOTS", bool(generate_plots)))
        m4.main()


def load_event0_artifacts():
    """Proxy to legacy event-0 artifact loader."""
    return _legacy_m4().load_event0_artifacts()


def build_estimated_bus_dataframes(simulation_dfs: dict, meta: dict, fault_bus: str):
    """Proxy to legacy estimated dataframe constructor."""
    return _legacy_m4().build_estimated_bus_dataframes(simulation_dfs, meta, str(fault_bus))


def compute_estimation_metrics(simulation_dfs: dict, estimated_dfs: dict):
    """Backwards-compatible proxy for legacy metric computation."""
    return _legacy_m4().compute_estimation_metrics(simulation_dfs, estimated_dfs)


def export_estimation_report(fault_bus: str, simulation_dfs: dict, estimated_dfs: dict) -> None:
    """Backwards-compatible proxy for legacy estimation report writer."""
    _legacy_m4().export_estimation_report(str(fault_bus), simulation_dfs, estimated_dfs)


# Backwards-compatibility alias used by older tests.
def run_type1_simulation(fault_bus: str | None = None) -> None:
    run_type1_fault_simulation(fault_bus=str(fault_bus or "39"))
