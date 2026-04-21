"""Top-level m3 event-0 calibration orchestration helpers."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from importlib import import_module
import json
import logging
from pathlib import Path
from typing import Any, Iterator
import sys

from src.calibration.andes_runtime import run_andes_normal
from src.calibration.andes_trajectories import build_all_trajectories
from src.calibration.raw_chunk_loader import discover_event0_chunks, load_event0_chunks_for_bus, normalize_bus_id


LOGGER = logging.getLogger(__name__)
PMU_BUSES = ["39", "29", "10", "22", "19", "2", "5", "6"]
ACTIVE_STATUSES = {"supported", "fallback_profile"}
ZERO_EVALUABLE_MSG = (
    "m3 found zero evaluable PMU buses. Check raw_chunks_dir, event-0 discovery, "
    "bus-id key typing, and trajectory loading."
)


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


def _legacy_m3():
    workspace_root = Path(__file__).resolve().parents[2]
    root_str = str(workspace_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return import_module("m3_andes_calibration_raw")


def sim_source_for_spec(spec, traj, current_mapping):
    """Legacy-compatible signal-source resolver."""
    return _legacy_m3().sim_source_for_spec(spec, traj, current_mapping)


def prepare_calibrated_series(bus_id, spec, real_aggregate, sim_source, sim_t, noise_layer, rng, chunks):
    """Legacy-compatible calibrated-series preparation helper."""
    return _legacy_m3().prepare_calibrated_series(bus_id, spec, real_aggregate, sim_source, sim_t, noise_layer, rng, chunks)


def evaluate_supported_signal(bus_id, spec, chunks, traj, current_mapping, noise_layer, rng):
    """Legacy-compatible per-signal evaluation helper."""
    return _legacy_m3().evaluate_supported_signal(bus_id, spec, chunks, traj, current_mapping, noise_layer, rng)


def _record_evaluation_guard(output_dir: Path) -> None:
    results_path = output_dir / "metrics" / "calibration_results.json"
    if not results_path.exists():
        raise RuntimeError(f"Missing calibration results artifact: {results_path}")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    status_counts: dict[str, int] = {}
    for rec in records:
        status = str(rec.get("status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
    active_count = sum(status_counts.get(s, 0) for s in ACTIVE_STATUSES)
    if records and active_count == 0 and status_counts.get("unsupported", 0) == len(records):
        raise RuntimeError(f"{ZERO_EVALUABLE_MSG} status_counts={status_counts}")


def _preflight_inputs(raw_chunks_dir: Path) -> tuple[list[Path], dict[str, list[dict]], dict[str, dict], Any]:
    event0_chunk_dirs = discover_event0_chunks(raw_chunks_dir)
    if not event0_chunk_dirs:
        raise RuntimeError(f"No event-0 chunk directories found under raw_chunks_dir='{raw_chunks_dir}'.")

    all_chunks: dict[str, list[dict]] = {}
    for bus_id in PMU_BUSES:
        bus = normalize_bus_id(bus_id)
        all_chunks[bus] = load_event0_chunks_for_bus(bus, raw_chunks_dir)

    if not any(len(chunks) > 0 for chunks in all_chunks.values()):
        raise RuntimeError(f"No Bus*.csv event-0 chunks loaded for PMU buses in raw_chunks_dir='{raw_chunks_dir}'.")

    system = run_andes_normal()
    trajectories = {normalize_bus_id(k): v for k, v in build_all_trajectories(system, PMU_BUSES).items()}
    if not trajectories:
        raise RuntimeError("No ANDES trajectories were built for PMU buses.")

    traj_keys = sorted(trajectories.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    chunk_keys = sorted(all_chunks.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    LOGGER.info("raw_chunks_dir=%s", raw_chunks_dir)
    LOGGER.info("event0_chunk_dir_count=%s", len(event0_chunk_dirs))
    LOGGER.info("PMU_BUSES=%s", PMU_BUSES)
    LOGGER.info("trajectory_keys=%s", traj_keys)
    LOGGER.info("trajectory_key_types=%s", sorted({type(k).__name__ for k in trajectories.keys()}))
    LOGGER.info("all_chunks_keys=%s", chunk_keys)
    LOGGER.info("all_chunks_key_types=%s", sorted({type(k).__name__ for k in all_chunks.keys()}))

    evaluable_bus_count = 0
    for bus_id in PMU_BUSES:
        bus = normalize_bus_id(bus_id)
        traj_found = bus in trajectories
        chunk_count = len(all_chunks.get(bus, []))
        if traj_found and chunk_count > 0:
            evaluable_bus_count += 1
        LOGGER.info("bus_id=%s traj_found=%s chunk_count=%s", bus, traj_found, chunk_count)

    if evaluable_bus_count == 0:
        raise RuntimeError(ZERO_EVALUABLE_MSG)

    return event0_chunk_dirs, all_chunks, trajectories, system


def run_event0_calibration_core(
    raw_chunks_dir: str | Path,
    raw_profile_file: str | Path,
    output_dir: str | Path,
    event_label: int = 0,
    drift_mode: str = "fft_residual_synthesis",
) -> None:
    """Run full event-0 calibration with legacy-equivalent behavior and configurable paths."""
    raw_chunks_path = Path(raw_chunks_dir).resolve()
    raw_profile_path = Path(raw_profile_file).resolve()
    output_path = Path(output_dir).resolve()
    _, _, _, preflight_system = _preflight_inputs(raw_chunks_path)

    m3 = _legacy_m3()
    with ExitStack() as stack:
        stack.enter_context(_temporary_attr(m3, "RAW_CHUNKS_DIR", str(raw_chunks_path)))
        stack.enter_context(_temporary_attr(m3, "RAW_PROFILE_FILE", str(raw_profile_path)))
        stack.enter_context(_temporary_attr(m3, "OUTPUT_DIR", str(output_path)))
        stack.enter_context(_temporary_attr(m3, "EVENT_LABEL", int(event_label)))
        stack.enter_context(_temporary_attr(m3, "DRIFT_MODE", str(drift_mode)))
        stack.enter_context(_temporary_attr(m3, "run_andes_normal", lambda: preflight_system))
        m3.run_raw_event0_calibration()
    _record_evaluation_guard(output_path)
