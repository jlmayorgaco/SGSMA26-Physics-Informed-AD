
# -----------------------------------------------------------------------------
# Output paths
# -----------------------------------------------------------------------------

from pathlib import Path

from src.config.config import AnalysisConfig
from src.utils.filesystem import ensure_dir
from src.utils.naming import event_folder_name


def build_root_dirs(config: AnalysisConfig) -> dict[str, Path]:
    roots = {
        "dataset": config.output_dataset_dir,
        "general": config.output_general_dir,
        "buses": config.output_buses_dir,
        "events": config.output_events_dir,
    }
    for root in roots.values():
        for child in ["csv", "plots", "reports", "json"]:
            ensure_dir(root / child)
    return roots


def build_bus_dirs(config: AnalysisConfig, bus_id: str) -> dict[str, Path]:
    root = config.output_buses_dir / bus_id
    dirs = {"root": root}
    for child in ["csv", "plots", "reports", "json"]:
        path = root / child
        ensure_dir(path)
        dirs[child] = path
    return dirs


def build_event_dirs(config: AnalysisConfig, event_index: int, label: str, bus_id: str | None = None) -> dict[str, Path]:
    event_root = config.output_events_dir / event_folder_name(event_index, label)
    if bus_id is None:
        root = event_root / "general"
    else:
        root = event_root / "buses" / bus_id

    dirs = {"root": root, "event_root": event_root}
    for child in ["csv", "plots", "reports", "json"]:
        path = root / child
        ensure_dir(path)
        dirs[child] = path
    return dirs