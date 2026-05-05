from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "report"
WORKBENCH_DIR = ROOT / "workbench"

DEFAULT_RAW_DIR = DATA_DIR / "RAW0001"
DEFAULT_TOPOLOGY_DIR = DATA_DIR / "topology" / "ieee39"
DEFAULT_FINAL_MODEL_DIR = MODEL_DIR


def resolve_path(path: Path | str) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return ROOT / value


def ensure_dir(path: Path | str) -> Path:
    value = resolve_path(path)
    value.mkdir(parents=True, exist_ok=True)
    return value
