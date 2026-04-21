"""Infrastructure helpers for PMU location metadata loading."""

from __future__ import annotations

from pathlib import Path

from src.metadata.pmu_location_parser import parse_pmu_location_file


def load_pmu_metadata(path: str | Path) -> dict:
    """Load canonical PMU metadata payload."""
    return parse_pmu_location_file(Path(path))

