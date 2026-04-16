

from typing import Any

from analysis.integrity import align_integrity
from config.config import AnalysisConfig
from config.constants import MEASUREMENT_COLUMNS, OPTIONAL_COLUMNS
from config.models import BusData


def build_dataset_integrity(buses: list[BusData], config: AnalysisConfig) -> dict[str, Any]:
    return {
        "input_dir": str(config.input_dir),
        "file_count": int(len(buses)),
        "bus_ids": [bus.bus_id for bus in buses],
        "alignment": align_integrity(buses),
        "expected_columns": MEASUREMENT_COLUMNS + OPTIONAL_COLUMNS,
    }