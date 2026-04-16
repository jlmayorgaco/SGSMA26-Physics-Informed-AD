

from typing import Any

from src.analysis.integrity import align_integrity
from src.config.config import AnalysisConfig
from src.config.constants import MEASUREMENT_COLUMNS, OPTIONAL_COLUMNS
from src.config.models import BusData


def build_dataset_integrity(buses: list[BusData], config: AnalysisConfig) -> dict[str, Any]:
    return {
        "input_dir": str(config.input_dir),
        "file_count": int(len(buses)),
        "bus_ids": [bus.bus_id for bus in buses],
        "alignment": align_integrity(buses),
        "expected_columns": MEASUREMENT_COLUMNS + OPTIONAL_COLUMNS,
    }