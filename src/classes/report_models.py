from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class BinaryMetrics:
    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    accuracy: float


@dataclass(slots=True)
class Event7BusThresholds:
    outlier_rate_q95: float
    jump_rate_q95: float
    stuck_flag_q95: float
    composite_q995: float


@dataclass(slots=True)
class RawCyberQualityReport:
    input_raw_dir: str
    output_dir: str
    buses: list[str]
    n_frames: int
    event5_metrics: BinaryMetrics
    event7_metrics: BinaryMetrics
    cyber_metrics: BinaryMetrics
    event7_bus_localization_accuracy: float
    counts: dict[str, int]
    event7_thresholds_by_bus: dict[str, Event7BusThresholds]
    artifact_paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["event5_metrics"] = asdict(self.event5_metrics)
        payload["event7_metrics"] = asdict(self.event7_metrics)
        payload["cyber_metrics"] = asdict(self.cyber_metrics)
        payload["event7_thresholds_by_bus"] = {
            bus: asdict(thresholds) for bus, thresholds in self.event7_thresholds_by_bus.items()
        }
        return payload

