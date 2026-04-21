from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class DetectorMetricsV2:
    f1_abnormal: float
    precision_abnormal: float
    recall_abnormal: float
    balanced_accuracy: float
    false_positives_per_minute: float
    detection_delay_s: float | None
    roc_auc: float | None
    pr_auc: float | None
    brier_score: float | None
    ece: float | None
    support_abnormal: int
    support_normal: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

