from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class DetectionOutput:
    p_abnormal: np.ndarray
    p_cyber: np.ndarray
    p_physical: np.ndarray
    y_pred_frame: np.ndarray
    y_pred_stable: np.ndarray
    is_abnormal: np.ndarray | None = None
    confidence: np.ndarray | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.p_abnormal = np.asarray(self.p_abnormal, dtype=float)
        self.p_cyber = np.asarray(self.p_cyber, dtype=float)
        self.p_physical = np.asarray(self.p_physical, dtype=float)
        self.y_pred_frame = np.asarray(self.y_pred_frame, dtype=int)
        self.y_pred_stable = np.asarray(self.y_pred_stable, dtype=int)
        n = len(self.p_abnormal)
        for name, values in (
            ("p_cyber", self.p_cyber),
            ("p_physical", self.p_physical),
            ("y_pred_frame", self.y_pred_frame),
            ("y_pred_stable", self.y_pred_stable),
        ):
            if len(values) != n:
                raise ValueError(f"{name} length must match p_abnormal length")
        if self.is_abnormal is None:
            self.is_abnormal = self.y_pred_stable.astype(int)
        else:
            self.is_abnormal = np.asarray(self.is_abnormal, dtype=int)
        if self.confidence is None:
            self.confidence = np.clip(np.abs(self.p_abnormal - 0.5) * 2.0, 0.0, 1.0)
        else:
            self.confidence = np.asarray(self.confidence, dtype=float)
