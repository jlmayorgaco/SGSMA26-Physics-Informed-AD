from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.detectors.cyber.event5_state_model import Event5StateModel
from src.detectors.domain.models.detection_input import DetectionInput


@dataclass(slots=True)
class Event5DetectorOutput:
    p_event5: np.ndarray
    states: list[str]
    evidence: list[dict[str, float]]


@dataclass(slots=True)
class Event5Detector:
    model: Event5StateModel = field(default_factory=Event5StateModel)
    partial_dropout_threshold: float = 0.20

    def _data_present_series(self, inputs: DetectionInput) -> np.ndarray:
        if "DATA_PRESENT" not in inputs.feature_names:
            return np.ones((inputs.x_windows.shape[0],), dtype=float)
        idx = inputs.feature_names.index("DATA_PRESENT")
        return np.clip(inputs.x_windows[:, :, idx].mean(axis=1), 0.0, 1.0)

    def _nan_series(self, inputs: DetectionInput) -> np.ndarray:
        nan_cols = [i for i, name in enumerate(inputs.feature_names) if name.endswith("__is_nan")]
        if not nan_cols:
            return np.zeros((inputs.x_windows.shape[0],), dtype=float)
        return np.clip(inputs.x_windows[:, :, nan_cols].mean(axis=(1, 2)), 0.0, 1.0)

    def detect(self, inputs: DetectionInput) -> Event5DetectorOutput:
        data_present = self._data_present_series(inputs)
        nan_ratio = self._nan_series(inputs)
        full_ratio = np.clip(np.maximum(1.0 - data_present, nan_ratio), 0.0, 1.0)
        partial_ratio = np.where(
            (full_ratio >= self.partial_dropout_threshold) & (full_ratio < self.model.full_start_threshold),
            full_ratio,
            0.0,
        )
        states, probs = self.model.run(partial_ratio=partial_ratio.tolist(), full_ratio=full_ratio.tolist())
        evidence = [
            {
                "data_present_mean": float(data_present[i]),
                "nan_ratio": float(nan_ratio[i]),
                "partial_dropout_ratio": float(partial_ratio[i]),
                "full_dropout_ratio": float(full_ratio[i]),
            }
            for i in range(len(probs))
        ]
        return Event5DetectorOutput(p_event5=np.asarray(probs, dtype=float), states=states, evidence=evidence)
