from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.detectors.domain.models.detection_input import DetectionInput


@dataclass(slots=True)
class PhysicalChangeDetectorOutput:
    p_physical: np.ndarray
    evidence: list[dict[str, float]]


@dataclass(slots=True)
class PhysicalChangeDetector:
    freq_weight: float = 0.35
    rocof_weight: float = 0.35
    magnitude_weight: float = 0.30

    def detect(self, inputs: DetectionInput, *, aux_physical_prob: np.ndarray | None = None) -> PhysicalChangeDetectorOutput:
        x = np.asarray(inputs.x_windows, dtype=float)
        if x.size == 0:
            return PhysicalChangeDetectorOutput(p_physical=np.zeros((0,), dtype=float), evidence=[])
        names = list(inputs.feature_names)
        freq_cols = [i for i, n in enumerate(names) if "FREQ" in n.upper()]
        rocof_cols = [i for i, n in enumerate(names) if "ROCOF" in n.upper()]
        mag_cols = [i for i, n in enumerate(names) if n.upper().endswith("_MAG")]
        if not mag_cols:
            mag_cols = [i for i, n in enumerate(names) if "__IS_NAN" not in n.upper() and "DATA_PRESENT" not in n.upper()]

        def _disp(cols: list[int]) -> np.ndarray:
            if not cols:
                return np.zeros((x.shape[0],), dtype=float)
            xx = x[:, :, cols]
            return np.clip(np.abs(xx[:, -1, :] - xx[:, 0, :]).mean(axis=1), 0.0, 1.0)

        freq_disp = _disp(freq_cols)
        rocof_disp = _disp(rocof_cols)
        mag_disp = _disp(mag_cols)
        base_score = np.clip(
            self.freq_weight * freq_disp + self.rocof_weight * rocof_disp + self.magnitude_weight * mag_disp,
            0.0,
            1.0,
        )
        if aux_physical_prob is not None and len(aux_physical_prob) == len(base_score):
            p_physical = np.clip(0.60 * base_score + 0.40 * np.asarray(aux_physical_prob, dtype=float), 0.0, 1.0)
        else:
            p_physical = base_score
        evidence = [
            {
                "freq_change": float(freq_disp[i]),
                "rocof_change": float(rocof_disp[i]),
                "magnitude_change": float(mag_disp[i]),
            }
            for i in range(len(p_physical))
        ]
        return PhysicalChangeDetectorOutput(p_physical=p_physical, evidence=evidence)

