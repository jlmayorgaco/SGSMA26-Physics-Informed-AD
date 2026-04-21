from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.detectors.domain.models.detection_input import DetectionInput
from src.detectors.physical.heuristics.current_surge_rules import current_surge_evidence
from src.detectors.physical.heuristics.estimator_innovation_rules import estimator_innovation_evidence
from src.detectors.physical.heuristics.freq_rocof_rules import freq_rocof_evidence
from src.detectors.physical.heuristics.voltage_sag_rules import voltage_sag_evidence


@dataclass(slots=True)
class PhysicalFeatureBatch:
    temporal_x: np.ndarray
    aux_x: np.ndarray
    aux_feature_names: list[str]
    heuristic_evidence: dict[str, np.ndarray]
    heuristic_score: np.ndarray


class PhysicalFeatureExtractor:
    def extract(self, inputs: DetectionInput) -> PhysicalFeatureBatch:
        windows = inputs.x_windows
        sag = voltage_sag_evidence(windows, inputs.feature_names)
        surge = current_surge_evidence(windows, inputs.feature_names)
        freq = freq_rocof_evidence(windows, inputs.feature_names)
        innovation = estimator_innovation_evidence(inputs.metadata)
        aux_x = np.column_stack(
            [
                sag["sag_ratio"],
                sag["sag_depth"],
                surge["surge_ratio"],
                surge["surge_strength"],
                freq["freq_dev"],
                freq["rocof_dev"],
                innovation["innovation_score"],
            ]
        )
        heuristic_score = np.clip(
            0.30 * sag["score"] + 0.25 * surge["score"] + 0.30 * freq["score"] + 0.15 * innovation["score"],
            0.0,
            1.0,
        )
        return PhysicalFeatureBatch(
            temporal_x=windows.astype(float),
            aux_x=aux_x.astype(float),
            aux_feature_names=[
                "sag_ratio",
                "sag_depth",
                "surge_ratio",
                "surge_strength",
                "freq_dev",
                "rocof_dev",
                "innovation_score",
            ],
            heuristic_evidence={
                **sag,
                **surge,
                **freq,
                **innovation,
            },
            heuristic_score=heuristic_score,
        )

