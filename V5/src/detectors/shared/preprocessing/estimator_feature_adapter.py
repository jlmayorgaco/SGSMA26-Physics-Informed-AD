from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd


DEFAULT_ESTIMATOR_FIELDS = {
    "ESTIMATOR_DIFFICULTY_SCORE": 0.0,
    "DETECTOR_DIFFICULTY_SCORE": 0.0,
    "CLASSIFIER_DIFFICULTY_SCORE": 0.0,
    "LOCALIZER_DIFFICULTY_SCORE": 0.0,
    "OVERALL_TRAINING_VALUE_SCORE": 0.0,
}


@dataclass(slots=True)
class EstimatorFeatureAdapter:
    enabled: bool = True

    def enrich_frame(self, frame: pd.DataFrame, scenario_dir: Path | None = None) -> pd.DataFrame:
        out = frame.copy()
        if not self.enabled:
            return out
        values = dict(DEFAULT_ESTIMATOR_FIELDS)
        if scenario_dir is not None:
            scoring = scenario_dir / "metadata" / "scenario_scoring.json"
            if scoring.exists():
                try:
                    payload = json.loads(scoring.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    payload = {}
                values["ESTIMATOR_DIFFICULTY_SCORE"] = float(payload.get("estimator_difficulty_score", 0.0) or 0.0)
                values["DETECTOR_DIFFICULTY_SCORE"] = float(payload.get("detector_difficulty_score", 0.0) or 0.0)
                values["CLASSIFIER_DIFFICULTY_SCORE"] = float(payload.get("classifier_difficulty_score", 0.0) or 0.0)
                values["LOCALIZER_DIFFICULTY_SCORE"] = float(payload.get("localizer_difficulty_score", 0.0) or 0.0)
                values["OVERALL_TRAINING_VALUE_SCORE"] = float(payload.get("overall_training_value_score", 0.0) or 0.0)
        for key, value in values.items():
            out[key] = float(value)
        return out

