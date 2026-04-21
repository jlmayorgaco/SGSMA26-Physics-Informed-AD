from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.detectors.shared.preprocessing.estimator_feature_adapter import EstimatorFeatureAdapter


def test_estimator_adapter_is_predictable_without_scoring_file(tmp_path: Path) -> None:
    frame = pd.DataFrame({"TIMESTAMP": [0.0, 0.1], "EVENT": [0, 1]})
    scenario_dir = tmp_path / "SIMA"
    (scenario_dir / "metadata").mkdir(parents=True, exist_ok=True)
    out = EstimatorFeatureAdapter(enabled=True).enrich_frame(frame, scenario_dir=scenario_dir)
    assert "ESTIMATOR_DIFFICULTY_SCORE" in out.columns
    assert float(out["ESTIMATOR_DIFFICULTY_SCORE"].iloc[0]) == 0.0


def test_estimator_adapter_can_be_disabled() -> None:
    frame = pd.DataFrame({"TIMESTAMP": [0.0], "EVENT": [0]})
    out = EstimatorFeatureAdapter(enabled=False).enrich_frame(frame, scenario_dir=Path("."))
    assert "ESTIMATOR_DIFFICULTY_SCORE" not in out.columns

