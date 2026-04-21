from __future__ import annotations

import pandas as pd

from src.detectors.training.datasets.split_rebuilder import rebuild_split_v3


def test_family_coverage_rebuild() -> None:
    frame = pd.DataFrame(
        [
            {"scenario_id": "N1", "event_coarse": 0, "difficulty_level": "easy"},
            {"scenario_id": "P1", "event_coarse": 2, "difficulty_level": "easy"},
            {"scenario_id": "C1", "event_coarse": 5, "difficulty_level": "medium"},
            {"scenario_id": "CC1", "event_coarse": 6, "difficulty_level": "hard"},
            {"scenario_id": "P2", "event_coarse": 3, "difficulty_level": "hard"},
        ]
    )
    for col in ["scenario_dir", "template_name", "scenario_family", "seed_family"]:
        frame[col] = "x"
    train = frame.copy(); train["split"] = "train"
    val = train.iloc[[0]].copy(); val["split"] = "val"
    test = train.iloc[[1]].copy(); test["split"] = "test"

    result = rebuild_split_v3(train, val, test)
    coverage = pd.DataFrame(result.manifest["coverage"])
    assert not coverage.empty
    assert set(coverage["split"].unique()) <= {"train", "val", "test"}
