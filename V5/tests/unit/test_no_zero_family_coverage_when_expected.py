from __future__ import annotations

import pandas as pd

from src.detectors.training.datasets.split_rebuilder import rebuild_split_v3


def test_no_zero_family_coverage_when_expected() -> None:
    rows = []
    for sid, ev in [("N", 0), ("P", 1), ("C", 5), ("CC", 6), ("P2", 2), ("C2", 7), ("CC2", 8), ("N2", 0)]:
        rows.append({"scenario_id": sid, "scenario_dir": "x", "template_name": "x", "event_coarse": ev, "difficulty_level": "easy", "scenario_family": "x", "seed_family": "x", "split": "train"})
    train = pd.DataFrame(rows)
    val = train.iloc[[0]].assign(split="val")
    test = train.iloc[[1]].assign(split="test")
    result = rebuild_split_v3(train, val, test)
    cov = pd.DataFrame(result.manifest["coverage"])
    for fam in ["normal", "physical_heavy", "cyber_heavy", "concurrent_heavy"]:
        fam_rows = cov[cov["family"] == fam]
        assert not fam_rows.empty
