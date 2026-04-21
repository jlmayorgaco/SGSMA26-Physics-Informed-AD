from __future__ import annotations

import json

from src.simulation.m9.final_polish import build_splits_v2


def test_split_coverage_constraints(tmp_path) -> None:
    scenarios = []
    for i in range(10):
        scenarios.append(
            {
                "scenario_id": f"SIM{i:04d}",
                "scenario_dir": f"data/scenarios/SIM{i:04d}",
                "template_name": f"SUB{i % 3}",
                "event_coarse": i % 5,
                "difficulty_level": ["easy", "medium", "hard"][i % 3],
                "scenario_family": f"F{i // 2}",
            }
        )
    root = tmp_path / "data" / "scenarios"
    root.mkdir(parents=True, exist_ok=True)
    (root / "scenario_registry.json").write_text(json.dumps({"scenarios": scenarios}), encoding="utf-8")
    out = build_splits_v2(root, tmp_path, split_strategy="family_grouped_balanced", seed=11)
    report = out["report"]
    assert "missing_in_val_test" in report
    assert "labels" in report["missing_in_val_test"]
    assert "difficulty_levels" in report["missing_in_val_test"]

