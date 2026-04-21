from __future__ import annotations

import json

from src.simulation.m9.final_polish import build_splits_v2


def test_split_balance_v2(tmp_path) -> None:
    scenarios = []
    for i in range(36):
        scenarios.append(
            {
                "scenario_id": f"SIM{i:04d}",
                "scenario_dir": f"data/scenarios/SIM{i:04d}",
                "template_name": f"T{i % 6}",
                "event_coarse": i % 9,
                "difficulty_level": ["easy", "medium", "hard", "adversarial"][i % 4],
                "scenario_family": f"FAM{i}",
            }
        )
    root = tmp_path / "data" / "scenarios"
    root.mkdir(parents=True, exist_ok=True)
    (root / "scenario_registry.json").write_text(json.dumps({"scenarios": scenarios}), encoding="utf-8")
    out = build_splits_v2(root, tmp_path, split_strategy="leakage_safe_balanced", seed=7)
    report = out["report"]
    assert report["no_scenario_leakage"] is True
    assert report["no_family_leakage"] is True
    assert report["label_coverage_score"] > 0.5
    assert (tmp_path / "split_balance_report_v2.json").exists()

