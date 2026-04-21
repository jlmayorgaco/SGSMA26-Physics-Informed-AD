from __future__ import annotations

import json

from tests.helpers.m9_test_utils import generate_templates


def test_generate_event_templates(tmp_path) -> None:
    manifests, scenarios_root = generate_templates(tmp_path)
    assert len(manifests) == 10
    for name in [
        "SIM0001",
        "SIM0002",
        "SIM0003",
        "SIM0004",
        "SIM0005",
        "SIM0006",
        "SIM0007",
        "SIM0008",
        "SIM0009",
        "SIM0010",
    ]:
        assert (scenarios_root / name / "scenario_manifest.json").exists()
    registry = json.loads((scenarios_root / "scenario_registry.json").read_text(encoding="utf-8"))
    assert len(registry["scenarios"]) == 10
