from __future__ import annotations

import json

from tests.helpers.m9_test_utils import generate_single


def test_raw_vs_sim_statistics(tmp_path) -> None:
    _, scenario_dir = generate_single(tmp_path, "TEMPLATE_OFFICIAL_STYLE_MULTI_EVENT", scenario_id="SIM0001")
    path = scenario_dir / "metadata" / "raw_vs_sim_comparison.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "per_bus" in payload
    assert "distance_summary" in payload
    assert "pass_realism" in payload
