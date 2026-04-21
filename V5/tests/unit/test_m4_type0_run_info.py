from __future__ import annotations

from pathlib import Path

from src.application.use_cases.simulate_normal_event import build_type0_run_info
from src.simulation.type0_config import Type0Config


def test_run_info_contains_type0_fields() -> None:
    cfg = Type0Config()
    run_info = build_type0_run_info(
        cfg=cfg,
        run_dir=Path("output/SIM0001_NORMAL_Type0_NEWARCH"),
        profile_path=Path("a.json"),
        mapping_csv=Path("b.csv"),
        support_csv=Path("c.csv"),
        calibration_json=Path("d.json"),
    )
    assert run_info["scenario_type"] == "type0"
    assert "sim_tf" in run_info and "sim_tstep" in run_info
    assert "event_label_mode" in run_info
    assert run_info["fault_bus"] is None
