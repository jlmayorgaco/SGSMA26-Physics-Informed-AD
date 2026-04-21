from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np

import src.application.use_cases.m8_hybrid_dynamic_benchmark as m8
from src.simulation.andes_ieee39_runner import AndesTruth


def test_m8_includes_m7_reference(monkeypatch, tmp_path: Path) -> None:
    def fake_truth(tf: float = 2.0, tstep: float = 1 / 30, stride: int = 1):
        t = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4])
        buses = [f"BUS{i}" for i in range(1, 40)]
        v = np.ones((len(t), len(buses)), dtype=complex)
        return AndesTruth(timestamps=t, bus_ids=buses, voltage_complex_pu=v)

    monkeypatch.setattr(m8, "run_andes_ieee39_truth", fake_truth)
    m7_json = tmp_path / "m7_results.json"
    m7_json.write_text(json.dumps({"best_estimator": {"name": "PMU_VOLTAGE_CURRENT_SMOOTHED"}}), encoding="utf-8")
    out = Path("output") / "_test_runs" / f"m8_ref_{uuid4().hex[:8]}"
    m8.run_m8_hybrid_dynamic_benchmark_use_case(
        raw_path=Path("data/metadata/IEEE_39_Bus_Power_System.raw"),
        pmu_location_path=Path("data/metadata/PMUbus_ Location.txt"),
        output_dir=out,
        use_andes_truth=True,
        m7_results_path=m7_json,
    )
    payload = json.loads((out / "report" / "m8_benchmark_results.json").read_text(encoding="utf-8"))
    assert "M8_M7_BEST_REFERENCE" in payload["estimators"]

