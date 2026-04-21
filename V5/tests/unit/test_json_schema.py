from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np

import src.application.use_cases.m7_state_estimator_benchmark as m7
from src.simulation.andes_ieee39_runner import AndesTruth


def test_benchmark_results_json_schema(monkeypatch) -> None:
    def fake_truth(tf: float = 1.0, tstep: float = 1 / 30, stride: int = 1):
        t = np.asarray([0.0, 0.1, 0.2])
        buses = [f"BUS{i}" for i in range(1, 40)]
        v = np.ones((len(t), len(buses)), dtype=complex)
        return AndesTruth(timestamps=t, bus_ids=buses, voltage_complex_pu=v)

    monkeypatch.setattr(m7, "run_andes_ieee39_truth", fake_truth)
    out = Path("output") / "_test_runs" / f"m7_schema_{uuid4().hex[:8]}"
    m7.run_m7_state_estimator_benchmark_use_case(
        raw_path=Path("data/metadata/IEEE_39_Bus_Power_System.raw"),
        pmu_location_path=Path("data/metadata/PMUbus_ Location.txt"),
        output_dir=out,
        use_andes_truth=True,
        stride=1,
    )
    payload = json.loads((out / "report" / "benchmark_results.json").read_text(encoding="utf-8"))
    for key in ["run_metadata", "data_sources", "network", "scenarios", "estimators", "ranking", "best_estimator", "overall_conclusion"]:
        assert key in payload

