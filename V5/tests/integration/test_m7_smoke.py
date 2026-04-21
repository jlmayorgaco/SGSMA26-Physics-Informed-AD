from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np

import src.application.use_cases.m7_state_estimator_benchmark as m7
from src.simulation.andes_ieee39_runner import AndesTruth


def test_m7_smoke(monkeypatch) -> None:
    def fake_truth(tf: float = 1.0, tstep: float = 1 / 30, stride: int = 1):
        t = np.asarray([0.0, 0.1, 0.2, 0.3])
        buses = [f"BUS{i}" for i in range(1, 40)]
        v = np.ones((len(t), len(buses)), dtype=complex)
        return AndesTruth(timestamps=t, bus_ids=buses, voltage_complex_pu=v)

    monkeypatch.setattr(m7, "run_andes_ieee39_truth", fake_truth)
    out = Path("output") / "_test_runs" / f"m7_smoke_{uuid4().hex[:8]}"
    summary = m7.run_m7_state_estimator_benchmark_use_case(
        raw_path=Path("data/metadata/IEEE_39_Bus_Power_System.raw"),
        pmu_location_path=Path("data/metadata/PMUbus_ Location.txt"),
        output_dir=out,
        use_andes_truth=True,
    )
    assert Path(summary["output_dir"]).exists()
    assert (out / "report" / "benchmark_results.json").exists()
    assert (out / "metrics" / "global_metrics.csv").exists()

