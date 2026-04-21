from __future__ import annotations

import argparse
from pathlib import Path

from src.simulation.m9.hardening import run_m9_1_hardening


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M9.1 hardening pipeline")
    p.add_argument("--raw-path", type=Path, default=Path("data/metadata/IEEE_39_Bus_Power_System.raw"))
    p.add_argument("--pmu-location-path", type=Path, default=Path("data/metadata/PMUbus_ Location.txt"))
    p.add_argument("--reference-pmu-dir", type=Path, default=Path("data/RAW0001"))
    p.add_argument("--output-root", type=Path, default=Path("."))
    p.add_argument("--scenario-template", type=str, default="TEMPLATE_OFFICIAL_STYLE_MULTI_EVENT")
    p.add_argument("--batch-mode", type=str, default="balanced_core")
    p.add_argument("--n-scenarios", type=int, default=24)
    p.add_argument("--difficulty-mode", type=str, default="uniform")
    p.add_argument("--run-estimator-scoring", action="store_true", default=True)
    p.add_argument("--build-splits", action="store_true", default=True)
    p.add_argument("--save-json", action="store_true", default=True)
    p.add_argument("--save-csv", action="store_true", default=True)
    p.add_argument("--save-plots", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    report = run_m9_1_hardening(
        raw_path=a.raw_path,
        pmu_location_path=a.pmu_location_path,
        reference_pmu_dir=a.reference_pmu_dir,
        output_root=a.output_root,
        scenario_template=a.scenario_template,
        batch_mode=a.batch_mode,
        n_scenarios=a.n_scenarios,
        difficulty_mode=a.difficulty_mode,
        run_estimator_scoring_flag=a.run_estimator_scoring,
        build_splits_flag=a.build_splits,
        seed=a.seed,
        save_plots=a.save_plots,
    )
    return 0 if report.get("overall_verdict", {}).get("m9_1_hardened", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())

