from __future__ import annotations

import argparse
from pathlib import Path

from src.simulation.m9.final_polish import run_m9_2_final_polish


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M9.2 final polish pipeline")
    p.add_argument("--raw-path", type=Path, default=Path("data/metadata/IEEE_39_Bus_Power_System.raw"))
    p.add_argument("--pmu-location-path", type=Path, default=Path("data/metadata/PMUbus_ Location.txt"))
    p.add_argument("--reference-pmu-dir", type=Path, default=Path("data/RAW0001"))
    p.add_argument("--output-root", type=Path, default=Path("."))
    p.add_argument("--batch-mode", type=str, default="balanced_core")
    p.add_argument("--n-scenarios", type=int, default=24)
    p.add_argument("--difficulty-mode", type=str, default="curriculum_easy_to_hard")
    p.add_argument("--split-strategy", type=str, default="leakage_safe_balanced")
    p.add_argument("--recompute-angular-realism", action="store_true", default=True)
    p.add_argument("--recompute-freq-rocof", action="store_true", default=True)
    p.add_argument("--recalibrate-cyber", action="store_true", default=True)
    p.add_argument("--rebuild-splits", action="store_true", default=True)
    p.add_argument("--save-json", action="store_true", default=True)
    p.add_argument("--save-csv", action="store_true", default=True)
    p.add_argument("--save-plots", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    report = run_m9_2_final_polish(
        raw_path=a.raw_path,
        pmu_location_path=a.pmu_location_path,
        reference_pmu_dir=a.reference_pmu_dir,
        output_root=a.output_root,
        n_scenarios=a.n_scenarios,
        batch_mode=a.batch_mode,
        difficulty_mode=a.difficulty_mode,
        split_strategy=a.split_strategy,
        seed=a.seed,
        recompute_angular_realism=a.recompute_angular_realism,
        recompute_freq_rocof=a.recompute_freq_rocof,
        recalibrate_cyber=a.recalibrate_cyber,
        rebuild_splits=a.rebuild_splits,
        save_plots=a.save_plots,
    )
    return 0 if report.get("overall_verdict", {}).get("m9_2_finalized", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())

