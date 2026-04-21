from __future__ import annotations

import argparse
from pathlib import Path

from src.simulation.m9.raw_informed_pipeline import run_m2_raw_informed_cyber


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M2 RAW-informed stochastic cyber layer pipeline")
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("output/M2_RAW_INFORMED_CYBER"))
    parser.add_argument("--chunks-root", type=Path, default=Path("output/M0_RAW0001_NEWARCH/chunks"))
    parser.add_argument("--raw-path", type=Path, default=Path("data/metadata/IEEE_39_Bus_Power_System.raw"))
    parser.add_argument("--pmu-location-path", type=Path, default=Path("data/metadata/PMUbus_ Location.txt"))
    parser.add_argument("--reference-pmu-dir", type=Path, default=Path("data/RAW0001"))
    parser.add_argument("--raw-holdout-root", type=Path, default=Path("data/RAW0001"))
    parser.add_argument("--baseline-model-path", type=Path, default=Path("output/detector_m10_2_ready/models/detector_model.pkl"))
    parser.add_argument("--baseline-threshold-path", type=Path, default=Path("output/detector_m10_2_ready/config/threshold_config_v4.json"))
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--event5-count", type=int, default=30)
    parser.add_argument("--event7-count", type=int, default=30)
    parser.add_argument("--mixed-count", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_m2_raw_informed_cyber(
        workspace_root=args.workspace_root.resolve(),
        output_root=args.output_root.resolve(),
        chunks_root=args.chunks_root.resolve(),
        raw_path=args.raw_path.resolve(),
        pmu_location_path=args.pmu_location_path.resolve(),
        reference_pmu_dir=args.reference_pmu_dir.resolve(),
        raw_holdout_root=args.raw_holdout_root.resolve(),
        baseline_model_path=args.baseline_model_path.resolve(),
        baseline_threshold_path=args.baseline_threshold_path.resolve(),
        seed=int(args.seed),
        event5_count=int(args.event5_count),
        event7_count=int(args.event7_count),
        mixed_count=int(args.mixed_count),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

