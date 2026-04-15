"""Train several V2 PMU grid models from existing synthetic CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pmu_grid_benchmark import BenchmarkConfig, PmuGridBenchmarkPipeline, SUPPORTED_MODELS  # noqa: E402


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_models(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-dir", type=Path, default=PROJECT_ROOT / "data" / "synthetic_v2")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models_benchmark")
    parser.add_argument("--models", default=",".join(SUPPORTED_MODELS))
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--samples-per-event", type=int, default=3)
    parser.add_argument("--normal-samples-per-scenario", type=int, default=8)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260412)
    parser.add_argument("--skip-completed", action="store_true", help="Reuse completed per-model summaries in the output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PmuGridBenchmarkPipeline(
        BenchmarkConfig(
            synthetic_dir=project_path(args.synthetic_dir),
            output_dir=project_path(args.output_dir),
            models=parse_models(args.models),
            window_sec=args.window_sec,
            samples_per_event=args.samples_per_event,
            normal_samples_per_scenario=args.normal_samples_per_scenario,
            max_scenarios=args.max_scenarios,
            train_fraction=args.train_fraction,
            runs=args.runs,
            seed=args.seed,
            skip_completed=args.skip_completed,
        )
    ).run()


if __name__ == "__main__":
    main()
