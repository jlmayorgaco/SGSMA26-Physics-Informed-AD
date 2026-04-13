"""Simple V2 pipeline: generate synthetic data, train, and infer from 8 PMUs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PROJECT_ROOT / "pipelines"
for path in (PROJECT_ROOT, PIPELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from make_synth_data import generate_dataset, load_config, parse_plot_bus_option, resolve_project_path  # noqa: E402
from src.ml.pmu_grid_benchmark import (  # noqa: E402
    BenchmarkConfig,
    PmuGridBenchmarkPipeline,
    SUPPORTED_MODELS,
)
from src.ml.pmu_grid_pipeline import (  # noqa: E402
    InferenceConfig,
    PmuGridInferencePipeline,
    PmuGridTrainingPipeline,
    TrainingConfig,
)


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate synthetic V2 scenarios.")
    add_generation_args(gen)

    train = sub.add_parser("train", help="Train/test the 8-PMU to 39-bus model.")
    add_training_args(train, include_synthetic_dir=True)

    benchmark_parser = sub.add_parser("benchmark", help="Train several compact models and select the best.")
    add_benchmark_args(benchmark_parser)

    infer = sub.add_parser("infer", help="Infer events from raw or synthetic PMU CSVs.")
    add_inference_args(infer)

    run_all = sub.add_parser("run-all", help="Generate synthetic data, then train/test.")
    add_generation_args(run_all)
    add_training_args(run_all, include_synthetic_dir=False)
    return parser.parse_args()


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "pipelines" / "make_synth_data.json")
    parser.add_argument("--n-scenarios", type=int, default=None)
    parser.add_argument("--synthetic-dir", type=Path, default=PROJECT_ROOT / "data" / "synthetic_v2")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-all", action="store_true", help="Write review plots/reports for every generated scenario.")
    parser.add_argument("--plot-first-n", type=int, default=None, help="Write review plots/reports for the first N scenarios.")
    parser.add_argument("--plot-buses", default=None, help='Review buses to plot: "pmu", "all", or comma-separated bus numbers.')


def add_training_args(parser: argparse.ArgumentParser, include_synthetic_dir: bool) -> None:
    if include_synthetic_dir:
        parser.add_argument("--synthetic-dir", type=Path, default=PROJECT_ROOT / "data" / "synthetic_v2")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--model", choices=["lightgbm", "histgb", "extratrees"], default="lightgbm")
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--samples-per-event", type=int, default=3)
    parser.add_argument("--normal-samples-per-scenario", type=int, default=8)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--train-seed", type=int, default=20260412)


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--synthetic-dir", type=Path, default=PROJECT_ROOT / "data" / "synthetic_v2")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models_benchmark")
    parser.add_argument("--models", default=",".join(SUPPORTED_MODELS))
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--samples-per-event", type=int, default=3)
    parser.add_argument("--normal-samples-per-scenario", type=int, default=8)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--train-seed", type=int, default=20260412)


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "pmu_grid_model.pkl")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "models" / "pmu_grid_predictions.json")
    parser.add_argument("--window-sec", type=float, default=None)
    parser.add_argument("--stride-sec", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--max-windows", type=int, default=None)


def generate(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    config = json.loads(json.dumps(config))
    if args.n_scenarios is not None:
        config["n_scenarios"] = int(args.n_scenarios)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.duration_sec is not None:
        config["duration_sec"] = float(args.duration_sec)
    review = config.setdefault("review", {})
    if args.plot_all:
        review["plot_first_n_scenarios"] = "all"
    elif args.plot_first_n is not None:
        review["plot_first_n_scenarios"] = int(args.plot_first_n)
    if args.plot_buses is not None:
        review["plot_buses_per_scenario"] = parse_plot_bus_option(
            args.plot_buses,
            [int(bus) for bus in config.get("pmu_buses", [])],
        )
    config["output_dir"] = str(project_path(args.synthetic_dir))
    output_dir = resolve_project_path(config["output_dir"])
    return generate_dataset(config, output_dir, make_plots=not args.no_plots)


def train(args: argparse.Namespace) -> dict:
    config = TrainingConfig(
        synthetic_dir=project_path(args.synthetic_dir),
        output_dir=project_path(args.output_dir),
        model_name=args.model,
        window_sec=args.window_sec,
        samples_per_event=args.samples_per_event,
        normal_samples_per_scenario=args.normal_samples_per_scenario,
        max_scenarios=args.max_scenarios,
        train_fraction=args.train_fraction,
        runs=args.runs,
        seed=args.train_seed,
    )
    return PmuGridTrainingPipeline(config).run()


def benchmark(args: argparse.Namespace) -> dict:
    config = BenchmarkConfig(
        synthetic_dir=project_path(args.synthetic_dir),
        output_dir=project_path(args.output_dir),
        models=[item.strip().lower() for item in args.models.split(",") if item.strip()],
        window_sec=args.window_sec,
        samples_per_event=args.samples_per_event,
        normal_samples_per_scenario=args.normal_samples_per_scenario,
        max_scenarios=args.max_scenarios,
        train_fraction=args.train_fraction,
        runs=args.runs,
        seed=args.train_seed,
    )
    return PmuGridBenchmarkPipeline(config).run()


def infer(args: argparse.Namespace) -> dict:
    config = InferenceConfig(
        model_path=project_path(args.model_path),
        input_dir=project_path(args.input_dir),
        output_path=project_path(args.out),
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        threshold=args.threshold,
        max_windows=args.max_windows,
    )
    return PmuGridInferencePipeline(config).run()


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        generate(args)
    elif args.command == "train":
        train(args)
    elif args.command == "benchmark":
        benchmark(args)
    elif args.command == "infer":
        infer(args)
    elif args.command == "run-all":
        generate(args)
        train(args)
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
