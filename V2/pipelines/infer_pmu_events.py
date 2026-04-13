"""Infer event type, time interval, location, and 39-bus state from 8 PMUs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pmu_grid_pipeline import InferenceConfig, PmuGridInferencePipeline  # noqa: E402


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "pmu_grid_model.pkl")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "models" / "pmu_grid_predictions.json")
    parser.add_argument("--window-sec", type=float, default=None)
    parser.add_argument("--stride-sec", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--max-windows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = InferenceConfig(
        model_path=project_path(args.model),
        input_dir=project_path(args.input_dir),
        output_path=project_path(args.out),
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        threshold=args.threshold,
        max_windows=args.max_windows,
    )
    PmuGridInferencePipeline(config).run()


if __name__ == "__main__":
    main()
