from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.helpers.paths import DEFAULT_RAW_DIR, DEFAULT_TOPOLOGY_DIR, resolve_path
from src.models.hybrid_submission import run_hybrid_submission_prediction


DEFAULT_BUS_AGNOSTIC_MODEL_DIR = Path(__file__).resolve().parent / "models_bus_agnostic"
DEFAULT_ML_MODEL_DIR = Path(__file__).resolve().parent / "models"


def _normalize_raw_dir(path: Path) -> Path:
    if path.exists():
        return path
    name = path.name.lower()
    if name in {"raw001", "raw0001"}:
        candidates = [
            path.parent / "RAW0001",
            path.parent / "RAW001",
            path.parent / "raw0001",
            path.parent / "raw001",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run bus-agnostic SGSMA 2026 PMU anomaly inference. "
            "By default reads data/RAW0001 and writes predictions.csv in that same folder."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Folder containing Bus*.csv PMU files. Defaults to data/RAW0001.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <input-dir>/predictions.csv.",
    )
    parser.add_argument(
        "--topology-dir",
        type=Path,
        default=DEFAULT_TOPOLOGY_DIR,
        help="IEEE-39 topology folder with branches_physical.csv.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_BUS_AGNOSTIC_MODEL_DIR,
        help="Fallback bus-agnostic physics model/config directory. Defaults to models_bus_agnostic.",
    )
    parser.add_argument(
        "--ml-model-dir",
        type=Path,
        default=DEFAULT_ML_MODEL_DIR,
        help="Validated ML model bundle. Defaults to models when present.",
    )
    parser.add_argument(
        "--ml-max-duration-s",
        type=float,
        default=120.0,
        help="Use the ML scenario/chunk model for inputs up to this duration; longer RAW streams use the physics runtime.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = _normalize_raw_dir(resolve_path(args.input_dir))
    topology_dir = resolve_path(args.topology_dir)
    model_dir = resolve_path(args.model_dir)
    output = None if args.output is None else resolve_path(args.output)
    ml_model_dir = resolve_path(args.ml_model_dir)
    _, diagnostics = run_hybrid_submission_prediction(
        input_dir=input_dir,
        output_csv=output,
        topology_dir=topology_dir,
        physics_model_dir=model_dir,
        ml_model_dir=ml_model_dir,
        ml_max_duration_s=args.ml_max_duration_s,
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
