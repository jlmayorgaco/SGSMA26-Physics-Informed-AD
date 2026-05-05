from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.helpers.paths import DEFAULT_RAW_DIR, DEFAULT_TOPOLOGY_DIR, resolve_path
from src.models.bus_agnostic import run_bus_agnostic_prediction


DEFAULT_BUS_AGNOSTIC_MODEL_DIR = Path(__file__).resolve().parent / "models_bus_agnostic"


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
        help="Bus-agnostic model/config directory. Defaults to models_bus_agnostic.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = _normalize_raw_dir(resolve_path(args.input_dir))
    topology_dir = resolve_path(args.topology_dir)
    model_dir = resolve_path(args.model_dir)
    output = None if args.output is None else resolve_path(args.output)
    _, diagnostics = run_bus_agnostic_prediction(
        input_dir=input_dir,
        output_csv=output,
        topology_dir=topology_dir,
        model_dir=model_dir,
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
