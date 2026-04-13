"""Regenerate IEEE 39 event diagrams from existing synthetic scenario JSON files.

This is a fast review utility: it does not rerun the simulator and does not
touch the bus CSVs. It only reads each ``SIM_####.json`` file and redraws the
professional single-line diagram in that scenario's ``plots`` folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plotters import plot_ieee39_diagram  # noqa: E402


DEFAULT_PMU_BUSES = [2, 5, 6, 10, 19, 22, 29, 39]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "synthetic_v2",
        help="Synthetic dataset directory containing SIM_#### folders.",
    )
    parser.add_argument(
        "--pattern",
        default="SIM_*/SIM_*.json",
        help="Glob pattern, relative to --dataset-dir, used to find scenario JSON files.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of scenarios to plot.")
    parser.add_argument(
        "--filename",
        default="ieee39_event_diagram.png",
        help="Output PNG filename inside each scenario's plots directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional shared output directory. If omitted, each diagram is written to SIM_####/plots.",
    )
    return parser.parse_args()


def load_scenario(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def scenario_output_dir(scenario_json: Path, shared_out_dir: Path | None) -> Path:
    if shared_out_dir is not None:
        return shared_out_dir
    return scenario_json.parent / "plots"


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else PROJECT_ROOT / args.dataset_dir
    scenario_files = sorted(dataset_dir.glob(args.pattern))
    if args.limit is not None:
        scenario_files = scenario_files[: args.limit]
    if not scenario_files:
        raise FileNotFoundError(f"No scenario JSON files found with pattern {dataset_dir / args.pattern}")

    written: list[Path] = []
    for scenario_json in scenario_files:
        scenario = load_scenario(scenario_json)
        branches = scenario.get("ieee39_branches")
        if not branches:
            raise ValueError(f"{scenario_json} does not contain ieee39_branches")
        pmu_buses = scenario.get("pmu_buses", DEFAULT_PMU_BUSES)
        events = scenario.get("events", [])
        out_dir = scenario_output_dir(scenario_json, args.out_dir)
        filename = args.filename
        if args.out_dir is not None and len(scenario_files) > 1:
            filename = f"{scenario.get('scenario_id', scenario_json.parent.name)}_{args.filename}"
        path = plot_ieee39_diagram(
            branches=branches,
            pmu_buses=pmu_buses,
            events=events,
            output_dir=out_dir,
            scenario_id=scenario.get("scenario_id", scenario_json.parent.name),
            filename=filename,
        )
        written.append(path)
        print(f"Wrote {path}")

    print(f"Finished {len(written)} IEEE 39 diagram(s).")


if __name__ == "__main__":
    main()
