from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.application.use_cases.m7_state_estimator_benchmark import run_m7_state_estimator_benchmark_use_case


LOGGER = logging.getLogger("m7_state_estimator_benchmark")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M7 benchmark for state estimators on IEEE-39 with ANDES truth.")
    parser.add_argument("--raw-path", type=Path, default=Path("data/metadata/IEEE_39_Bus_Power_System.raw"))
    parser.add_argument("--pmu-location-path", type=Path, default=Path("data/metadata/PMUbus_ Location.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/M7_STATE_ESTIMATOR_BENCHMARK"))
    parser.add_argument("--use-andes-truth", action="store_true", default=True)
    parser.add_argument("--scenario-set", type=str, default="default")
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=10.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--estimators", nargs="*", default=[])
    parser.add_argument("--default-estimator", type=str, default="PMU_VOLTAGE_CURRENT_WLS")
    parser.add_argument("--save-csv", action="store_true", default=True)
    parser.add_argument("--save-json", action="store_true", default=True)
    parser.add_argument("--save-plots", action="store_true", default=True)
    parser.add_argument("--save-report", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    summary = run_m7_state_estimator_benchmark_use_case(
        raw_path=args.raw_path.resolve(),
        pmu_location_path=args.pmu_location_path.resolve(),
        output_dir=args.output_dir.resolve(),
        use_andes_truth=bool(args.use_andes_truth),
        scenario_set=args.scenario_set,
        start_time=args.start_time,
        end_time=args.end_time,
        stride=max(1, int(args.stride)),
        estimators=args.estimators,
        default_estimator=args.default_estimator,
    )
    LOGGER.info("M7 benchmark completed: %s", summary["output_dir"])
    LOGGER.info("Best estimator: %s", summary["best_estimator"])
    LOGGER.info("Estimator family working: %s", summary["overall_conclusion"]["is_estimator_family_working"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

