from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.application.use_cases.m6_topology_aware_state_estimation import (
    run_m6_topology_aware_state_estimation_use_case,
)


LOGGER = logging.getLogger("m6_topology_aware_state_estimation")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M6 topology-aware PMU-only state estimation pipeline.")
    parser.add_argument("--raw-path", type=Path, default=Path("data/metadata/IEEE_39_Bus_Power_System.raw"))
    parser.add_argument("--pmu-location-path", type=Path, default=Path("data/metadata/PMUbus_ Location.txt"))
    parser.add_argument("--pmu-data-dir", type=Path, default=Path("tests/fixtures/raw_small"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/M6_STATE_ESTIMATION"))
    parser.add_argument("--diagnostic", action="store_true", default=True)
    parser.add_argument("--use-andes-truth", action="store_true")
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--lambda-reg", type=float, default=5e-2)
    parser.add_argument("--mu-reg", type=float, default=1e-3)
    parser.add_argument("--save-json", action="store_true", default=True)
    parser.add_argument("--save-csv", action="store_true", default=True)
    parser.add_argument("--save-plots", action="store_true", default=True)
    parser.add_argument("--save-report", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    summary = run_m6_topology_aware_state_estimation_use_case(
        raw_path=args.raw_path.resolve(),
        pmu_location_path=args.pmu_location_path.resolve(),
        pmu_data_dir=args.pmu_data_dir.resolve() if args.pmu_data_dir is not None else None,
        output_dir=args.output_dir.resolve(),
        diagnostic=bool(args.diagnostic),
        use_andes_truth=bool(args.use_andes_truth),
        start_time=args.start_time,
        end_time=args.end_time,
        stride=max(1, int(args.stride)),
        lambda_reg=float(args.lambda_reg),
        mu_reg=float(args.mu_reg),
    )
    LOGGER.info("M6 completed: output_dir=%s", summary["output_dir"])
    LOGGER.info(
        "timestamps=%s buses=%s pmu_buses=%s use_andes_truth=%s",
        summary["timestamp_count"],
        summary["bus_count"],
        summary["pmu_bus_count"],
        summary["use_andes_truth"],
    )
    if summary.get("global_metrics"):
        LOGGER.info("global_metrics=%s", summary["global_metrics"])
    if int(summary.get("min_pmus_used", 0)) <= 0:
        LOGGER.error("M6 failed acceptance: zero PMUs used in at least one frame.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
