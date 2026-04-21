from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.application.use_cases.build_metadata_model import run_build_metadata_model_use_case


LOGGER = logging.getLogger("m5_pipeline_metadata_model")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build M5 metadata/model bundle from PMU and RAW metadata files.")
    parser.add_argument("--pmu-location-path", type=Path, default=Path("data/metadata/PMUbus_ Location.txt"))
    parser.add_argument("--raw-path", type=Path, default=Path("data/metadata/IEEE_39_Bus_Power_System.raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/M5_METADATA_MODEL"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    summary = run_build_metadata_model_use_case(
        pmu_location_path=args.pmu_location_path.resolve(),
        raw_path=args.raw_path.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    LOGGER.info("M5 metadata/model build completed.")
    LOGGER.info("output_dir=%s", summary["output_dir"])
    LOGGER.info("bus_alignment_ok=%s ready_for_estimation=%s", summary["bus_alignment_ok"], summary["ready_for_estimation"])
    LOGGER.info("ybus_shape=%s zbus_shape=%s", summary["ybus_shape"], summary["zbus_shape"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
