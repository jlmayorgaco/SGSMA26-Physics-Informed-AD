from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.application.use_cases.simulate_normal_event import run_simulate_type0_use_case
from src.simulation.comparison_reports import compare_runs


LOGGER = logging.getLogger("m4_pipeline_type0_normal")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _default_newarch_m3_root() -> Path:
    return Path("output/ANDES_CALIBRATION_RAW_NEWARCH")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run migrated m4 type0 normal (no-fault) pipeline.")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--event0-profile-path", type=Path, default=Path("output/M2_RAW0001_NOISE_PROFILE_NEWARCH/dataset_profiles_raw.json"))
    parser.add_argument("--event0-current-mapping-csv", type=Path, default=_default_newarch_m3_root() / "current_mapping_selection.csv")
    parser.add_argument("--event0-support-matrix-csv", type=Path, default=_default_newarch_m3_root() / "signal_support_matrix.csv")
    parser.add_argument("--event0-calibration-json", type=Path, default=_default_newarch_m3_root() / "metrics" / "calibration_results.json")
    parser.add_argument("--compare-with-run-dir", type=Path, default=None)
    parser.add_argument("--comparison-output-dir", type=Path, default=Path("output/M4_TYPE0_TYPE1_COMPARISON"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    _require(args.event0_profile_path, "event0 profile")
    _require(args.event0_current_mapping_csv, "event0 current mapping")
    _require(args.event0_support_matrix_csv, "event0 support matrix")
    _require(args.event0_calibration_json, "event0 calibration json")

    summary = run_simulate_type0_use_case(
        output_root=args.output_root.resolve(),
        event0_profile_path=args.event0_profile_path.resolve(),
        event0_current_mapping_csv=args.event0_current_mapping_csv.resolve(),
        event0_support_matrix_csv=args.event0_support_matrix_csv.resolve(),
        event0_calibration_json=args.event0_calibration_json.resolve(),
        generate_plots=True,
    )
    run_dir = Path(summary["run_dir"])
    LOGGER.info("type0 run completed: %s", run_dir)

    if args.compare_with_run_dir is not None:
        cmp_summary = compare_runs(
            run_a_dir=run_dir,
            run_b_dir=args.compare_with_run_dir.resolve(),
            output_dir=args.comparison_output_dir.resolve(),
        )
        LOGGER.info("comparison generated: %s", cmp_summary["comparison_dir"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
