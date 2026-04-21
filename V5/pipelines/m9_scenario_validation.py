from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.simulation.m9.validation import run_m9_validation_suite, validate_scenario

LOGGER = logging.getLogger("m9_scenario_validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M9 scenario validation suite and readiness report.")
    parser.add_argument("--raw-path", type=Path, default=Path("data/metadata/IEEE_39_Bus_Power_System.raw"))
    parser.add_argument("--pmu-location-path", type=Path, default=Path("data/metadata/PMUbus_ Location.txt"))
    parser.add_argument("--reference-pmu-dir", type=Path, default=Path("data/RAW0001"))
    parser.add_argument("--output-root", type=Path, default=Path("data"), help="Root containing/receiving scenarios and report folders.")
    parser.add_argument("--scenario-dir", type=Path, default=None, help="Validate one existing scenario instead of running the full matrix.")
    parser.add_argument("--scenario-template", type=str, default="ALL_EVENT_TEMPLATES")
    parser.add_argument("--scenario-id", type=str, default="SIM0001")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--use-andes", action="store_true")
    parser.add_argument("--noise-profile", type=str, default="raw0001_empirical")
    parser.add_argument("--cyber-profile", type=str, default="template")
    parser.add_argument("--physical-profile", type=str, default="template")
    parser.add_argument("--n-scenarios", type=int, default=1)
    parser.add_argument("--validate", action="store_true", default=True)
    parser.add_argument("--save-json", action="store_true", default=True)
    parser.add_argument("--save-csv", action="store_true", default=True)
    parser.add_argument("--save-plots", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    if args.scenario_dir is not None:
        result = validate_scenario(args.scenario_dir, reference_pmu_dir=args.reference_pmu_dir, save_json=args.save_json)
        LOGGER.info("Scenario valid: %s", result.get("scenario_valid"))
        return 0 if result.get("scenario_valid") else 2
    report = run_m9_validation_suite(
        raw_path=args.raw_path,
        pmu_location_path=args.pmu_location_path,
        reference_pmu_dir=args.reference_pmu_dir,
        output_root=args.output_root,
        seed=args.seed,
        fps=args.fps,
        duration_s=args.duration_s,
        use_andes=args.use_andes,
        save_plots=args.save_plots,
    )
    working = bool(report.get("overall_verdict", {}).get("m9_simulator_working"))
    LOGGER.info("M9 simulator working: %s", working)
    return 0 if working else 2


if __name__ == "__main__":
    raise SystemExit(main())
