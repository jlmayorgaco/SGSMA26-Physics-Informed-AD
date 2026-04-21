from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.simulation.m9.generator import generate_scenario, generate_template_matrix

LOGGER = logging.getLogger("m9_scenario_generator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M9 synthetic PMU scenario generator for IEEE-39.")
    parser.add_argument("--raw-path", type=Path, default=Path("data/metadata/IEEE_39_Bus_Power_System.raw"))
    parser.add_argument("--pmu-location-path", type=Path, default=Path("data/metadata/PMUbus_ Location.txt"))
    parser.add_argument("--reference-pmu-dir", type=Path, default=Path("data/RAW0001"))
    parser.add_argument("--output-root", type=Path, default=Path("data/scenarios"))
    parser.add_argument("--scenario-template", type=str, default="TEMPLATE_OFFICIAL_STYLE_MULTI_EVENT")
    parser.add_argument("--scenario-id", type=str, default="SIM0001")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--use-andes", action="store_true")
    parser.add_argument("--noise-profile", type=str, default="raw0001_empirical")
    parser.add_argument("--cyber-profile", type=str, default="template")
    parser.add_argument("--physical-profile", type=str, default="template")
    parser.add_argument("--n-scenarios", type=int, default=1)
    parser.add_argument("--validate", action="store_true", help="Run scenario validation after generation.")
    parser.add_argument("--save-json", action="store_true", default=True)
    parser.add_argument("--save-csv", action="store_true", default=True)
    parser.add_argument("--save-plots", action="store_true", default=True)
    parser.add_argument("--all-templates", action="store_true", help="Generate SIM0001 plus Event 0..8 template scenarios.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    if args.all_templates or args.scenario_template.upper() in {"ALL", "ALL_EVENT_TEMPLATES", "MATRIX"}:
        manifests = generate_template_matrix(
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
        LOGGER.info("Generated %d M9 scenarios under %s", len(manifests), args.output_root)
        return 0

    for i in range(max(1, int(args.n_scenarios))):
        scenario_id = args.scenario_id if i == 0 else f"SIM{int(args.scenario_id.replace('SIM', '') or 1) + i:04d}"
        manifest = generate_scenario(
            raw_path=args.raw_path,
            pmu_location_path=args.pmu_location_path,
            reference_pmu_dir=args.reference_pmu_dir,
            output_root=args.output_root,
            scenario_template=args.scenario_template,
            scenario_id=scenario_id,
            seed=args.seed + i,
            fps=args.fps,
            duration_s=args.duration_s,
            use_andes=args.use_andes,
            noise_profile=args.noise_profile,
            cyber_profile=args.cyber_profile,
            physical_profile=args.physical_profile,
            save_json=args.save_json,
            save_csv=args.save_csv,
            save_plots=args.save_plots,
        )
        LOGGER.info("Generated %s at %s", manifest["scenario_id"], args.output_root / manifest["scenario_id"])
        if args.validate:
            from src.simulation.m9.validation import validate_scenario

            result = validate_scenario(args.output_root / manifest["scenario_id"], reference_pmu_dir=args.reference_pmu_dir)
            LOGGER.info("Scenario valid: %s", result.get("scenario_valid"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
