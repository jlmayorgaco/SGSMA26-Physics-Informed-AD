from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys

import pandas as pd
from pandas.testing import assert_frame_equal


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.application.use_cases.simulate_physical_event import run_simulate_type1_fault_use_case
from src.infrastructure.legacy.m4_adapter import run_type1_fault_simulation
from src.simulation.type1_config import Type1Config


LOGGER = logging.getLogger("m4_pipeline_type1_fault")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _default_newarch_m3_root() -> Path:
    return Path("output/ANDES_CALIBRATION_RAW_NEWARCH")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run migrated m4 type-1 fault simulation pipeline.")
    parser.add_argument("--fault-bus", default="39")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--event0-profile-path", type=Path, default=Path("output/M2_RAW0001_NOISE_PROFILE_NEWARCH/dataset_profiles_raw.json"))
    parser.add_argument("--event0-current-mapping-csv", type=Path, default=_default_newarch_m3_root() / "current_mapping_selection.csv")
    parser.add_argument("--event0-support-matrix-csv", type=Path, default=_default_newarch_m3_root() / "signal_support_matrix.csv")
    parser.add_argument("--event0-calibration-json", type=Path, default=_default_newarch_m3_root() / "metrics" / "calibration_results.json")
    parser.add_argument("--compare-legacy", action="store_true")
    parser.add_argument("--legacy-output-root", type=Path, default=Path("output/_legacy_m4_compare"))
    parser.add_argument("--copy-debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def validate_run_outputs(run_dir: Path) -> None:
    required = [
        run_dir / "simulation",
        run_dir / "estimated",
        run_dir / "run_info.json",
        run_dir / "estimated" / "estimation_metrics_long.csv",
        run_dir / "estimated" / "estimation_summary_by_bus.csv",
        run_dir / "estimated" / "estimation_summary_by_signal.csv",
        run_dir / "estimated" / "estimation_report.json",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")
    sim_count = len(list((run_dir / "simulation").glob("BUS*_Competition_Data_nanmask.csv")))
    est_count = len(list((run_dir / "estimated").glob("BUS*_Competition_Data_nanmask.csv")))
    if sim_count == 0 or est_count == 0:
        raise RuntimeError("Expected simulation/estimated BUS csv files; got zero.")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_runs(new_run: Path, legacy_run: Path) -> None:
    for rel in [
        "estimated/estimation_metrics_long.csv",
        "estimated/estimation_summary_by_bus.csv",
        "estimated/estimation_summary_by_signal.csv",
    ]:
        assert_frame_equal(
            pd.read_csv(new_run / rel),
            pd.read_csv(legacy_run / rel),
            check_dtype=False,
            rtol=1e-6,
            atol=1e-8,
            check_like=True,
        )
    new_info = _load_json(new_run / "run_info.json")
    old_info = _load_json(legacy_run / "run_info.json")
    for key in ["fault_bus", "event_label_mode", "folders"]:
        if key not in new_info or key not in old_info:
            raise AssertionError(f"run_info.json missing key {key!r}")
    if len(list((new_run / "simulation").glob("BUS*_Competition_Data_nanmask.csv"))) != len(
        list((legacy_run / "simulation").glob("BUS*_Competition_Data_nanmask.csv"))
    ):
        raise AssertionError("Simulation BUS csv count mismatch vs legacy")
    if len(list((new_run / "estimated").glob("BUS*_Competition_Data_nanmask.csv"))) != len(
        list((legacy_run / "estimated").glob("BUS*_Competition_Data_nanmask.csv"))
    ):
        raise AssertionError("Estimated BUS csv count mismatch vs legacy")


def _copy_debug(new_run: Path, legacy_run: Path | None) -> Path:
    debug_root = REPO_ROOT / "debug_parity_outputs" / "m4"
    if debug_root.exists():
        shutil.rmtree(debug_root)
    debug_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(new_run, debug_root / "migrated")
    if legacy_run and legacy_run.exists():
        shutil.copytree(legacy_run, debug_root / "legacy")
    return debug_root


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    _require(args.event0_profile_path, "event0 profile")
    _require(args.event0_current_mapping_csv, "event0 current mapping")
    _require(args.event0_support_matrix_csv, "event0 support matrix")
    _require(args.event0_calibration_json, "event0 calibration json")

    LOGGER.info("fault_bus=%s", args.fault_bus)
    summary = run_simulate_type1_fault_use_case(
        fault_bus=args.fault_bus,
        output_root=args.output_root.resolve(),
        event0_profile_path=args.event0_profile_path.resolve(),
        event0_current_mapping_csv=args.event0_current_mapping_csv.resolve(),
        event0_support_matrix_csv=args.event0_support_matrix_csv.resolve(),
        event0_calibration_json=args.event0_calibration_json.resolve(),
        generate_plots=True,
    )
    run_dir = Path(summary["run_dir"])
    validate_run_outputs(run_dir)
    LOGGER.info("m4 migrated run completed. run_dir=%s", run_dir)

    legacy_run: Path | None = None
    if args.compare_legacy:
        run_type1_fault_simulation(
            fault_bus=args.fault_bus,
            output_root=args.legacy_output_root.resolve(),
            event0_profile_path=args.event0_profile_path.resolve(),
            event0_current_mapping_csv=args.event0_current_mapping_csv.resolve(),
            event0_support_matrix_csv=args.event0_support_matrix_csv.resolve(),
            event0_calibration_json=args.event0_calibration_json.resolve(),
            generate_plots=True,
        )
        legacy_run = args.legacy_output_root.resolve() / Type1Config(fault_bus=str(args.fault_bus)).run_folder_name()
        validate_run_outputs(legacy_run)
        _compare_runs(run_dir, legacy_run)
        LOGGER.info("legacy parity comparison passed")

    if args.copy_debug:
        debug_root = _copy_debug(run_dir, legacy_run)
        LOGGER.info("debug outputs copied to %s", debug_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
