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

from src.application.use_cases.calibrate_event0 import run_calibrate_event0_use_case
from src.infrastructure.legacy.m3_adapter import run_raw_event0_calibration


LOGGER = logging.getLogger("m3_pipeline_event0_calibration")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run migrated m3 raw event-0 calibration pipeline.")
    parser.add_argument("--raw-chunks-dir", type=Path, default=Path("output/SCENARIO_RAW0001/chunks"))
    parser.add_argument("--raw-profile-file", type=Path, default=Path("output/SCENARIO_RAW0001/dataset_profiles_raw.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/ANDES_CALIBRATION_RAW_NEWARCH"))
    parser.add_argument("--compare-legacy", action="store_true")
    parser.add_argument("--legacy-output-dir", type=Path, default=Path("output/ANDES_CALIBRATION_RAW_LEGACY_COMPARE"))
    parser.add_argument("--copy-debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_m3_outputs(output_dir: Path) -> None:
    required = [
        output_dir / "current_mapping_selection.csv",
        output_dir / "signal_support_matrix.csv",
        output_dir / "metrics" / "calibration_metrics_long.csv",
        output_dir / "metrics" / "calibration_summary_by_bus.csv",
        output_dir / "metrics" / "calibration_summary_by_signal.csv",
        output_dir / "metrics" / "calibration_summary_chunk0.csv",
        output_dir / "metrics" / "calibration_summary_signal_family.csv",
        output_dir / "metrics" / "current_mapping_diagnostics.csv",
        output_dir / "metrics" / "calibration_results.json",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")
    payload = _load_json(output_dir / "metrics" / "calibration_results.json")
    for k in ["generated_at_utc", "event_label", "drift_mode", "records"]:
        if k not in payload:
            raise ValueError(f"calibration_results.json missing key '{k}'")
    records = payload.get("records", [])
    status_counts: dict[str, int] = {}
    for rec in records:
        status = str(rec.get("status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
    evaluated_rows = status_counts.get("supported", 0) + status_counts.get("fallback_profile", 0)
    if records and evaluated_rows == 0:
        raise RuntimeError(
            "m3 found zero evaluable rows after calibration. "
            f"status_counts={status_counts}. Check raw_chunks_dir and bus-id key consistency."
        )


def _compare_outputs(new_dir: Path, legacy_dir: Path) -> None:
    assert_frame_equal(
        pd.read_csv(new_dir / "current_mapping_selection.csv"),
        pd.read_csv(legacy_dir / "current_mapping_selection.csv"),
        check_dtype=False,
        check_like=False,
    )
    assert_frame_equal(
        pd.read_csv(new_dir / "signal_support_matrix.csv"),
        pd.read_csv(legacy_dir / "signal_support_matrix.csv"),
        check_dtype=False,
        check_like=False,
    )
    assert_frame_equal(
        pd.read_csv(new_dir / "metrics" / "calibration_metrics_long.csv"),
        pd.read_csv(legacy_dir / "metrics" / "calibration_metrics_long.csv"),
        check_dtype=False,
        check_like=False,
    )
    if _load_json(new_dir / "metrics" / "calibration_results.json")["records"] != _load_json(
        legacy_dir / "metrics" / "calibration_results.json"
    )["records"]:
        raise AssertionError("calibration_results.json records differ")


def _copy_debug_outputs(new_dir: Path, legacy_dir: Path | None = None) -> Path:
    debug_root = REPO_ROOT / "debug_parity_outputs" / "m3"
    if debug_root.exists():
        shutil.rmtree(debug_root)
    debug_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(new_dir, debug_root / "migrated")
    if legacy_dir and legacy_dir.exists():
        shutil.copytree(legacy_dir, debug_root / "legacy")
    return debug_root


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    LOGGER.info("raw_chunks_dir=%s", args.raw_chunks_dir.resolve())
    summary = run_calibrate_event0_use_case(
        raw_chunks_dir=args.raw_chunks_dir.resolve(),
        raw_profile_file=args.raw_profile_file.resolve(),
        output_dir=args.output_dir.resolve(),
        event_label=0,
    )
    out_dir = Path(summary["output_dir"])
    validate_m3_outputs(out_dir)
    LOGGER.info("Migrated m3 calibration completed.")
    LOGGER.info("output_dir=%s", out_dir)
    LOGGER.info("record_count=%s", summary["record_count"])
    LOGGER.info("evaluated_row_count=%s", summary.get("evaluated_row_count"))
    LOGGER.info("status_counts=%s", summary.get("status_counts", {}))

    legacy_out: Path | None = None
    if args.compare_legacy:
        legacy_out = args.legacy_output_dir.resolve()
        run_raw_event0_calibration(
            raw_chunks_dir=str(args.raw_chunks_dir.resolve()),
            raw_profile_file=str(args.raw_profile_file.resolve()),
            output_dir=str(legacy_out),
            event_label=0,
        )
        validate_m3_outputs(legacy_out)
        _compare_outputs(out_dir, legacy_out)
        LOGGER.info("Legacy comparison passed.")

    if args.copy_debug:
        debug_root = _copy_debug_outputs(out_dir, legacy_out)
        LOGGER.info("Debug artifacts copied to: %s", debug_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
