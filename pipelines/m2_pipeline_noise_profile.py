from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.application.use_cases.profile_noise import run_profile_noise_use_case
from src.infrastructure.legacy.m2_adapter import run_noise_profile_pipeline


LOGGER = logging.getLogger("m2_pipeline_noise_profile")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run migrated m2 raw-noise profiling pipeline.")
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("output/SCENARIO_RAW0001/chunks"),
        help="Directory containing chunk*_event_*/Bus*.csv files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("output/M2_RAW0001_NOISE_PROFILE_NEWARCH/dataset_profiles_raw.json"),
        help="Output JSON path for migrated profile payload.",
    )
    parser.add_argument(
        "--compare-legacy",
        action="store_true",
        help="Run legacy m2 profiler and compare generated profile payload.",
    )
    parser.add_argument(
        "--legacy-output-path",
        type=Path,
        default=Path("output/M2_RAW0001_NOISE_PROFILE_LEGACY_COMPARE/dataset_profiles_raw.json"),
        help="Output JSON path for legacy comparison run.",
    )
    parser.add_argument(
        "--copy-debug",
        action="store_true",
        help="Copy migrated/legacy JSON outputs to debug_parity_outputs/m2.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_profile_contract(output_path: Path) -> None:
    if not output_path.exists():
        raise FileNotFoundError(f"Missing output JSON: {output_path}")
    payload = _load_json(output_path)
    if not isinstance(payload, dict):
        raise ValueError("dataset_profiles_raw.json must be a JSON object")
    for event_type, buses in payload.items():
        if not isinstance(event_type, str):
            raise ValueError("Top-level event_type keys must be strings")
        if not isinstance(buses, dict):
            raise ValueError("Second-level bus mapping must be a dict")
        for bus_id, signals in buses.items():
            if not isinstance(bus_id, str):
                raise ValueError("bus_id keys must be strings")
            if not isinstance(signals, dict):
                raise ValueError("Signal mapping must be a dict")
            for _, profile in signals.items():
                for key in ["eda_stats", "noise_model", "diagnostics", "sample_count"]:
                    if key not in profile:
                        raise ValueError(f"Profile missing key '{key}'")


def _copy_debug_outputs(new_path: Path, legacy_path: Path | None = None) -> Path:
    debug_root = REPO_ROOT / "debug_parity_outputs" / "m2"
    if debug_root.exists():
        shutil.rmtree(debug_root)
    debug_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(new_path, debug_root / "migrated_dataset_profiles_raw.json")
    if legacy_path is not None and legacy_path.exists():
        shutil.copy2(legacy_path, debug_root / "legacy_dataset_profiles_raw.json")
    return debug_root


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    chunks_dir = args.chunks_dir.resolve()
    output_path = args.output_path.resolve()
    profiles = run_profile_noise_use_case(chunks_dir=chunks_dir, output_path=output_path)
    validate_profile_contract(output_path)
    LOGGER.info("Migrated m2 profiling completed.")
    LOGGER.info("output_path=%s", output_path)
    LOGGER.info("event_type_count=%s", len(profiles))

    legacy_output_path: Path | None = None
    if args.compare_legacy:
        legacy_output_path = args.legacy_output_path.resolve()
        run_noise_profile_pipeline(chunks_dir=str(chunks_dir), out_path=str(legacy_output_path))
        validate_profile_contract(legacy_output_path)
        if _load_json(output_path) != _load_json(legacy_output_path):
            raise AssertionError("Migrated and legacy profile JSON payloads differ")
        LOGGER.info("Legacy comparison passed.")

    if args.copy_debug:
        debug_root = _copy_debug_outputs(output_path, legacy_output_path)
        LOGGER.info("Debug artifacts copied to: %s", debug_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
