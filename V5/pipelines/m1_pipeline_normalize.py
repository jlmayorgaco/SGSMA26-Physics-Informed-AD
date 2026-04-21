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

from src.application.use_cases.normalize_data import run_normalize_data_use_case
from src.infrastructure.legacy.m1_adapter import run_normalization_pipeline


LOGGER = logging.getLogger("m1_pipeline_normalize")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run migrated m1 normalization pipeline with optional parity compare."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/RAW0001"),
        help="Directory containing *_nanmask.csv raw inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/M1_RAW0001_NORMALIZED_NEWARCH"),
        help="Output directory for migrated normalization artifacts.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip chunk signal plots and timeline plots.",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=["legacy_replace_angles", "augment_angles"],
        default="augment_angles",
        help="Feature construction mode for normalized output.",
    )
    parser.add_argument(
        "--compare-legacy",
        action="store_true",
        help="Run legacy m1 pipeline and compare key artifacts.",
    )
    parser.add_argument(
        "--legacy-output-dir",
        type=Path,
        default=Path("output/M1_RAW0001_NORMALIZED_LEGACY_COMPARE"),
        help="Output directory for legacy compare run.",
    )
    parser.add_argument(
        "--copy-debug",
        action="store_true",
        help="Copy outputs to debug_parity_outputs/m1 for inspection.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_m1_output_contract(output_dir: Path) -> None:
    baselines_path = output_dir / "normalization_baselines.csv"
    chunks_metadata_path = output_dir / "chunks_metadata.json"
    chunk_index_path = output_dir / "normalized_chunk_index.json"
    chunks_dir = output_dir / "chunks"

    for path in [baselines_path, chunks_metadata_path, chunk_index_path, chunks_dir]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    baseline_df = pd.read_csv(baselines_path)
    required_baseline_columns = {
        "bus_id",
        "raw_signal",
        "output_signal",
        "signal_family",
        "transform",
        "baseline_method",
        "baseline_value",
        "n_normal_samples",
        "baseline_source",
        "notes",
    }
    missing = required_baseline_columns - set(baseline_df.columns)
    if missing:
        raise ValueError(f"normalization_baselines.csv missing columns: {sorted(missing)}")

    chunk_index = _load_json(chunk_index_path)
    if int(chunk_index.get("focus_event_label", -1)) != 0:
        raise ValueError("normalized_chunk_index.json must set focus_event_label=0")
    if "event0_chunk_ids" not in chunk_index or not isinstance(chunk_index["event0_chunk_ids"], list):
        raise ValueError("normalized_chunk_index.json missing event0_chunk_ids list")
    if "chunks" not in chunk_index or not isinstance(chunk_index["chunks"], list):
        raise ValueError("normalized_chunk_index.json missing chunks list")

    chunk_csvs = list(chunks_dir.glob("chunk*/Bus*_normalized.csv"))
    if not chunk_csvs:
        raise ValueError("No normalized chunk CSV files were produced")

    sample_df = pd.read_csv(chunk_csvs[0])
    for col in ["DATA_PRESENT", "Event"]:
        if col not in sample_df.columns:
            raise ValueError(f"Normalized chunk CSV missing required column: {col}")


def _compare_artifacts(new_out: Path, legacy_out: Path) -> None:
    def _normalize_index_paths(payload: dict) -> dict:
        payload = dict(payload)
        normalized_chunks = []
        for chunk in payload.get("chunks", []):
            item = dict(chunk)
            item["chunk_path"] = str(Path(item["chunk_path"]).name)
            normalized_chunks.append(item)
        payload["chunks"] = normalized_chunks
        return payload

    assert_frame_equal(
        pd.read_csv(new_out / "normalization_baselines.csv"),
        pd.read_csv(legacy_out / "normalization_baselines.csv"),
        check_dtype=False,
        check_like=False,
    )
    if _load_json(new_out / "chunks_metadata.json") != _load_json(legacy_out / "chunks_metadata.json"):
        raise AssertionError("chunks_metadata.json differs from legacy output")
    if _normalize_index_paths(_load_json(new_out / "normalized_chunk_index.json")) != _normalize_index_paths(
        _load_json(legacy_out / "normalized_chunk_index.json")
    ):
        raise AssertionError("normalized_chunk_index.json differs from legacy output")

    new_chunk_csvs = sorted((new_out / "chunks").glob("chunk*/Bus*_normalized.csv"))
    legacy_chunk_csvs = sorted((legacy_out / "chunks").glob("chunk*/Bus*_normalized.csv"))
    if len(new_chunk_csvs) != len(legacy_chunk_csvs):
        raise AssertionError("Different number of normalized chunk CSV files vs legacy")
    for i in range(min(2, len(new_chunk_csvs))):
        assert_frame_equal(
            pd.read_csv(new_chunk_csvs[i]),
            pd.read_csv(legacy_chunk_csvs[i]),
            check_dtype=False,
            check_like=False,
        )


def _copy_debug_outputs(new_out: Path, legacy_out: Path | None = None) -> Path:
    debug_root = REPO_ROOT / "debug_parity_outputs" / "m1"
    if debug_root.exists():
        shutil.rmtree(debug_root)
    debug_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(new_out, debug_root / "migrated")
    if legacy_out and legacy_out.exists():
        shutil.copytree(legacy_out, debug_root / "legacy")
    return debug_root


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    output_dir = args.output_dir.resolve()
    input_dir = args.input_dir.resolve()

    summary = run_normalize_data_use_case(
        input_dir=input_dir,
        output_dir=output_dir,
        generate_plots=not args.skip_plots,
        feature_mode=args.feature_mode,
    )
    validate_m1_output_contract(output_dir)
    LOGGER.info("Migrated m1 normalization completed.")
    LOGGER.info("output_dir=%s", output_dir)
    LOGGER.info("chunk_count=%s", summary["chunk_count"])
    LOGGER.info("event0_chunk_count=%s", summary["event0_chunk_count"])

    legacy_output_dir: Path | None = None
    if args.compare_legacy:
        if args.feature_mode != "legacy_replace_angles":
            raise ValueError(
                "--compare-legacy requires --feature-mode legacy_replace_angles for parity-safe comparison."
            )
        legacy_output_dir = args.legacy_output_dir.resolve()
        run_normalization_pipeline(
            input_dir=str(input_dir),
            output_dir=str(legacy_output_dir),
            generate_plots=not args.skip_plots,
        )
        validate_m1_output_contract(legacy_output_dir)
        _compare_artifacts(output_dir, legacy_output_dir)
        LOGGER.info("Legacy comparison passed.")

    if args.copy_debug:
        debug_root = _copy_debug_outputs(output_dir, legacy_output_dir)
        LOGGER.info("Debug artifacts copied to: %s", debug_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
