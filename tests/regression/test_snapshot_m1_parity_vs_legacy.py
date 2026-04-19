from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from src.application.use_cases.normalize_data import run_normalize_data_use_case
from src.infrastructure.legacy.m1_adapter import run_normalization_pipeline


def _workspace_test_dir(prefix: str) -> Path:
    out_dir = Path("output") / "_test_runs" / f"{prefix}_{uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_json(path: Path) -> dict:
    assert path.exists(), f"Missing JSON artifact: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_required_artifacts_exist(out_dir: Path) -> None:
    assert out_dir.exists(), f"Output dir does not exist: {out_dir}"
    assert (out_dir / "chunks").exists(), f"Missing chunks dir in: {out_dir}"
    assert (out_dir / "normalization_baselines.csv").exists(), f"Missing normalization_baselines.csv in: {out_dir}"
    assert (out_dir / "chunks_metadata.json").exists(), f"Missing chunks_metadata.json in: {out_dir}"
    assert (out_dir / "normalized_chunk_index.json").exists(), f"Missing normalized_chunk_index.json in: {out_dir}"


def _assert_normalized_chunk_index_contract(payload: dict) -> None:
    assert isinstance(payload, dict), "normalized_chunk_index.json must be a JSON object"
    assert payload.get("focus_event_label") == 0, "focus_event_label must equal 0"
    assert isinstance(payload.get("event0_chunk_ids"), list), "event0_chunk_ids must be a list"
    chunks = payload.get("chunks")
    assert isinstance(chunks, list), "chunks must be a list"

    required_chunk_keys = {
        "chunk_id",
        "chunk_path",
        "label",
        "label_name",
        "duration_s",
        "start_time_s",
        "end_time_s",
        "buses_available",
        "bus_columns",
    }
    for i, entry in enumerate(chunks, start=1):
        missing = required_chunk_keys - set(entry.keys())
        assert not missing, f"normalized index entry #{i} missing keys: {sorted(missing)}"


def _assert_baselines_contract(df: pd.DataFrame) -> None:
    required_columns = {
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
    missing = required_columns - set(df.columns)
    assert not missing, f"normalization_baselines.csv missing columns: {sorted(missing)}"
    assert not df.empty, "normalization_baselines.csv is empty"


@pytest.mark.regression
def test_m1_migrated_pipeline_matches_legacy_artifacts(raw_small_dir: Path) -> None:
    out_dir = _workspace_test_dir("m1_regression")

    run_normalize_data_use_case(
        input_dir=raw_small_dir,
        output_dir=out_dir,
        generate_plots=False,
    )
    _assert_required_artifacts_exist(out_dir)

    migrated_baselines = pd.read_csv(out_dir / "normalization_baselines.csv")
    migrated_meta = _load_json(out_dir / "chunks_metadata.json")
    migrated_index = _load_json(out_dir / "normalized_chunk_index.json")
    migrated_csvs = sorted((out_dir / "chunks").glob("chunk*/Bus*_normalized.csv"))
    migrated_csv_samples = [
        pd.read_csv(path) for path in migrated_csvs[: min(2, len(migrated_csvs))]
    ]

    _assert_baselines_contract(migrated_baselines)
    _assert_normalized_chunk_index_contract(migrated_index)

    run_normalization_pipeline(
        input_dir=str(raw_small_dir),
        output_dir=str(out_dir),
        generate_plots=False,
    )
    _assert_required_artifacts_exist(out_dir)

    legacy_baselines = pd.read_csv(out_dir / "normalization_baselines.csv")
    legacy_meta = _load_json(out_dir / "chunks_metadata.json")
    legacy_index = _load_json(out_dir / "normalized_chunk_index.json")
    legacy_csvs = sorted((out_dir / "chunks").glob("chunk*/Bus*_normalized.csv"))

    _assert_baselines_contract(legacy_baselines)
    _assert_normalized_chunk_index_contract(legacy_index)

    assert_frame_equal(migrated_baselines, legacy_baselines, check_dtype=False, check_like=False)
    assert migrated_meta == legacy_meta, "chunks_metadata.json differs from legacy"
    assert migrated_index == legacy_index, "normalized_chunk_index.json differs from legacy"
    assert len(migrated_csv_samples) == min(2, len(legacy_csvs)), "Representative CSV sample mismatch"
    for i, migrated_df in enumerate(migrated_csv_samples):
        assert_frame_equal(migrated_df, pd.read_csv(legacy_csvs[i]), check_dtype=False, check_like=False)
