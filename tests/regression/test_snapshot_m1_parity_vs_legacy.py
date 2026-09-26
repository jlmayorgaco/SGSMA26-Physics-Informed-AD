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
    root = Path("output") / "_test_runs" / f"{prefix}_{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_required_artifacts_exist(out_dir: Path) -> None:
    assert (out_dir / "normalization_baselines.csv").exists()
    assert (out_dir / "chunks_metadata.json").exists()
    assert (out_dir / "normalized_chunk_index.json").exists()
    assert (out_dir / "chunks").exists()


def _normalize_chunk_paths(payload: dict) -> dict:
    # chunk_path embeds output root (migrated vs legacy). Normalize for portable parity assertion.
    payload = dict(payload)
    chunks = []
    for item in payload.get("chunks", []):
        row = dict(item)
        row["chunk_path"] = str(Path(row["chunk_path"]).name)
        chunks.append(row)
    payload["chunks"] = chunks
    return payload


@pytest.mark.regression
def test_m1_migrated_pipeline_matches_legacy_artifacts(raw_small_dir: Path) -> None:
    root = _workspace_test_dir("m1_regression")
    migrated_out = root / "migrated"
    legacy_out = root / "legacy"

    run_normalize_data_use_case(
        input_dir=raw_small_dir,
        output_dir=migrated_out,
        generate_plots=False,
        feature_mode="legacy_replace_angles",
    )
    run_normalization_pipeline(
        input_dir=str(raw_small_dir),
        output_dir=str(legacy_out),
        generate_plots=False,
    )

    _assert_required_artifacts_exist(migrated_out)
    _assert_required_artifacts_exist(legacy_out)

    assert_frame_equal(
        pd.read_csv(migrated_out / "normalization_baselines.csv"),
        pd.read_csv(legacy_out / "normalization_baselines.csv"),
        check_dtype=False,
        check_like=False,
    )

    migrated_meta = _load_json(migrated_out / "chunks_metadata.json")
    legacy_meta = _load_json(legacy_out / "chunks_metadata.json")
    assert migrated_meta == legacy_meta

    migrated_index = _load_json(migrated_out / "normalized_chunk_index.json")
    legacy_index = _load_json(legacy_out / "normalized_chunk_index.json")
    assert _normalize_chunk_paths(migrated_index) == _normalize_chunk_paths(legacy_index)

    migrated_csvs = sorted((migrated_out / "chunks").glob("chunk*/Bus*_normalized.csv"))
    legacy_csvs = sorted((legacy_out / "chunks").glob("chunk*/Bus*_normalized.csv"))
    assert migrated_csvs and legacy_csvs
    assert len(migrated_csvs) == len(legacy_csvs)
    for i in range(min(2, len(migrated_csvs))):
        assert_frame_equal(
            pd.read_csv(migrated_csvs[i]),
            pd.read_csv(legacy_csvs[i]),
            check_dtype=False,
            check_like=False,
        )
