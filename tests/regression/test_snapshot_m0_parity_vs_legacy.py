from __future__ import annotations

from contextlib import ExitStack, contextmanager
import importlib
import json
import shutil
from pathlib import Path
import sys
from typing import Any, Iterator

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from src.application.use_cases.chunk_raw_data import run_chunk_raw_data_use_case


@contextmanager
def _temporary_attr(obj: Any, attr: str, value: Any) -> Iterator[None]:
    had_attr = hasattr(obj, attr)
    old_value = getattr(obj, attr, None)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        if had_attr:
            setattr(obj, attr, old_value)
        else:
            delattr(obj, attr)


def _run_legacy_m0(input_dir: Path, out_dir: Path) -> None:
    workspace_root = Path(__file__).resolve().parents[3]
    root_str = str(workspace_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        m0 = importlib.import_module("m0_chunks")
    except ModuleNotFoundError as exc:
        if exc.name == "m0_chunks":
            pytest.skip("legacy m0_chunks.py is not present in this checkout")
        raise
    with ExitStack() as stack:
        stack.enter_context(_temporary_attr(m0, "INPUT_DIR", str(input_dir)))
        stack.enter_context(_temporary_attr(m0, "OUTPUT_DIR", str(out_dir)))
        stack.enter_context(_temporary_attr(m0, "CHUNKS_DIR", str(out_dir / "chunks")))
        stack.enter_context(
            _temporary_attr(
                m0,
                "RAW_SANITY_SUMMARY_CSV",
                str(out_dir / "raw_signal_sanity_summary.csv"),
            )
        )
        stack.enter_context(
            _temporary_attr(
                m0,
                "CHUNK0_INDEX_JSON",
                str(out_dir / "chunk0_index.json"),
            )
        )
        m0.process_pipeline(enable_sanity_check=True, generate_plots=False)


def _print_tree(root: Path, label: str) -> None:
    print(f"\n[{label}] {root}")
    if not root.exists():
        print("  (missing)")
        return

    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        kind = "DIR " if path.is_dir() else "FILE"
        print(f"  {kind}  {rel}")


def _assert_required_artifacts_exist(out_dir: Path) -> None:
    assert out_dir.exists(), f"Output dir does not exist: {out_dir}"
    assert (out_dir / "chunks").exists(), f"Missing chunks dir in: {out_dir}"
    assert (out_dir / "chunks_metadata.json").exists(), f"Missing chunks_metadata.json in: {out_dir}"
    assert (out_dir / "chunk0_index.json").exists(), f"Missing chunk0_index.json in: {out_dir}"
    assert (out_dir / "raw_signal_sanity_summary.csv").exists(), (
        f"Missing raw_signal_sanity_summary.csv in: {out_dir}"
    )


def _load_json(path: Path) -> dict:
    assert path.exists(), f"JSON file not found: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_chunks_metadata_contract(meta: dict) -> None:
    assert isinstance(meta, dict), "chunks_metadata.json must be a JSON object"
    assert "chunks" in meta, "chunks_metadata.json must contain key 'chunks'"
    assert isinstance(meta["chunks"], list), "'chunks' must be a list"
    assert len(meta["chunks"]) > 0, "Expected at least one chunk in chunks_metadata.json"

    required_keys = {
        "chunk_order",
        "label",
        "label_name",
        "chunk_type",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "affected_buses",
        "labels_present",
        "per_bus_labels",
        "chunk_dir",
    }

    for i, chunk in enumerate(meta["chunks"], start=1):
        missing = required_keys - set(chunk.keys())
        assert not missing, f"Chunk #{i} missing keys: {sorted(missing)}"
        assert chunk["end_time_s"] >= chunk["start_time_s"], (
            f"Chunk #{i} has invalid time interval: {chunk['start_time_s']} -> {chunk['end_time_s']}"
        )
        assert chunk["duration_s"] >= 0, f"Chunk #{i} has negative duration"
        assert isinstance(chunk["affected_buses"], list), f"Chunk #{i} affected_buses must be a list"
        assert isinstance(chunk["labels_present"], list), f"Chunk #{i} labels_present must be a list"
        assert isinstance(chunk["per_bus_labels"], dict), f"Chunk #{i} per_bus_labels must be a dict"


def _assert_chunk0_contract(chunk0: dict) -> None:
    assert isinstance(chunk0, dict), "chunk0_index.json must be a JSON object"
    assert "event_label" in chunk0, "chunk0_index.json must contain event_label"
    assert "chunks" in chunk0, "chunk0_index.json must contain chunks"
    assert chunk0["event_label"] == 0, "chunk0_index.json event_label must be 0"
    assert isinstance(chunk0["chunks"], list), "chunk0_index.json 'chunks' must be a list"

    for i, item in enumerate(chunk0["chunks"], start=1):
        for key in ["chunk_order", "chunk_dir", "start_time_s", "end_time_s", "duration_s", "affected_buses"]:
            assert key in item, f"chunk0 item #{i} missing key: {key}"


def _assert_sanity_contract(df: pd.DataFrame) -> None:
    required_columns = {
        "chunk_order",
        "chunk_dir",
        "bus_id",
        "dominant_event_label",
        "dominant_event_name",
        "timestamp_start_s",
        "timestamp_end_s",
        "n_rows",
        "signal_name",
        "n_valid",
        "n_nan",
        "nan_pct",
        "mean",
        "std",
        "min",
        "max",
        "p01",
        "p50",
        "p99",
    }

    missing = required_columns - set(df.columns)
    assert not missing, f"raw_signal_sanity_summary.csv missing columns: {sorted(missing)}"
    assert len(df) > 0, "raw_signal_sanity_summary.csv is empty"
    assert (df["n_rows"] >= 0).all(), "Found negative n_rows"
    assert (df["n_valid"] >= 0).all(), "Found negative n_valid"
    assert (df["n_nan"] >= 0).all(), "Found negative n_nan"
    assert ((df["nan_pct"] >= 0) & (df["nan_pct"] <= 100)).all(), "nan_pct must be in [0, 100]"


def _assert_chunk_dir_names_match_metadata(meta: dict, out_dir: Path) -> None:
    chunks_dir = out_dir / "chunks"
    actual_dirs = sorted(p.name for p in chunks_dir.iterdir() if p.is_dir())
    expected_dirs = sorted(chunk["chunk_dir"] for chunk in meta["chunks"])
    assert actual_dirs == expected_dirs, (
        "Chunk directory names differ from chunks_metadata.json\n"
        f"Actual:   {actual_dirs}\n"
        f"Expected: {expected_dirs}"
    )


def _persist_debug_outputs(migrated_out: Path, legacy_out: Path, debug_root: Path) -> None:
    if debug_root.exists():
        shutil.rmtree(debug_root)
    debug_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(migrated_out, debug_root / "migrated")
    shutil.copytree(legacy_out, debug_root / "legacy")


@pytest.mark.regression
def test_m0_migrated_pipeline_matches_legacy_artifacts(
    raw_small_dir: Path,
    tmp_path: Path,
) -> None:
    migrated_out = tmp_path / "migrated"
    legacy_out = tmp_path / "legacy"

    print(f"\nraw_small_dir = {raw_small_dir}")
    print(f"tmp_path      = {tmp_path}")
    print(f"migrated_out  = {migrated_out}")
    print(f"legacy_out    = {legacy_out}")

    run_chunk_raw_data_use_case(
        input_dir=str(raw_small_dir),
        output_dir=str(migrated_out),
        enable_sanity_check=True,
        generate_plots=False,
    )
    _run_legacy_m0(raw_small_dir, legacy_out)

    _print_tree(migrated_out, "MIGRATED TREE")
    _print_tree(legacy_out, "LEGACY TREE")

    _assert_required_artifacts_exist(migrated_out)
    _assert_required_artifacts_exist(legacy_out)

    migrated_meta = _load_json(migrated_out / "chunks_metadata.json")
    legacy_meta = _load_json(legacy_out / "chunks_metadata.json")

    _assert_chunks_metadata_contract(migrated_meta)
    _assert_chunks_metadata_contract(legacy_meta)

    migrated_chunk0 = _load_json(migrated_out / "chunk0_index.json")
    legacy_chunk0 = _load_json(legacy_out / "chunk0_index.json")

    _assert_chunk0_contract(migrated_chunk0)
    _assert_chunk0_contract(legacy_chunk0)

    migrated_sanity = pd.read_csv(migrated_out / "raw_signal_sanity_summary.csv")
    legacy_sanity = pd.read_csv(legacy_out / "raw_signal_sanity_summary.csv")

    _assert_sanity_contract(migrated_sanity)
    _assert_sanity_contract(legacy_sanity)

    _assert_chunk_dir_names_match_metadata(migrated_meta, migrated_out)
    _assert_chunk_dir_names_match_metadata(legacy_meta, legacy_out)

    # Extra invariants before exact parity
    assert len(migrated_meta["chunks"]) == len(legacy_meta["chunks"]), (
        "Different number of chunks between migrated and legacy"
    )
    assert len(migrated_chunk0["chunks"]) == len(legacy_chunk0["chunks"]), (
        "Different number of event-0 chunks between migrated and legacy"
    )
    assert list(migrated_sanity.columns) == list(legacy_sanity.columns), (
        "Sanity CSV columns differ between migrated and legacy"
    )
    assert migrated_sanity.shape == legacy_sanity.shape, (
        f"Sanity CSV shape differs. "
        f"migrated={migrated_sanity.shape}, legacy={legacy_sanity.shape}"
    )

    # Final strict parity checks
    assert migrated_meta == legacy_meta, "chunks_metadata.json differs from legacy"
    assert migrated_chunk0 == legacy_chunk0, "chunk0_index.json differs from legacy"
    assert_frame_equal(
        migrated_sanity,
        legacy_sanity,
        check_dtype=False,
        check_like=False,
    )

    # Optional: persist debug outputs for manual inspection
    debug_root = Path("debug_parity_outputs") / "m0"
    _persist_debug_outputs(migrated_out, legacy_out, debug_root)
    print(f"\nDebug parity artifacts copied to: {debug_root.resolve()}")
