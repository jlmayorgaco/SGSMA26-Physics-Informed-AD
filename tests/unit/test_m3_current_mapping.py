from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.calibration.current_mapping import choose_best_current_mapping_for_bus, select_current_mappings


def _workspace_dir(prefix: str) -> Path:
    root = Path("output") / "_test_runs" / f"{prefix}_{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _chunk_with_or_without_currents(include_currents: bool) -> list[dict]:
    cols = {"BUS10_VA_MAG": np.ones(30)}
    if include_currents:
        cols.update({"BUS10_IA_MAG": np.linspace(10, 12, 30), "BUS10_IB_MAG": np.linspace(11, 13, 30), "BUS10_IC_MAG": np.linspace(9, 11, 30)})
    df = pd.DataFrame(cols)
    return [{"chunk_id": "chunk01", "df": df}]


def test_choose_best_current_mapping_for_bus_returns_none_when_no_real_phases() -> None:
    best, rows = choose_best_current_mapping_for_bus("10", _chunk_with_or_without_currents(False), {"current_candidates": {}})
    assert best is None
    assert rows == []


def test_choose_best_current_mapping_for_bus_returns_sorted_rows() -> None:
    traj = {"current_candidates": {"c1": np.linspace(10, 12, 30), "c2": np.linspace(9, 11, 30)}}
    best, rows = choose_best_current_mapping_for_bus("10", _chunk_with_or_without_currents(True), traj)
    assert best is not None
    scores = [r["composite_score"] for r in rows]
    assert scores == sorted(scores)


def test_select_current_mappings_writes_selection_csv() -> None:
    out = _workspace_dir("m3_curr_map")
    all_chunks = {"10": _chunk_with_or_without_currents(True)}
    trajectories = {"10": {"current_candidates": {"c1": np.linspace(10, 12, 30)}}}
    select_current_mappings(all_chunks, trajectories, out)
    assert (out / "current_mapping_selection.csv").exists()


def test_selected_mapping_stored_per_bus() -> None:
    out = _workspace_dir("m3_curr_map")
    all_chunks = {"10": _chunk_with_or_without_currents(True)}
    trajectories = {"10": {"current_candidates": {"c1": np.linspace(10, 12, 30)}}}
    chosen = select_current_mappings(all_chunks, trajectories, out)
    assert "10" in chosen
