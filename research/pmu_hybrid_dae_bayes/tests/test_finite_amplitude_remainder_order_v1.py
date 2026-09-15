"""Contract checks for FINITE-AMPLITUDE-REMAINDER-ORDER-V1 artifacts."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39"
OUT = ROOT / "output" / "finite_amplitude_remainder_order_v1"
RES = OUT / "results"


def test_preregistration_and_selection_are_frozen():
    cfg = json.loads((RES / "preregistration_manifest.json").read_text())
    cases = pd.read_csv(RES / "selected_case_manifest.csv")
    assert cfg["target_counts"] == {"tail": 18, "mid": 6, "low": 6}
    assert cases.stratum.value_counts().to_dict() == {"tail": 18, "mid": 6, "low": 6}
    digest = hashlib.sha256((RES / "preregistration_manifest.csv").read_bytes()).hexdigest()
    assert digest == (RES / "preregistration_manifest.sha256").read_text().strip()


def test_homotopy_grid_and_no_duplicate_point_ids():
    m = pd.read_csv(RES / "amplitude_homotopy_manifest.csv")
    assert set(np.round(m.lambda_value.unique(), 6)) == {0.125, 0.25, 0.5, 0.75, 1.0}
    assert m.point_id.is_unique
    assert set(m.kind) == {"true", "competitor"}
    assert (m[m.precision == "tight"].lambda_value.isin([0.125, 0.25])).all()


def test_all_new_tds_points_completed_and_excluded():
    m = pd.read_csv(RES / "amplitude_homotopy_manifest.csv")
    ex = pd.read_csv(RES / "v3_exclusion_manifest_additions.csv")
    assert len(ex) == len(m)
    assert ex.future_v3_excluded.all()
    generated = ex[ex.new_tds]
    assert len(generated) == 284
    assert generated.status.str.contains("SUCCESS|CHECKPOINT").all()
    assert generated.sha256.str.len().gt(32).all()


def test_remainder_outputs_and_floor_are_finite():
    floor = pd.read_csv(RES / "numerical_floor.csv")
    r1 = pd.read_csv(RES / "first_order_remainder.csv")
    r2 = pd.read_csv(RES / "second_order_remainder.csv")
    ex = pd.read_csv(RES / "horizon_order_analysis.csv")
    assert len(floor) == 36 and np.isfinite(floor.norm_diff).all()
    assert len(r1) and len(r2) and np.isfinite(r1.norm).all() and np.isfinite(r2.norm).all()
    assert set(ex.horizon) == {30, 45, 60, 90, 120}


def test_no_cubic_model_or_estimator_mutation():
    # This campaign is diagnostic only: a cubic coefficient table may be
    # measured, but no accepted cubic dictionary is created.
    assert not (RES / "cubic_dictionary.npz").exists()
    cfg = json.loads((RES / "preregistration_manifest.json").read_text())
    assert "D" in cfg["frozen_models"] and "Q" in cfg["frozen_models"]
