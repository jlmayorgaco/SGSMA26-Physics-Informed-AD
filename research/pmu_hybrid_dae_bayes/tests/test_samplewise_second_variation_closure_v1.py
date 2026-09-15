"""Regression checks for SAMPLEWISE-SECOND-VARIATION-CLOSURE-V1."""
from pathlib import Path
import hashlib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39" / "output" / "samplewise_second_variation_closure_v1"
RES = ROOT / "results"


def test_preregistered_direction_and_hash_are_frozen():
    mf = pd.read_csv(RES / "samplewise_manifest.csv")
    assert len(mf) == 144
    assert set(mf.kind) == {"self", "cross"}
    assert set(mf[mf.kind == "self"].bus_i.astype(int)) == {7, 26, 3, 16}
    assert set(mf[mf.kind == "cross"].support) == {"7-12", "26-28", "3-18", "16-18"}
    digest = hashlib.sha256((RES / "preregistration_manifest.csv").read_bytes()).hexdigest()
    assert digest == (RES / "preregistration_manifest.sha256").read_text().strip()


def test_only_frozen_fine_stencil_was_executed_successfully():
    ex = pd.read_csv(RES / "tds_execution_manifest.csv")
    assert len(ex) == 72
    assert set(ex.status) == {"EXECUTED_SUCCESS"}
    assert ex.point_id.is_unique


def test_samplewise_contract_shape_and_first_sample_finiteness():
    d = pd.read_csv(RES / "samplewise_hessian_comparison.csv")
    assert len(d) == 3 * 8 * 45
    assert set(d.sample_index) == set(range(1, 46))
    assert d.time.min() > 2.0
    assert np.isfinite(d[["raw_error_norm", "whitened_error_norm", "relative_error"]].to_numpy()).all()
    first = d[d.sample_index == 1]
    assert len(first) == 24


def test_required_review_artifacts_and_no_cubic_dictionary():
    required = [
        "production_first_step_derivatives.csv",
        "aligned_vs_existing_vs_tds.csv",
        "homogeneous_error_dynamics.csv",
        "numerical_uncertainty.csv",
        "first_divergent_sample.csv",
        "asymptotic_A2_A3_A4.csv",
    ]
    assert all((RES / f).exists() for f in required)
    a2 = pd.read_csv(RES / "asymptotic_A2_A3_A4.csv")
    assert "fit_status" in a2
    # The audit must not materialize a cubic estimator/dictionary.
    assert not any("cubic" in p.name.lower() and p.suffix in {".npz", ".npy"} for p in ROOT.rglob("*"))

