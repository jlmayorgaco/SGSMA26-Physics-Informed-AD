from pathlib import Path
import hashlib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39" / "output" / "targeted_nonlinear_margin_closure_v1"
RES = ROOT / "results"


def test_preregistration_hash_and_target_strata():
    digest = hashlib.sha256((RES / "preregistration_manifest.csv").read_bytes()).hexdigest()
    assert digest == (RES / "preregistration_manifest.sha256").read_text().strip()
    d = pd.read_csv(RES / "target_case_manifest.csv")
    assert len(d) == 48
    assert (d.eta_model > 1).sum() == 18
    assert ((d.eta_model > .5) & (d.eta_model <= 1)).sum() == 6
    assert ((d.eta_model > .1) & (d.eta_model <= .5)).sum() == 12
    assert (d.eta_model < .01).sum() == 12
    assert {"26-28", "3-18", "16-18", "7-12"}.issubset(set(d.true_support))


def test_stage_a_triangle_certificates():
    d = pd.read_csv(RES / "stage_a_center_check.csv")
    assert len(d) == 48 * 5
    assert d.triangle_pass.all()
    assert ((d.triangle_abs_gap <= d.triangle_bound + 1e-8)).all()


def test_new_trajectories_are_successful_and_hashed():
    d = pd.read_csv(RES / "tds_execution_manifest.csv")
    assert len(d) == 306
    assert set(d.status) == {"EXECUTED_SUCCESS"}
    dup = d.duplicate_of.fillna("")
    assert d.sha256.nunique() == 294
    assert int((dup != "").sum()) == 12
    for r in d.itertuples():
        p = Path(r.path)
        assert p.exists()
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r.sha256


def test_future_v3_exclusion_and_tail_classification():
    ex = pd.read_csv(RES / "v3_exclusion_manifest_additions.csv")
    cl = pd.read_csv(RES / "target_case_classification.csv")
    assert len(ex) == 306 and ex.future_v3_excluded.all()
    assert (cl[cl.eta_model > 1].classification == "D_RESIDUAL_MANIFOLD_ERROR").all()
    assert len(cl) == 48


def test_stage_b_activation_is_preregistered_subset():
    a = pd.read_csv(RES / "stage_b_activation.csv")
    p = pd.read_csv(RES / "stage_b_active_points.csv")
    frozen = pd.read_csv(RES / "local_stencil_manifest.csv")
    assert len(a) == 48 and int(a.stage_b_active.sum()) == 30
    keys = set(zip(frozen.op_tag, frozen.competitor_support, frozen.b1.round(14), frozen.b2.round(14)))
    assert all((r.op_tag, r.competitor_support, round(r.b1,14), round(r.b2,14)) in keys for r in p.itertuples())
