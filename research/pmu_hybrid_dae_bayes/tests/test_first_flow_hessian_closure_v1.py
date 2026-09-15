from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "powerdynamics_ieee39" / "output" / "first_flow_hessian_closure_v1"
RES = OUT / "results"


def test_complete_state_map_inventory():
    m = pd.read_csv(RES / "first_flow_state_map_manifest.csv")
    assert len(m) == 3072
    assert set(m.n_state) == {192}
    assert bool(m.finite.all())
    assert set(m.op_tag) == {"op_m035", "op_m085", "op_m125"}


def test_all_source_derivative_rows_and_finite_values():
    d = pd.read_csv(RES / "derivative_validation.csv")
    assert len(d[d.derivative_order == 1]) == 48
    assert len(d[(d.derivative_order == 2) & d.direction.str.startswith("self")]) == 48
    assert len(d[(d.derivative_order == 2) & d.direction.str.startswith("cross")]) == 360
    assert np.isfinite(d[["relative_error", "cosine", "uncertainty_relative"]].to_numpy()).all()
    assert d[d.derivative_order == 1].relative_error.max() < 1e-5


def test_ad_probe_and_corrected_dictionary_contract():
    ad = pd.read_csv(RES / "ad_feasibility.csv")
    assert ad.iloc[0].status == "NOT_SUPPORTED"
    for op in ("op_m035", "op_m085", "op_m125"):
        z = np.load(RES / f"corrected_dictionary_{op}.npz")
        assert z["D"].shape == (3840, 16)
        assert z["Q"].shape == (3840, 16)
        assert z["Qcross"].shape == (3840, 120)
        assert np.isfinite(z["D"]).all() and np.isfinite(z["Q"]).all() and np.isfinite(z["Qcross"]).all()


def test_replay_is_stored_tds_only_and_v3_not_claimed():
    r = pd.read_csv(RES / "corrected_t120_replay.csv")
    assert not r.empty and (r.new_tds_generated == False).all()
    report = (OUT / "reports" / "first_flow_hessian_closure_v1.md").read_text(encoding="utf-8")
    assert "V3_READINESS = NOT_READY" in report
