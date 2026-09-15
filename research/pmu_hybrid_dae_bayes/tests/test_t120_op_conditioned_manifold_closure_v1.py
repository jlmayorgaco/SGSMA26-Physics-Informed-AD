from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39"
OUT = ROOT / "output" / "t120_op_conditioned_manifold_closure_v1"
RES = OUT / "results"


def test_op_conditioned_exports_and_prefixes_are_complete():
    m = pd.read_csv(RES / "op_conditioned_dictionary_manifest.csv")
    assert set(m.op_tag) == {"op_m035", "op_m085", "op_m125"}
    assert m.available.all()
    assert (m.n_D == 3840).all() and (m.n_Q == 3840).all() and (m.n_Qij == 120).all()
    p = pd.read_csv(RES / "dictionary_prefix_consistency.csv")
    assert set(p.horizon) == {30, 45, 60, 90, 120}
    assert p.exact_prefix.all() and (~p.independent_refit).all()


def test_fixed_eta_tail_is_recomputed_without_dropping_cases():
    d = pd.read_csv(RES / "eta_model_fixed_vs_conditioned.csv")
    assert len(d) == 2640
    assert (d.eta_fixed > 1).sum() == 323
    assert ((d.eta_fixed > 1) & (d.eta_conditioned > 1)).sum() == 0


def test_gamma_positive_and_report_blocks_prospective_v3():
    g = pd.read_csv(RES / "resolvability_op_conditioned.csv")
    assert len(g) == 3
    assert (g.gamma4_infinity > 0).all()
    report = (OUT / "reports" / "t120_op_conditioned_manifold_closure_v1.md").read_text(encoding="utf-8")
    assert "V3_READINESS = NOT_READY" in report
    assert "no push" in report


def test_margin_reference_is_not_fabricated():
    d = pd.read_csv(RES / "model_vs_tds_margin.csv")
    assert d.J_TDS_reference.isna().all()
    assert (d.reference_status == "NOT_RECOMPUTED_THIS_RUN_HISTORICAL_REFERENCE_ONLY").all()
