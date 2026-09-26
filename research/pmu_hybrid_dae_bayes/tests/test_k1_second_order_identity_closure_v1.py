from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39"
OUT = ROOT / "output" / "k1_second_order_identity_closure_v1"
RES = OUT / "results"


def test_k1_artifacts_exist_and_are_frozen():
    for name in (
        "production_output_identity.csv",
        "same_stencil_state_output_hessian.csv",
        "measurement_chain_rule.csv",
        "factor_convention.csv",
        "a2_deltaq_identity.csv",
        "k1_manifest.csv",
    ):
        assert (RES / name).exists(), name


def test_value_identity_and_factor_contract():
    ident = pd.read_csv(RES / "production_output_identity.csv")
    assert len(ident) == 48
    assert ident.max_abs_state_to_canonical.max() < 2e-10
    assert ident.independent_production_output_available.sum() == 48
    factors = pd.read_csv(RES / "factor_convention.csv")
    assert set(factors.check) == {"PASS"}
    assert set(factors[factors.term.str.startswith("self")].factor) == {0.5}
    assert set(factors[factors.term.str.startswith("cross")].factor) == {1.0}


def test_chain_rule_and_same_stencil_coverage_are_explicit():
    chain = pd.read_csv(RES / "measurement_chain_rule.csv")
    assert set(chain[chain.term.isin(["h_p", "h_up", "h_pu"])].max_abs) == {0.0}
    same = pd.read_csv(RES / "same_stencil_state_output_hessian.csv")
    assert same.independent_output.sum() == 12
    assert (same[~same.independent_output].status == "NOT_ASSESSED_MATCHING_STENCIL").all()


def test_a2_self_identity_and_cross_limit_are_not_conflated():
    a2 = pd.read_csv(RES / "a2_deltaq_identity.csv")
    assert len(a2) == 480
    self_rows = a2[a2.kind == "self"]
    assert (self_rows.groupby("sample_index").cosine.median() > 0.999).all()
    cross_rows = a2[a2.kind == "cross"]
    assert (cross_rows.uncertainty.str.contains("two scales")).all()


def test_k1_script_does_not_contain_tds_execution():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "k1_second_order_identity_closure_v1.py").read_text()
    assert "subprocess.run" not in script
    assert "Julia" not in script
