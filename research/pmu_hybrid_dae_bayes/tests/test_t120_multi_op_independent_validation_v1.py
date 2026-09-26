from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "powerdynamics_ieee39" / "output" / "t120_multi_op_independent_validation_v1"
RES = OUT / "results"

def test_distance_contract_is_explicit():
    d = pd.read_csv(RES / "canonical_distance_definitions.csv")
    assert set(["CASE_PROFILED_MAHALANOBIS_SQ", "DICTIONARY_SPARSE_MARGIN"]).issubset(set(d.name))
    m = pd.read_csv(RES / "historical_distance_mapping.csv")
    assert set(m.canonical_name) >= {"CASE_PROFILED_MAHALANOBIS_SQ", "DICTIONARY_SPARSE_MARGIN"}

def test_fresh_bank_and_exclusion_manifest():
    op = pd.read_csv(RES / "operating_points.csv")
    ex = pd.read_csv(RES / "v3_exclusion_manifest_delta.csv")
    assert set(op.op_tag) == {"op_m035", "op_m085", "op_m125"}
    assert len(ex) >= 3 * 16
    assert ex.excluded_from_v3.all()
    assert ex.sha256.str.len().eq(64).all()

def test_information_and_eta_horizons_are_finite():
    info = pd.read_csv(RES / "information_growth_fresh.csv")
    assert list(info.horizon) == [30, 45, 60, 90, 120]
    assert info.median_delta_sq.diff().dropna().ge(-1e-10).all()
    eta = pd.read_csv(RES / "eta_model_distribution.csv")
    assert set(eta.horizon) == {30, 45, 60, 90, 120}
    assert eta.eta_model.notna().all()

