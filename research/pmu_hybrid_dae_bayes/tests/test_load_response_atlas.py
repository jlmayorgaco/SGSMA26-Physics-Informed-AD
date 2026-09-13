from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'powerdynamics_ieee39' / 'output' / 'load_response_atlas_v1'
RES = ROOT / 'results'

def test_native_atlas_has_two_complete_candidate_groups():
    m = pd.read_csv(ROOT / 'simulation_manifest_native.csv')
    assert len(m) == 20
    assert set(m.candidate_bus) == {7, 12}
    assert set(m.status) == {'EXECUTED_SUCCESS'}

def test_fd_operator_is_voltage_only_and_dimensioned():
    o = pd.read_parquet(RES / 'fd_candidate_operators.parquet')
    assert set(o.channel) <= set(range(16))
    assert set(o.candidate_bus) == {7, 12}
    assert len(o) == 2 * 4 * 30 * 16

def test_fd_derivative_converges_and_units_are_fractional():
    c = pd.read_csv(RES / 'fd_derivative_convergence.csv')
    one = c[c.comparison == 'one_sided']
    assert (one.central_vs_plus_relerr < 0.01).all()
    assert (one.central_vs_minus_relerr < 0.01).all()
    assert '0.10 = 10%' in (ROOT / 'reports' / 'temporal_operator_unit_audit.md').read_text()

def test_tangent_not_claimed_without_native_state_export():
    t = pd.read_csv(RES / 'tangent_vs_fd.csv').iloc[0]
    assert t.status == 'NOT_IMPLEMENTED'

def test_atlas_does_not_promote_unvalidated_evi():
    s = pd.read_csv(RES / 'atlas_summary.csv').iloc[0]
    assert s.temporal_operator_validation == 'PARTIAL'
    assert s.source_inference == 'NOT_RUN'
