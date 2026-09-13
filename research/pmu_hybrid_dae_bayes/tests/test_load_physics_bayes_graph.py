from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'powerdynamics_ieee39' / 'output' / 'load_event_physics_bayes_graph_v1'
RES = ROOT / 'results'

def test_final_package_freezes_all_16_candidates_and_native_cases():
    s = pd.read_csv(RES / 'campaign_summary.csv').iloc[0]
    assert int(s.candidate_count) == 16
    assert int(s.executed_native_trajectories) == 64
    assert s.LOAD_ATLAS_COMPLETE == 'PARTIAL'

def test_downstream_bayes_and_graph_claims_are_blocked_by_tangent_gate():
    s = pd.read_csv(RES / 'campaign_summary.csv').iloc[0]
    assert s.TRAJECTORY_TANGENT_VALIDATION == 'NOT_RUN'
    assert s.SOURCE_BAYES_EVIDENCE == 'NOT_RUN'
    assert s.GSP_EVENT_OPERATOR == 'NOT_SUPPORTED'

def test_fd_and_linearity_gates_are_materialized():
    s = pd.read_csv(RES / 'campaign_summary.csv').iloc[0]
    assert s.FD_REFERENCE_OPERATOR == 'PASS'
    assert s.LOAD_LOCAL_LINEARITY == 'PASS'
    a = pd.read_csv(RES / 'bus7_amplitude_validation.csv').iloc[0]
    assert abs(float(a.estimated_fraction) - .10) < .005
