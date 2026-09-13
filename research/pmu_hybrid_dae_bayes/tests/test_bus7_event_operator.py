from pathlib import Path
import pandas as pd

BASE=Path(__file__).resolve().parents[1]/'powerdynamics_ieee39/output/results/full_field_event_reconstruction_v1'
AUDIT=Path(__file__).resolve().parents[1]/'powerdynamics_ieee39/output/full_field_event_reconstruction_v1/results'

def test_bus7_ten_percent_sign_semantics():
    m=pd.read_csv(BASE/'bus7_event_metadata.csv').iloc[0]
    assert m.p_post < m.p_pre and abs(m.p_post/m.p_pre-1.1)<1e-6
    assert m.q_post < m.q_pre and abs(m.q_post/m.q_pre-1.1)<1e-6

def test_native_event_replay_is_deterministic():
    r=pd.read_csv(AUDIT/'bus7_event_replay.csv').iloc[0]
    assert r.status == 'PASS' and r.max_numeric_replay_error == 0

def test_event_is_after_pre_event_interval():
    tr=pd.read_csv(BASE/'bus7_full_truth.csv')
    assert tr.time.min() < 2.0 < tr.time.max()
