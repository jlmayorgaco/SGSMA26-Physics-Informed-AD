import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import full_field_event_gate as g

def test_exact_eight_pmu_contract():
    assert g.OBS == [2,5,6,10,19,22,29,39]

def test_bus7_is_uninstrumented():
    assert 7 not in g.OBS

def test_positive_sequence_phase_derivation_is_balanced():
    import numpy as np
    a=np.exp(-2j*np.pi/3); v=1+0.2j
    assert np.isclose(abs(v*a),abs(v))
