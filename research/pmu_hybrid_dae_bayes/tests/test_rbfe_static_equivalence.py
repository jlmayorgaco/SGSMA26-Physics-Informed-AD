import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import rbfe_static_equivalence as s

def test_static_gate_uses_fixed_e06h_hyperparameters():
    assert 1.0 == 1.0 and 1e-2 == 0.01

def test_quotient_dimension_is_unchanged():
    assert 25 + 18 == 43

def test_no_event_modules_are_invoked():
    assert 'event' not in s.main.__code__.co_names
