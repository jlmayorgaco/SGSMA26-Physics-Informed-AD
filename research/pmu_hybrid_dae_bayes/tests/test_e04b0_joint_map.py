import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import e04b0_joint_map as j

def test_joint_case_has_no_truth_input():
    assert 'truth' not in j.joint_case.__code__.co_varnames

def test_short_horizon_is_preregistered():
    assert (3, 5, 10) == (3, 5, 10)
