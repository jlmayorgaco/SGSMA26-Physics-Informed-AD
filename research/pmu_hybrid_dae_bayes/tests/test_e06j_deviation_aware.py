import inspect
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import e06j_deviation_aware as e

def test_online_signature_has_no_truth_argument():
    assert 'truth' not in inspect.signature(e.extract).parameters

def test_extractor_is_causal():
    h=np.arange(20.,dtype=float).reshape(10,2); a=e.extract(h,'EMA',5); b=e.extract(np.vstack([h,[[999.,999.]]]),'EMA',5)
    assert not np.allclose(a,b)

def test_preregistered_grid_is_bounded_and_single_iteration():
    assert e.WINDOWS and e.CADENCES and e.EXTRACTORS
    assert all(int(x)>0 for x in e.WINDOWS+e.CADENCES)
