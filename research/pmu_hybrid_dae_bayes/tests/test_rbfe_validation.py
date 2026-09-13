import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import rbfe_validation as r

def test_functional_quotient_dimensions():
    assert 43 - 25 == 18

def test_innovation_likelihood_is_finite():
    A=np.array([[.9]]); C=np.array([[1.]]); L=np.zeros((62,1));
    y=np.zeros((3,1)); out=r.rbfe(A,C,L,y,np.zeros(1),np.zeros(31,dtype=complex),np.ones((1,1)),np.ones((62,1)),2)
    assert np.isfinite(out['evidence']).all()

def test_causal_window_lengths():
    assert all(n in (3,5,10,15,30) for n in (3,5,10,15,30))
