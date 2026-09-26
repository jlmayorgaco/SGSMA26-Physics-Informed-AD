import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_delta_i_identity():
    rng=np.random.default_rng(1); Y=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)); d=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)); v=rng.normal(size=4)+1j*rng.normal(size=4)
    assert np.allclose((Y+d)@v-Y@v,d@v)

def test_whitened_quotient_orthogonality():
    q,_=np.linalg.qr(np.random.default_rng(2).normal(size=(10,3))); u,s,v=np.linalg.svd(q,full_matrices=True)
    assert np.linalg.norm(u[:,3:].T@q) < 1e-10

def test_posterior_normalization():
    z=np.array([0.2,0.8]); p=np.exp(z-np.max(z)); assert np.isclose((p/p.sum()).sum(),1.0)
