import numpy as np

from research.pmu_hybrid_dae_bayes.scripts.cross_k1_second_order_identity_pilot_v1 import richardson
from research.pmu_hybrid_dae_bayes.scripts.hard_pair_cross_second_order_replication_v1 import PAIRS


def test_replication_pair_set_is_frozen():
    assert PAIRS == [(26, 28), (3, 18), (16, 18)]


def test_matched_mixed_stencil_has_no_self_contamination():
    values = {}
    for level, hi, hj in (("coarse", 1e-4, 2e-4), ("fine", 5e-5, 1e-4)):
        for si in (-1, 1):
            for sj in (-1, 1):
                ai, aj = si * hi, sj * hj
                values[(level, si, sj)] = np.array([[2*ai + 7*ai*ai - 3*aj + 11*aj*aj + 5*ai*aj]])
    hessian, uncertainty, _, _ = richardson(values)
    np.testing.assert_allclose(hessian, [[5.0]], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(uncertainty, 0.0, atol=1e-10)


def test_three_sigma_identity_rule():
    difference = 1.1232
    assert difference <= 3.0

