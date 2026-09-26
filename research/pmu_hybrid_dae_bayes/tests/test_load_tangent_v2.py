from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'powerdynamics_ieee39' / 'output' / 'load_tangent_v2'
RES = ROOT / 'results'


def test_central_fd_reference_has_all_candidates_and_full_pmu_channels():
    z = np.load(RES / 'load_fd_central_operator.npz')
    assert z['central'].shape == (16, 30, 32)
    assert float(z['epsilon']) == 0.005
    c = pd.read_csv(RES / 'load_fd_central_consistency.csv')
    assert len(c) == 16
    assert c.central_vs_plus_relerr.max() < 0.01
    assert c.plus_minus_cosine.min() > 0.999


def test_descriptor_inventory_is_frozen_but_tangent_is_not_promoted():
    m = pd.read_csv(RES / 'load_descriptor_metadata.csv')
    assert len(m) == 192
    assert (m.kind == 'differential').sum() == 114
    assert (m.kind == 'algebraic_zero_mass').sum() == 78
    t = pd.read_csv(RES / 'load_tangent_metrics.csv')
    assert len(t) == 16
    assert set(t.status) == {'NOT_RUN'}


def test_remaining_requests_are_rejected_without_false_independence():
    m = pd.read_csv(RES / 'load_atlas_manifest_v2.csv')
    assert (m.status == 'EXECUTED_SUCCESS').sum() == 64
    assert (m.status == 'REJECTED_DETERMINISTIC_DUPLICATE').sum() == 48
