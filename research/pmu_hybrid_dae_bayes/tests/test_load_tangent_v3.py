from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'powerdynamics_ieee39' / 'output' / 'load_tangent_v3'
RES = ROOT / 'results'


def test_v3_atlas_and_event_semantics_are_frozen():
    a = pd.read_csv(RES / 'load_event_semantics.csv')
    assert a.iloc[0].classification == 'TRUE_TIME_LOCAL'
    m = pd.read_csv(RES / 'load_native_state_order.csv')
    assert len(m) == 192


def test_composed_pmu_jacobian_has_tight_fd_parity():
    z = np.load(RES / 'load_native_C_pmu.npz')
    assert z['C'].shape == (32, 78)
    a = pd.read_csv(RES / 'load_C_audit.csv')
    assert a.relative_frobenius_difference.iloc[-1] < 1e-8


def test_missing_native_parameter_derivatives_block_tangent():
    t = pd.read_csv(RES / 'load_tangent_metrics_v3.csv')
    assert len(t) == 16
    assert set(t.status) == {'NOT_RUN'}
    b = np.load(RES / 'load_native_Bg.npz')
    assert b['status'] == 'NOT_EXPORTED'
