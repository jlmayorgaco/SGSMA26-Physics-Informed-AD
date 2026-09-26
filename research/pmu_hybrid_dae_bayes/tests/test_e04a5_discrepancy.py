from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pmu_hybrid.e04a5_discrepancy import (
    marginal_hidden_covariance, conditional_physical_covariance,
    build_physical_dictionary, propagate_hidden_process_covariance,
    augmented_process_prediction,
    hidden_component_nees,
)


def test_augmented_marginal_is_not_conditional():
    Pxx = np.array([[2.0, .2], [.2, 1.0]])
    Pxc = np.array([[.6], [.1]])
    Pcc = np.array([[1.0]])
    P = np.block([[Pxx, Pxc], [Pxc.T, Pcc]])
    L = np.eye(2)
    got = marginal_hidden_covariance(L, P, physical_dim=2)
    cond = conditional_physical_covariance(Pxx, Pxc, Pcc)
    assert np.allclose(got, Pxx)
    assert not np.allclose(got, cond)


def test_physical_dictionary_dimensions_and_labels():
    root = Path(__file__).parents[1] / "powerdynamics_ieee39" / "output" / "results"
    B, table = build_physical_dictionary(root / "pd_descriptor_inventory.csv")
    assert B.shape[0] == 114 and B.shape[1] >= 10
    assert np.allclose(np.linalg.norm(B, axis=0), 1.0)
    assert table.direction_id.is_unique
    assert {"governor_mechanical_reference", "load_active_power_equivalent"}.issubset(set(table.type))


def test_process_forcing_reaches_hidden_marginal():
    Pxx = np.eye(2) * .1; Pxc = np.array([[.02], [.0]]); Pcc = np.array([[.5]])
    Bc = np.array([[1.], [0.]]); Qc = np.array([[.2]])
    L = np.array([[1., 0.]])
    out = propagate_hidden_process_covariance(Pxx, Pxc, Pcc, Bc, Qc, L)
    assert out[0, 0] > Pxx[0, 0]


def test_cal_manifest_has_no_test_seed_overlap():
    root = Path(__file__).parents[1] / "powerdynamics_ieee39" / "output" / "results"
    cal = pd.read_csv(root / "e04a3_cal_split_manifest.csv")
    frozen = pd.read_csv(root / "e04_pd_dataset.csv")
    assert set(cal.seed).isdisjoint(set(frozen.loc[frozen.split == "TEST", "seed"]))


def test_augmented_process_kalman_prediction_reference():
    A = np.array([[.9]]); B = np.array([[.5]]); F = np.array([[.8]])
    P = np.diag([.2, .3])
    out = augmented_process_prediction(A, B, F, np.array([[.01]]), np.array([[.04]]), P)
    expected = np.array([[.9*.9*.2 + .5*.5*.3 + .01, .5*.8*.3],
                         [.5*.8*.3, .8*.8*.3 + .04]])
    assert np.allclose(out, expected)


def test_hidden_nees_toy_uses_marginal_diagonal():
    e = np.array([[[2.0, 0.0]]])
    cov = np.array([[[[4.0, 0.0], [0.0, 9.0]]]])
    n = hidden_component_nees(e, cov)
    assert n[0, 0, 0] == pytest.approx(1.0)
