"""Estimator registry for the POC ablation."""

from poc.estimators.estimator_e1_basic_lse import BasicLSEEstimator
from poc.estimators.estimator_e2_ekf import ExtendedKalmanEstimator
from poc.estimators.estimator_e3_ukf import UnscentedKalmanEstimator
from poc.estimators.estimator_e4_gnn import GraphNeuralImputerEstimator

ESTIMATORS = {
    "E1_LSE": BasicLSEEstimator,
    "E2_EKF": ExtendedKalmanEstimator,
    "E3_UKF": UnscentedKalmanEstimator,
    "E4_GNN": GraphNeuralImputerEstimator,
}

__all__ = [
    "BasicLSEEstimator",
    "ExtendedKalmanEstimator",
    "UnscentedKalmanEstimator",
    "GraphNeuralImputerEstimator",
    "ESTIMATORS",
]

