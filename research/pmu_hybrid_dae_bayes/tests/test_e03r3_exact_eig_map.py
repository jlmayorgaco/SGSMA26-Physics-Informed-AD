from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "output" / "results"


def test_native_eig_as_parity():
    row = pd.read_csv(RESULTS / "e03r3_native_eig_parity.csv").iloc[0]
    assert row.parity == "PASS"
    assert row.frobenius_relative < 1e-10
    assert row.eigen_matching_max_abs < 1e-8


def test_zero_tf_fold_and_constraint_extraction():
    assert len(pd.read_csv(RESULTS / "e03r3_zero_tf_states.csv")) == 50
    assert len(pd.read_csv(RESULTS / "e03r3_state_constraints.csv")) == 10


def test_constraint_substitution_residual():
    row = pd.read_csv(RESULTS / "e03r3_lift_residuals.csv").iloc[0]
    assert row.C_R_l2 < 1e-10


def test_lift_dimensions_and_folded_residual_are_recorded():
    row = pd.read_csv(RESULTS / "e03r3_lift_residuals.csv").iloc[0]
    assert (row.original_x, row.original_y, row.state_dimension) == (220, 479, 160)
    assert np.isfinite(row.folded_algebraic_residual_l2)


def test_consistent_measurement_fd_is_small_at_intermediate_eps():
    frame = pd.read_csv(RESULTS / "e03r3_measurement_lift_fd.csv")
    assert frame.measurement_fd_error.iloc[1] < 1e-6


def test_no_input_native_timestamp_campaign_and_slope_are_recorded():
    frame = pd.read_csv(RESULTS / "e03r3_no_input_tds_convergence.csv")
    assert len(frame) == 6 and frame.tds_ok.all()
    assert 0.5 < frame.slope_p.iloc[0] < 1.5


def test_positive_mode_validation_is_explicit():
    row = pd.read_csv(RESULTS / "e03r3_positive_mode.csv").iloc[0]
    assert row.classification in {"NATIVE_MODE_CONFIRMED", "NATIVE_MODE_NOT_REPRODUCED"}


def test_native_positive_mode_value_is_not_tuned():
    row = pd.read_csv(RESULTS / "e03r3_positive_mode.csv").iloc[0]
    assert abs(row.native_real - 1.0327798136) < 1e-6
