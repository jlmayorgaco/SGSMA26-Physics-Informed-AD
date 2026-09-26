"""Read-only contracts for MULTI-LIKELIHOOD-CALIBRATION-V1.

The tests inspect frozen artifacts only; they never start PowerDynamics or
generate new trajectories.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39"
OUT = ROOT / "output" / "multi_likelihood_calibration_v1"
RES = OUT / "results"


def test_no_new_tds_and_frozen_start_head():
    s = pd.read_csv(RES / "multi_likelihood_calibration_summary.csv").iloc[0]
    assert bool(s.NO_NEW_TDS)
    assert s.HEAD_START == "e138e51de"
    assert s.analytic_dae_tangent == "PENDING"


def test_exact_replay_and_frozen_dictionary_contract():
    d = pd.read_csv(RES / "exact_model_replay.csv")
    assert len(d) == 1536  # 192 physical cases x 4 candidates x 2 noise replays
    assert set(d.noise_kind) == {"FROZEN_EXISTING", "EXACT_COVARIANCE"}
    assert set(d["mode"]) == {"L2", "L3", "L2-D3", "L3-D"}
    assert (d.quadrature == "adaptive_local").all()
    assert (d.integration_grid_n >= 1_000).all()


def test_lambda_projection_decomposition_and_files():
    d = pd.read_csv(RES / "physical_truncation_residual.csv")
    assert len(d) == 192 and np.max(np.abs(d.lambda_decomp_error)) < 1e-10
    assert (d.lambda_model >= -1e-12).all()
    assert (RES / "noiseless_physical_bias.csv").exists()
    p = pd.read_csv(RES / "truncation_projection.csv")
    assert np.max(np.abs(d.lambda_parallel.to_numpy() - p.lambda_parallel.to_numpy())) < 1e-10


def test_fisher_orientation_is_explicit_and_historical_score_preserved():
    d = pd.read_csv(RES / "fisher_orientation_audit.csv")
    assert set(d.orientation) == {"REVERSED"}
    assert (d.auc_1mp > d.auc_p).all()
    assert (d.corrected_auc >= .5).all()
    assert (RES / "fisher_orientation_examples.csv").exists()


def test_grouped_split_has_no_physical_path_leakage():
    d = pd.read_csv(RES / "dev_group_split.csv")
    assert len(d) == 192
    assert d.groupby("physical_path")["split"].nunique().max() == 1
    assert set(d["split"]) == {"DEV_FIT", "DEV_VAL"}


def test_likelihood_comparison_and_exact_calibration():
    d = pd.read_csv(RES / "likelihood_dev_comparison.csv")
    assert set(d["mode"]) == {"L2", "L3", "L2-D3", "L3-D"}
    ex = pd.read_csv(RES / "likelihood_exact_replay.csv")
    l2 = ex[(ex.noise_kind == "EXACT_COVARIANCE") & (ex["mode"] == "L2")].iloc[0]
    assert l2.c95i > .85 and l2.c95j > .85


def test_retrospective_and_numerical_stability_artifacts():
    ret = pd.read_csv(RES / "retrospective_true_support.csv")
    mix = pd.read_csv(RES / "retrospective_model_averaged.csv")
    stab = pd.read_csv(RES / "numerical_stability.csv")
    assert len(ret) == 3840
    assert (ret.label == "RETROSPECTIVE_TEST_DIAGNOSTIC").all()
    assert len(mix) > 0 and (mix.label == "RETROSPECTIVE_TEST_DIAGNOSTIC").all()
    assert len(stab) == 12 and (stab.mean_diff_over_sd_max >= 0).all()


def test_required_diagnostic_artifacts_exist():
    required = {
        "fisher_orientation_audit.csv", "exact_model_replay.csv",
        "noiseless_physical_bias.csv", "physical_truncation_residual.csv",
        "amplitude_jacobian_conditioning.csv", "truncation_projection.csv",
        "local_bias_prediction.csv", "truncation_order.csv",
        "single_cubic_Ri.csv", "pair_cross_cubic_Riij_Rijj.csv",
        "cubic_identifiability.csv", "dev_group_split.csv",
        "likelihood_dev_comparison.csv", "likelihood_exact_replay.csv",
        "retrospective_true_support.csv", "retrospective_model_averaged.csv",
        "numerical_stability.csv",
    }
    assert required.issubset({p.name for p in RES.glob("*.csv")})
