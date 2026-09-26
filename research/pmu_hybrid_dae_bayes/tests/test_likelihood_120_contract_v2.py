"""Regression tests for the frozen likelihood-120 contract artifacts."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PD = ROOT / "powerdynamics_ieee39"
OUT = PD / "output" / "likelihood_120_contract_v2"
spec = importlib.util.spec_from_file_location("l120", ROOT / "scripts" / "likelihood_120_contract_v2.py")
l120 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l120)


def test_t120_dictionary_shapes_and_prefixes():
    z = np.load(OUT / "results" / "dictionary_120.npz")
    assert z["D"].shape == (3840, 16)
    assert z["Q"].shape == (3840, 16)
    assert sum(k.startswith("qij_") for k in z.files) == 120
    p = pd.read_csv(OUT / "results" / "dictionary_prefix_consistency.csv")
    assert p.exact_prefix.all() and p["T"].tolist() == l120.HORIZONS


def test_ar1_innovation_matches_dense():
    var = pd.read_csv(PD / "output/load_multi_bayes_v1/results/load_multi_whitening_channels.csv").sort_values("channel").variance.to_numpy()
    rng = np.random.default_rng(7); r = rng.normal(size=960)
    assert abs(l120.ar1_loglike(r, var) - l120.dense_loglike(r, var, 30)) < 1e-8


def test_required_contract_outputs_and_exclusion():
    required = ["event_map_regression.csv", "dictionary_120_manifest.csv", "dictionary_prefix_consistency.csv", "t30_backward_compatibility.csv", "physical_manifold_validation_by_horizon.csv", "model_error_relative_to_margin.csv", "ar1_likelihood_validation.csv", "gh31_t120_validation.csv", "information_growth.csv", "marginal_information_gain.csv", "equilibrium_event_signatures.csv", "gamma_infinity.csv", "equilibrium_pair_resolvability.csv", "structural_candidates.csv", "runtime_scaling.csv", "v3_exclusion_manifest.csv"]
    assert all((OUT / "results" / x).exists() for x in required)
    ex = pd.read_csv(OUT / "results" / "v3_exclusion_manifest.csv")
    assert ex.excluded_from_future_v3.all() and len(ex) >= 500


def test_information_prefix_is_monotone_and_gamma4_positive():
    g = pd.read_csv(OUT / "results" / "information_growth.csv")
    assert g.monotone_global_prefix.all()
    gi = pd.read_csv(OUT / "results" / "gamma_infinity.csv")
    assert float(gi.loc[gi.k == 4, "gamma_infinity"].iloc[0]) > 0
