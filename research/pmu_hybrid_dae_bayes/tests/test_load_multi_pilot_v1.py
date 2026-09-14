from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39" / "output" / "load_multi_pilot_v1" / "results"


def test_pilot_selection_is_frozen_restricted_and_includes_bus7_bus12():
    d = pd.read_csv(ROOT / "load_multi_pilot_pair_selection.csv")
    assert len(d) == 12
    assert {(7, 12)}.issubset(set(zip(d.source_i, d.source_j)))
    assert set(d.selection_source) == {"V1_EXPLORATORY_ONLY"}
    assert (ROOT / "load_multi_pilot_pair_selection.sha256").exists()


def test_pilot_physical_manifests_have_no_failed_cases():
    q = pd.read_csv(ROOT / "load_multi_pilot_qij_manifest.csv")
    dev = pd.read_csv(ROOT / "load_multi_pilot_dev_manifest.csv")
    assert len(q) == 48 and set(q.status) == {"EXECUTED_SUCCESS"}
    assert len(dev) == 192 and all(Path(p).exists() for p in dev.physical_path)


def test_pilot_posterior_normalization_and_h0():
    d = pd.read_parquet(ROOT / "load_multi_pilot_posterior.parquet")
    pcols = [c for c in d.columns if c.startswith("p_H")]
    assert len(pcols) == 29
    assert np.allclose(d[pcols].sum(axis=1), 1.0, atol=2e-8)
    h0 = d[d.true_M == 0]
    assert len(h0) == 200 and float(h0.p_M0.mean()) > 0.9


def test_conditional_information_and_step_sensitivity_are_finite():
    info = pd.read_csv(ROOT / "load_multi_pilot_info.csv")
    sens = pd.read_csv(ROOT / "load_multi_pilot_qij_step_sensitivity.csv")
    assert len(info) == 12 * 2 * 8
    assert not info.degenerate.any()
    assert np.isfinite(info.I_j_given_i).all()
    assert float(sens.relative_to_h005.max()) < 1e-2


def test_pilot_split_manifest_seed_ranges_are_disjoint():
    d = pd.read_csv(ROOT / "load_multi_pilot_test_manifest.csv")
    h0 = set(d.loc[d.case_type == "H0", "noise_seed"])
    one = set(d.loc[d.case_type == "SINGLE", "noise_seed"])
    two = set(d.loc[d.case_type == "DOUBLE", "noise_seed"])
    assert h0.isdisjoint(one) and h0.isdisjoint(two) and one.isdisjoint(two)


def test_pilot_gaussian_support_probabilities_are_explicit():
    d = pd.read_parquet(ROOT / "load_multi_pilot_posterior.parquet")
    assert {"p_M0", "p_M1", "p_M2"}.issubset(d.columns)
    assert np.all((d[["p_M0", "p_M1", "p_M2"]] >= 0).to_numpy())
    assert np.allclose(d[["p_M0", "p_M1", "p_M2"]].sum(axis=1), 1.0, atol=2e-8)
