"""Contract tests for MULTI-INDEPENDENT-INTEGRATOR-V1 artifacts.

The campaign is reference-only and reuses frozen confirmatory trajectories;
these tests intentionally do not launch PowerDynamics or regenerate TDS data.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "powerdynamics_ieee39/output/multi_independent_integrator_v1"
RES = OUT / "results"


def _summary():
    return pd.read_csv(RES / "multi_independent_integrator_summary.csv").iloc[0]


def test_selected_manifest_is_frozen_and_stratified():
    selected = pd.read_csv(RES / "selected_cases.csv")
    assert len(selected) == 64
    assert selected.groupby("regime").size().to_dict() == {
        "WEAK_WEAK": 16, "WEAK_STRONG": 16, "MODERATE": 16, "FINITE": 16
    }
    assert selected[["case_id", "noise_seed"]].duplicated().sum() == 0
    # The summary hashes the exact frozen CSV bytes (rather than pandas' type
    # re-serialization, which can change integer formatting on read-back).
    digest = hashlib.sha256((RES / "selected_cases.csv").read_bytes()).hexdigest()
    assert digest == _summary().selected_case_manifest_hash


def test_double_and_best_single_support_contract():
    supports = pd.read_csv(RES / "selected_supports.csv")
    ev = pd.read_csv(RES / "gh_vs_gk_evidence.csv")
    singles = pd.read_csv(RES / "best_single_evidence.csv")
    assert len(supports) == 323
    assert len(ev) == 259 and len(singles) == 64
    assert set(singles.support_label) == {"BEST_SINGLE"}
    assert (ev.support_i != ev.support_j).all()
    assert set(ev.support_label).issuperset({"TRUE_DOUBLE", "GH_TOP3", "LOW_EVIDENCE_DOUBLE"})


def test_gk_evidence_and_moments_agree_with_gh31():
    ev = pd.read_csv(RES / "gh_vs_gk_evidence.csv")
    mo = pd.read_csv(RES / "gh_vs_gk_moments.csv")
    assert np.isfinite(ev[["logZ_GH31", "logZ_GK", "delta_logZ"]]).all().all()
    assert np.isfinite(mo[["mean_i_GH31", "mean_i_GK", "sd_i_GH31", "sd_i_GK"]]).all().all()
    assert ev.abs_delta_logZ.median() < 1e-4
    assert ev.abs_delta_logZ.quantile(.95) < 1e-3
    assert ev.abs_delta_logZ.max() < 1e-2
    assert max(mo.mean_i_shift_GK_SD.median(), mo.mean_j_shift_GK_SD.median()) < .01
    assert max(mo.mean_i_shift_GK_SD.quantile(.95), mo.mean_j_shift_GK_SD.quantile(.95)) < .05
    assert max(mo.sd_i_rel_diff.median(), mo.sd_j_rel_diff.median()) < 1e-3
    assert max(mo.sd_i_rel_diff.quantile(.95), mo.sd_j_rel_diff.quantile(.95)) < .01


def test_gk_tail_and_tolerance_refinement():
    s = _summary()
    assert float(s.omitted_prior_tail_probability) < 1e-20
    tol = pd.read_csv(RES / "gk_tolerance_convergence.csv")
    assert len(tol) == 8 * 3
    assert set(tol.tol) == {"T1", "T2", "T3"}
    spread = tol.groupby("case_id").logZ.agg(lambda x: x.max() - x.min())
    assert spread.max() < 1e-6
    assert (tol.quad_err < 1e-6).all()


def test_cardinality_and_support_decisions_are_stable():
    card = pd.read_csv(RES / "cardinality_crosscheck.csv")
    sup = pd.read_csv(RES / "support_crosscheck.csv")
    assert len(card) == 16
    assert card.map_flip.sum() == 0
    assert card[["abs_diff_M0", "abs_diff_M1", "abs_diff_M2"]].max().max() < .01
    assert len(sup) == 16
    assert sup[["p_true_GK", "top1_GK"]].notna().all().all()


def test_old_grid_is_historical_control_only():
    old = pd.read_csv(RES / "old31_vs_gh_vs_gk.csv")
    assert len(old) == 259
    d = np.abs(old.logZ_old31 - old.logZ_GK)
    assert np.isfinite(d).all()
    assert d.median() > 1.0
    assert _summary().old31_grid == "CONFIRMED_INADEQUATE"


def test_bus7_bus12_and_runtime_artifacts():
    bus = pd.read_csv(RES / "bus7_bus12_crosscheck.csv")
    runtime = pd.read_csv(RES / "runtime.csv")
    assert len(bus) >= 4
    assert set(runtime.component) == {"GK2D_REFERENCE", "GH31"}
    assert (runtime.median_runtime_s > 0).all()
    assert runtime.loc[runtime.component == "GK2D_REFERENCE", "median_runtime_s"].iloc[0] > runtime.loc[runtime.component == "GH31", "median_runtime_s"].iloc[0]


def test_report_statuses_and_required_plots():
    s = _summary()
    assert s.independent_integrator_agreement == "PASS"
    assert s.evidence_stability == "PASS"
    assert s.posterior_moment_stability == "PASS"
    assert s.cardinality_decision_stability == "PASS"
    assert s.support_decision_stability == "PASS"
    assert s.analytic_dae_tangent == "PENDING"
    report = (OUT / "reports/multi_independent_integrator_v1.md").read_text(encoding="utf-8")
    assert "No PowerDynamics TDS was generated" in report
    assert "+log(1/sigma)" in report
    for name in [
        "logZ_gh_vs_gk.png", "posterior_mean_shift_sd.png", "posterior_sd_ratio.png",
        "cardinality_gh_vs_gk.png", "support_probability_gh_vs_gk.png",
        "old31_gh_gk_comparison.png", "runtime_comparison.png", "bus7_bus12_gh_vs_gk.png",
    ]:
        assert (OUT / "plots" / name).exists()


def test_gaussian_tail_reference_value():
    # Contract for the standardized [-10,10] integration domain.
    assert math.erfc(10 / math.sqrt(2)) == pytest.approx(1.523970604832119e-23, rel=1e-12)
