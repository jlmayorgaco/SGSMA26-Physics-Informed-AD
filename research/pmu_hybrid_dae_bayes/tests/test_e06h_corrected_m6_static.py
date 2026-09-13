from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39"
RES = ROOT / "output" / "results"
REPORT = ROOT / "output" / "reports" / "e06h_corrected_m6_static_recentering.md"


def read(name):
    p = RES / name
    assert p.exists(), p
    return pd.read_csv(p)


def test_fresh_dev_test_counts_and_disjoint_seeds():
    dev, test = read("e06h_dev_manifest.csv"), read("e06h_test_manifest.csv")
    assert len(dev) == 30 and len(test) == 60
    assert set(dev.seed).isdisjoint(set(test.seed))
    assert set(dev.m) == {0.5, 1.0, 1.5}
    assert set(test.m) == {0.5, 1.0, 1.5}


def test_correct_slack_and_pv_semantics_are_documented():
    text = REPORT.read_text()
    assert "bus 31 as Slack" in text
    assert "Bus 39 remains PV" in text


def test_extended_basis_has_43_coordinates():
    text = REPORT.read_text()
    assert "basis has 43 nuisance coordinates" in text


def test_nominal_control_is_fixed_point():
    df = read("e06h_nominal_control.csv").iloc[0]
    assert bool(df.converged)
    assert df.hidden_tve_percent < 1e-8
    assert df.pmu_residual < 1e-10
    assert df.ac_residual < 1e-10
    assert df.d_norm < 1e-8


def test_map_convergence_gate_and_rejections():
    t = read("e06h_test_per_case.csv")
    m = t[t.method == "S0-MAP-CORRECTED"]
    assert len(m) == 60
    assert m.converged.all()
    assert m.ac_residual.max() <= 5e-5
    assert len(read("e06h_rejections.csv")) == 0


def test_corrected_map_closure_is_strong_by_scale():
    c = read("e06h_closure.csv")
    assert len(c) == 3
    assert (c.closure_median >= 0.8).all()
    assert (c.closure_ci95_low >= 0.8).all()


def test_functional_identifiability_reports_rank_deficiency():
    i = read("e06h_identifiability.csv")
    assert len(i) == 60
    assert (i.information_rank == 25).all()
    assert (i.parameter_dimension == 43).all()
    assert (i.nullspace_dimension == 18).all()


def test_nullspace_aware_uncertainty_outputs_are_finite():
    u = read("e06h_uncertainty.csv")
    assert len(u) == 60
    assert u.coverage95.notna().all()
    assert u.NLL.notna().all()
    assert u.NEES_like.notna().all()


def test_independent_solver_subset_has_two_cases_per_scale():
    s = read("e06h_solver_subset.csv")
    assert len(s) == 6
    assert s.groupby("m").size().to_dict() == {0.5: 2, 1.0: 2, 1.5: 2}
    assert s.independent_ac_residual.max() < 1e-6


def test_claim_cleanup_withdraws_m6_network_parameter_claim():
    c = read("e06h_claim_cleanup.csv")
    assert "SUPERSEDED_BY_E06G" in set(c.status)
    assert "WITHDRAWN_FOR_M6_PENDING_E06H" in set(c.status)


def test_required_plots_exist():
    plots = ROOT / "output" / "plots"
    names = ["e06h_nominal_vs_corrected_map.png", "e06h_closure_vs_scale.png", "e06h_hidden_tve_distribution.png", "e06h_ac_residual.png", "e06h_identifiability_vs_error.png", "e06h_runtime.png"]
    assert all((plots / n).exists() for n in names)

