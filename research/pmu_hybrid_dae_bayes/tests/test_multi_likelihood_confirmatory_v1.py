from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "powerdynamics_ieee39/output/multi_likelihood_confirmatory_v1"
RES = OUT / "results"

def test_confirmatory_artifacts_and_manifest():
    mf = pd.read_csv(RES / "physical_manifest.csv")
    assert len(mf) == 384
    assert (mf.status == "EXECUTED_SUCCESS").all()
    assert set(mf.regime) == {"WEAK_WEAK", "WEAK_STRONG", "MODERATE", "FINITE"}
    assert len(mf.case_id.unique()) == 384
    digest = hashlib.sha256(mf.to_csv(index=False).encode()).hexdigest()
    assert digest == (RES / "physical_manifest.sha256").read_text().strip()

def test_noise_manifest_separates_physical_paths_and_h0():
    mf = pd.read_csv(RES / "physical_manifest.csv")
    nm = pd.read_csv(RES / "noise_manifest.csv")
    event = nm[nm.regime != "H0"]
    assert len(event) == 384 * 20
    assert event.case_id.nunique() == 384
    assert len(nm[nm.regime == "H0"]) == 200
    assert not set(event.case_id).intersection(set(nm[nm.regime == "H0"].case_id))
    assert set(event.case_id) == set(mf.case_id)

def test_adaptive_convergence_and_required_outputs():
    q = pd.read_csv(RES / "quadrature_stability.csv")
    assert len(q) == 24
    assert q.mean_diff_21_31.median() < 1e-5
    assert q.sd_diff_21_31.median() < 1e-5
    assert pd.read_csv(RES / "edge_mass.csv").iloc[0].subset_n == 24
    for name in ["old_grid_vs_adaptive.csv", "true_support_calibration.csv",
                 "known_cardinality_support.csv", "full_bayes_cardinality.csv",
                 "full_bayes_support.csv", "model_averaged_amplitude.csv",
                 "conditional_fisher.csv", "nested_manifold_distance.csv",
                 "multiplicity_decomposition.csv", "failure_root_cause.csv"]:
        assert (RES / name).exists()
    for name in ["true_support_coverage.png", "model_averaged_coverage.png",
                 "cardinality_confusion.png", "support_vs_amplitude.png",
                 "old_vs_adaptive_evidence.png", "old_vs_adaptive_posterior_sd.png",
                 "conditional_information_vs_merge.png", "nested_distance_vs_merge.png",
                 "multiplicity_decomposition.png"]:
        assert (OUT / "plots" / name).exists()

def test_scope_and_leakage_contract():
    s = pd.read_csv(RES / "multi_likelihood_confirmatory_summary.csv").iloc[0]
    assert s.global_support_recovery == "NOT_ESTABLISHED"
    assert s.analytic_dae_tangent == "PENDING"
    assert bool(s.NO_NEW_TDS) is False
    assert s.pair_supports == 24
    report = (OUT / "reports/multi_likelihood_confirmatory_v1.md").read_text()
    assert "No V1/V2 TEST row" in report
    assert "five-seed-per-trajectory" in report
