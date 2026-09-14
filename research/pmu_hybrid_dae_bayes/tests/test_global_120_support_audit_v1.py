"""Contract tests for the retrospective GLOBAL-120 audit artifacts."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
OUT=HERE/"powerdynamics_ieee39/output/global_120_support_audit_v1"
RES=OUT/"results"; PLOTS=OUT/"plots"

def test_global_manifest_and_support_cardinality():
    m=pd.read_csv(RES/"global_case_manifest.csv"); s=pd.read_csv(RES/"support_per_case.csv")
    assert len(m)==10440 and m.campaign.eq("RETROSPECTIVE_GLOBAL_120_SUPPORT_AUDIT").all()
    assert len(s)==10440*137
    assert set(s.groupby("case_id").size())=={137}

def test_posterior_normalization_and_hypothesis_counts():
    s=pd.read_csv(RES/"support_per_case.csv")
    sums=s.groupby("case_id").posterior.sum().to_numpy()
    assert np.max(np.abs(sums-1.0))<2e-10
    assert (s.support_type=="SINGLE").groupby(s.case_id).sum().eq(16).all()
    assert (s.support_type=="DOUBLE").groupby(s.case_id).sum().eq(120).all()

def test_evidence_dimension_and_multiplicity_identity():
    n=pd.read_csv(RES/"evidence_dimension_normalization.csv")
    assert n.status.eq("PASS").all() and np.max(np.abs(n.integral_prior-1))<1e-12
    d=pd.read_csv(RES/"multiplicity_decomposition.csv")
    assert np.max(np.abs(d.decomposition_residual))<1e-8
    assert np.allclose(d.multiplicity_term.iloc[0],-np.log(120/16))

def test_retro_global_and_gk_reference():
    rm=pd.read_csv(RES/"run_manifest.csv").iloc[0]
    assert bool(rm.retrospective) and int(rm.new_tds)==0 and int(rm.hypotheses_per_case)==137
    g=pd.read_csv(RES/"gk_reference_check.csv")
    assert len(g)==8 and np.max(g.abs_delta_logZ)<1e-7

def test_required_artifacts_exist():
    names=["cardinality_summary.csv","support_summary.csv","source_inclusion.csv","model_averaged_amplitude.csv","restricted_vs_full.csv","weak_weak_audit.csv","conditional_fisher.csv","pair_difficulty_atlas.csv","graph_metadata.csv","nested_single_manifold_distance.csv","pair_manifold_distance.csv"]
    assert all((RES/n).exists() for n in names)
    assert len(list(PLOTS.glob("*.png")))>=13
