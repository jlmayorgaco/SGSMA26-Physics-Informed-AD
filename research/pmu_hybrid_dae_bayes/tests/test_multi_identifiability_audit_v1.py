"""Contract tests for the post-pilot multiplicity/uncertainty audit.

These tests are intentionally read-only: they inspect frozen audit artifacts
and never launch PowerDynamics or create trajectories.
"""
from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39"
RES = ROOT / "output" / "load_multi_pilot_v1" / "results"


def test_support_multiplicity_contract_and_identity():
    d = pd.read_csv(RES / "load_multi_multiplicity.csv")
    assert set(d.candidate_space) == {"RESTRICTED", "FULL"}
    assert set(d.m) == {0, 1, 2}
    assert d.loc[(d.candidate_space == "RESTRICTED") & (d.m == 2), "N_m"].iat[0] == 12
    assert d.loc[(d.candidate_space == "FULL") & (d.m == 2), "N_m"].iat[0] == 120
    lhs = d.log_evidence_given_M.to_numpy()
    rhs = d.best_support_term.to_numpy() + d.multiplicity_term.to_numpy() + d.support_volume_term.to_numpy()
    assert np.max(np.abs(lhs - rhs)) < 1e-8


def test_uniform_support_count_reference():
    r = pd.read_csv(RES / "load_multi_multiplicity_runtime.csv").iloc[0]
    assert int(r.restricted_pairs) == 12 and int(r.full_pairs) == 120
    assert np.isclose(float(r.pure_log120_over_12), np.log(10.0))


def test_fisher_incremental_and_within_strata_are_frozen_evaluations():
    x = pd.read_csv(RES / "load_multi_fisher_incremental.csv")
    assert set(x.model) == {"MODEL_AMP", "MODEL_INFO"}
    assert len(x) == 2 and (x.test_n == 3840).all() and (x.dev_n == 192).all()
    s = pd.read_csv(RES / "load_multi_fisher_within_stratum.csv")
    assert set(s.regime) == {"FINITE", "MODERATE", "WEAK_STRONG", "WEAK_WEAK"}


def test_c1_c2_c3_c4_outputs_and_atom_bounds():
    oracle = pd.read_csv(RES / "load_multi_amplitude_oracle.csv")
    maps = pd.read_csv(RES / "load_multi_amplitude_map_support.csv")
    mix = pd.read_csv(RES / "load_multi_amplitude_model_averaged.csv")
    assert len(oracle) == 3840  # both amplitudes for all frozen double rows
    assert len(maps) == 3840
    assert set(mix.true_present) == {0, 1}
    assert ((mix.atom_mass >= -1e-12) & (mix.atom_mass <= 1 + 1e-12)).all()
    assert maps.map_double.sum() >= maps.map_support_correct.sum()


def test_audit_summary_preserves_blocked_future_claims():
    s = pd.read_csv(RES / "load_multi_identifiability_audit_summary.csv").iloc[0]
    assert bool(s.no_new_TDS)
    assert s.GLOBAL_SUPPORT_RECOVERY == "NOT_ESTABLISHED"
    assert s.ANALYTIC_DAE_TANGENT == "PENDING"
    assert s.support_space_multiplicity in {"SMALL", "MODERATE", "LARGE"}
    assert s.conditional_fisher_incremental_value in {"SUPPORTED", "INCONCLUSIVE", "NOT_SUPPORTED"}
    assert s.amplitude_undercoverage_root_cause in {"LIKELIHOOD", "POST_SELECTION", "WRONG_SUPPORT", "MIXED"}
