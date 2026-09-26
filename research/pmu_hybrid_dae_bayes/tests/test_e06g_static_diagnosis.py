from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39"
RES = ROOT / "output" / "results"


def _read(name):
    path = RES / name
    assert path.exists(), path
    return pd.read_csv(path)


def test_m6_required_injection_reconstruction():
    df = _read("e06g_required_injections.csv")
    assert len(df) == 30 * 39
    assert set(df.role) == {"PQ", "PV", "SLACK"}
    bus3 = df[df.bus == 3]
    assert bus3.deltaP_required.abs().median() > 1e-4


def test_current_basis_representability_is_falsified():
    df = _read("e06g_basis_representability.csv")
    cur = df[df.basis == "CURRENT_D_BASIS"]
    assert cur.relative_residual.median() > 1e-2


def test_extended_physical_basis_representability():
    df = _read("e06g_basis_representability.csv")
    phy = df[df.basis == "M6_PHYSICAL_BASIS"]
    assert phy.relative_residual.max() < 1e-10


def test_known_feasible_forward_inverse_round_trip():
    df = _read("e06g_known_solution_cases.csv")
    assert len(df) == 27
    assert df[df.method == "B0_FULL_VOLTAGE"].forward_ac_residual.median() < 1e-6


def test_full_voltage_inverse_pf_gate():
    df = _read("e06g_known_solution_cases.csv")
    b0 = df[df.method == "B0_FULL_VOLTAGE"]
    assert b0.converged.all()
    assert b0.hidden_V_TVE_percent.median() < 1e-4
    assert b0.ac_residual.median() < 1e-6


def test_eight_pmu_true_initialization_stationarity():
    df = _read("e06g_known_solution_cases.csv")
    b1 = df[df.method == "B1_8PMU_TRUE_INIT"]
    assert b1.converged.all()
    assert b1.hidden_V_TVE_percent.median() < 1e-8


def test_exact_jacobian_matches_finite_difference():
    df = _read("e06g_jacobian_audit.csv")
    assert df.relative_frobenius_difference.max() < 1e-5


def test_independent_solver_comparison_is_recorded():
    df = _read("e06g_solver_comparison.csv")
    assert len(df) == 9
    assert set(df.independent_method) == {"SLSQP_FULL_VOLTAGE_CONSTRAINED"}


def test_continuation_consistency():
    df = _read("e06g_continuation.csv")
    assert len(df) == 6
    assert df.converged.all()
    assert df.hidden_tve_percent.iloc[-1] < df.hidden_tve_percent.iloc[0]


def test_functional_hidden_voltage_identifiability_is_reported():
    df = _read("e06g_identifiability.csv")
    assert len(df) == 9
    assert (df.parameter_dimension > df.information_rank).all()
    assert (df.nullspace_dimension > 0).all()


def test_real_m6_not_run_before_synthetic_gates():
    summary = __import__("json").loads((RES / "e06g_summary.json").read_text())
    assert summary["REAL_M6_STATIC_RECENTERING"] == "NOT_RUN"

