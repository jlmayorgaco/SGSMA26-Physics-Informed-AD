from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm


def test_exact_augmented_exponential_input() -> None:
    from pmu_hybrid.experiments.e03r_closure import _linear_response

    A = np.array([[-1.0]]); C = np.array([[1.0]])
    response = _linear_response(A, C, np.array([0.0]), np.array([2.0]), 0, 4, 0.1, 1.0)
    expected = np.array([(2.0 * (1.0 - np.exp(-0.1 * k))) for k in range(4)])
    assert np.allclose(response[:, 0], expected, atol=1e-10)


def test_timed_alter_and_tds_timeseries_smoke() -> None:
    pytest.importorskip("andes")
    import andes

    case = andes.load(andes.get_case("ieee39/ieee39_full.xlsx"), setup=False, no_output=True)
    case.add("Alter", model="TGOV1N", dev="TGOV1_1", src="wref0", t=0.10, method="+", amount=1e-4)
    case.setup(); assert case.PFlow.run()
    case.TDS.config.tf = 0.12; case.TDS.config.tstep = 0.01; case.TDS.config.criteria = 0
    case.TDS.init(); assert case.TDS.run()
    values = case.TDS.get_timeseries(case.Bus.v)
    assert values.shape[1] == 39 and len(values) >= 10
    assert len(case.dae.ts.t) == len(values)


def test_information_gramian_toy_problem_is_positive_semidefinite() -> None:
    A = np.array([[-1.0, 0.0], [0.0, -2.0]])
    C = np.eye(2); Ad = expm(A / 30.0); J = np.zeros((2, 2)); Ak = np.eye(2)
    for _ in range(11):
        J += Ak.T @ C.T @ C @ Ak; Ak = Ak @ Ad
    assert np.all(np.linalg.eigvalsh(J) > 0)
    assert np.all(np.diag(np.linalg.pinv(J)) > 0)


def test_e03r_artifacts_record_all_three_cases() -> None:
    path = Path(__file__).resolve().parents[1] / "output" / "results" / "e03r_linear_vs_tds_per_case.csv"
    if not path.exists():
        pytest.skip("E03-R campaign artifacts not generated")
    import pandas as pd
    frame = pd.read_csv(path)
    assert set(frame.case) == {"GOVERNOR_REFERENCE_STEP", "SMALL_SHUNT_LOAD_STEP", "INITIAL_STATE_PERTURBATION"}
    assert frame.tds_ok.all()
