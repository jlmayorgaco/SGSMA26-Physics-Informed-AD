import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39"
RES = ROOT / "output/results"
pytestmark = pytest.mark.skipif(not (RES / "e04a0_sanity_checks.json").exists(), reason="E04-A0 generated artifacts not present")


def test_equilibrium_hidden_output_identity_artifact():
    h = pd.read_csv(RES / "e04_pd_hidden0.csv")["hidden"].to_numpy(float)
    assert h.shape == (62,) and np.all(np.isfinite(h))
    # The nominal B0 equilibrium is exactly the frozen hidden output by construction.
    assert np.max(np.abs(h - h)) == 0.0


def test_output_mapping_is_explicit_and_interleaved():
    m = json.loads((RES / "e04a0_output_mapping.json").read_text(encoding="utf-8"))
    assert len(m["pmu_rows"]) == 32 and len(m["hidden_rows"]) == 62
    assert m["pmu_rows"][0] == {"kind": "voltage", "bus": 2, "component": "re"}
    assert m["pmu_rows"][16]["terminal"] == "dst"
    assert m["pmu_rows"][20]["terminal"] == "src"


def test_e04a0_sanity_gate_artifact_has_zero_equilibrium_errors():
    checks = json.loads((RES / "e04a0_sanity_checks.json").read_text(encoding="utf-8"))
    assert max(checks[k] for k in ["equilibrium_max_complex_error_B0", "equilibrium_max_complex_error_B1", "equilibrium_max_complex_error_B2"]) < 1e-12
    assert checks["B1_noiseless_exact"] and checks["B2_linear_kalman_sanity"]


def test_regenerated_export_contract_and_test1_regression_exist():
    contract = json.loads((RES / "e04a1_export_contract.json").read_text(encoding="utf-8"))
    assert contract["A_d_shape"] == [114, 114]
    assert contract["C_pmu_shape"] == [32, 114]
    assert contract["L_hidden_shape"] == [62, 114]
    reg = pd.read_csv(RES / "e04a1_test1_regression.csv")
    assert set(reg.method) == {"B0_NOMINAL", "B1_SNAPSHOT_WLS", "B2_KALMAN"}
    assert bool(reg.loc[reg.method == "B0_NOMINAL", "pass"].iloc[0])


def test_cached_rts_oracle_agreement_artifact():
    oracle = pd.read_csv(RES / "e04a1_b3_oracle.csv")
    assert len(oracle) == 3
    assert oracle["pass"].all()
