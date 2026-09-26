from pathlib import Path
import hashlib
import pandas as pd


def test_nominal_parity_and_all_family_gates_pass():
    root = Path(__file__).parents[1] / "powerdynamics_ieee39" / "output" / "results"
    gates = pd.read_csv(root / "e06c_family_gates.csv")
    assert gates.loc[gates.family == "HARNESS_NOMINAL_PARITY", "gate"].iloc[0] == "PASS"
    assert set(gates.loc[gates.family != "HARNESS_NOMINAL_PARITY", "gate"]) == {"PASS"}


def test_source_hashes_are_stable_and_reachability_is_recorded():
    root = Path(__file__).parents[1] / "powerdynamics_ieee39" / "output" / "results"
    hashes = pd.read_csv(root / "e06c_source_hashes.csv")
    assert len(hashes) == 10 and hashes.sha256.str.len().eq(64).all()
    reach = pd.read_csv(root / "e06c_parameter_reachability.csv")
    assert set(reach.status) == {"REBUILD_REACHED"}


def test_oracle_is_isolated_from_harness_gate():
    root = Path(__file__).parents[1] / "powerdynamics_ieee39" / "output" / "results"
    oracle = pd.read_csv(root / "e06c_oracle_relinearized.csv")
    assert set(oracle.status) == {"NOT_RUN"}
