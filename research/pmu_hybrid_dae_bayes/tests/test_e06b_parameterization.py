from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from e06b_validate_parameterization import ybus


def test_network_ybus_finite_difference_is_nonzero():
    root = Path(__file__).parents[1] / "powerdynamics_ieee39" / "output" / "results"
    # Use the source branch table copied by the installed PowerDynamics example.
    data = Path(r"C:\Users\walla\.julia\packages\PowerDynamics\VzOiZ\docs\examples\ieee39data")
    b = pd.read_csv(data / "branch.csv")
    r = b.iloc[0]; eps = 1e-6 * max(abs(float(r.R)), 1e-3)
    bp, bm = b.copy(), b.copy(); bp.loc[0, "R"] += eps; bm.loc[0, "R"] -= eps
    assert np.linalg.norm((ybus(bp) - ybus(bm)) / (2 * eps)) > 0


def test_parameter_registry_gate_is_explicitly_not_surrogate_verified():
    root = Path(__file__).parents[1] / "powerdynamics_ieee39" / "output" / "results"
    reg = pd.read_csv(root / "e06b_parameter_registry.csv")
    assert len(reg) > 0
    assert not reg.actual_mutation_verified.any()
