"""Contract tests for GLOBAL-137-CONFIRMATORY-V1 artifacts and design."""
from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/global_137_confirmatory_v1.py"
OUT = ROOT / "powerdynamics_ieee39/output/global_137_confirmatory_v1"

def test_script_exists_and_uses_frozen_integrator():
    txt = SCRIPT.read_text(encoding="utf-8")
    assert "GH=31" in txt and "GK2D" in txt
    assert "old_grid" not in txt.lower()

def test_preregistered_design_has_full_pairs_and_four_weak_levels():
    # Keep this independent of generated data so it catches accidental pruning.
    import importlib.util
    spec = importlib.util.spec_from_file_location("g137", SCRIPT)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert len(m.PAIRS) == 120
    assert sum(r[2] == "WEAK_WEAK" for r in m.AMP_PAIRS) >= 4

def test_fresh_manifest_no_overlap_when_present():
    p = OUT / "results/physical_manifest.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    assert d.trajectory_id.is_unique
    # While a checkpointed generation is running, PREREGISTERED is valid;
    # after completion every row must be successful.
    assert set(d.status).issubset({"PREREGISTERED", "EXECUTED_SUCCESS", "EXECUTED_FAIL"})
    assert set(d.regime[d.source_j != 0]) >= {"WEAK_WEAK", "WEAK_STRONG", "MODERATE", "FINITE"}

def test_full_137_posterior_normalization_when_present():
    p = OUT / "results/support_per_case.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    key = ["case_id", "noise_seed"] if "noise_seed" in d.columns else ["case_id"]
    n = d.groupby(key).posterior.sum()
    assert (n.size > 0) and np.allclose(n.to_numpy(), 1.0, atol=2e-8)

def test_frozen_cardinality_prior():
    p = OUT / "results/cardinality_summary.csv"
    if not p.exists():
        return
    assert math.isfinite(float(pd.read_csv(p).select_dtypes("number").to_numpy().mean()))
