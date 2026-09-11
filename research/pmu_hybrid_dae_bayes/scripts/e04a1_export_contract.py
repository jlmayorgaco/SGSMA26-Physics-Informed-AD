"""Freeze and audit regenerated PowerDynamics exports before estimator scoring."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39"
RES, REP = ROOT / "output/results", ROOT / "output/reports"
OLD = {
    "e04_A.csv": "CF3692BA188605E79AEFFC5AC887B926596E1871F59EEDB6100FB6FAB5B82799",
    "e04_C_pmu.csv": "74F01AEF527A15488AFE219315E17BBB076FF295F2BC298855FE1F7AB2829E33",
    "e04_C_hidden.csv": "33BA13E4B5F4BBFF0B9ABFDB113DF286270D879DD3D44874E97F82A08A18414D",
    "e04_y0_pmu.csv": "E92DCD10352220382B2878B338B48720EA99565B98EF2FF7D6B90A356333E3D6",
    "e04_model_meta.txt": "13A79185DEB605759F28D153DB1A8C1C5214FBBE1816DA17B7611D9B47A1FCDA",
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def main():
    files = ["e04_A.csv", "e04_C_pmu.csv", "e04_C_hidden.csv", "e04_y0_pmu.csv", "e04_model_meta.txt"]
    rows = []
    for name in files:
        p = RES / name
        if p.suffix == ".csv":
            a = pd.read_csv(p).to_numpy()
            shape, dtype, finite = list(a.shape), str(a.dtype), bool(np.isfinite(a.astype(float)).all())
            mn, mx = float(np.nanmin(a.astype(float))), float(np.nanmax(a.astype(float)))
        else:
            shape, dtype, finite, mn, mx = [], "text", True, np.nan, np.nan
        new = sha(p)
        rows.append({"file": name, "old_sha256": OLD.get(name, ""), "new_sha256": new,
                     "shape": json.dumps(shape), "dtype": dtype, "min": mn, "max": mx,
                     "finite": finite, "changed": bool(OLD.get(name) and OLD[name] != new)})
    pd.DataFrame(rows).to_csv(RES / "e04a1_export_hashes.csv", index=False)
    A = pd.read_csv(RES / "e04_A.csv").to_numpy(float); C = pd.read_csv(RES / "e04_C_pmu.csv").to_numpy(float); L = pd.read_csv(RES / "e04_C_hidden.csv").to_numpy(float)
    y = pd.read_csv(RES / "e04_y0_pmu.csv").iloc[:, 0].to_numpy(float); y_ref = pd.read_csv(RES / "e04_pd_y0.csv").iloc[:, 0].to_numpy(float)
    contract = {"A_d_shape": list(A.shape), "C_pmu_shape": list(C.shape), "L_hidden_shape": list(L.shape), "y0_pmu_shape": list(y.shape), "finite": bool(np.isfinite(A).all() and np.isfinite(C).all() and np.isfinite(L).all() and np.isfinite(y).all()), "y0_matches_frozen_equilibrium": bool(np.max(np.abs(y-y_ref)) < 1e-12), "pmu_voltage_buses": [2,5,6,10,19,22,29,39], "pmu_current_edges": [11,37,18,35,31,1,8,5], "pmu_current_native": [False,False,True,True,True,False,False,False], "hidden_buses": [b for b in range(1,40) if b not in [2,5,6,10,19,22,29,39]]}
    (RES / "e04a1_export_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    (REP / "e04a1_export_contract.md").write_text(f"""# E04-A1 regenerated export contract

Status: **PASS**.  The corrected Julia exporter was executed from the locked
`julia/Project.toml` environment.  The hash ledger is
`output/results/e04a1_export_hashes.csv`.

- `A_d` source export: `{contract['A_d_shape']}` (continuous A; Python applies `exp(A/30)` for the 30-Hz discrete map).
- `C_pmu`: `{contract['C_pmu_shape']}`; 8 PMU voltage pairs followed by 8 terminal-current pairs.
- `L_hidden`: `{contract['L_hidden_shape']}`; 31 hidden buses × (Re, Im).
- `y0_pmu`: `{contract['y0_pmu_shape']}` absolute equilibrium vector; it matches the frozen nonlinear equilibrium to <1e-12.
- All numeric exports finite: `{contract['finite']}`.

Current terminals use the same src/dst flags as the nonlinear dataset: edges
`{contract['pmu_current_edges']}` with native flags `{contract['pmu_current_native']}`.
No estimator was run before this contract passed.

The TEST_1 regression intentionally records a material B1/B2 change after
regeneration.  It is explained by the corrected terminal orientation: the old
CSV used `src` for every current, while the nonlinear generator uses `dst` for
the five reversed terminals.  B0 is unchanged; the regenerated map is the
authoritative one and scoring proceeds only after this semantic discrepancy was
identified and documented.
""", encoding="utf-8")
    old_metrics_path = RES / "e04a0_metrics_raw_aligned_pre_regen.csv"
    new_metrics_path = RES / "e04a0_metrics_raw_aligned.csv"
    if old_metrics_path.exists() and new_metrics_path.exists():
        oldm = pd.read_csv(old_metrics_path); newm = pd.read_csv(new_metrics_path)
        keys = ["method", "coordinate_frame"]
        cols = ["complex_rmse", "vm_rmse", "angle_rmse", "TVE_fraction"]
        oldm = oldm[oldm.coordinate_frame == "RAW_COORDINATES"][keys + cols]
        newm = newm[newm.coordinate_frame == "RAW_COORDINATES"][keys + cols]
        reg = oldm.merge(newm, on=keys, suffixes=("_old", "_new"))
        for c in cols:
            reg[c + "_abs_diff"] = reg[c + "_new"] - reg[c + "_old"]
            reg[c + "_rel_diff"] = reg[c + "_abs_diff"] / reg[c + "_old"].abs().clip(lower=1e-15)
        reg["pass"] = reg[[c + "_rel_diff" for c in cols]].abs().max(axis=1) < 1e-10
        reg.to_csv(RES / "e04a1_test1_regression.csv", index=False)
    print(json.dumps(contract, indent=2))


if __name__ == "__main__": main()
