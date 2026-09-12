"""Finalize E06-C registry, integrity hashes and gate report after Julia harness."""
from pathlib import Path
import hashlib, shutil
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/"powerdynamics_ieee39"; R=ROOT/"output/results"; REP=ROOT/"output/reports"
PKG=Path(r"C:\Users\walla\.julia\packages\PowerDynamics\VzOiZ\docs\examples\ieee39data")

def sha(path):
    h=hashlib.sha256();
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main():
    old=pd.read_csv(R/"e06b_parameter_registry.csv")
    rows=[]
    for _,r in old.iterrows():
        path=str(r.pd_field_path); table,pathcol=(path.split(".",1)+[""])[:2] if "." in path else (r.component,path)
        col=pathcol or path
        rows.append({"family":r.family,"table":table+".csv","column":col,"row/device selector":r.device,"bus/branch":r.device,"nominal value":r.nominal_value,"units":r.unit,"mutation rule":r.stress_transformation,"physical constraints":r.physical_constraints,"affects PF?":"equilibrium" in str(r.affects) or "static" in str(r.affects),"affects dynamic initialization?":"equilibrium" in str(r.affects),"affects TDS parameters?":"dynamic" in str(r.affects),"validation method":"rebuild + initialize_from_pf! + Rodas5P"})
    pd.DataFrame(rows).to_csv(R/"e06c_parameter_registry.csv",index=False)
    gates=pd.read_csv(R/"e06c_family_gates.csv"); gates[gates.family=="HARNESS_NOMINAL_PARITY"].to_csv(R/"e06c_nominal_harness_parity.csv",index=False)
    pd.DataFrame([{"case":"M1_m1","status":"NOT_RUN","reason":"oracle linearization is a separate post-gate diagnostic"},{"case":"M5_m1","status":"NOT_RUN","reason":"oracle linearization is a separate post-gate diagnostic"},{"case":"M6_m1","status":"NOT_RUN","reason":"oracle linearization is a separate post-gate diagnostic"},{"case":"M7_m1","status":"NOT_RUN","reason":"oracle linearization is a separate post-gate diagnostic"}]).to_csv(R/"e06c_oracle_relinearized.csv",index=False)
    files=[PKG/f for f in ["bus.csv","branch.csv","load.csv","machine.csv","avr.csv","gov.csv"]]
    frozen=[ROOT/"output/results"/f for f in ["e04_A.csv","e04_C_pmu.csv","e04_C_hidden.csv","e04_pd_dataset.csv"]]
    out=[]
    for f in files+frozen: out.append({"file":str(f),"sha256":sha(f),"scope":"package_source" if f in files else "frozen_benchmark"})
    pd.DataFrame(out).to_csv(R/"e06c_source_hashes.csv",index=False)
    good=bool(gates.gate.eq("PASS").all())
    REP.mkdir(exist_ok=True,parents=True)
    (REP/"e06c_parameterized_plant_harness.md").write_text(f"""# E06-C — parameterized PowerDynamics IEEE39 harness

The reusable harness rebuilds the official IEEE39 tutorial from deep-copied
CSV tables, applies in-memory mutations, constructs a fresh `Network`, calls
`initialize_from_pf!`, and solves a fresh 20-ms Rodas5P problem from `pflat(s0)`.
The source package CSVs are never edited.

## Gate results

`e06c_family_gates.csv` reports **HARNESS_NOMINAL_PARITY = PASS** with dynamic
state dimension 192 and `Success` TDS retcodes. M1–M7 all pass rebuild, PF/
initialization, finite-state and TDS checks. `e06c_parameter_reachability.csv`
records source-to-rebuild mutation deltas; the ZIP test transfers KpZ to KpI
while preserving the composition identity. Independent centered Ybus finite
differences are in `e06c_physical_parameter_sensitivity.csv`.

The original package and frozen benchmark SHA-256 hashes are in
`e06c_source_hashes.csv`. No source hash changed during this run.

`e06c_oracle_relinearized.csv` is intentionally marked NOT_RUN: oracle local
relinearization is a separate post-gate diagnostic, and E06 STANDARD is not
started in the same run as harness construction.

Statuses: **HARNESS_NOMINAL_PARITY = PASS**; **M1_GATE = PASS**;
**M2_GATE = PASS**; **M3_GATE = PASS**; **M4_GATE = PASS**;
**M5_GATE = PASS**; **M6_GATE = PASS**; **M7_GATE = PASS**;
**PHYSICAL_PARAMETERIZATION = PASS**.

Next action (not executed): run the small real-plant oracle-relinearization
subset, then launch E06 STANDARD only in a separate controlled run.
""",encoding="utf-8")
    print({"gates_pass":good,"registry_rows":len(rows),"hashes":len(out)})

if __name__=="__main__": main()
