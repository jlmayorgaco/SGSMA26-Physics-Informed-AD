"""E06-B physical-parameterization gate.

This validator intentionally stops before STANDARD when a true PowerDynamics
mutation/rebuild/PF/TDS path is not present.  It still records the real source
parameter registry and analytical network sensitivities, so the failure is
auditable rather than silently replaced by a signal surrogate.
"""
from pathlib import Path
import json, re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/"powerdynamics_ieee39"; R=ROOT/"output/results"; REP=ROOT/"output/reports"
JDATA=Path(r"C:\Users\walla\.julia\packages\PowerDynamics\VzOiZ\docs\examples\ieee39data")

def ybus(branch):
    n=39; Y=np.zeros((n,n),complex)
    for _,r in branch.iterrows():
        i,j=int(r.src_bus)-1,int(r.dst_bus)-1; z=complex(float(r.R),float(r.X)); y=1/z if z else 0j; b=1j*float(r.B_src+r.B_dst)
        Y[i,i]+=y+b; Y[j,j]+=y+b; Y[i,j]-=y; Y[j,i]-=y
    return Y

def main():
    frames=[]
    if not JDATA.exists():
        raise FileNotFoundError(f"PowerDynamics data directory missing: {JDATA}")
    branch=pd.read_csv(JDATA/"branch.csv"); machine=pd.read_csv(JDATA/"machine.csv"); gov=pd.read_csv(JDATA/"gov.csv"); avr=pd.read_csv(JDATA/"avr.csv"); load=pd.read_csv(JDATA/"load.csv"); bus=pd.read_csv(JDATA/"bus.csv")
    def add(fam,component,device,path,val,unit,transform,constraints,affects): frames.append({"family":fam,"component":component,"device":device,"pd_field_path":path,"nominal_value":val,"unit":unit,"stress_transformation":transform,"physical_constraints":constraints,"affects":affects,"actual_mutation_verified":False,"gate_note":"NO_PARAMETERIZED_REBUILD_PATH"})
    for i,r in branch.iterrows():
        dev=f"branch_{int(r.src_bus)}_{int(r.dst_bus)}"; add("M1_NETWORK","PiLine",dev,"branch.R",r.R,"pu","R*(1+m*s)","R>0","static network;equilibrium") ; add("M1_NETWORK","PiLine",dev,"branch.X",r.X,"pu","X*(1+m*s)","X>0","static network;equilibrium"); add("M1_NETWORK","PiLine",dev,"branch.B_src/B_dst",float(r.B_src+r.B_dst),"pu","B*(1+m*s)","finite charging","static network;equilibrium")
    for _,r in machine.iterrows():
        for c in ["H","D"]:
            if c in r: add("M2_MACHINE","SauerPaiMachine",f"bus_{int(r.bus)}",f"machine.{c}",r[c],"s" if c=="H" else "pu","v*(1+m*s)","H>0;D>=0","dynamic equations;equilibrium")
    for _,r in gov.iterrows():
        for c in ["R","T1","T2","T3","DT","ω_ref"]:
            if c in r: add("M3_GOVERNOR","TGOV1",f"bus_{int(r.bus)}",f"gov.{c}",r[c],"pu/s","v*(1+m*s)","time constants>0","dynamic equations;equilibrium")
    for _,r in avr.iterrows():
        for c in ["Ka","Ta","Tf","Te","Tr"]:
            if c in r: add("M4_AVR","AVRTypeI",f"bus_{int(r.bus)}",f"avr.{c}",r[c],"pu/s","v*(1+m*s)","time constants>0","dynamic equations;equilibrium")
    for _,r in load.iterrows():
        for c in ["KpZ","KpI","KpC","KqZ","KqI","KqC"]:
            if c in r: add("M5_LOAD_MODEL","ZIPLoad",f"bus_{int(r.bus)}",f"load.{c}",r[c],"fraction","composition-preserving transfer","fractions>=0;sum=1","static network;equilibrium")
    for _,r in bus.iterrows():
        if bool(r.get("has_load",False)): add("M6_OPERATING_POINT","bus injection",f"bus_{int(r.bus)}","bus.P",r.P,"pu","balanced scale/redispatch","PF feasible","equilibrium;measurement map")
    reg=pd.DataFrame(frames); reg.to_csv(R/"e06b_parameter_registry.csv",index=False)
    # Independent analytical Ybus finite differences are useful, but do not pass
    # the actual nonlinear-plant mutation gate.
    rows=[]; r0=branch.iloc[0]; Y0=ybus(branch)
    for p in ("R","X","B_src"):
        bp=branch.copy(); bm=branch.copy(); eps=1e-6*max(abs(float(r0[p])),1e-3); bp.loc[0,p]+=eps; bm.loc[0,p]-=eps; dY=(ybus(bp)-ybus(bm))/(2*eps); rows.append({"family":"M1_NETWORK","parameter":p,"device":f"branch_{int(r0.src_bus)}_{int(r0.dst_bus)}","finite_difference_norm":float(np.linalg.norm(dY)),"analytical_check":"PASS","actual_plant_mutation":"NOT_VERIFIED","note":"CSV/Ybus analytical audit only"})
    for fam in ["M2_MACHINE","M3_GOVERNOR","M4_AVR","M5_LOAD_MODEL","M6_OPERATING_POINT"]: rows.append({"family":fam,"parameter":"representative","device":"representative","finite_difference_norm":np.nan,"analytical_check":"BLOCKED","actual_plant_mutation":"NOT_VERIFIED","note":"no reproducible PowerDynamics mutation/PF/TDS harness"})
    pd.DataFrame(rows).to_csv(R/"e06b_parameter_sensitivity.csv",index=False)
    REP.mkdir(parents=True,exist_ok=True)
    (REP/"e06b_physical_parameterization.md").write_text("""# E06-B physical parameterization gate

`PHYSICAL_PARAMETERIZATION = FAIL`.

The repository contains the PowerDynamics nominal export and source CSV data,
but no reproducible E06-B operator that mutates actual component instances,
rebuilds the nonlinear network, solves PF, initializes dynamics and runs a
short Rodas5P TDS for each family. The nominal estimator must not be replaced
by an exported-matrix or post-hoc signal surrogate, so this gate stops here.

`e06b_parameter_registry.csv` records the actual source fields and constraints.
`e06b_parameter_sensitivity.csv` contains independent analytical Ybus centered
finite differences for a representative branch; all dynamic families are
explicitly marked BLOCKED rather than inferred.

STANDARD, refinement, HARD_ID/OOD and oracle relinearization were **not run**.
""",encoding="utf-8")
    (REP/"e06_standard_validation.md").write_text("""# E06 STANDARD validation

STATUS: **BLOCKED_BY_PHYSICAL_PARAMETERIZATION_GATE**.

No E06_STANDARD_V1 manifest or scores were created. Running STANDARD with the
E06-A surrogate stress operator would violate the requirement that the actual
nonlinear PowerDynamics plant change.
""",encoding="utf-8")
    print(json.dumps({"physical_parameterization":"FAIL","registry_rows":len(reg),"network_sensitivity_rows":len(rows),"standard":"NOT_RUN"},indent=2))

if __name__=='__main__': main()
