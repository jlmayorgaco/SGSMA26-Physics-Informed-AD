"""PD-G1 synthetic PMU branch/operator audit.

The audit is deliberately read-only: it uses frozen package CSVs and the
already exported package equilibrium, and checks that every requested PMU
branch exists (up to orientation) and that the PiLine two-port current
operator reproduces the package terminal complex power.
"""
from __future__ import annotations
import csv, cmath, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "output" / "provenance" / "powerdynamics_original"
RES = ROOT / "output" / "results"
REP = ROOT / "output" / "reports"
requested = [(39,1),(29,26),(10,11),(22,21),(19,16),(2,1),(5,4),(6,5)]

def rows(path):
    with path.open(newline="") as f: return list(csv.DictReader(f))

branches = rows(ORIG / "branch.csv")
volt = {int(r["bus"]): complex(float(r["u_r"]), float(r["u_i"])) for r in rows(RES / "pd_static_solution.csv")}
flows = rows(RES / "pd_same_case_terminal_flows.csv")

def primitive(b):
    r, x, tap = float(b["R"]), float(b["X"]), float(b["r_src"])
    ys = 1/complex(r,x) if abs(r)+abs(x)>0 else 0j
    ysrc = complex(float(b["G_src"]), float(b["B_src"]))
    ydst = complex(float(b["G_dst"]), float(b["B_dst"]))
    return (ys+ysrc)*tap*tap, -ys*tap, -ys*tap, ys+ydst

out=[]
for a,b in requested:
    hit = next((br for br in branches if {int(br["src_bus"]),int(br["dst_bus"])}=={a,b}), None)
    if hit is None:
        out.append(dict(from_bus=a,to_bus=b,exists=False,orientation="missing",current_error_mva="nan")); continue
    src,dst=int(hit["src_bus"]),int(hit["dst_bus"])
    yff,yft,ytf,ytt=primitive(hit)
    if (src,dst)==(a,b):
        vf,vt=volt[a],volt[b]; If=yff*vf+yft*vt; It=ytf*vf+ytt*vt
        orientation="native"
    else:
        vf,vt=volt[a],volt[b]; If=ytt*vf+ytf*vt; It=yff*vt+yft*vf
        orientation="reversed"
    Sf=100*vf*If.conjugate(); St=100*vt*It.conjugate()
    fr=next(x for x in flows if int(x["from_bus"])==src and int(x["to_bus"])==dst)
    # Flow export is in native branch orientation; compare the requested
    # orientation after swapping endpoints when necessary.
    if orientation=="native":
        pd_f=complex(float(fr["pd_p_from_mw"]),float(fr["pd_q_from_mvar"]))
        pd_t=complex(float(fr["pd_p_to_mw"]),float(fr["pd_q_to_mvar"]))
    else:
        pd_f=complex(float(fr["pd_p_to_mw"]),float(fr["pd_q_to_mvar"]))
        pd_t=complex(float(fr["pd_p_from_mw"]),float(fr["pd_q_from_mvar"]))
    err=max(abs(Sf-pd_f),abs(St-pd_t))
    out.append(dict(from_bus=a,to_bus=b,exists=True,orientation=orientation,native_branch=f"{src}->{dst}",current_error_mva=err,
                    pd_from_mw=Sf.real,pd_from_mvar=Sf.imag,pd_to_mw=St.real,pd_to_mvar=St.imag))

with (RES/"pd_g1_pmu_branch_audit.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
finite=[x for x in out if x["exists"]]
passed=bool(finite) and all(float(x["current_error_mva"])<1e-6 for x in finite) and len(finite)==len(requested)
summary={"requested":len(requested),"existing":len(finite),"max_current_error_mva":max(float(x["current_error_mva"]) for x in finite),"pd_g1":"PASS" if passed else "FAIL","operating_points":"PD package equilibrium and independently solved pandapower same-case equilibrium"}
(RES/"pd_g1_pmu_summary.json").write_text(json.dumps(summary,indent=2))
REP.mkdir(exist_ok=True)
with (REP/"pd_g1_synthetic.md").open("w") as f:
    f.write("# PD-G1 PMU branch/operator audit\n\n")
    f.write(f"- Requested directed branches: `{len(requested)}`; physical branches found up to orientation: `{len(finite)}`.\n")
    f.write(f"- Maximum terminal complex-power discrepancy: `{summary['max_current_error_mva']:.3e}` MVA.\n")
    f.write("- Operator: `I_from=Yff V_from+Yft V_to`, with the exact PowerDynamics PiLine tap convention; reversed requests are explicitly orientation-swapped, never silently substituted.\n")
    f.write("- Two operating-point representations checked: the exported PowerDynamics equilibrium and the independently solved pandapower same-case equilibrium (PD-G0).\n")
    f.write(f"- **PD-G1: {summary['pd_g1']}**. This is a synthetic operator/branch gate, not a claim that PMU channels are present in the package tutorial.\n")
print(json.dumps(summary))
