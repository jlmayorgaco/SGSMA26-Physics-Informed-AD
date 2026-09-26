from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]/"powerdynamics_ieee39"; R=ROOT/"output/results"; REP=ROOT/"output/reports"
per=pd.read_parquet(R/"e06_standard_per_case.parquet"); b=per[per.method=="B2"].copy(); nominal=b[b.m==0].TVE_fraction.median(); b["canonical_ratio"]=b.TVE_fraction/nominal
rows=[]
for (f,m),g in b.groupby(["family","m"]): rows.append({"family":f,"m":m,"nominal_reference_TVE_fraction":nominal,"nominal_reference_TVE_percent":100*nominal,"aggregation":"median of per-trajectory ratios","numerator":"per-case hidden-31 B2 TVE","denominator":"median B2 TVE across all STANDARD m=0 cases","excitation_stratum":"all E-A..E-D; 20 seeds/cell","seed_subset":"1..20 main grid","grid":"main","median_ratio":g.canonical_ratio.median(),"mean_ratio":g.canonical_ratio.mean(),"n":len(g)})
audit=pd.DataFrame(rows); audit.to_csv(R/"e06e_breakpoint_audit.csv",index=False); br=[]
for f,g in audit.groupby("family"):
 for th in (2,5,10):
  q=g.sort_values("m"); hit=q[q.median_ratio>=th]; hi=float(hit.m.iloc[0]) if len(hit) else np.nan; lo=float(q[q.m<hi].m.max()) if len(hit) and len(q[q.m<hi]) else np.nan; br.append({"family":f,"threshold":f"{th}x","lower_m":lo,"upper_m":hi,"interval":"not_reached" if np.isnan(hi) else f"[{lo if not np.isnan(lo) else 0},{hi}]"})
pd.DataFrame(br).to_csv(R/"e06e_breakpoints_canonical.csv",index=False)
(REP/"e06e_breakpoint_audit.md").write_text(f"""# E06-E breakpoint consistency audit\n\nCanonical definition: for each STANDARD B2 trajectory, `r_i(m)=TVE_i(m)/median(TVE_nominal)` where the denominator is the median hidden-31 TVE over all 140 m=0 STANDARD cases. Cell values are the median of these per-trajectory ratios across the 20 independent seeds, retaining the E-A…E-D strata. No mean-based breakpoint is mixed into the canonical table.\n\nThe previous M7 `[0.00,0.25]` label came from the separate refinement-seed table and a different direct-median aggregation; it is not comparable to the main-grid table. Canonical breakpoints are in `e06e_breakpoints_canonical.csv`.\n\nReference TVE = **{100*nominal:.8g}%**. Full audit fields are in `e06e_breakpoint_audit.csv`.\n""",encoding="utf-8")
print(audit[audit.median_ratio>=2].to_string(index=False))
