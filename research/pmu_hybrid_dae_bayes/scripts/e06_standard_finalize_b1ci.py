from pathlib import Path
import numpy as np, pandas as pd
R=Path(__file__).resolve().parents[1]/"powerdynamics_ieee39"/"output"/"results"; p=pd.read_csv(R/"e06_standard_b1_vs_b2.csv"); rng=np.random.default_rng(20260912); rows=[]
for (f,m),g in p.groupby(["family","m"]):
 d=g.TVE_B2_minus_B1.to_numpy(); a=g.angle_B2_minus_B1.to_numpy(); bd=[np.mean(rng.choice(d,len(d),replace=True)) for _ in range(1000)]; ba=[np.mean(rng.choice(a,len(a),replace=True)) for _ in range(1000)]; rows.append({"family":f,"m":m,"n":len(g),"TVE_B2_minus_B1_mean":d.mean(),"TVE_ci_low":np.quantile(bd,.025),"TVE_ci_high":np.quantile(bd,.975),"angle_B2_minus_B1_mean":a.mean(),"angle_ci_low":np.quantile(ba,.025),"angle_ci_high":np.quantile(ba,.975),"temporal_class":"BENEFICIAL" if d.mean()<-1e-8 else "HARMFUL" if d.mean()>1e-8 else "NEUTRAL"})
pd.DataFrame(rows).to_csv(R/"e06_standard_b1_vs_b2_ci.csv",index=False); print("B1/B2 CIs finalized")
