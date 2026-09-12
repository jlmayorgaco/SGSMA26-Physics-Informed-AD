from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]/"powerdynamics_ieee39"; R=ROOT/"output/results"; P=ROOT/"output/plots"; P.mkdir(parents=True,exist_ok=True)
a=pd.read_csv(R/"e06_standard_adequacy.csv"); plt.figure(figsize=(8,4));
for f,g in a.groupby("family"): plt.plot(g.m,g.AUROC,"o-",label=f)
plt.axhline(.5,color="k",lw=.7); plt.xlabel("mismatch m"); plt.ylabel("CAL-only AUROC"); plt.legend(fontsize=7,ncol=2); plt.tight_layout(); plt.savefig(P/"e06_standard_adequacy_detector.png",dpi=140); plt.close()
o=pd.read_csv(R/"e06d_family_summary.csv"); plt.figure(figsize=(8,4)); x=range(len(o)); plt.plot(x,o.O1_TVE_percent,"s-",label="O1 recenter"); plt.plot(x,o.O2_TVE_percent,"^-",label="O2 relinearize"); plt.xticks(list(x),o.family,rotation=25); plt.ylabel("oracle TVE (%)"); plt.legend(); plt.tight_layout(); plt.savefig(P/"e06_standard_oracle_n0_o1_o2.png",dpi=140); plt.close()
plt.figure(figsize=(8,4)); plt.bar([i-.15 for i in x],o.RECOVERY_RECENTER_TVE,.3,label="O1"); plt.bar([i+.15 for i in x],o.RECOVERY_RELINEARIZE_TVE,.3,label="O2"); plt.xticks(list(x),o.family,rotation=25); plt.ylabel("recovery fraction"); plt.legend(); plt.tight_layout(); plt.savefig(P/"e06_standard_recovery_o1_o2_by_family.png",dpi=140); plt.close()
bus=pd.read_csv(R/"e06_standard_per_bus.csv"); bus=bus[(bus.method=="B2") & bus.hidden_bus.isin([33,34,20,37,38])]; plt.figure(figsize=(8,4));
for b,g in bus.groupby("hidden_bus"): plt.plot(g.m,g.TVE_fraction*100,"o-",label=f"bus {b}")
plt.xlabel("mismatch m"); plt.ylabel("TVE (%)"); plt.legend(ncol=2,fontsize=8); plt.tight_layout(); plt.savefig(P/"e06_standard_weak_buses_vs_mismatch.png",dpi=140); plt.close()
per=pd.read_parquet(R/"e06_standard_per_case.parquet"); g=per[(per.family=="M7_COUPLED")&(per.method=="B2")].groupby("m").TVE_percent.median(); plt.figure(figsize=(6,4)); plt.plot(g.index,g.values,"o-"); plt.axvline(1,color="k",ls="--",label="HARD_ID m=1"); plt.axvline(1.5,color="r",ls="--",label="OOD m=1.5"); plt.xlabel("M7 mismatch m"); plt.ylabel("TVE (%)"); plt.legend(); plt.tight_layout(); plt.savefig(P/"e06_standard_m7_hard_ood.png",dpi=140); plt.close()
print("plots finalized")
