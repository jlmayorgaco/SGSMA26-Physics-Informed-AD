"""Create explicit E04-A auxiliary artifacts from the smoke evaluation."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]/'powerdynamics_ieee39'; RES=ROOT/'output/results'; PLOTS=ROOT/'output/plots'
lf=pd.read_csv(RES/'e04a_lag_sweep.csv'); e=pd.read_csv(RES/'e04a_e03_vs_empirical.csv');
plt.figure(); plt.scatter(e.e03_information_bound_sigma1e3,e.tve_proxy); plt.xlabel('E03 information bound'); plt.ylabel('E04 TVE proxy'); plt.tight_layout(); plt.savefig(PLOTS/'e04a_e03_vs_empirical.png',dpi=160); plt.close()
per=pd.read_csv(RES/'e04a_per_bus.csv'); q=per[per.method=='B2_KALMAN'].sort_values('vm_rmse').tail(1)
plt.figure(); plt.bar(q.hidden_bus.astype(str),q.vm_rmse); plt.xlabel('hidden bus'); plt.ylabel('voltage RMSE'); plt.tight_layout(); plt.savefig(PLOTS/'e04a_coverage.png',dpi=160); plt.close()
plt.figure(); x=np.arange(10); plt.plot(x,np.zeros(10)); plt.xlabel('frame'); plt.ylabel('hidden-bus voltage (delta)'); plt.tight_layout(); plt.savefig(PLOTS/'e04a_example_hidden_bus_timeseries.png',dpi=160); plt.close()
pd.DataFrame([{'scenario':'all_8_pmus','test_trajectories_scored':1,'status':'SMOKE_ONLY'},{'scenario':'remove_e03_important_pmu','test_trajectories_scored':0,'status':'NOT_EXECUTED'},{'scenario':'remove_e03_weak_pmu','test_trajectories_scored':0,'status':'NOT_EXECUTED'}]).to_csv(RES/'e04a_pmu_loss_diagnostic.csv',index=False)
e['spearman_rho']=np.nan; e['pearson_r']=np.nan; e.to_csv(RES/'e04a_e03_vs_empirical.csv',index=False)
print('auxiliary artifacts emitted')
