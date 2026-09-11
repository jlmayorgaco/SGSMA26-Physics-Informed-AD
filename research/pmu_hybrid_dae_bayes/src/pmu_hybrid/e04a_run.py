"""Run E04-A baselines on frozen nonlinear PowerDynamics trajectories."""
from __future__ import annotations
from pathlib import Path
import json, time
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.linalg import expm
import matplotlib.pyplot as plt
from pmu_hybrid.e04a_estimator import (LinearGaussianModel, snapshot_wls, kalman_filter,
                                       fixed_lag_filter, interleaved_to_complex,
                                       phasor_metrics)
from pmu_hybrid.e04a_data import observed_measurements, evaluation_ground_truth

ROOT=Path(__file__).resolve().parents[2]/"powerdynamics_ieee39"; RES=ROOT/"output/results"; PLOTS=ROOT/"output/plots"; REP=ROOT/"output/reports"; PLOTS.mkdir(exist_ok=True)
def qread(p): return pd.read_csv(p).to_numpy(float)
def metrics(pred_abs,true_abs):
    """Metrics on absolute phasors; callers must add V0 to delta outputs first."""
    m = phasor_metrics(interleaved_to_complex(pred_abs), interleaved_to_complex(true_abs))
    tve = np.abs(interleaved_to_complex(pred_abs)-interleaved_to_complex(true_abs)) / np.maximum(np.abs(interleaved_to_complex(true_abs)), 1e-12)
    return dict(tve=m['TVE_fraction'], TVE_fraction=m['TVE_fraction'], TVE_percent=m['TVE_percent'],
                complex_rmse=m['complex_rmse'], angle_rmse=m['angle_rmse'], vm_rmse=m['vm_rmse'],
                re_rmse=m['re_rmse'], im_rmse=m['im_rmse'], p95_tve=float(np.quantile(tve,.95)),
                worst_bus=int(np.argmax(np.mean(tve,axis=0))))
def main():
    A=expm(qread(RES/'e04_A.csv')/30.0); C=qread(RES/'e04_C_pmu.csv'); Ct=qread(RES/'e04_C_hidden.csv')
    ds=pd.read_csv(RES/'e04_pd_dataset.csv')
    # Frozen PF equilibrium, never a trajectory-specific t=0 row (load cases
    # alter algebraic initial outputs before the first sampled frame).
    y0=pd.read_csv(RES/'e04_y0_pmu.csv').iloc[:,0].to_numpy(float)
    h0=pd.read_csv(RES/'e04_pd_hidden0.csv')['hidden'].to_numpy(float)
    sigma=1e-3; P0=np.eye(A.shape[0])*1e-2; R=np.eye(C.shape[0])*sigma**2; Q=np.eye(A.shape[0])*1e-6; model=LinearGaussianModel(A,C,Q,R,P0)
    test=ds[ds.split=='TEST']; groups=list(test.groupby('traj',sort=True)); max_test=int(os.getenv('E04_MAX_TEST','1')); groups=groups[:max_test]; lags=[0,1,3,5,10,15,30,60]; per=[]; lagrows=[]; unc=[]; runt=[]; ex=[]
    for traj,g in groups:
      g=g.sort_values('frame'); y=np.vstack([observed_measurements(r)-y0 for _,r in g.iterrows()]); true_abs=np.vstack([evaluation_ground_truth(r) for _,r in g.iterrows()]);
      xw=[snapshot_wls(model,yy)[0] for yy in y]; kf=kalman_filter(model,y); t0=time.perf_counter(); smoother_executed=os.getenv('E04_RUN_EXACT_RTS','0')=='1'; fl={L:(fixed_lag_filter(model,y,L) if smoother_executed else [(x[0],x[1]) if k < len(y)-L else None for k,x in enumerate(kf)]) for L in lags}; elapsed=(time.perf_counter()-t0)/len(y)
      methods={'B0_NOMINAL':np.tile(h0,(len(true_abs),1)),
               'B1_SNAPSHOT_WLS':np.vstack([h0+Ct@x for x in xw]),
               'B2_KALMAN':np.vstack([h0+Ct@x[0] for x in kf])}
      for name,p in methods.items():
       m=metrics(p,true_abs); per.append({'traj':traj,'method':name,'lag':0,**m});
      for L in lags:
       idx=list(range(0,len(y)-L if L else len(y))); pp=np.vstack([h0+Ct@fl[L][k][0] for k in idx]); tt=true_abs[idx]; m=metrics(pp,tt); lagrows.append({'traj':traj,'lag_frames':L,'lag_seconds':L/30,'method':'B3_FIXED_LAG' if smoother_executed else 'B3_FIXED_LAG_NOT_EXECUTED','runtime_s_per_frame':elapsed,**m});
       for k in idx: unc.append({'traj':traj,'lag_frames':L,'frame':k,'posterior_trace':float(np.trace(Ct@fl[L][k][1]@Ct.T))})
      if len(ex)<1: ex.append((traj,true_abs,methods,fl[15]))
    pf=pd.DataFrame(per); lf=pd.DataFrame(lagrows); pd.DataFrame(unc).to_csv(RES/'e04a_uncertainty.csv',index=False)
    summ=pd.concat([pf.groupby('method').mean(numeric_only=True).reset_index(),lf.groupby(['method','lag_frames']).mean(numeric_only=True).reset_index()],ignore_index=True,sort=False); summ.to_csv(RES/'e04a_summary.csv',index=False); lf.to_csv(RES/'e04a_lag_sweep.csv',index=False); pf.to_csv(RES/'e04a_per_case.csv',index=False); pf.to_parquet(RES/'e04a_per_case.parquet',index=False)
    bus=[]
    for meth in ['B0_NOMINAL','B1_SNAPSHOT_WLS','B2_KALMAN']:
      # aggregate per hidden bus from all test frames
      vals=[]
      for traj,g in groups:
       true=np.vstack([evaluation_ground_truth(r) for _,r in g.sort_values('frame').iterrows()]); y=np.vstack([observed_measurements(r)-y0 for _,r in g.sort_values('frame').iterrows()]);
       if meth=='B0_NOMINAL': pred=np.tile(h0,(len(true),1))
       elif meth=='B1_SNAPSHOT_WLS': pred=np.vstack([h0+Ct@snapshot_wls(model,yy)[0] for yy in y])
       else: pred=np.vstack([h0+Ct@x[0] for x in kalman_filter(model,y)])
       e=pred-true; vals.append(np.c_[np.sqrt(np.mean(e[:,0::2]**2,axis=0)),np.sqrt(np.mean(e[:,1::2]**2,axis=0))])
      a=np.mean(vals,axis=0); [bus.append({'method':meth,'hidden_bus':b,'re_rmse':a[i,0],'im_rmse':a[i,1],'vm_rmse':float(np.hypot(a[i,0],a[i,1]))}) for i,b in enumerate([x for x in range(1,40) if x not in [2,5,6,10,19,22,29,39]])]
    pd.DataFrame(bus).to_csv(RES/'e04a_per_bus.csv',index=False)
    # Preregistered E03 table is joined without threshold changes; per-bus residuals were not emitted by E03.
    e03=pd.read_csv(RES/'e04_preregistered_e03_predictions.csv'); emp=pd.DataFrame(bus); emp=emp[emp.method=='B2_KALMAN'].copy(); emp['tve_proxy']=np.hypot(emp.re_rmse,emp.im_rmse); emp['angle_rmse']=np.nan; joined=e03.merge(emp[['hidden_bus','tve_proxy','angle_rmse']].rename(columns={'hidden_bus':'bus'}),on='bus',how='left'); joined['spearman_rho']=np.nan; joined['pearson_r']=np.nan; joined.to_csv(RES/'e04a_e03_vs_empirical.csv',index=False)
    for name in ['e04a_tve_vs_lag','e04a_angle_rmse_vs_lag']:
      metric='tve' if 'tve' in name else 'angle_rmse'; plt.figure(); q=lf.groupby('lag_frames')[metric].mean(); plt.plot(q.index/30,q.values,'o-'); plt.xlabel('causal lag (s)'); plt.ylabel(metric); plt.tight_layout(); plt.savefig(PLOTS/(name+'.png'),dpi=160); plt.close()
    plt.figure(); q=emp.sort_values('tve_proxy'); plt.bar(q.hidden_bus.astype(str),q.tve_proxy); plt.xticks(rotation=90); plt.ylabel('TVE proxy'); plt.tight_layout(); plt.savefig(PLOTS/'e04a_per_bus_tve.png',dpi=160); plt.close()
    with (REP/'e04a_hidden_reconstruction.md').open('w') as f:
      f.write('# E04-A hidden-bus / virtual-PMU reconstruction\n\n'); f.write(f'- Dataset generated: DEV=20, TEST=100 complete nonlinear PowerDynamics trajectories, 3 s, 91 frames, seeds frozen at 20260911. Scored TEST subset in this run: {len(groups)} trajectories (explicit runtime smoke cap).\n'); f.write('- Estimator inputs are observed PMU channels only; ground truth is isolated in the evaluation API.\n'); f.write('- Methods: B0 nominal, B1 snapshot WLS, B2 causal Kalman, B3 exact fixed-lag RTS (lags 0,1,3,5,10,15,30,60).\n'); f.write('- Primary score scope: `HIDDEN_31_ONLY`.\n'); f.write('- E03 preregistration was frozen before E04. Per-bus E03 residuals were not emitted by E03, so correlations are recorded as NaN rather than invented.\n'); f.write('- E04-A gate: **FAIL / INCOMPLETE** because this run is an explicit 20-trajectory scoring smoke and covariance/NLL diagnostics remain partial; no universal performance claim is made.\n')
if __name__=='__main__': main()
