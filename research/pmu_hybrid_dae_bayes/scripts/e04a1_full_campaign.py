"""Full frozen TEST B0/B1/B2 campaign and calibration/theory diagnostics.

Runs only the linear-Gaussian estimators; B3 is intentionally left for the
post-gate smoother task.  Q/R/P0 are read from the frozen nominal configuration.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import spearmanr, pearsonr
from pmu_hybrid.e04a_data import observed_measurements, evaluation_ground_truth
from pmu_hybrid.e04a_estimator import LinearGaussianModel, snapshot_wls, kalman_filter, interleaved_to_complex, wrapped_angle_error

ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39"; RES = ROOT / "output/results"; REP = ROOT / "output/reports"; PLOTS = ROOT / "output/plots"
OBS = [2,5,6,10,19,22,29,39]; HIDDEN = [b for b in range(1,40) if b not in OBS]

def ci(values, seed=20260911, n=2000):
    a=np.asarray(values,float); rng=np.random.default_rng(seed); means=np.mean(a[rng.integers(0,len(a),(n,len(a)))],axis=1); return float(np.quantile(means,.025)),float(np.quantile(means,.975))

def score(zp, zt):
    e=zp-zt; dth=wrapped_angle_error(np.angle(zp),np.angle(zt)); tve=np.abs(e)/np.maximum(np.abs(zt),1e-12)
    return dict(complex_rmse=float(np.sqrt(np.mean(np.abs(e)**2))), vm_rmse=float(np.sqrt(np.mean((np.abs(zp)-np.abs(zt))**2))), angle_rmse=float(np.sqrt(np.mean(np.rad2deg(dth)**2))), TVE_fraction=float(np.mean(tve)), TVE_percent=float(100*np.mean(tve)))

def run_model(model, y, h0, Ct):
    t0=time.perf_counter(); xs=[snapshot_wls(model,v)[0] for v in y]; b1=time.perf_counter()-t0
    t0=time.perf_counter(); kf=kalman_filter(model,y); b2=time.perf_counter()-t0
    p1=h0+np.vstack(xs)@Ct.T; p2=h0+np.vstack([q[0] for q in kf])@Ct.T
    return p1,p2,kf,b1,b2

def main():
    A=expm(pd.read_csv(RES/'e04_A.csv').to_numpy(float)/30); C=pd.read_csv(RES/'e04_C_pmu.csv').to_numpy(float); Ct=pd.read_csv(RES/'e04_C_hidden.csv').to_numpy(float)
    y0=pd.read_csv(RES/'e04_y0_pmu.csv').iloc[:,0].to_numpy(float); h0=pd.read_csv(RES/'e04_pd_hidden0.csv').iloc[:,0].to_numpy(float); ds=pd.read_csv(RES/'e04_pd_dataset.csv'); test=ds[ds.split=='TEST']
    model=LinearGaussianModel(A,C,np.eye(114)*1e-6,np.eye(32)*1e-6,np.eye(114)*1e-2)
    records=[]; bus_rows=[]; uncertainty=[]; innovations=[]; runt=[]; outputs={}
    for ti,(traj,g) in enumerate(test.groupby('traj',sort=True)):
        g=g.sort_values('frame'); y=np.vstack([observed_measurements(r)-y0 for _,r in g.iterrows()]); true=np.vstack([evaluation_ground_truth(r) for _,r in g.iterrows()]); zt=interleaved_to_complex(true)
        p1,p2,kf,t1,t2=run_model(model,y,h0,Ct); p0=np.tile(h0,(len(g),1)); outputs[traj]=(g,true,p0,p1,p2,kf)
        runt += [{'traj':traj,'method':'B1_SNAPSHOT_WLS','seconds_per_frame':t1/len(g)}, {'traj':traj,'method':'B2_KALMAN','seconds_per_frame':t2/len(g)}]
        for frame in range(len(g)):
            for meth,p in [('B0_NOMINAL',p0),('B1_SNAPSHOT_WLS',p1),('B2_KALMAN',p2)]:
                zp=interleaved_to_complex(p[frame]); e=zp-zt[frame]; tv=np.abs(e)/np.maximum(np.abs(zt[frame]),1e-12); da=np.rad2deg(wrapped_angle_error(np.angle(zp),np.angle(zt[frame])))
                records.append({'traj':traj,'seed':int(g.seed.iloc[0]),'scenario':str(g.scenario.iloc[0]),'amplitude_a':float(g.amplitude_a.iloc[0]),'amplitude_b':float(g.amplitude_b.iloc[0]),'solver_status':str(g.retcode.iloc[0]),'frame':frame,'time_s':float(g.time.iloc[frame]),'method':meth,'complex_rmse':float(np.sqrt(np.mean(np.abs(e)**2))),'vm_rmse':float(np.sqrt(np.mean((np.abs(zp)-np.abs(zt[frame]))**2))),'angle_rmse':float(np.sqrt(np.mean(da**2))),'TVE_fraction':float(np.mean(tv)),'TVE_percent':float(100*np.mean(tv))})
            # B2 componentwise 95% intervals and 2D NLL per hidden bus.
            x,P,innov,S=kf[frame]; Ph=Ct@P@Ct.T
            for j,bus in enumerate(HIDDEN):
                d=true[frame,2*j:2*j+2]-(h0[2*j:2*j+2]+(Ct@x)[2*j:2*j+2]); cov=Ph[2*j:2*j+2,2*j:2*j+2]; sd=np.sqrt(np.maximum(np.diag(cov),1e-15)); uncertainty.append({'traj':traj,'frame':frame,'hidden_bus':bus,'coverage_re':bool(abs(d[0])<=1.96*sd[0]),'coverage_im':bool(abs(d[1])<=1.96*sd[1]),'nll':float(0.5*(2*np.log(2*np.pi)+np.linalg.slogdet(cov+np.eye(2)*1e-12)[1]+d@np.linalg.solve(cov+np.eye(2)*1e-12,d))),'predicted_variance':float(np.trace(cov)/2),'abs_error':float(np.linalg.norm(d))})
            innovations.append({'traj':traj,'frame':frame,'nis':float(innov@np.linalg.solve(S,innov)),'innovation_norm':float(np.linalg.norm(innov))})
    rf=pd.DataFrame(records); rf.to_parquet(RES/'e04a_full_per_case.parquet',index=False); rf.to_csv(RES/'e04a_full_per_case.csv',index=False)
    # Aggregate summaries and bootstrap confidence intervals by trajectory.
    rows=[]
    for method,g in rf.groupby('method'):
        tg=g.groupby('traj')[['complex_rmse','vm_rmse','angle_rmse','TVE_fraction','TVE_percent']].mean()
        row={'method':method,'n_trajectories':len(tg)}
        for c in tg: row[c]=float(tg[c].mean()); row[c+'_median']=float(tg[c].median()); row[c+'_p95']=float(tg[c].quantile(.95)); row[c+'_ci95_low'],row[c+'_ci95_high']=ci(tg[c].to_numpy())
        row['worst_trajectory']=str(tg['TVE_fraction'].idxmax()); rows.append(row)
    pd.DataFrame(rows).to_csv(RES/'e04a_b0_b1_b2_summary.csv',index=False)
    # Per-bus metrics and uncertainty calibration.
    bus=[]
    for method in ['B0_NOMINAL','B1_SNAPSHOT_WLS','B2_KALMAN']:
        for i,bus_id in enumerate(HIDDEN):
            vals=[]; angs=[]
            for traj,g in outputs.items():
                _,true,p0,p1,p2,_=g; p={'B0_NOMINAL':p0,'B1_SNAPSHOT_WLS':p1,'B2_KALMAN':p2}[method]; z=interleaved_to_complex(p)[:,i]; t=interleaved_to_complex(true)[:,i]; vals.extend(np.abs(z-t)/np.maximum(np.abs(t),1e-12)); angs.extend(np.rad2deg(wrapped_angle_error(np.angle(z),np.angle(t))))
            bus.append({'method':method,'hidden_bus':bus_id,'TVE_fraction':float(np.mean(vals)),'TVE_percent':float(100*np.mean(vals)),'angle_rmse':float(np.sqrt(np.mean(np.asarray(angs)**2)))})
    busdf=pd.DataFrame(bus); busdf.to_csv(RES/'e04a_full_per_bus.csv',index=False)
    ud=pd.DataFrame(uncertainty); ud.to_csv(RES/'e04a_uncertainty.csv',index=False)
    # Innovation diagnostics.
    inn=pd.DataFrame(innovations); inn.to_csv(RES/'e04a_innovations.csv',index=False)
    acf_raw=inn.innovation_norm.autocorr(); acf=float(0.0 if pd.isna(acf_raw) else acf_raw) if len(inn)>1 else 0.0
    paired=[]
    tg=rf.groupby(['method','traj'])[['TVE_fraction','angle_rmse','vm_rmse']].mean().reset_index(); piv=tg.pivot(index='traj',columns='method')
    for a,b in [('B1_SNAPSHOT_WLS','B0_NOMINAL'),('B2_KALMAN','B0_NOMINAL'),('B2_KALMAN','B1_SNAPSHOT_WLS')]:
        for c in ['TVE_fraction','angle_rmse','vm_rmse']:
            d=piv[c][a]-piv[c][b]; lo,hi=ci(d.to_numpy()); paired.append({'comparison':a+' - '+b,'metric':c,'mean_difference':float(d.mean()),'paired_bootstrap_ci95_low':lo,'paired_bootstrap_ci95_high':hi})
    pd.DataFrame(paired).to_csv(RES/'e04a_paired_comparisons.csv',index=False)
    # E03 -> E04 bus-level theory test.
    e03=pd.read_csv(RES/'pd_e03_per_bus_predictions.csv'); e03=e03[e03.horizon_frames==180].copy(); b1=busdf[busdf.method=='B1_SNAPSHOT_WLS'][['hidden_bus','TVE_fraction','angle_rmse']].rename(columns={'TVE_fraction':'B1_TVE','angle_rmse':'B1_angle_rmse'}); b2=busdf[busdf.method=='B2_KALMAN'][['hidden_bus','TVE_fraction','angle_rmse']].rename(columns={'TVE_fraction':'B2_TVE','angle_rmse':'B2_angle_rmse'}); cov=ud.groupby('hidden_bus').agg(coverage=('coverage_re','mean'),posterior_variance=('predicted_variance','mean')).reset_index(); theory=e03.merge(b1,on='hidden_bus').merge(b2,on='hidden_bus').merge(cov,on='hidden_bus');
    for x in ['B1_TVE','B2_TVE','B1_angle_rmse','B2_angle_rmse','coverage','posterior_variance']:
        for y in ['functional_residual','information_bound_sigma1e3']:
            r=spearmanr(theory[x],theory[y]); theory[f'spearman_{x}_vs_{y}']=float(r.statistic)
    theory.to_csv(RES/'e04a1_e03_vs_e04.csv',index=False)
    # PMU-loss diagnostic on first 20 TEST trajectories; rebuild C by deleting PMU pairs.
    important=[39,2,29,5,6,10,19,22]; loss=[]
    for bus_id in [None,39,2]:
        keep=np.ones(32,dtype=bool)
        if bus_id is not None:
            j=OBS.index(bus_id); keep[2*j:2*j+2]=False; keep[16+2*j:16+2*j+2]=False
        cm=LinearGaussianModel(A,C[keep],np.eye(114)*1e-6,np.eye(int(keep.sum()))*1e-6,np.eye(114)*1e-2)
        for traj,g in list(test.groupby('traj',sort=True))[:20]:
            g=g.sort_values('frame'); yy=np.vstack([observed_measurements(r)[keep]-y0[keep] for _,r in g.iterrows()]); tt=interleaved_to_complex(np.vstack([evaluation_ground_truth(r) for _,r in g.iterrows()])); p1,p2,_,_,_=run_model(cm,yy,h0,Ct)
            for m,p in [('B1',p1),('B2',p2)]: loss.append({'pmu_mask':'all_8' if bus_id is None else 'remove_bus_'+str(bus_id),'method':m,**score(interleaved_to_complex(p),tt)})
    pd.DataFrame(loss).to_csv(RES/'e04a_pmu_loss.csv',index=False)
    rt=pd.DataFrame(runt); rt.groupby('method').seconds_per_frame.agg(['median',lambda x:np.quantile(x,.95)]).rename(columns={'<lambda_0>':'p95'}).reset_index().to_csv(RES/'e04a_runtime.csv',index=False)
    summary={'test_trajectories':int(test.traj.nunique()),'test_frames':int(len(rf)),'innovation_mean_nis':float(inn.nis.mean()),'innovation_variance_nis':float(inn.nis.var()),'innovation_norm_lag1_autocorrelation':acf,'coverage_re_mean':float(ud.coverage_re.mean()),'coverage_im_mean':float(ud.coverage_im.mean()),'nll_mean':float(ud.nll.mean()),'worst_coverage_bus':int(ud.groupby('hidden_bus').coverage_re.mean().sub(.95).abs().idxmax())}
    (RES/'e04a_full_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (REP/'e04a_full_validation.md').write_text(f"""# E04-A full B0/B1/B2 validation

This run scored all **{summary['test_trajectories']} frozen TEST trajectories**
({summary['test_frames']} trajectory-frame-method rows) after the regenerated
PowerDynamics export contract passed.  Q/R/P0 were preserved from the frozen
nominal configuration; no TEST data were used for selection.

Primary outputs are `e04a_b0_b1_b2_summary.csv`, `e04a_full_per_case.parquet`,
`e04a_full_per_bus.csv`, `e04a_uncertainty.csv`, `e04a1_e03_vs_e04.csv`, and
`e04a_pmu_loss.csv`.  Paired bootstrap comparisons use seed 20260911.

Aggregate uncertainty diagnostics: mean component coverage
Re={summary['coverage_re_mean']:.4f}, Im={summary['coverage_im_mean']:.4f}; mean
hidden 2-D NLL={summary['nll_mean']:.6g}; mean observed NIS={summary['innovation_mean_nis']:.6g}.
Innovation lag-1 autocorrelation of the norm is {summary['innovation_norm_lag1_autocorrelation']:.6g}.

B3 was not executed in this stage; its fixed-lag optimization/oracle validation
remains the explicit remaining gate item.
""",encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
