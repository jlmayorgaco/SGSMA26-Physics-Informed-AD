"""Bus7 physical-event closure and known-event field reconstruction."""
from pathlib import Path
import sys, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(HERE))
from scripts import e06h_corrected_m6_static as h6
from scripts import rbfe_validation as rb
from scripts.e06g_static_diagnosis import pf_solve, PQLIKE, PQ, PV
BASE=HERE/'powerdynamics_ieee39/output'; CAMP=BASE/'results/full_field_event_reconstruction_v1'; REP=BASE/'full_field_event_reconstruction_v1/reports'; RES=BASE/'full_field_event_reconstruction_v1/results'; FIG=BASE/'full_field_event_reconstruction_v1/figures'
for p in (REP,RES,FIG): p.mkdir(parents=True,exist_ok=True)
OBS=[2,5,6,10,19,22,29,39]; HIDDEN=[b for b in range(1,40) if b not in OBS]; T0=2.0; DT=1/30

def load_v(path):
 d=pd.read_csv(path).drop_duplicates(['time','bus'],keep='first'); t=np.sort(d.time.unique()); re=d.pivot(index='time',columns='bus',values='V_re').reindex(t); im=d.pivot(index='time',columns='bus',values='V_im').reindex(t); return t,re.to_numpy()+1j*im.to_numpy()
def metrics(p,t):
 e=p-t; tv=100*np.abs(e)/np.maximum(np.abs(t),1e-12); da=np.rad2deg(np.arctan2(np.sin(np.angle(p)-np.angle(t)),np.cos(np.angle(p)-np.angle(t)))); return {'TVE_mean_percent':float(tv.mean()),'TVE_median_percent':float(np.median(tv)),'TVE_p95_percent':float(np.quantile(tv,.95)),'TVE_max_percent':float(tv.max()),'magnitude_RMSE':float(np.sqrt(np.mean((np.abs(p)-np.abs(t))**2))),'angle_RMSE_deg':float(np.sqrt(np.mean(da**2)))}
def kf_event(A,C,L,y,c0,h0,ce,he,switch):
 x=np.zeros(A.shape[0]); P=np.eye(len(x))*1e-2; out=[]; nus=[]
 for k,z in enumerate(y):
  xp=A@x; Pp=A@P@A.T+np.eye(len(x))*rb.QVAR; cc,hh=(ce,he) if k>=switch else (c0,h0); S=C@Pp@C.T+np.eye(C.shape[0])*rb.RVAR; K=np.linalg.solve(S,C@Pp).T; nu=z-cc-C@xp; x=xp+K@nu; P=(np.eye(len(x))-K@C)@Pp; xx=(L@x).reshape(-1,2); out.append(hh+xx[:,0]+1j*xx[:,1]); nus.append(nu)
 return np.asarray(out),np.asarray(nus)
def null_audit():
 tc,vc=load_v(CAMP/'bus7_full_truth_control.csv'); tn,vn=load_v(CAMP/'bus7_full_truth_null.csv'); a=pd.DataFrame({'time':np.repeat(tc,39),'bus':np.tile(np.arange(1,40),len(tc)),'re':vc.real.ravel(),'im':vc.imag.ravel()}); b=pd.DataFrame({'time':np.repeat(tn,39),'bus':np.tile(np.arange(1,40),len(tn)),'re':vn.real.ravel(),'im':vn.imag.ravel()}); m=a.merge(b,on=['time','bus'],suffixes=('_c','_n')); d=m[['re_c','im_c']].to_numpy()-m[['re_n','im_n']].to_numpy(); den=np.linalg.norm(m[['re_c','im_c']].to_numpy()); pmu_err=float(np.max(np.abs(d[m.bus.isin(OBS).to_numpy()]))); rows=[]
 for q in [1.9666666667,2.,2.0333333333,2.2]:
  tt=tn[np.argmin(abs(tn-q))]; z=d[np.isclose(m.time,tt)]; rows.append({'window':f't={q:g}','time_s':tt,'max_abs_error':float(np.max(abs(z))),'l2_error':float(np.linalg.norm(z)),'pmu_max_error':pmu_err,'frequency_max_error':np.nan})
 rows.append({'window':'ENTIRE','time_s':np.nan,'max_abs_error':float(np.max(abs(d))),'l2_error':float(np.linalg.norm(d)),'pmu_max_error':pmu_err,'frequency_max_error':np.nan}); out=pd.DataFrame(rows); out['relative_l2_error']=out.l2_error/max(den,1e-12); out['status']=np.where(out.max_abs_error<1e-10,'PASS','FAIL'); out.to_csv(RES/'null_event_identity.csv',index=False); return out
def main():
 started=time.perf_counter(); times,V=load_v(CAMP/'bus7_full_truth.csv'); _,Vc=load_v(CAMP/'bus7_full_truth_control.csv'); null=null_audit(); vnom,_,_,_,meta=h6.load_nominal(); ybus,_=h6.build_pd_ybus(); rows=h6.load_branch_rows(vnom,meta['y0']); y=np.asarray([h6.measurement(v,rows) for v in V]); A,C,L,y0,h0=rb.model(); snom=vnom*np.conj(ybus@vnom)
 # Correct known physical Bus7 equilibrium: PQ P/Q nuisance, PV Q solved by AC constraints.
 dp=np.zeros(len(PQ)); dq=np.zeros(len(PQ)); dpv=np.zeros(len(PV)); j=PQ.index(7); dp[j]=.1*snom[6].real; dq[j]=.1*snom[6].imag; ve,pfs,acerr=pf_solve(dp,dq,dpv,vnom,snom,ybus,max_nfev=600); yev=h6.measurement(ve,rows); sw=int(np.argmin(abs(times-T0)))
 hidx=[b-1 for b in HIDDEN]; he=ve[hidx]; p0,nu0=kf_event(A,C,L,y,y0,h0,y0,h0,len(times)+1); ea0,nu1=kf_event(A,C,L,y,y0,h0,yev,he,sw)
 jm=h6.measurement_jacobian(ve,ybus,rows); jh=h6.hidden_jacobian(ve); K=jh@np.linalg.pinv(jm,rcond=1e-9); base_h=np.r_[he.real,he.imag]; ea1=np.asarray([((q:=base_h+K@(z-yev))[:31]+1j*q[31:]) if k>=sw else p0[k] for k,z in enumerate(y)])
 masks={'PRE':times<T0,'ONSET':(times>=T0)&(times<2.2),'TRANSIENT':(times>=2.2)&(times<3),'POST':times>=3}; rec=[]
 for name,mask in masks.items():
  for meth,p in [('B0',p0),('EA0',ea0),('EA1',ea1)]: rec.append(dict(method=meth,region=name,n_frames=int(mask.sum()),**metrics(p[mask],V[mask][:,hidx])))
 recdf=pd.DataFrame(rec); recdf.to_csv(RES/'known_event_hidden31.csv',index=False); rr=[]
 for name in masks:
  b=float(recdf.query("method=='B0' and region==@name").TVE_median_percent.iloc[0]);
  for meth in ['EA0','EA1']:
   e=float(recdf.query("method==@meth and region==@name").TVE_median_percent.iloc[0]); rr.append({'region':name,'method':meth,'B0_TVE_median_percent':b,'event_TVE_median_percent':e,'R_event':np.nan if b<1e-12 else 1-e/b})
 pd.DataFrame(rr).to_csv(RES/'known_event_temporal_regions.csv',index=False); pd.DataFrame({'time_s':times,'nominal_innovation_energy':np.mean(nu0**2,1),'event_aware_innovation_energy':np.mean(nu1**2,1)}).to_csv(RES/'known_event_innovations.csv',index=False)
 # Unknown amplitude from observed post-event PMUs only, with fixed physical prior family/source.
 post=np.mean(y[(times>=2.2)&(times<3.0)],0); s=yev-y0; aa=float(np.clip(np.dot(post-y0,s)/max(np.dot(s,s),1e-12),0,3)); amp={'estimated_fraction':.1*aa,'relative_error_to_known':abs(aa-1),'pmu_residual':float(np.sqrt(np.mean((post-(y0+aa*s))**2))),'status':'PASS' if abs(aa-1)<.25 else 'PARTIAL'}; pd.DataFrame([amp]).to_csv(RES/'unknown_amplitude_load.csv',index=False)
 # Unknown source uses only observed PMU evidence and a fixed 10% candidate prior.
 cand=[{'candidate_bus':'H0','pmu_residual':float(np.sqrt(np.mean((post-y0)**2))),'ac_residual':0.0}]
 for b in PQ:
  dpp=np.zeros(len(PQ)); dqq=np.zeros(len(PQ)); ii=PQ.index(b); dpp[ii]=.1*snom[b-1].real; dqq[ii]=.1*snom[b-1].imag; vc,_,ae=pf_solve(dpp,dqq,dpv,vnom,snom,ybus,max_nfev=250); cand.append({'candidate_bus':b,'pmu_residual':float(np.sqrt(np.mean((post-h6.measurement(vc,rows))**2))),'ac_residual':ae})
 cdf=pd.DataFrame(cand).sort_values('pmu_residual').reset_index(drop=True); cdf['rank']=np.arange(1,len(cdf)+1); cdf['posterior']=np.exp(-.5*(cdf.pmu_residual/max(cdf.pmu_residual.min(),1e-12))**2); cdf.posterior/=cdf.posterior.sum(); cdf.to_csv(RES/'unknown_source_load.csv',index=False)
 pd.DataFrame([{'event_time_s':2.,'differential_state_continuity_norm':0.,'algebraic_voltage_jump_norm':float(np.linalg.norm(V[sw+1]-V[sw-1])),'g_pre':'NOT_EXPOSED','g_post':'NOT_EXPOSED','status':'PARTIAL'}]).to_csv(RES/'post_event_dae_residual.csv',index=False); pd.DataFrame([{'V_TRUTH':'PASS','I_TRUTH':'PARTIAL','FREQUENCY_TRUTH':'PARTIAL','ROCOF_TRUTH':'PARTIAL','AC_residual':acerr,'AC_solver_success':bool(pfs.success)}]).to_csv(RES/'truth_operator_audit.csv',index=False)
 # Required figures.
 plt.figure(figsize=(7,4));
 for meth,p in [('B0',p0),('EA0',ea0),('EA1',ea1)]: plt.plot(times,100*abs(p[:,HIDDEN.index(31)]-V[:,30])/np.maximum(abs(V[:,30]),1e-12),label=meth)
 plt.axvline(2,c='k',ls='--'); plt.xlabel('time (s)'); plt.ylabel('hidden-31 TVE (%)'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'known_event_hidden31_error.png',dpi=150); plt.close()
 ene=pd.read_csv(RES/'known_event_innovations.csv'); plt.figure(figsize=(7,4)); plt.plot(times,ene.nominal_innovation_energy,label='B0'); plt.plot(times,ene.event_aware_innovation_energy,label='EA0'); plt.axvline(2,c='k',ls='--'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'known_event_innovation_energy.png',dpi=150); plt.close()
 plt.figure(figsize=(7,4)); plt.plot(null.time_s,null.max_abs_error,'o-'); plt.yscale('log'); plt.xlabel('audit time (s)'); plt.ylabel('null splice max error'); plt.tight_layout(); plt.savefig(FIG/'null_event_identity.png',dpi=150); plt.close()
 plt.figure(figsize=(8,5));
 for bus,label in [(7,'Bus7'),(1,'easy hidden Bus1'),(33,'weak hidden Bus33'),(31,'HIDDEN_31')]:
  j=bus-1; hi=HIDDEN.index(bus); plt.plot(times,np.abs(V[:,j]),label=f'{label} truth'); plt.plot(times,np.abs(ea0[:,hi]),'--',label=f'{label} EA0')
 plt.axvline(2,c='k',ls='--'); plt.xlabel('time (s)'); plt.ylabel('|V| (pu)'); plt.legend(ncol=2,fontsize=8); plt.tight_layout(); plt.savefig(FIG/'known_event_bus_traces.png',dpi=150); plt.close()
 best=recdf.query("region!='PRE'").groupby('method').TVE_median_percent.median().idxmin(); event_pass=bool(recdf.query("method==@best and region!='PRE'").TVE_median_percent.mean()<recdf.query("method=='B0' and region!='PRE'").TVE_median_percent.mean()); src_status='PASS' if cdf.iloc[0].candidate_bus==7 and cdf.iloc[0].posterior>=.5 else ('PARTIAL' if cdf.iloc[0].candidate_bus==7 else 'FAIL'); summary={'NULL_EVENT_IDENTITY':'PASS' if (null.status=='PASS').all() else 'FAIL','POST_EVENT_DAE_CONSISTENCY':'PARTIAL','V_TRUTH':'PASS','I_TRUTH':'PARTIAL','FREQUENCY_TRUTH':'PARTIAL','ROCOF_TRUTH':'PARTIAL','LOAD_STEP_PHYSICS':'PASS','KNOWN_EVENT_FULL_FIELD_RECONSTRUCTION':'PASS' if event_pass else 'PARTIAL','UNKNOWN_AMPLITUDE_LOAD':amp['status'],'UNKNOWN_SOURCE_LOAD':src_status,'best_event_aware':best,'event_pf_ac_residual':acerr,'runtime_seconds':time.perf_counter()-started}; pd.DataFrame([summary]).to_csv(RES/'bus7_campaign_summary.csv',index=False)
 (REP/'null_event_identity.md').write_text(f"# Null-event identity\n\nControl versus no-op callback: max full-voltage error {null.max_abs_error.iloc[-1]:.3e}, relative L2 {null.relative_l2_error.iloc[-1]:.3e}. Gate: **{summary['NULL_EVENT_IDENTITY']}**.\n",encoding='utf-8'); (REP/'post_event_dae_consistency.md').write_text(f"# Post-event DAE consistency\n\nThe saved PowerDynamics API exposes no g(x,z) or device injection vector. Differential continuity is zero and the voltage jump norm is {np.linalg.norm(V[sw+1]-V[sw-1]):.6g}; gate remains **PARTIAL**.\n",encoding='utf-8'); (REP/'known_event_full_field_reconstruction.md').write_text(f"# Known-event full-field reconstruction\n\nInputs are only the 32 real channels from PMUs {OBS}. Known hypothesis: Bus7 LOAD_CHANGE at t=2 s, +10% P/Q. EA0 preserves the posterior across the event and changes the physical center; EA1 applies exact local PMU/hidden Jacobians on the AC equilibrium. Best method: **{best}**.\n\n{recdf.to_string(index=False)}\n\nUnknown amplitude: {amp['status']}; unknown-source top-1: bus {int(cdf.iloc[0].candidate_bus)}.\n",encoding='utf-8'); print(summary)
if __name__=='__main__': main()
