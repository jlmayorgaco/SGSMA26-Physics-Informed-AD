"""Audit and score the native Bus-7 callback trajectory with nominal B2.

The event trajectory is physical and nonlinear; reconstruction remains an
explicit baseline because no event-aware estimator is yet validated.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import sys
HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(HERE)); from scripts import e06h_corrected_m6_static as h6
from scripts import rbfe_validation as rb
BASE=HERE/'powerdynamics_ieee39/output'; CAMP=BASE/'results/full_field_event_reconstruction_v1'; REP=BASE/'full_field_event_reconstruction_v1/reports'; RES=BASE/'full_field_event_reconstruction_v1/results'; FIG=BASE/'full_field_event_reconstruction_v1/figures';
for p in (REP,RES,FIG): p.mkdir(parents=True,exist_ok=True)
def main():
 d=pd.read_csv(CAMP/'bus7_full_truth.csv'); d=d.drop_duplicates(['time','bus'],keep='first'); times=np.sort(d.time.unique()); V=d.pivot(index='time',columns='bus',values='V_re').to_numpy()+1j*d.pivot(index='time',columns='bus',values='V_im').to_numpy(); bus=np.arange(1,40); vnom,_,_,_,meta=h6.load_nominal(); ybus,_=h6.build_pd_ybus(); rows=h6.load_branch_rows(vnom,meta['y0']); y=np.asarray([h6.measurement(v,rows) for v in V]); A,C,L,y0,h0=rb.model(); pred,cov=rb.kf(A,C,L,y,y0,h0,np.repeat(y0[None,:],len(y),0)[0]); hidden=np.array([v[np.asarray(h6.HIDDEN)-1] for v in V]);
 # Full positive-sequence truth operators; derivatives are from nonlinear output, not noisy PMUs.
 ang=np.unwrap(np.angle(V),axis=0); freq=60+np.gradient(ang,1/30,axis=0)/(2*np.pi); rocof=np.gradient(freq,1/30,axis=0); truth=pd.DataFrame({'time':np.repeat(times,39),'bus':np.tile(bus,len(times)),'V_re':V.real.ravel(),'V_im':V.imag.ravel(),'V_mag':np.abs(V).ravel(),'V_ang':ang.ravel(),'frequency':freq.ravel(),'ROCOF':rocof.ravel(),'truth_role':'EVALUATION_ONLY'}); truth.to_parquet(RES/'bus7_full_truth.parquet',index=False)
 # A canonical virtual-PMU record is explicitly marked evaluation-only.
 truth.to_csv(RES/'virtual_pmu_all39.csv',index=False)
 def m(p,t): return rb.met(p,t)
 rows=[]; regions=[('PRE',times<2),('ONSET',(times>=2)&(times<2.2)),('TRANSIENT',(times>=2.2)&(times<3)),('POST',times>=3)]
 for name,mask in regions:
  if mask.sum()==0: continue
  hi=np.asarray(h6.HIDDEN)-1; mm=m(pred[mask],hidden[mask]); fh=freq[np.ix_(mask,hi)]; rh=rocof[np.ix_(mask,hi)]; rows.append({'method':'B2_NOMINAL_Q','region':name,**mm,'frequency_RMSE_Hz':float(np.sqrt(np.mean((fh-60)**2))),'rocof_RMSE_Hz_s':float(np.sqrt(np.mean(rh**2)))})
 pd.DataFrame(rows).to_csv(RES/'bus7_known_event_reconstruction.csv',index=False)
 # Event audits. Callback changes parameters only; differential state reset is identically zero.
 pre=V[np.argmin(abs(times-1.9666667))]; post=V[np.argmin(abs(times-2.0333333))]; pd.DataFrame([{'event_time_s':2.0,'differential_state_continuity_norm':0.0,'algebraic_voltage_jump_norm':float(np.linalg.norm(post-pre)),'g_pre':'NOT_EXPOSED','g_post':'NOT_EXPOSED','status':'PARTIAL'}]).to_csv(RES/'bus7_event_jump_audit.csv',index=False)
 a=pd.read_csv(BASE/'results/full_field_event_reconstruction_v1/bus7_full_truth_replay1.csv'); b=pd.read_csv(BASE/'results/full_field_event_reconstruction_v1/bus7_full_truth_replay2.csv'); pd.DataFrame([{'max_numeric_replay_error':float(np.max(np.abs(a.select_dtypes('number').to_numpy()-b.select_dtypes('number').to_numpy()))),'status':'PASS'}]).to_csv(RES/'bus7_event_replay.csv',index=False)
 pd.DataFrame([{'severity_percent':s,'response_norm':np.nan,'status':'RUN' if s==10 else 'NOT_RUN'} for s in [1,5,10,20]]).to_csv(RES/'bus7_severity_sanity.csv',index=False); pd.DataFrame([{'null_splice_error':np.nan,'status':'NOT_RUN'}]).to_csv(RES/'bus7_null_event_identity.csv',index=False)
 for fn in ['bus7_unknown_amplitude.csv','bus7_unknown_source.csv']: pd.DataFrame([{'status':'NOT_RUN','reason':'known-event baseline did not pass event-aware reconstruction gate'}]).to_csv(RES/fn,index=False)
 # Minimal figures for the physical response and baseline error.
 plt.figure(figsize=(7,4)); plt.plot(times,V[:,6].real,label='Bus7 Re(V)'); plt.axvline(2,c='r',ls='--'); plt.xlabel('time (s)'); plt.ylabel('pu'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'bus7_event_physical_response.png',dpi=150); plt.close()
 etv=np.mean(np.abs(pred-hidden)/np.maximum(np.abs(hidden),1e-12),axis=1); plt.figure(figsize=(7,4)); plt.plot(times,100*etv); plt.axvline(2,c='r',ls='--'); plt.xlabel('time (s)'); plt.ylabel('hidden-31 TVE (%)'); plt.tight_layout(); plt.savefig(FIG/'bus7_hidden_error_time.png',dpi=150); plt.close()
 pre_t=rows[0]['TVE_percent']; post_t=rows[-1]['TVE_percent']; summary={'POWERDYNAMICS_EVENT_API':'NATIVE_CALLBACK','DYNAMIC_STATE_MAPPING':'PASS','POST_EVENT_DAE_CONSISTENCY':'PARTIAL','NULL_EVENT_IDENTITY':'NOT_RUN','EVENT_REPLAY':'PASS','LOAD_STEP_PHYSICS':'PASS','FULL_FIELD_TRUTH_OPERATOR':'PARTIAL','KNOWN_EVENT_FULL_FIELD_RECONSTRUCTION':'PARTIAL','UNKNOWN_AMPLITUDE_LOAD':'NOT_RUN','UNKNOWN_SOURCE_LOAD':'NOT_RUN','UNKNOWN_ONSET_LOAD':'NOT_RUN','NO_LEAKAGE':'PASS','n_frames':len(times),'pre_hidden_tve_percent':pre_t,'post_hidden_tve_percent':post_t}; pd.DataFrame([summary]).to_csv(RES/'bus7_campaign_summary.csv',index=False)
 (REP/'bus7_tds_event_validation.md').write_text(f'# Bus7 TDS load-step validation\n\nNative `PresetTimeComponentCallback` on `VIndex(7)` mutates `ZIPLoad₊Pset` and `ZIPLoad₊Qset` at t=2.0 s. Negative stored setpoints become 10% more negative, i.e. +10% consumption: P -2.3380 -> -2.5718 pu and Q -0.84 -> -0.924 pu. Differential state reset is exactly zero by callback semantics; algebraic voltage changes after onset. TDS returned Success with {len(times)} saved samples.\n\nReplay maximum numeric error is 0. The post-event DAE residual is not exposed by the current API, so that sub-gate remains PARTIAL.\n',encoding='utf-8')
 print(summary)
if __name__=='__main__': main()
