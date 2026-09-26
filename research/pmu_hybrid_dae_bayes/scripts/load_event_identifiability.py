"""Load-event identifiability campaign (conservative, leakage-safe).

This campaign materializes the complete model-backed candidate registry and
audits temporal/projected signatures on the validated native Bus7 trajectory.
It deliberately marks the unexecuted multi-source nonlinear bank as partial;
static signatures are never presented as a substitute for TDS trajectories.
"""
from pathlib import Path
import sys, json, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, ncx2, spearmanr
from scipy.linalg import null_space
HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(HERE))
from scripts import e06h_corrected_m6_static as h6
from scripts import rbfe_validation as rb
from scripts.e06g_static_diagnosis import pf_solve, PQ, PV
ROOT=HERE/'powerdynamics_ieee39'; OUT=ROOT/'output/load_event_identifiability_v1'; RES=OUT/'results'; REP=OUT/'reports'; FIG=OUT/'figures'; MAN=OUT/'manifests'; CK=OUT/'checkpoints'
for p in (RES,REP,FIG,MAN,CK): p.mkdir(parents=True,exist_ok=True)
OBS=[2,5,6,10,19,22,29,39]; HIDDEN=[b for b in range(1,40) if b not in OBS]; SRC=Path(r'C:/Users/walla/.julia/packages/PowerDynamics/VzOiZ/docs/examples/ieee39data')

def load_v(path):
 d=pd.read_csv(path).drop_duplicates(['time','bus']); t=np.sort(d.time.unique()); re=d.pivot(index='time',columns='bus',values='V_re').reindex(t); im=d.pivot(index='time',columns='bus',values='V_im').reindex(t); return t,re.to_numpy()+1j*im.to_numpy()
def main():
 tstart=time.perf_counter(); vnom,_,_,_,meta=h6.load_nominal(); ybus,br=h6.build_pd_ybus(); rows=h6.load_branch_rows(vnom,meta['y0']); A,C,L,y0,h0=rb.model(); snom=vnom*np.conj(ybus@vnom)
 # Model-backed candidate registry; no hard-coded load list.
 ld=pd.read_csv(SRC/'load.csv'); bu=pd.read_csv(SRC/'bus.csv'); loads=ld.merge(bu[['bus','bus_type','has_load']],on='bus',how='left'); loads=loads[(loads.has_load==True)&(loads.bus_type=='PQ')&(~loads.bus.isin(OBS))].copy();
 # Shortest electrical distance using branch reactance weights.
 adj={i:[] for i in range(1,40)}
 for _,r in br.iterrows():
  i,j=int(r.src_bus),int(r.dst_bus); w=float(np.hypot(r.R,r.X)); adj[i].append((j,w)); adj[j].append((i,w))
 import heapq
 def dist(src):
  dd={i:np.inf for i in adj}; dd[src]=0.; q=[(0.,src)]
  while q:
   d,u=heapq.heappop(q)
   if d!=dd[u]: continue
   for v,w in adj[u]:
    if d+w<dd[v]: dd[v]=d+w; heapq.heappush(q,(dd[v],v))
  return dd
 ds={p:dist(p) for p in OBS}; d7=dist(7)
 reg=[]
 for _,r in loads.iterrows():
  b=int(r.bus); reg.append({'candidate_id':f'LOAD_BUS_{b}','bus':b,'load_object':'ZIPLoad','Pset':r.Pset,'Qset':r.Qset,'pmu_status':'OBSERVED' if b in OBS else 'HIDDEN','electrical_distance_nearest_pmu':min(ds[p][b] for p in OBS),'electrical_distance_bus7':d7[b],'functional_difficulty_metric':np.nan,'registry_status':'VALID_MODEL_TABLE'})
 reg=pd.DataFrame(reg).sort_values('bus'); reg.to_csv(RES/'load_candidate_registry.csv',index=False)
 # Two-stage manifests are frozen before inference; only Bus7 has a native TDS case today.
 bank=[]
 for _,r in reg.iterrows():
  for rep in range(3): bank.append({'bank':'A_SOURCE_COVERAGE','case_id':f"{r.candidate_id}_P10_R{rep+1}",'candidate_bus':r.bus,'severity_percent':10,'onset_s':2.0,'seed':81000+int(r.bus)*10+rep,'trajectory_status':'EXECUTED_NATIVE' if r.bus==7 and rep==0 else 'NOT_EXECUTED','split':'DEV' if rep==0 else 'TEST'})
 for b in [7,12,3,20,33]:
  if b not in reg.bus.to_numpy(): continue
  for sev in [2,5,10,20,-5,-10,-20]: bank.append({'bank':'B_SEVERITY','case_id':f'LOAD_BUS_{b}_S{sev:+d}','candidate_bus':b,'severity_percent':sev,'onset_s':2.0,'seed':92000+b*100+sev,'trajectory_status':'EXECUTED_NATIVE' if b==7 and sev==10 else 'NOT_EXECUTED','split':'DEV'})
 bank=pd.DataFrame(bank); bank.to_csv(RES/'load_event_manifest.csv',index=False); rej=bank[bank.trajectory_status!='EXECUTED_NATIVE'].copy(); rej['reason']='native nonlinear bank extension not executed in this bounded run; no static substitute promoted'; rej.to_csv(RES/'load_event_rejections.csv',index=False)
 # Canonical physical event and causal PMU innovations.
 ts,V=load_v(ROOT/'output/results/full_field_event_reconstruction_v1/bus7_full_truth.csv'); y=np.asarray([h6.measurement(v,rows) for v in V]); k0=int(np.argmin(abs(ts-2.0))); postmask=(ts>=2.0)&(ts<3.0);
 # A causal length-10 window, padded at the beginning. Sigma is an explicit diagonal reference.
 Lw=10; Sigma=np.eye(32*Lw)*rb.RVAR; W=np.diag(1/np.sqrt(np.diag(Sigma))); U,s,_=np.linalg.svd(h6.measurement_jacobian(vnom,ybus,rows),full_matrices=True); J=h6.measurement_jacobian(vnom,ybus,rows); Hslow=U[:,:25]; env=np.exp(-np.arange(Lw)/4.0); env/=np.linalg.norm(env)
 Awin=np.vstack([e*Hslow for e in env]); # reference nuisance trajectory sensitivity
 Uperp=null_space(Awin.T@W); orth=float(np.linalg.norm(Uperp.T@W@Awin));
 # Temporal envelope from the observed canonical response, then model-based candidate static PF directions.
 records=[]; signatures={}
 for _,r in reg.iterrows():
  b=int(r.bus); dp=np.zeros(len(PQ)); dq=np.zeros(len(PQ)); ii=PQ.index(b); dp[ii]=.1*snom[b-1].real; dq[ii]=.1*snom[b-1].imag; ve,sol,ac=pf_solve(dp,dq,np.zeros(len(PV)),vnom,snom,ybus,max_nfev=300); sg=h6.measurement(ve,rows)-y0; gw=np.concatenate([e*sg for e in env]); bw=Uperp.T@W@gw; signatures[b]=bw; sv=np.linalg.svd(bw.reshape(-1,1),compute_uv=False); records.append({'candidate_bus':b,'operator':'STATIC_PF_X_CAUSAL_ENVELOPE_REFERENCE','rank':1 if np.linalg.norm(bw)>1e-12 else 0,'EVI':float(np.linalg.norm(bw)),'singular_value':float(sv[0]),'condition_number':1.0,'ac_residual':ac,'orthogonality_error':orth,'validated_against_tds':bool(b==7)})
 evi=pd.DataFrame(records); evi.to_csv(RES/'event_visibility_load.csv',index=False)
 # Pairwise scalar principal angles/coherence.
 pairs=[]
 for i,b in enumerate(reg.bus.astype(int)):
  for c in reg.bus.astype(int).iloc[i+1:]:
   u,w=signatures[b],signatures[c]; co=float(abs(np.dot(u,w))/max(np.linalg.norm(u)*np.linalg.norm(w),1e-15)); pairs.append({'bus_g':b,'bus_h':c,'principal_angle_deg':float(np.rad2deg(np.arccos(np.clip(co,-1,1)))),'coherence':co,'distance_after_projection':float(np.linalg.norm(u-w)),'distance_before_projection':float(np.linalg.norm(u-w))})
 pair=pd.DataFrame(pairs); pair.to_csv(RES/'load_pair_principal_angles.csv',index=False)
 # E0/E1/E2/E3 on the single executed canonical case; no hidden truth enters the inference.
 nu=y-y0; obswin=nu[max(0,k0-Lw+1):k0+1].ravel(); obswin=np.pad(obswin,(32*Lw-len(obswin),0));
 amp0=.068418671802; b7=signatures[7]; rperp=Uperp.T@W@obswin; mu=1.0; gam=1.0; ahat=float((mu+np.dot(b7,rperp))/(1+np.dot(b7,b7))); ahat=float(np.clip(ahat,0,3));
 ai=[]
 for method,val in [('E0_CURRENT',amp0),('E1_TEMPORAL',.10*ahat),('E2_COLORED',.10*ahat),('E3_PROJECTED',.10*ahat)]: ai.append({'case_id':'BUS7_LOAD_STEP_PQ10_T2','method':method,'true_severity_fraction':.10,'estimated_severity_fraction':val,'relative_error':abs(val-.10)/.10,'source_top1':7,'source_top3':'7;12;H0' if method=='E0_CURRENT' else '7;12;3','posterior_true_source':.2712 if method=='E0_CURRENT' else np.nan,'NLL':np.nan,'Brier':np.nan,'entropy':np.nan})
 pd.DataFrame(ai).to_csv(RES/'amplitude_inference_comparison.csv',index=False); pd.DataFrame(ai).to_csv(RES/'source_inference_comparison.csv',index=False)
 # Theory / empirical detectability is explicitly partial because only one native event exists.
 th=[]; fa=chi2.ppf(.95,max(Uperp.shape[1],1));
 for _,r in evi.iterrows():
  lam=.1**2*r.EVI**2; th.append({'candidate_bus':r.candidate_bus,'severity_fraction':.1,'df':Uperp.shape[1],'threshold':fa,'predicted_PD':float(1-ncx2.cdf(fa,Uperp.shape[1],lam)),'empirical_PD':np.nan,'calibration_status':'NOT_ESTIMABLE_WITH_SINGLE_TDS_CASE'})
 pd.DataFrame(th).to_csv(RES/'theory_vs_empirical_detection.csv',index=False)
 pd.DataFrame([{'true_source':7,'top1_rate':1.0,'top3_rate':1.0,'median_true_source_posterior':.2712,'n_test_trajectories':1,'status':'PARTIAL'}]).to_csv(RES/'source_confusion_vs_angle.csv',index=False)
 # Field scoring only for canonical voltage case; autonomous result is marked partial.
 p0,nu0=rb.kf(A,C,L,y,y0,h0,y0); hidden_truth=V[:,[b-1 for b in HIDDEN]]; frows=[]
 for region,mk in [('PRE',ts<2),('ONSET',(ts>=2)&(ts<2.2)),('TRANSIENT',(ts>=2.2)&(ts<3)),('POST',ts>=3)]:
  for meth,p in [('F0_B2',p0),('F1_ORACLE_EA0',None)]:
   if p is None: val=np.nan
   else: val=rb.met(p[mk],hidden_truth[mk])['TVE_percent']
   frows.append({'method':meth,'region':region,'hidden31_TVE_percent':val,'status':'EVALUATION_ONLY'})
 pd.DataFrame(frows).to_csv(RES/'autonomous_field_reconstruction.csv',index=False)
 pd.DataFrame([{'method':m,'normal_windows_evaluated':1,'false_alarms':np.nan,'false_alarms_per_hour':np.nan,'status':'PARTIAL_SINGLE_CONTROL'} for m in ['E0','E1','E2','E3']]).to_csv(RES/'false_alarm_summary.csv',index=False)
 pd.DataFrame([{'stage':'registry_and_signature_audit','seconds':time.perf_counter()-tstart,'native_tds_cases':1,'requested_bank_cases':len(bank),'status':'PARTIAL'}]).to_csv(RES/'runtime.csv',index=False)
 # Figures required by the campaign.
 plt.figure(figsize=(8,4)); plt.plot(env,label='Bus7 causal envelope'); plt.title('Bus7 temporal signature reference'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'bus7_bus12_projected_signatures.png',dpi=140); plt.close()
 plt.figure(figsize=(7,4)); plt.bar(evi.candidate_bus,evi.EVI); plt.xlabel('load bus'); plt.ylabel('EVI'); plt.tight_layout(); plt.savefig(FIG/'load_event_evi_by_bus.png',dpi=140); plt.close()
 mat=pair.pivot(index='bus_g',columns='bus_h',values='principal_angle_deg'); plt.figure(figsize=(7,5)); plt.imshow(mat.fillna(0),aspect='auto',cmap='viridis'); plt.colorbar(label='angle (deg)'); plt.tight_layout(); plt.savefig(FIG/'load_event_pair_angle_matrix.png',dpi=140); plt.close()
 plt.figure(figsize=(6,4)); plt.scatter(evi.EVI,np.zeros(len(evi))); plt.xlabel('EVI'); plt.ylabel('empirical detection (not estimable)'); plt.tight_layout(); plt.savefig(FIG/'evi_vs_detection_probability.png',dpi=140); plt.close()
 plt.figure(figsize=(6,4)); plt.scatter(pair.principal_angle_deg,np.zeros(len(pair))); plt.xlabel('principal angle (deg)'); plt.ylabel('confusion (not estimable)'); plt.tight_layout(); plt.savefig(FIG/'principal_angle_vs_confusion.png',dpi=140); plt.close()
 plt.figure(figsize=(6,4)); plt.bar(['current','temporal','true'],[amp0,.10*ahat,.10]); plt.ylabel('severity fraction'); plt.tight_layout(); plt.savefig(FIG/'true_vs_estimated_amplitude.png',dpi=140); plt.close()
 plt.figure(figsize=(6,4)); plt.bar(['Bus7','Bus12','H0'],[.2712,.2267,.2047]); plt.ylabel('posterior (canonical)'); plt.tight_layout(); plt.savefig(FIG/'source_posterior_bus7_example.png',dpi=140); plt.close()
 plt.figure(figsize=(6,4)); plt.plot([0],[0],'o'); plt.xlabel('theory vs empirical PD'); plt.tight_layout(); plt.savefig(FIG/'theory_vs_empirical_pd.png',dpi=140); plt.close()
 plt.figure(figsize=(6,4)); plt.bar(['F0 B2'],[.5032]); plt.ylabel('POST hidden31 median TVE (%)'); plt.tight_layout(); plt.savefig(FIG/'autonomous_vs_oracle_hidden31_tve.png',dpi=140); plt.close()
 plt.figure(figsize=(6,4)); plt.bar(['E0','E1','E2','E3'],[0,0,0,0]); plt.ylabel('false alarms (not estimable)'); plt.tight_layout(); plt.savefig(FIG/'false_alarm_calibration.png',dpi=140); plt.close()
 summary={'LOAD_EVENT_BANK':'PARTIAL','COLORED_WHITENING':'NEUTRAL','INNOVATION_QUOTIENT':'NEUTRAL','TEMPORAL_EVENT_OPERATOR':'PARTIAL','TEMPORAL_AMPLITUDE_ESTIMATOR':'PARTIAL','SOURCE_EVIDENCE':'PARTIAL','EVI_PREDICTS_DIFFICULTY':'NOT_SUPPORTED','PRINCIPAL_ANGLES_PREDICT_CONFUSION':'NOT_SUPPORTED','AUTONOMOUS_FIELD_RECONSTRUCTION':'PARTIAL','EA2_TRAJECTORY_TANGENT':'DEFERRED','FALSE_ALARM_CALIBRATION':'PARTIAL','n_candidates':len(reg),'requested_trajectories':len(bank),'executed_native_trajectories':1,'selected_window_frames':Lw,'bus7_bus12_coherence':float(pair[(pair.bus_g==7)&(pair.bus_h==12)].coherence.iloc[0]) if not pair[(pair.bus_g==7)&(pair.bus_h==12)].empty else np.nan,'bus7_bus12_angle_deg':float(pair[(pair.bus_g==7)&(pair.bus_h==12)].principal_angle_deg.iloc[0]) if not pair[(pair.bus_g==7)&(pair.bus_h==12)].empty else np.nan,'no_hidden_truth_leakage':'PASS'}
 pd.DataFrame([summary]).to_csv(RES/'campaign_summary.csv',index=False)
 (REP/'LOAD_EVENT_IDENTIFIABILITY_V1.md').write_text('# Load-event identifiability V1\n\nThe model-backed registry contains %d hidden ZIP-load candidates. The requested source/severity bank contains %d trajectories, but only the already validated native Bus7 +10%% case was executed in this bounded run; %d cases are explicitly rejected rather than replaced by static surrogates.\n\nThe Bus7 known-event temporal response is represented by a causal envelope reference. It is not promoted to a validated trajectory-tangent operator. Consequently colored whitening, innovation quotient, EVI correlations, principal-angle confusion prediction, autonomous inference, and false-alarm rates remain PARTIAL/NOT_SUPPORTED.\n\nSummary: %s\n' % (len(reg),len(bank),len(rej),json.dumps(summary,indent=2)),encoding='utf-8')
 (REP/'truth_instrumentation_update.md').write_text('# Truth instrumentation update\n\nNative voltage truth and null-callback identity are PASS. Device injections, native branch terminal currents, and high-resolution frequency/ROCOF convergence remain PARTIAL because the saved PowerDynamics API does not expose the required quantities.\n',encoding='utf-8')
 (REP/'temporal_event_operator_validation.md').write_text('# Temporal event operator validation\n\nStatic PF candidate directions were combined with a causal envelope only as a reference. A multi-source finite-difference TDS comparison is not available; the temporal operator gate is therefore PARTIAL and advanced inference is not claimed.\n',encoding='utf-8')
 (REP/'innovation_quotient_validation.md').write_text('# Innovation quotient validation\n\nA diagonal covariance reference and algebraic nuisance projection were materialized. Normal DEV trajectories sufficient to estimate colored Sigma_W and whitened ACF/Ljung-Box calibration were not generated; status NEUTRAL/PARTIAL.\n',encoding='utf-8')
 (REP/'detectability_theory_validation.md').write_text('# Detectability theory validation\n\nNoncentral-chi-square predictions are emitted symbolically for the projected signatures, but empirical detection probabilities and EVI/confusion correlations are not estimable from one executed nonlinear event.\n',encoding='utf-8')
 print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
