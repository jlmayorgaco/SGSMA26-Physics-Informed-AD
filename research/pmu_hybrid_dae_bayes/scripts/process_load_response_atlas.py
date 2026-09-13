"""Process the native load-response atlas without running source inference."""
from pathlib import Path
import sys, hashlib, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(HERE))
from scripts import e06h_corrected_m6_static as h6
from scripts.e06g_static_diagnosis import PQ, PV, pf_solve
OUT=HERE/'powerdynamics_ieee39/output/load_response_atlas_v1'; RES=OUT/'results'; REP=OUT/'reports'; FIG=OUT/'figures'; MAN=OUT/'manifests'; CK=OUT/'checkpoints';
for p in (RES,REP,FIG,MAN,CK): p.mkdir(parents=True,exist_ok=True)
OBS=[2,5,6,10,19,22,29,39]; EPS=[.0025,.005,.01,.02]
def load_case(path):
 d=pd.read_csv(path).drop_duplicates(['time','bus']); t=np.sort(d.time.unique()); re=d.pivot(index='time',columns='bus',values='V_re').reindex(t); im=d.pivot(index='time',columns='bus',values='V_im').reindex(t); return t,re.to_numpy()+1j*im.to_numpy()
def main():
 vnom,_,_,_,meta=h6.load_nominal(); ybus,_=h6.build_pd_ybus(); rows=h6.load_branch_rows(vnom,meta['y0']); mf=pd.read_csv(OUT/'simulation_manifest_native.csv'); reg=pd.read_csv(OUT/'candidate_registry_native.csv');
 # Freeze registry hash before analysis.
 reg.to_csv(MAN/'candidate_registry_frozen.csv',index=False); sha=hashlib.sha256((MAN/'candidate_registry_frozen.csv').read_bytes()).hexdigest();
 operators=[]; conv=[]; nonlinear=[]; oldcmp=[]; amp=[]; tang=[]; fdrefs={}
 for b in sorted(mf.candidate_bus.unique()):
  paths={float(a):OUT/'results'/f'LOAD_BUS_{int(b)}_A{("m" if a<0 else "").join(str(abs(a)).replace(".","p").split("e"))}_R1.csv' for a in []}
  sub=mf[mf.candidate_bus==b]; files={float(r.amplitude):Path(r.path) for _,r in sub.iterrows()};
  if 0.0 not in files: continue
  t,V0=load_case(files[0.0]); y0=np.asarray([h6.measurement(v,rows)[:16] for v in V0]); post=np.arange(np.argmin(abs(t-2.0)),min(len(t),np.argmin(abs(t-2.0))+30));
  for e in EPS:
   if e not in files or -e not in files: continue
   tp,Vp=load_case(files[e]); tm,Vm=load_case(files[-e]); yp=np.asarray([h6.measurement(v,rows)[:16] for v in Vp]); ym=np.asarray([h6.measurement(v,rows)[:16] for v in Vm]);
   g=(yp[post]-ym[post])/(2*e); fdrefs.setdefault(int(b),{})[e]=g; gp=(yp[post]-y0[post])/e; gm=(ym[post]-y0[post])/(-e); flat=g.ravel(); operators += [{'candidate_bus':int(b),'epsilon':e,'frame':int(k),'channel':int(c),'value':float(g[i,c])} for i,k in enumerate(post) for c in range(16)]; conv.append({'candidate_bus':int(b),'epsilon':e,'comparison':'one_sided','central_vs_plus_relerr':float(np.linalg.norm(g-gp)/max(np.linalg.norm(g),1e-12)),'central_vs_minus_relerr':float(np.linalg.norm(g-gm)/max(np.linalg.norm(g),1e-12)),'pairwise_relerr':np.nan,'pairwise_cosine':np.nan,'norm':float(np.linalg.norm(g))})
 for b,refs in fdrefs.items():
  for e,g in refs.items():
   for f,q in refs.items():
    if f>=e: continue
    conv.append({'candidate_bus':b,'epsilon':e,'comparison':f'vs_{f:g}','central_vs_plus_relerr':np.nan,'central_vs_minus_relerr':np.nan,'pairwise_relerr':float(np.linalg.norm(g-q)/max(np.linalg.norm(q),1e-12)),'pairwise_cosine':float(np.dot(g.ravel(),q.ravel())/(max(np.linalg.norm(g)*np.linalg.norm(q),1e-12))),'norm':float(np.linalg.norm(g))})
  # 10% nonlinear response against the 0.5% derivative reference.
  if .1 in files:
   t10,V10=load_case(files[.1]); y10=np.asarray([h6.measurement(v,rows)[:16] for v in V10]); e=.005; tp,Vp=load_case(files[e]); tm,Vm=load_case(files[-e]); gf=(np.asarray([h6.measurement(v,rows)[:16] for v in Vp])[post]-np.asarray([h6.measurement(v,rows)[:16] for v in Vm])[post])/(2*e); d10=y10[post]-y0[post]; pred=.1*gf; nonlinear.append({'candidate_bus':int(b),'relative_trajectory_error':float(np.linalg.norm(d10-pred)/max(np.linalg.norm(d10),1e-12)),'cosine_similarity':float(np.dot(d10.ravel(),pred.ravel())/max(np.linalg.norm(d10)*np.linalg.norm(pred),1e-12))})
   if b==7:
    ah=float(np.dot(gf.ravel(),d10.ravel())/max(np.dot(gf.ravel(),gf.ravel()),1e-12)); amp.append({'method':'FD_REFERENCE','true_fraction':.10,'estimated_fraction':ah,'relative_error':abs(ah-.10)/.10,'epsilon_reference':.005})
 # Compare old static-envelope operator against FD for Bus7 only.
 if 7 in mf.candidate_bus.unique():
  t,V=load_case(Path(mf[(mf.candidate_bus==7)&(mf.amplitude==0)].iloc[0].path)); y=np.asarray([h6.measurement(v,rows)[:16] for v in V]); post=np.arange(np.argmin(abs(t-2.0)),min(len(t),np.argmin(abs(t-2.0))+30)); e=.005; vp=load_case(Path(mf[(mf.candidate_bus==7)&(mf.amplitude==e)].iloc[0].path))[1]; vm=load_case(Path(mf[(mf.candidate_bus==7)&(mf.amplitude==-e)].iloc[0].path))[1]; gf=(np.asarray([h6.measurement(v,rows)[:16] for v in vp])[post]-np.asarray([h6.measurement(v,rows)[:16] for v in vm])[post])/(2*e); dp=np.zeros(len(PQ)); dq=np.zeros(len(PQ)); dp[PQ.index(7)]=.1*(vnom*np.conj(ybus@vnom))[6].real; dq[PQ.index(7)]=.1*(vnom*np.conj(ybus@vnom))[6].imag; ve,_,_=pf_solve(dp,dq,np.zeros(len(PV)),vnom,vnom*np.conj(ybus@vnom),ybus,max_nfev=300); sg=h6.measurement(ve,rows)[:16]-h6.measurement(vnom,rows)[:16]; env=np.exp(-np.arange(len(post))/4); old=np.concatenate([z*sg for z in env]).reshape(gf.shape); gamma=float(np.dot(old.ravel(),gf.ravel())/max(np.dot(old.ravel(),old.ravel()),1e-12)); oldcmp.append({'candidate_bus':7,'relative_frobenius_error':float(np.linalg.norm(old-gf)/np.linalg.norm(gf)),'cosine_similarity':float(np.dot(old.ravel(),gf.ravel())/(np.linalg.norm(old)*np.linalg.norm(gf))),'best_scalar_gain':gamma,'error_after_rescaling':float(np.linalg.norm(gamma*old-gf)/np.linalg.norm(gf)),'direction':'MATCHED_REFERENCE_ONLY','failure_class':'SCALE_AND_TEMPORAL_PROFILE_UNVALIDATED'})
 pd.DataFrame(conv).to_csv(RES/'fd_derivative_convergence.csv',index=False); pd.DataFrame(operators).to_parquet(RES/'fd_candidate_operators.parquet',index=False); pd.DataFrame(nonlinear).to_csv(RES/'nonlinear_10pct_vs_linear.csv',index=False); pd.DataFrame(oldcmp).to_csv(RES/'old_operator_vs_fd.csv',index=False); pd.DataFrame(amp).to_csv(RES/'bus7_amplitude_revalidation.csv',index=False); pd.DataFrame(columns=['candidate_bus','frame','channel','value']).to_parquet(RES/'tangent_candidate_operators.parquet',index=False); pd.DataFrame([{'status':'NOT_IMPLEMENTED','reason':'trajectory-tangent requires native differential/algebraic state export'}]).to_csv(RES/'tangent_vs_fd.csv',index=False)
 # Validated visibility is intentionally withheld until all candidate TDS groups exist.
 pd.DataFrame(columns=['candidate_bus','EVI','validated']).to_csv(RES/'validated_event_visibility.csv',index=False); pd.DataFrame(columns=['bus_g','bus_h','angle_deg','coherence','validated']).to_csv(RES/'validated_pair_angles.csv',index=False); pd.DataFrame([{'status':'NOT_RUN','reason':'source inference forbidden before atlas completion'}]).to_csv(RES/'validated_source_evidence.csv',index=False)
 # Figures from executed physical signatures only.
 cdf=pd.DataFrame(conv); plt.figure(figsize=(7,4));
 for b,g in cdf.groupby('candidate_bus'): plt.plot(g.epsilon,g.central_vs_plus_relerr,'o-',label=f'Bus{b}');
 plt.xscale('log'); plt.xlabel('epsilon'); plt.ylabel('central vs + one-sided relative error'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'fd_convergence_by_epsilon.png',dpi=140); plt.close()
 if operators:
  op=pd.DataFrame(operators); plt.figure(figsize=(7,4));
  for b,g in op.groupby('candidate_bus'): plt.plot(g.groupby('frame').value.apply(lambda x:np.linalg.norm(x)),label=f'Bus{b}');
  plt.xlabel('post-event frame'); plt.ylabel('||G_FD||'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'bus7_fd_response_vs_time.png',dpi=140); plt.close()
 for name in ['bus7_old_operator_vs_fd.png','bus7_tangent_vs_fd.png','candidate_operator_cosine_similarity.png','candidate_operator_relative_error.png','nonlinear_10pct_vs_linear_prediction.png','validated_bus7_bus12_signatures.png','bus7_amplitude_before_after_validation.png']:
  plt.figure(figsize=(5,3)); plt.text(.5,.5,'Validation pending complete native atlas',ha='center',va='center'); plt.axis('off'); plt.tight_layout(); plt.savefig(FIG/name,dpi=120); plt.close()
 status={'candidate_count':len(reg),'executed_trajectory_count':len(mf),'executed_success':int((mf.status=='EXECUTED_SUCCESS').sum()),'registry_sha256':sha,'completed_candidates':sorted(mf.candidate_bus.unique().tolist()),'requested_full_atlas':'16 candidates x 3 realizations x paired amplitudes','selected_epsilon':.005,'temporal_operator_validation':'PARTIAL','tangent_validation':'NOT_RUN','source_inference':'NOT_RUN'}; pd.DataFrame([status]).to_csv(RES/'atlas_summary.csv',index=False)
 (REP/'LOAD_RESPONSE_ATLAS_V1.md').write_text(f"# Physical Load Response Atlas V1\n\nFrozen model-table registry: {len(reg)} hidden ZIP-load candidates (SHA-256 `{sha}`). Native TDS executed {len(mf)} trajectories for complete groups Bus7 and Bus12, one realization each; the remaining candidate groups were not executed in this bounded run.\n\nFinite-difference voltage-only operators were generated for amplitudes +/-0.25, +/-0.5, +/-1 and +/-2 percent plus +10 percent. The atlas is therefore **PARTIAL** and no EVI or Bayesian source claim is promoted.\n\n{json.dumps(status,indent=2)}\n",encoding='utf-8'); (REP/'temporal_operator_unit_audit.md').write_text('# Unit audit\n\nAmplitude `a` is dimensionless fractional load scaling (0.10 = 10%). Measurements are rectangular PMU voltage Re/Im in pu. `G_FD = dY/da` therefore has pu-voltage per fractional severity. Percent values are converted to fractions exactly once.\n',encoding='utf-8'); (REP/'fd_derivative_validation.md').write_text('# Finite-difference validation\n\nCentral paired derivatives use identical pre-event operating points and native callback trajectories. Convergence statistics are reported in `fd_derivative_convergence.csv`; only Bus7 and Bus12 have executed native groups.\n',encoding='utf-8'); (REP/'trajectory_tangent_validation.md').write_text('# Trajectory-tangent validation\n\nDeferred: native differential/algebraic state export is required to integrate the DAE sensitivity equations. No tangent operator is claimed.\n',encoding='utf-8'); print(json.dumps(status,indent=2))
if __name__=='__main__': main()
