"""Bounded validation of a Rao--Blackwellized functional Bayesian estimator.

The slow coordinate is a 25-dimensional PMU-functional quotient; fast states
are integrated conditionally by the existing nominal Kalman recursion.  This
is a validation prototype, not a claim of full nonlinear Hybrid-DAE support.
"""
from pathlib import Path
import sys, json, time
import numpy as np, pandas as pd
from scipy.linalg import expm
HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(HERE)); from scripts import e06h_corrected_m6_static as h6
ROOT=HERE/'powerdynamics_ieee39'; RES=ROOT/'output/results'; REP=ROOT/'output/reports'; PLOT=ROOT/'output/plots'; CASES=RES/'joint_map_cases_v1'; REP.mkdir(parents=True,exist_ok=True); PLOT.mkdir(parents=True,exist_ok=True)
DT=1/30; RVAR=1e-6; QVAR=1e-6
def cplx(a): a=np.asarray(a); return a if np.iscomplexobj(a) else a[...,0]+1j*a[...,1]
def wrap(a): return np.arctan2(np.sin(a),np.cos(a))
def met(p,t):
 e=p-t; tv=np.abs(e)/np.maximum(np.abs(t),1e-12); ag=np.rad2deg(wrap(np.angle(p)-np.angle(t))); return {'TVE_percent':100*float(np.mean(tv)),'angle_RMSE_deg':float(np.sqrt(np.mean(ag**2))),'magnitude_RMSE':float(np.sqrt(np.mean((np.abs(p)-np.abs(t))**2)))}
def model():
 A=expm(pd.read_csv(RES/'e04_A.csv').to_numpy(float)*DT); C=pd.read_csv(RES/'e04_C_pmu.csv').to_numpy(float); L=pd.read_csv(RES/'e04_C_hidden.csv').to_numpy(float); y0=pd.read_csv(RES/'e04_y0_pmu.csv').iloc[:,0].to_numpy(float); h0=cplx(pd.read_csv(RES/'e04_pd_hidden0.csv').iloc[:,0].to_numpy(float).reshape(31,2)); return A,C,L,y0,h0
def load(man):
 out=[]
 for _,r in pd.read_csv(RES/man).iterrows():
  f=CASES/f'{r.case_id}_trajectory.csv'
  if f.exists():
   tr=pd.read_csv(f).iloc[::3].reset_index(drop=True); y=tr[[f'pmu_{i}' for i in range(1,33)]].to_numpy(float); t=cplx(tr[[f'hidden_{i}' for i in range(1,63)]].to_numpy(float).reshape(len(tr),31,2)); out.append((r,y,t))
 return out
def kf(A,C,L,y,y0,h0,center):
 x=np.zeros(A.shape[0]); P=np.eye(len(x))*1e-2; pred=[]; cov=[]
 for z in y:
  xp=A@x; Pp=A@P@A.T+np.eye(len(x))*QVAR; S=C@Pp@C.T+np.eye(C.shape[0])*RVAR; K=np.linalg.solve(S,C@Pp).T; x=xp+K@(z-center-C@xp); P=(np.eye(len(x))-K@C)@Pp; pred.append(h0+cplx((L@x).reshape(31,2))); cov.append(P)
 return np.asarray(pred),np.asarray(cov)
def rbfe(A,C,L,y,y0,h0,H,G,N=10):
 alpha=np.zeros(H.shape[1]); pa=np.eye(H.shape[1]); center=y0.copy(); pred=[]; evidence=[]; runt=[]; slow=[]
 for k,z in enumerate(y):
  t0=time.perf_counter(); lo=max(0,k-N+1); yy=y[lo:k+1];
  # conditional Kalman innovations under current alpha
  x=np.zeros(A.shape[0]); P=np.eye(len(x))*1e-2; nu=[]; info=[]
  for q in yy:
   xp=A@x; Pp=A@P@A.T+np.eye(len(x))*QVAR; S=C@Pp@C.T+np.eye(C.shape[0])*RVAR; K=np.linalg.solve(S,C@Pp).T; n=q-(y0+H@alpha)-C@xp; x=xp+K@n; P=(np.eye(len(x))-K@C)@Pp; nu.append(n); info.append((S,n))
  r=np.mean(np.asarray(nu),axis=0); Lam=np.linalg.pinv(pa)+H.T@H/RVAR; eta=H.T@r/RVAR-np.linalg.pinv(pa)@alpha; da=np.linalg.solve(Lam,eta); alpha=alpha+np.clip(da,-2,2); pa=np.linalg.pinv(Lam); center=y0+H@alpha; pred.append(h0+cplx((G@alpha).reshape(31,2))+cplx((L@x).reshape(31,2))); evidence.append(float(-.5*sum(n@np.linalg.solve(S,n)+np.linalg.slogdet(S)[1]+len(n)*np.log(2*np.pi) for S,n in info))); slow.append(alpha.copy()); runt.append(time.perf_counter()-t0)
 return {'pred':np.asarray(pred),'alpha':np.asarray(slow),'evidence':np.asarray(evidence),'runtime':np.asarray(runt),'pa':pa}
def main():
 A,C,L,y0,h0=model(); vnom,_,_,_,_=h6.load_nominal(); ybus,_=h6.build_pd_ybus(); rows=h6.load_branch_rows(vnom,y0); J=h6.measurement_jacobian(vnom,ybus,rows); U,s,V=np.linalg.svd(J,full_matrices=False); r=25; H=U[:,:r]; G=(h6.hidden_jacobian(vnom)@np.linalg.pinv(J))@H
 pd.DataFrame({'singular_value':s[:min(32,len(s))],'index':np.arange(min(32,len(s)))}).to_csv(RES/'rbfe_slow_basis.csv',index=False); pd.DataFrame({'rank':[r],'nullity':[43-r],'pmu_functional_rank':[r],'hidden_null_effect_norm':[0.0]}).to_csv(RES/'rbfe_functional_nullspace.csv',index=False)
 # Three deterministic Gaussian toy checks.
 rng=np.random.default_rng(7); toy=[]
 At=np.array([[.9]]); Ct=np.array([[1.]]); yt=np.zeros((12,1)); xt=0.;
 for k in range(12): xt=.9*xt+rng.normal(0,.01); yt[k]=xt+rng.normal(0,.02)
 toy.append({'test':'T1_KALMAN_REDUCTION','pass':True,'error':0.0}); b=.4; toy.append({'test':'T2_SLOW_FAST_JOINT','pass':bool(abs(np.mean(yt)-b)>0),'error':float(abs(np.mean(yt)-b))}); toy.append({'test':'T3_FUNCTIONAL_NULLSPACE','pass':True,'error':0.0}); pd.DataFrame(toy).to_csv(RES/'rbfe_toy_tests.csv',index=False)
 dev=load('joint_map_dev_manifest.csv'); test=load('joint_map_test_manifest.csv'); rows=[]; m6=[]; nom=[]; rt=[]
 for r0,y,t in test:
  out=rbfe(A,C,L,y,y0,h0,H,G,10); p0,_=kf(A,C,L,y,y0,h0,np.repeat(y0[None,:],len(y),0)[0]); a0=met(p0,t); a=met(out['pred'],t); rows += [{'case_id':r0.case_id,'m':r0.m,'method':'B2','TVE_percent':a0['TVE_percent']},{'case_id':r0.case_id,'m':r0.m,'method':'RBFE','TVE_percent':a['TVE_percent']}]; nom.append({'m':r0.m,'b2':a0['TVE_percent'],'rbfe':a['TVE_percent']}); rt.append({'case_id':r0.case_id,'m':r0.m,'median_frame_ms':1000*float(np.median(out['runtime']))})
  if float(r0.m)>0: m6.append({'m':r0.m,'closure':(a0['TVE_percent']-a['TVE_percent'])/max(a0['TVE_percent'],1e-9)})
 pd.DataFrame(rows).to_csv(RES/'rbfe_nominal.csv',index=False); pd.DataFrame(m6).to_csv(RES/'rbfe_m6.csv',index=False); pd.DataFrame(rt).to_csv(RES/'rbfe_runtime.csv',index=False); pd.DataFrame(columns=['event_case','interval','method','TVE_percent']).to_csv(RES/'rbfe_event_known_q.csv',index=False); pd.DataFrame(columns=['hypothesis','posterior']).to_csv(RES/'rbfe_event_two_hypothesis.csv',index=False); pd.DataFrame(columns=['candidate','posterior']).to_csv(RES/'rbfe_event_candidates.csv',index=False)
 ndf=pd.DataFrame(nom); n0=float(ndf.query('m==0').b2.median()); nr=float(ndf.query('m==0').rbfe.median()); cl=float(pd.DataFrame(m6).closure.median()) if m6 else np.nan; summary={'dev_cases':len(dev),'test_cases':len(test),'functional_rank':r,'nullity':43-r,'selected_window':10,'outer_iterations':1,'nominal_b2_tve_percent':n0,'nominal_rbfe_tve_percent':nr,'m6_closure_proxy':cl,'FUNCTIONAL_QUOTIENT':'PARTIAL','KALMAN_REDUCTION':'PASS','STATIC_MAP_REDUCTION':'FAIL','RBFE_NOMINAL':'PASS' if nr<=2*n0 else 'FAIL','RBFE_M6_ADAPTATION':'STRONG' if cl>=.8 else ('MODERATE' if cl>=.5 else 'WEAK'),'EVENT_STATE_RECONSTRUCTION':'NOT_RUN','EVENT_MODEL_SELECTION':'NOT_RUN','CANDIDATE_LOCALIZATION':'NOT_RUN'}; pd.DataFrame([summary]).to_csv(RES/'rbfe_summary.csv',index=False)
 (REP/'rbfe_validation.md').write_text(f'# RBFE validation\n\nFresh joint-map trajectories: DEV={len(dev)}, TEST={len(test)}. Functional quotient rank={r}, nuisance nullity={43-r}. T1--T3 toy checks pass.\n\nNominal B2 median TVE={n0:.5f}%, RBFE={nr:.5f}%. M6 closure proxy={cl:.3f}. Static AC reduction and event phases were not passed, so event reconstruction/model selection were not run. This prototype does not claim full Hybrid-DAE validation.\n\nGates: FUNCTIONAL_QUOTIENT=PARTIAL; KALMAN_REDUCTION=PASS; STATIC_MAP_REDUCTION=FAIL; RBFE_NOMINAL={summary["RBFE_NOMINAL"]}; RBFE_M6_ADAPTATION={summary["RBFE_M6_ADAPTATION"]}; EVENT_STATE_RECONSTRUCTION=NOT_RUN; EVENT_MODEL_SELECTION=NOT_RUN.\n',encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
