"""E04-B0 bounded joint slow/fast fixed-lag MAP smoke.

This implementation solves the linearized joint Gaussian window (shared PMU
centre plus nominal dynamic deviations) by block-coordinate Gauss--Newton.
It is deliberately labelled a minimal approximation until AC-constrained
KKT residuals are added; no truth enters ``joint_case``.
"""
from pathlib import Path
import sys, time, json
import numpy as np, pandas as pd
from scipy.linalg import expm
HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(HERE)); from scripts import e06h_corrected_m6_static as h6
ROOT=HERE/'powerdynamics_ieee39'; RES=ROOT/'output/results'; REP=ROOT/'output/reports'; CASES=RES/'joint_map_cases_v1'; REP.mkdir(parents=True,exist_ok=True)
DT=1/30; Q=1e-6; R=1e-6
def cplx(a): a=np.asarray(a); return a if np.iscomplexobj(a) else a[...,0]+1j*a[...,1]
def wrap(a): return np.arctan2(np.sin(a),np.cos(a))
def metric(p,t):
 e=p-t; tv=np.abs(e)/np.maximum(np.abs(t),1e-12); ang=np.rad2deg(wrap(np.angle(p)-np.angle(t))); return {'TVE_percent':100*float(np.mean(tv)),'angle_RMSE_deg':float(np.sqrt(np.mean(ang**2))),'magnitude_RMSE':float(np.sqrt(np.mean((np.abs(p)-np.abs(t))**2)))}
def load_model():
 A=expm(pd.read_csv(RES/'e04_A.csv').to_numpy(float)*DT); C=pd.read_csv(RES/'e04_C_pmu.csv').to_numpy(float); L=pd.read_csv(RES/'e04_C_hidden.csv').to_numpy(float); y0=pd.read_csv(RES/'e04_y0_pmu.csv').iloc[:,0].to_numpy(float); h0=cplx(pd.read_csv(RES/'e04_pd_hidden0.csv').iloc[:,0].to_numpy(float).reshape(31,2)); return A,C,L,y0,h0
def load_cases(man):
 out=[]
 for _,r in pd.read_csv(RES/man).iterrows():
  f=CASES/f'{r.case_id}_trajectory.csv'
  if f.exists():
   tr=pd.read_csv(f).iloc[::3].reset_index(drop=True); y=tr[[f'pmu_{i}' for i in range(1,33)]].to_numpy(float); t=cplx(tr[[f'hidden_{i}' for i in range(1,63)]].to_numpy(float).reshape(len(tr),31,2)); out.append((r,y,t))
 return out
def kf(A,C,y,center):
 x=np.zeros(A.shape[0]); P=np.eye(A.shape[0])*1e-2; pred=[]
 for z in y:
  xp=A@x; Pp=A@P@A.T+np.eye(len(x))*Q; S=C@Pp@C.T+np.eye(C.shape[0])*R; K=np.linalg.solve(S,C@Pp).T; x=xp+K@(z-center-C@xp); P=(np.eye(len(x))-K@C)@Pp; pred.append(x.copy())
 return np.asarray(pred)
def joint_case(A,C,L,y0,h0,y,window=5):
 # One causal window at a time; shared centre and dynamic path are inferred jointly.
 centre=y0.copy(); xs=np.zeros((len(y),A.shape[0])); t0=time.perf_counter()
 for end in range(len(y)):
  lo=max(0,end-window+1); yy=y[lo:end+1]
  for _ in range(2):
   xx=kf(A,C,yy,centre); centre=centre + np.mean(yy-centre-(xx@C.T),axis=0)
  xs[lo:end+1]=xx
 p=np.asarray([h0+cplx((L@x).reshape(31,2)) for x in xs]); return {'pred':p,'centre':centre,'runtime':time.perf_counter()-t0,'iterations':2,'ac_residual':np.nan}
def frozen(A,C,L,y0,h0,y):
 x=kf(A,C,y,y0); return np.asarray([h0+cplx((L@q).reshape(31,2)) for q in x])
def main():
 A,C,L,y0,h0=load_model(); dev=load_cases('joint_map_dev_manifest.csv'); test=load_cases('joint_map_test_manifest.csv');
 # DEV chooses only the preregistered horizons; no hidden scores are used for TEST.
 devsel=[]
 for N in (3,5,10):
  vals=[]
  for r,y,t in dev[:4]: vals.append(metric(joint_case(A,C,L,y0,h0,y,N)['pred'],t)['TVE_percent'])
  devsel.append({'horizon':N,'median_dev_tve':float(np.median(vals))})
 pd.DataFrame(devsel).to_csv(RES/'e04b0_dev_selection.csv',index=False); N=int(min(devsel,key=lambda z:z['median_dev_tve'])['horizon'])
 rows=[]; closures=[]; runt=[]
 for r,y,t in test:
  a0=frozen(A,C,L,y0,h0,y); out=joint_case(A,C,L,y0,h0,y,N); a3=out['pred']; a1=np.repeat(t[0][None,:],len(y),0)
  m0=metric(a0,t); m1=metric(a1,t); m3=metric(a3,t)
  for name,m in [('A0_FROZEN_B2',m0),('A1_ORACLE_CENTER',m1),('A3_JOINT_MAP',m3)]: rows.append({'case_id':r.case_id,'m':r.m,'seed':r.seed,'method':name,**m})
  den=m0['TVE_percent']-m1['TVE_percent']; closures.append({'case_id':r.case_id,'m':r.m,'closure':(m0['TVE_percent']-m3['TVE_percent'])/den if den>1e-9 else np.nan}); runt.append({'case_id':r.case_id,'runtime_s':out['runtime'],'iterations':out['iterations']})
 per=pd.DataFrame(rows); per.to_csv(RES/'e04b0_test_per_case.csv',index=False); pd.DataFrame(closures).to_csv(RES/'e04b0_closure.csv',index=False); pd.DataFrame(runt).to_csv(RES/'e04b0_runtime.csv',index=False)
 n0=per.query("m==0 and method=='A0_FROZEN_B2'").TVE_percent.median(); n3=per.query("m==0 and method=='A3_JOINT_MAP'").TVE_percent.median(); cl=pd.DataFrame(closures).query('m>0').closure.median(); status='PASS' if cl>=.8 and n3<=2*n0 else ('PARTIAL' if cl>=.5 else 'FAIL'); summary={'dev_cases':len(dev),'test_cases':len(test),'selected_horizon':N,'closure_median':float(cl),'nominal_r0_tve_percent':float(n0),'nominal_a3_tve_percent':float(n3),'JOINT_SLOW_FAST_MAP':status,'NOMINAL_SAFETY':'PASS' if n3<=2*n0 else 'FAIL','AC_CONSTRAINTS':'NOT_IMPLEMENTED'}; pd.DataFrame([summary]).to_csv(RES/'e04b0_summary.csv',index=False); (REP/'e04b0_joint_map.md').write_text(f'# E04-B0 joint slow-fast fixed-lag MAP\n\nDEV={len(dev)}, TEST={len(test)}, horizon={N}. Median closure={cl:.3f}; nominal A0={n0:.4f}%, A3={n3:.4f}%.\n\nStatus: JOINT_SLOW_FAST_MAP={status}; NOMINAL_SAFETY={summary["NOMINAL_SAFETY"]}. This bounded implementation is linearized and does not claim AC-constrained Hybrid-DAE validation.\n',encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
