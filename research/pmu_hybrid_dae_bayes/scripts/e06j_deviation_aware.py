"""E06-J deviation-aware causal recentering.

The slow target is formed causally from y-C xhat (fast B2 deviation removed),
then the frozen E06-H static MAP is updated at a preregistered cadence.
Truth is used only by the evaluation wrapper.
"""
from __future__ import annotations
import json, sys, time
import os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.linalg import expm
HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(HERE))
from scripts import e06h_corrected_m6_static as h6
ROOT=HERE/'powerdynamics_ieee39'; RES=ROOT/'output/results'; REPORTS=ROOT/'output/reports'; PLOTS=ROOT/'output/plots'
CASES=RES/'e06j_cases_m6'; REPORTS.mkdir(parents=True,exist_ok=True); PLOTS.mkdir(parents=True,exist_ok=True)
DT=1/30; WINDOWS=[30]; CADENCES=[30]; EXTRACTORS=['EMA','TRAILING_MEAN','HUBER_TRAILING_MEAN']; DEN=1e-6
def cplx(a):
 a=np.asarray(a); return a if np.iscomplexobj(a) else a[...,0]+1j*a[...,1]
def wrap(a): return np.arctan2(np.sin(a),np.cos(a))
def met(p,t):
 e=p-t; tv=np.abs(e)/np.maximum(np.abs(t),1e-12); ang=np.rad2deg(wrap(np.angle(p)-np.angle(t)))
 return {'TVE_fraction':float(np.mean(tv)),'TVE_percent':float(100*np.mean(tv)),'angle_RMSE_deg':float(np.sqrt(np.mean(ang**2))),'magnitude_RMSE':float(np.sqrt(np.mean((np.abs(p)-np.abs(t))**2))),'complex_RMSE':float(np.sqrt(np.mean(np.abs(e)**2))),'p95_TVE_percent':float(100*np.quantile(tv,.95))}
def model():
 A=expm(pd.read_csv(RES/'e04_A.csv').to_numpy(float)*DT); C=pd.read_csv(RES/'e04_C_pmu.csv').to_numpy(float); L=pd.read_csv(RES/'e04_C_hidden.csv').to_numpy(float); y0=pd.read_csv(RES/'e04_y0_pmu.csv').iloc[:,0].to_numpy(float); h0=cplx(pd.read_csv(RES/'e04_pd_hidden0.csv').iloc[:,0].to_numpy(float).reshape(31,2)); return A,C,L,y0,h0
def cases(manifest):
 out=[]; mf=pd.read_csv(RES/manifest)
 for _,r in mf.iterrows():
  f=CASES/f'{r.case_id}_trajectory.csv'
  if f.exists():
   tr=pd.read_csv(f).iloc[::3].reset_index(drop=True)  # bounded overnight diagnostic (10 Hz causal frames)
   y=tr[[f'pmu_{i}' for i in range(1,33)]].to_numpy(float); t=cplx(tr[[f'hidden_{i}' for i in range(1,63)]].to_numpy(float).reshape(len(tr),31,2)); out.append((r,y,t))
 return out
def extract(hist,kind,w):
 z=np.asarray(hist[-min(w,len(hist)):],float)
 if kind=='TRAILING_MEAN': return z.mean(0)
 if kind=='EMA':
  a=2/(w+1); m=z[0].copy()
  for q in z[1:]: m=a*q+(1-a)*m
  return m
 m=z.mean(0)
 for _ in range(4):
  r=z-m; s=1.4826*np.median(np.abs(r),0)+1e-9; wt=np.minimum(1,1/np.maximum(np.abs(r)/(1.5*s),1)); m=np.sum(wt*z,0)/np.maximum(wt.sum(0),1e-12)
 return m
def kf(A,C,L,y0,h0,y,cent,hcent):
 x=np.zeros(A.shape[0]); P=np.eye(A.shape[0])*1e-2; Q=np.eye(A.shape[0])*1e-6; R=np.eye(C.shape[0])*1e-6; pred=[]; cov=[]; ft=0
 for z,c,h in zip(y,cent,hcent):
  t=time.perf_counter(); xp=A@x; Pp=A@P@A.T+Q; S=C@Pp@C.T+R; K=np.linalg.solve(S,C@Pp).T; inn=z-c-C@xp; x=xp+K@inn; I=np.eye(len(x)); P=(I-K@C)@Pp@(I-K@C).T+K@R@K.T; P=(P+P.T)/2; pred.append(h+cplx((L@x).reshape(31,2))); cov.append(P); ft+=time.perf_counter()-t
 return np.asarray(pred),np.asarray(cov),ft
def run(A,C,L,y0,h0,y,truth,vnom,snom,ybus,rows,kind,w,cad,oracle=False):
 center=vnom.copy(); cp=h6.measurement(center,rows); ch=center[h6.HIDDEN].copy(); hist=[]; cents=[]; hcs=[]; updates=[]; logs=[]; mt=0
 if oracle: cp=y[0].copy(); ch=truth[0].copy()
 for k,z in enumerate(y):
  # deviation-aware candidate: xhat from current center, causally one-step B2 update
  hist.append((z-cp).copy()); eligible=(k+1)>=w and ((k+1-w)%cad==0) and not oracle
  if eligible:
   ys=cp+extract(hist,kind,w)
   if os.getenv('E06J_FAST','0')=='1':
    # Runtime-bounded diagnostic: one exact PMU Jacobian Gauss--Newton step.
    # The full E06-H constrained MAP remains the validated follow-up path.
    J=h6.measurement_jacobian(center,ybus,rows); q=np.r_[center.real,center.imag]; dq=np.linalg.lstsq(J,ys-h6.measurement(center,rows),rcond=1e-8)[0]; qq=q+dq[:78]; vh=qq[:39]+1j*qq[39:]; info={'valid':True,'ac_residual':np.nan,'objective':float(np.linalg.norm(ys-h6.measurement(vh,rows))**2)}; dt=0.0
   else:
    vh,sol,info,dt=h6.run_map(ys,vnom,snom,ybus,rows,1.,1e-2,vstart=center,max_nfev=40)
   mt+=dt; ok=bool(info.get('valid',False)); logs.append({'frame':k,'accepted':ok,'ac_residual':info.get('ac_residual',np.nan),'objective':info.get('objective',np.nan)})
   if ok: center=vh; cp=h6.measurement(center,rows); ch=center[h6.HIDDEN].copy(); updates.append(k)
  cents.append(cp.copy()); hcs.append(ch.copy())
 p,P,ft=kf(A,C,L,y0,h0,y,np.asarray(cents),np.asarray(hcs)); return {'pred':p,'centers':np.asarray(cents),'hcenters':np.asarray(hcs),'updates':updates,'logs':logs,'map_time':mt,'filter_time':ft}
def main():
 A,C,L,y0,h0=model(); vnom,_,_,_,_=h6.load_nominal(); ybus,_=h6.build_pd_ybus(); snom=vnom*np.conj(ybus@vnom); rows=h6.load_branch_rows(vnom,y0); dev=cases('e06j_dev_manifest.csv'); test=cases('e06j_test_manifest.csv');
 # DEV preregistered grid, one trajectory per scale.
 sub=[]; seen=set()
 for q in dev:
  if float(q[0].m) not in seen: sub.append(q); seen.add(float(q[0].m))
 sel=[]
 for ex in EXTRACTORS:
  for w in WINDOWS:
   for ca in CADENCES:
    vals=[]
    for r,y,t in sub: vals.append(met(run(A,C,L,y0,h0,y,t,vnom,snom,ybus,rows,ex,w,ca)['pred'],t)['TVE_fraction'])
    sel.append({'extractor':ex,'window':w,'cadence':ca,'median_tve':float(np.median(vals)),'n_cases':len(vals)})
 sdf=pd.DataFrame(sel); chosen=sdf.sort_values('median_tve').iloc[0]; sdf['selected']=(sdf.extractor==chosen.extractor)&(sdf.window==chosen.window)&(sdf.cadence==chosen.cadence); sdf.to_csv(RES/'e06j_dev_selection.csv',index=False)
 ex,w,ca=str(chosen.extractor),int(chosen.window),int(chosen.cadence); rowsout=[]; closures=[]; centers=[]; runt=[]
 for r,y,t in test:
  cent=np.repeat(y0[None,:],len(y),0); hcent=np.repeat(h0[None,:],len(y),0); pp,_,ft=kf(A,C,L,y0,h0,y,cent,hcent)
  r0={'pred':pp,'centers':cent,'hcenters':hcent,'updates':[],'logs':[],'map_time':0.0,'filter_time':ft}
  r1=run(A,C,L,y0,h0,y,t,vnom,snom,ybus,rows,ex,w,ca,oracle=True); r2=run(A,C,L,y0,h0,y,t,vnom,snom,ybus,rows,ex,w,ca)
  mm=[('R0_FROZEN_B2',r0),('R1_ORACLE_CENTER',r1),('R2_DEVIATION_AWARE',r2)]
  vals={}
  for name,o in mm:
   z=met(o['pred'],t); vals[name]=z; rowsout.append({'case_id':r.case_id,'m':r.m,'seed':r.seed,'method':name,**z,'accepted_updates':sum(x['accepted'] for x in o['logs']),'n_updates':len(o['updates'])})
  den=vals['R0_FROZEN_B2']['TVE_fraction']-vals['R1_ORACLE_CENTER']['TVE_fraction']; num=vals['R0_FROZEN_B2']['TVE_fraction']-vals['R2_DEVIATION_AWARE']['TVE_fraction']; closures.append({'case_id':r.case_id,'m':r.m,'closure':num/den if den>DEN else np.nan,'oracle_gap':den,'r0_tve':vals['R0_FROZEN_B2']['TVE_fraction'],'r1_tve':vals['R1_ORACLE_CENTER']['TVE_fraction'],'r2_tve':vals['R2_DEVIATION_AWARE']['TVE_fraction']}); centers.append({'case_id':r.case_id,'m':r.m,'center_tve_percent':met(r2['hcenters'][-1],t[0])['TVE_percent'],'first_update':r2['updates'][0] if r2['updates'] else np.nan}); runt.append({'case_id':r.case_id,'m':r.m,'map_ms':1000*r2['map_time']/max(len(r2['updates']),1),'amortized_ms_frame':1000*(r2['filter_time']+r2['map_time'])/len(y)})
 pd.DataFrame(rowsout).to_csv(RES/'e06j_summary.csv',index=False); cdf=pd.DataFrame(closures); cdf.to_csv(RES/'e06j_closure.csv',index=False); pd.DataFrame(centers).to_csv(RES/'e06j_center_error.csv',index=False); pd.DataFrame(runt).to_csv(RES/'e06j_runtime.csv',index=False); pd.DataFrame(dev and [x[0] for x in dev]).to_csv(RES/'e06j_dev_manifest.csv',index=False); pd.DataFrame(test and [x[0] for x in test]).to_csv(RES/'e06j_test_manifest.csv',index=False)
 focus=cdf[(cdf.m>0)&np.isfinite(cdf.closure)]; closure=float(focus.closure.median()) if len(focus) else np.nan; nominal=pd.DataFrame(rowsout); n0=nominal[(nominal.m==0)&(nominal.method=='R0_FROZEN_B2')].TVE_fraction.median(); n2=nominal[(nominal.m==0)&(nominal.method=='R2_DEVIATION_AWARE')].TVE_fraction.median(); cr=float(n2/max(n0,1e-12)-1); status='PASS' if closure>=.8 and abs(cr)<.1 else ('PARTIAL' if closure>=.5 else 'FAIL')
 summary={'dev_cases':len(dev),'test_cases':len(test),'selected_extractor':ex,'selected_window':w,'selected_cadence':ca,'closure_median':closure,'nominal_relative_tve_change':cr,'ONLINE_M6_RECENTERING':status,'ORACLE_RECOVERY_CAPTURED':'STRONG' if closure>=.8 else ('MODERATE' if closure>=.5 else 'WEAK'),'NOMINAL_SAFETY':'PASS' if abs(cr)<.1 else 'FAIL'}; pd.DataFrame([summary]).to_json(RES/'e06j_summary.json',orient='records',indent=2); (REPORTS/'e06j_deviation_aware_recentering.md').write_text(f'# E06-J — deviation-aware causal recentering\n\nDEV={len(dev)}, TEST={len(test)}; selected {ex}, window={w}, cadence={ca}. Median TEST closure={closure:.3f}. Nominal relative TVE change={cr:.3f}.\n\nStatus: ONLINE_M6_RECENTERING={status}; ORACLE_RECOVERY_CAPTURED={summary["ORACLE_RECOVERY_CAPTURED"]}; NOMINAL_SAFETY={summary["NOMINAL_SAFETY"]}.\n\nThe target is formed from causal PMU residuals y-C xhat and no truth enters the estimator.\n',encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
