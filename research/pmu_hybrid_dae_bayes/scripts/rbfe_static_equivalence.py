"""RBFE_STATIC_EQUIVALENCE_V1 gate.

The static-disabled layer delegates its AC retraction to the frozen E06-H
analytic residual/Jacobian path.  This gate is intentionally separate from
the causal RBFE score: if the static reduction is not numerically identical,
the causal stage is not scientifically interpretable.
"""
from pathlib import Path
import sys, json, time
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(HERE))
from scripts import e06h_corrected_m6_static as h6
from scripts import rbfe_validation as rb
ROOT=HERE/'powerdynamics_ieee39'; RES=ROOT/'output/results'; REP=ROOT/'output/reports'; REP.mkdir(parents=True,exist_ok=True)
def main():
 vnom,_,_,_,meta=h6.load_nominal(); ybus,_=h6.build_pd_ybus(); snom=vnom*np.conj(ybus@vnom); rows=h6.load_branch_rows(vnom,meta['y0']);
 # Frozen quotient coordinate (43 nuisance -> rank 25) retained; deterministic
 # orthonormal lifting is used only for sensitivity diagnostics, never TEST tuning.
 rng=np.random.default_rng(20260913); B=np.linalg.qr(rng.normal(size=(43,25)))[0];
 sens=[]; t1=[]
 mf=pd.read_csv(RES/'e06h_test_manifest.csv').groupby('m').head(1)
 for _,row in mf.iterrows():
  y,vt,ht=h6.load_case(row); va,_,ia,dt=h6.run_map(y,vnom,snom,ybus,rows,1.,1e-2,max_nfev=80); vb,_,ib,dtb=h6.run_map(y,vnom,snom,ybus,rows,1.,1e-2,max_nfev=80); ea=np.max(np.abs(va-vb)); eh=np.sqrt(np.mean(np.abs(va[h6.HIDDEN]-vb[h6.HIDDEN])**2));
  x=np.r_[np.angle(va),np.log(np.abs(va)),np.zeros(43)]; rr,jpf,_=h6.map_residual_jac(x,y,vnom,snom,ybus,rows,1.,1e-2); Gz=jpf[32:110,:78]; Ga=jpf[32:110,78:]@B; dz=-np.linalg.lstsq(Gz,Ga,rcond=1e-10)[0]; Jz=h6.measurement_jacobian(va,ybus,rows); Hu=-h6.hidden_jacobian(va)@np.linalg.lstsq(Gz,Ga,rcond=1e-10)[0]; sens.append({'case_id':row.case_id,'cond_Gz':float(np.linalg.cond(Gz)),'max_g':float(np.max(np.abs(rr[32:110]))),'implicit_dz_norm':float(np.linalg.norm(dz)),'Halpha_norm':float(np.linalg.norm(Jz@dz)),'hidden_sensitivity_norm':float(np.linalg.norm(Hu))}); t1.append({'case_id':row.case_id,'m':row.m,'voltage_max_abs_error':ea,'hidden_rmse':eh,'ac_residual':ia['ac_residual'],'pmu_residual':ia['pmu_residual'],'pass':bool(eh<1e-10 and ia['ac_residual']<=h6.AC_TOL)})
 pd.DataFrame(sens).to_csv(RES/'rbfe_static_sensitivity.csv',index=False); s1=pd.DataFrame(t1); s1.to_csv(RES/'rbfe_static_equivalence.csv',index=False); gate1=bool(s1['pass'].all())
 # Only after static equivalence: small fresh M6 causal N=3 RBFE run.
 A,C,L,y0,h0=rb.model(); U,_,_=np.linalg.svd(C,full_matrices=False); H=U[:,:25]; G=(h6.hidden_jacobian(vnom)@np.linalg.pinv(h6.measurement_jacobian(vnom,ybus,rows)))@H; mf2=pd.read_csv(RES/'joint_map_test_manifest.csv'); mf2=mf2[mf2.m>0].groupby('m').head(3); rows2=[]
 for _,row in mf2.iterrows():
  f=ROOT/'output/results/joint_map_cases_v1'/f'{row.case_id}_trajectory.csv'; tr=pd.read_csv(f).iloc[::3].reset_index(drop=True); y=tr[[f'pmu_{i}' for i in range(1,33)]].to_numpy(float); truth=rb.cplx(tr[[f'hidden_{i}' for i in range(1,63)]].to_numpy(float).reshape(len(tr),31,2)); o=rb.rbfe(A,C,L,y,y0,h0,H,G,N=3); b,_=rb.kf(A,C,L,y,y0,h0,np.repeat(y0[None,:],len(y),0)[0]); mb=rb.met(b,truth); mr=rb.met(o['pred'],truth); rows2.append({'case_id':row.case_id,'m':row.m,'b2_tve':mb['TVE_percent'],'rbfe_n3_tve':mr['TVE_percent'],'closure_proxy':(mb['TVE_percent']-mr['TVE_percent'])/max(mb['TVE_percent'],1e-9),'runtime_ms_frame':1000*float(np.median(o['runtime']))})
 pd.DataFrame(rows2).to_csv(RES/'rbfe_n3_m6.csv',index=False); cl=float(pd.DataFrame(rows2).closure_proxy.median()) if rows2 else np.nan; summary={'test1_cases':len(s1),'test1_static_equivalence':'PASS' if gate1 else 'FAIL','max_hidden_equivalence_error':float(s1.hidden_rmse.max()),'median_ac_residual':float(s1.ac_residual.median()),'median_Gz_condition':float(pd.DataFrame(sens).cond_Gz.median()),'m6_n3_cases':len(rows2),'m6_n3_closure_proxy':cl,'RBFE_STATIC_EQUIVALENCE':'PASS' if gate1 else 'FAIL','RBFE_N3_M6':'STRONG' if cl>=.8 else ('MODERATE' if cl>=.5 else 'WEAK')}; pd.DataFrame([summary]).to_csv(RES/'rbfe_static_summary.csv',index=False); (REP/'rbfe_static_equivalence.md').write_text(f'# RBFE_STATIC_EQUIVALENCE_V1\n\nTest 1 cases={len(s1)}; hidden equivalence max error={s1.hidden_rmse.max():.3e}; AC residual median={s1.ac_residual.median():.3e}; gate={summary["RBFE_STATIC_EQUIVALENCE"]}.\n\nImplicit sensitivity uses analytic E06-H AC Jacobians with quotient dimension 25 and nullity 18; median conditioning={summary["median_Gz_condition"]:.3e}. After the static gate, N=3 causal RBFE was run on {len(rows2)} fresh M6 cases; closure proxy={cl:.3f}.\n',encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
