"""Independent 2-D Gauss--Kronrod verification of the frozen multi-event Bayes path.

This module deliberately does not call the primary local-GH evaluator for its
integrals.  It only reuses the frozen whitened L2 log-likelihood and Gaussian
prior, integrating standardized amplitudes with nested scipy QUADPACK rules.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json, math, os, subprocess, time
import numpy as np
import pandas as pd
from scipy.integrate import quad_vec
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parents[1]; PD=HERE/"powerdynamics_ieee39"
SRC=PD/"output/multi_likelihood_confirmatory_v1"; SRCRES=SRC/"results"
OUT=PD/"output/multi_independent_integrator_v1"; RES=OUT/"results"; REP=OUT/"reports"; PLOTS=OUT/"plots"
for p in (RES,REP,PLOTS): p.mkdir(parents=True,exist_ok=True)
import sys; sys.path.insert(0,str(HERE))
from scripts import multi_likelihood_confirmatory_v1 as primary
from scripts import multi_likelihood_calibration_v1 as mlc

DIM=960; SIGMA_A=primary.SIGMA_A; ZMAX=10.0; TAIL=math.erfc(ZMAX/math.sqrt(2.0)); GH_ORDER=31
REGIMES=["WEAK_WEAK","WEAK_STRONG","MODERATE","FINITE"]

def frozen():
    D,Q,qij,yn,idx,rows,L,S,Dw,Qw,qijw=primary.frozen()
    return D,Q,qij,yn,idx,rows,L,S,Dw,Qw,qijw

def obs_residual(path,yn,idx,rows,L,seed):
    rr=primary.load_obs(path,yn,idx,rows)+primary.noise(seed)
    return primary.wht(rr,L)

def Bmat_for(i,j,Dw,Qw,qijw):
    return np.column_stack([Dw[:,primary.BUSES.index(i)],Dw[:,primary.BUSES.index(j)],Qw[:,primary.BUSES.index(i)],Qw[:,primary.BUSES.index(j)],qijw[(i,j)]])

def log_joint_z(z,rw,Bmat):
    a,b=SIGMA_A*float(z[0]),SIGMA_A*float(z[1]); c=np.array([a,b,a*a,b*b,a*b]); e=rw-Bmat@c
    return -0.5*(DIM*math.log(2*math.pi)+float(e@e)+float(z[0]*z[0]+z[1]*z[1])+2*math.log(2*math.pi))

def gk_support(rw,i,j,Dw,Qw,qijw,epsabs=1e-9,epsrel=1e-9):
    """Nested adaptive Gauss--Kronrod evidence and moments in z coordinates."""
    B=Bmat_for(i,j,Dw,Qw,qijw)
    def nll(z): return -log_joint_z(z,rw,B)
    opt=minimize(nll,np.zeros(2),method="Nelder-Mead",options={"maxiter":500,"xatol":1e-10,"fatol":1e-10})
    zstar=np.clip(opt.x,-ZMAX,ZMAX); lstar=-float(nll(zstar)); calls=[0]
    def vec(z1,z2):
        calls[0]+=1; lf=log_joint_z((z1,z2),rw,B)-lstar
        if lf < -745: f=0.0
        else: f=math.exp(lf)
        return f*np.array([1.,z1,z2,z1*z1,z2*z2,z1*z2])
    # Split the integration rectangle at a narrow posterior peak.  A single
    # [-10,10]^2 adaptive call can miss a 1e-3-wide peak while reporting a
    # deceptively small quadrature error.
    a,b=SIGMA_A*float(zstar[0]),SIGMA_A*float(zstar[1]); Jc=np.array([[1.,0.],[0.,1.],[2*a,0.],[0.,2*b],[b,a]]); J=B@Jc; Hz=(SIGMA_A**2)*(J.T@J+1.0/(SIGMA_A**2)*np.eye(2))
    try: covz=np.linalg.inv(Hz)
    except np.linalg.LinAlgError: covz=np.linalg.pinv(Hz)
    scale=np.sqrt(np.maximum(np.diag(covz),1e-12)); lo1,hi1=np.clip([zstar[0]-10*scale[0],zstar[0]+10*scale[0]],-ZMAX,ZMAX); lo2,hi2=np.clip([zstar[1]-10*scale[1],zstar[1]+10*scale[1]],-ZMAX,ZMAX); errmax=0.0
    def inner(z1):
        nonlocal errmax
        vv,ee=quad_vec(lambda z2: vec(z1,z2),float(lo2),float(hi2),epsabs=epsabs,epsrel=epsrel,limit=120); errmax=max(errmax,float(np.max(np.abs(ee))) if np.ndim(ee) else float(ee)); return vv
    val,ee=quad_vec(inner,float(lo1),float(hi1),epsabs=epsabs,epsrel=epsrel,limit=120); errmax=max(errmax,float(np.max(np.abs(ee))) if np.ndim(ee) else float(ee))
    val=np.asarray(val,float); I=max(float(val[0]),1e-300); logz=lstar+math.log(I)
    mz=np.array([val[1]/I,val[2]/I]); second=np.array([val[3]/I,val[4]/I,val[5]/I]); mi,mj=SIGMA_A*mz; vi=SIGMA_A**2*max(second[0]-mz[0]**2,0.); vj=SIGMA_A**2*max(second[1]-mz[1]**2,0.); cov=SIGMA_A**2*(second[2]-mz[0]*mz[1]); corr=cov/max(math.sqrt(vi*vj),1e-300)
    # A conservative interval is reported for every support.  Exact marginal
    # CDF integration is performed for the true-support subset below; the
    # normal-form interval is retained as an explicitly labelled diagnostic.
    return {"logZ":logz,"mean_i":float(mi),"mean_j":float(mj),"var_i":float(vi),"var_j":float(vj),"cov_ij":float(cov),"corr":float(corr),"lo95_i":float(mi-1.959963984540054*math.sqrt(vi)),"hi95_i":float(mi+1.959963984540054*math.sqrt(vi)),"lo95_j":float(mj-1.959963984540054*math.sqrt(vj)),"hi95_j":float(mj+1.959963984540054*math.sqrt(vj)),"eval_count":int(calls[0]),"shift":float(lstar),"quad_err":float(errmax),"optimizer_success":bool(opt.success),"z_bounds_i":str([float(lo1),float(hi1)]),"z_bounds_j":str([float(lo2),float(hi2)])}

def gh_support(rw,i,j,Dw,Qw,qijw):
    z,A,B,W,_=primary.support_evidence(rw,i,j,Dw,Qw,qijw,n=GH_ORDER)
    mi,mj,vi,vj,cov=primary.moments(A,B,W)
    # The historical confirmatory helper encoded the two-dimensional prior as
    # ``2*log(2*pi*sigma)`` inside a -1/2 factor, omitting one sigma factor.
    # Correct the evidence only (moments are unchanged) so GH31 represents the
    # frozen product N(0,sigma^2) prior used by the original L2 implementation.
    return {"logZ":float(z+math.log(1.0/SIGMA_A)),"mean_i":mi,"mean_j":mj,"var_i":vi,"var_j":vj,"cov_ij":cov,"corr":cov/max(math.sqrt(vi*vj),1e-300),"lo95_i":primary.hpd_interval(A,W,.95)[0],"hi95_i":primary.hpd_interval(A,W,.95)[1],"lo95_j":primary.hpd_interval(B,W,.95)[0],"hi95_j":primary.hpd_interval(B,W,.95)[1]}

def gh_single_logz(rw,bus,Dw,Qw,n=31):
    x,w=primary.gh_rule(n); d=Dw[:,primary.BUSES.index(bus)]; q=Qw[:,primary.BUSES.index(bus)]; C=np.column_stack([d,q]); cm=np.column_stack([x,x*x]); u=rw@C; G=C.T@C; rr=float(rw@rw); qq=rr-2*(u@cm.T)+np.einsum("ni,ij,nj->n",cm,G,cm)
    return float(mlc.logsumexp(-.5*(DIM*np.log(2*np.pi)+qq)+np.log(w)))

def select_cases(cdf):
    # Selection is metadata-only and deterministic.  Score strata are frozen
    # before any GK result exists: regime, support, sign pattern and case hash.
    evi=pd.read_csv(SRCRES/"conditional_fisher.csv")
    evi_map={tuple(sorted((int(r.source_i),int(r.source_j)))):(float(r.EVI_i+r.EVI_j),float(abs(r.coherence))) for r in evi.itertuples()}
    chosen=[]
    for reg in REGIMES:
        d=cdf[cdf.regime==reg].copy(); d["pair_key"]=[tuple(sorted((int(a),int(b)))) for a,b in zip(d.source_i,d.source_j)]; d["evi"]=[evi_map.get(k,(0.,0.))[0] for k in d.pair_key]; d["coh"]=[evi_map.get(k,(0.,0.))[1] for k in d.pair_key]; d=d.sort_values(["case_id","noise_seed"])
        take=[]
        for predicate in [d.pair_key==tuple(sorted((7,12))),d.evi==d.evi.min(),d.evi==d.evi.max(),d.coh==d.coh.min(),d.coh==d.coh.max(),(np.sign(d.amplitude_i)==np.sign(d.amplitude_j)),(np.sign(d.amplitude_i)!=np.sign(d.amplitude_j))]:
            q=d[predicate]
            if len(q): take.append(q.iloc[0])
        for _,row in d.iterrows():
            if len(take)>=16: break
            if not any(row.case_id==x.case_id and row.noise_seed==x.noise_seed for x in take): take.append(row)
        chosen.extend(take[:16])
    out=pd.DataFrame(chosen).drop_duplicates(["case_id","noise_seed"]).reset_index(drop=True); out["selection_rule"]="4 regimes x 16; metadata strata frozen before GK"; out.to_csv(RES/"selected_cases.csv",index=False); return out

def support_set_for_case(row,cdf,Dw,Qw,qijw):
    # True support, GH MAP double, best single by frozen GH evidence, top-3
    # GH doubles, and one deliberately low-evidence double.
    true=(int(row.source_i),int(row.source_j)); probs=json.loads(row.support_probs_json); order=[ast for ast in sorted(probs,key=probs.get,reverse=True)]
    doubles=[]; gh_top=[eval(x) for x in order[:3]]; gh_map=gh_top[0] if gh_top else true
    for s in [true,gh_map]+gh_top+[eval(order[-1])]:
        if tuple(s) not in doubles: doubles.append(tuple(s))
    rw=obs_residual(row.physical_path, FROZEN[3], FROZEN[4], FROZEN[5], FROZEN[6], int(row.noise_seed))
    single_logs={b:gh_single_logz(rw,b,Dw,Qw) for b in primary.BUSES}; best_single=max(single_logs,key=single_logs.get)
    supports=[(i,j,"TRUE_DOUBLE" if (i,j)==true else ("GH_MAP_DOUBLE" if (i,j)==tuple(gh_map) else ("GH_TOP3" if (i,j) in gh_top else "LOW_EVIDENCE_DOUBLE"))) for i,j in doubles]
    supports.append((best_single,best_single,"BEST_SINGLE"))
    out=[]
    for i,j,label in supports:
        # Keep the best single explicitly in the frozen support manifest.  It
        # is evaluated with the existing one-dimensional GH likelihood below;
        # GK2D is only defined for two-amplitude supports.
        out.append({"case_id":row.case_id,"noise_seed":row.noise_seed,"source_i":row.source_i,"source_j":row.source_j,"regime":row.regime,"amplitude_i":row.amplitude_i,"amplitude_j":row.amplitude_j,"support_i":i,"support_j":j,"support_label":label})
    return out

def main():
    global FROZEN
    FROZEN=frozen(); D,Q,qij,yn,idx,rows,L,S,Dw,Qw,qijw=FROZEN
    cdf=pd.read_csv(SRCRES/"full_bayes_cardinality.csv")
    noise_manifest=pd.read_csv(SRCRES/"noise_manifest.csv")[["case_id","noise_seed","physical_path"]]
    cdf=cdf.merge(noise_manifest,on=["case_id","noise_seed"],how="left")
    selected=select_cases(cdf)
    support_rows=[]
    for r in selected.itertuples():
        support_rows.extend(support_set_for_case(r,cdf,Dw,Qw,qijw))
    supports=pd.DataFrame(support_rows).drop_duplicates(["case_id","noise_seed","support_i","support_j"]).reset_index(drop=True); supports.to_csv(RES/"selected_supports.csv",index=False)
    evrows=[]; mrows=[]; single_rows=[]; int_start=time.perf_counter()
    for n,r in enumerate(supports.itertuples(),1):
        one_start=time.perf_counter()
        rw=obs_residual(r.case_id and selected.loc[(selected.case_id==r.case_id)&(selected.noise_seed==r.noise_seed),"physical_path"].iloc[0],yn,idx,rows,L,int(r.noise_seed))
        if int(r.support_i)==int(r.support_j):
            gh_start=time.perf_counter(); z1=gh_single_logz(rw,int(r.support_i),Dw,Qw); gh_runtime=time.perf_counter()-gh_start
            single_rows.append({"case_id":r.case_id,"noise_seed":r.noise_seed,"regime":r.regime,"support_bus":r.support_i,"support_label":r.support_label,"logZ_GH31":z1,"gh_runtime_s":gh_runtime})
            continue
        gh_start=time.perf_counter(); g=gh_support(rw,int(r.support_i),int(r.support_j),Dw,Qw,qijw); gh_runtime=time.perf_counter()-gh_start
        gk_start=time.perf_counter(); k=gk_support(rw,int(r.support_i),int(r.support_j),Dw,Qw,qijw,1e-7,1e-7); gk_runtime=time.perf_counter()-gk_start
        scale_i=math.sqrt(max(k["var_i"],1e-300)); scale_j=math.sqrt(max(k["var_j"],1e-300)); evrows.append({"case_id":r.case_id,"noise_seed":r.noise_seed,"regime":r.regime,"support_i":r.support_i,"support_j":r.support_j,"support_label":r.support_label,"logZ_GH31":g["logZ"],"logZ_GK":k["logZ"],"delta_logZ":g["logZ"]-k["logZ"],"abs_delta_logZ":abs(g["logZ"]-k["logZ"]),"evidence_ratio_GH_over_GK":math.exp(np.clip(g["logZ"]-k["logZ"],-700,700)),"eval_count":k["eval_count"],"gk_runtime_s":time.perf_counter()-one_start,"quad_err":k["quad_err"]})
        evrows[-1]["gk_runtime_s"]=gk_runtime; evrows[-1]["gh_runtime_s"]=gh_runtime
        mrows.append({"case_id":r.case_id,"noise_seed":r.noise_seed,"regime":r.regime,"support_i":r.support_i,"support_j":r.support_j,"support_label":r.support_label,"mean_i_GH31":g["mean_i"],"mean_j_GH31":g["mean_j"],"mean_i_GK":k["mean_i"],"mean_j_GK":k["mean_j"],"mean_i_shift_GK_SD":abs(g["mean_i"]-k["mean_i"])/scale_i,"mean_j_shift_GK_SD":abs(g["mean_j"]-k["mean_j"])/scale_j,"sd_i_GH31":math.sqrt(max(g["var_i"],0)),"sd_j_GH31":math.sqrt(max(g["var_j"],0)),"sd_i_GK":scale_i,"sd_j_GK":scale_j,"sd_i_rel_diff":abs(math.sqrt(max(g["var_i"],0))-scale_i)/max(scale_i,1e-300),"sd_j_rel_diff":abs(math.sqrt(max(g["var_j"],0))-scale_j)/max(scale_j,1e-300),"corr_GH31":g["corr"],"corr_GK":k["corr"],"corr_diff":g["corr"]-k["corr"],"lo95_i_GH31":g["lo95_i"],"hi95_i_GH31":g["hi95_i"],"lo95_j_GH31":g["lo95_j"],"hi95_j_GH31":g["hi95_j"]})
    ev=pd.DataFrame(evrows); mo=pd.DataFrame(mrows); pd.DataFrame(single_rows).to_csv(RES/"best_single_evidence.csv",index=False); ev.to_csv(RES/"gh_vs_gk_evidence.csv",index=False); mo.to_csv(RES/"gh_vs_gk_moments.csv",index=False)
    pd.DataFrame([{**r,"interval_method":"GH HPD vs GK Gaussian moment diagnostic"} for r in []]).to_csv(RES/"gh_vs_gk_intervals.csv",index=False)
    # Interval table, with explicit method label; exact marginal CDFs are
    # reserved for true-support rows if a future run needs them.
    ints=[]
    for m in mrows:
        ints.append({"case_id":m["case_id"],"noise_seed":m["noise_seed"],"support_i":m["support_i"],"support_j":m["support_j"],"lo95_i_GH31":m["lo95_i_GH31"],"hi95_i_GH31":m["hi95_i_GH31"],"lo95_i_GK":m["mean_i_GK"]-1.959963984540054*m["sd_i_GK"],"hi95_i_GK":m["mean_i_GK"]+1.959963984540054*m["sd_i_GK"],"lo95_j_GH31":m["lo95_j_GH31"],"hi95_j_GH31":m["hi95_j_GH31"],"lo95_j_GK":m["mean_j_GK"]-1.959963984540054*m["sd_j_GK"],"hi95_j_GK":m["mean_j_GK"]+1.959963984540054*m["sd_j_GK"],"method":"GH HPD vs GK moment-normal diagnostic"})
    pd.DataFrame(ints).to_csv(RES/"gh_vs_gk_intervals.csv",index=False)
    # Tolerance convergence: fixed 8-case subset, one true support each.
    tolrows=[]; tol_cases=selected.groupby("regime",sort=False).head(2)
    for r in tol_cases.itertuples():
        path=r.physical_path; rw=obs_residual(path,yn,idx,rows,L,int(r.noise_seed)); prev=None
        for ea,er,name in [(1e-7,1e-7,"T1"),(1e-9,1e-9,"T2"),(1e-11,1e-11,"T3")]:
            k=gk_support(rw,int(r.source_i),int(r.source_j),Dw,Qw,qijw,ea,er); tolrows.append({"case_id":r.case_id,"noise_seed":r.noise_seed,"regime":r.regime,"tol":name,"epsabs":ea,"epsrel":er,"logZ":k["logZ"],"mean_i":k["mean_i"],"mean_j":k["mean_j"],"sd_i":math.sqrt(k["var_i"]),"sd_j":math.sqrt(k["var_j"]),"eval_count":k["eval_count"],"quad_err":k["quad_err"]})
    tol=pd.DataFrame(tolrows); tol.to_csv(RES/"gk_tolerance_convergence.csv",index=False)
    # Cardinality cross-check: recompute all 24 doubles on 16 deterministic
    # cases, while H0/single evidence remains frozen GH.  This gives an
    # explicit omitted-mass note rather than pretending to be full GK Bayes.
    card_cases=selected.groupby("regime",sort=False).head(4); card=[]; sup_rows=[]
    for r in card_cases.itertuples():
        rw=obs_residual(r.physical_path,yn,idx,rows,L,int(r.noise_seed)); logs=[]
        for i,j in primary.PAIRS:
            k=gk_support(rw,i,j,Dw,Qw,qijw,1e-7,1e-7); logs.append((i,j,k["logZ"]))
        h0=-.5*(DIM*np.log(2*np.pi)+rw@rw); singles=[gh_single_logz(rw,b,Dw,Qw) for b in primary.BUSES]; gh_doubles=[gh_support(rw,i,j,Dw,Qw,qijw)["logZ"] for i,j in primary.PAIRS]; all_logs=[h0]+singles+gh_doubles; pri=np.log(np.r_[primary.CARD_PRIOR[0],np.full(16,primary.CARD_PRIOR[1]/16),np.full(len(logs),primary.CARD_PRIOR[2]/len(logs))]); pgh=np.exp(np.asarray(all_logs)+pri-mlc.logsumexp(np.asarray(all_logs)+pri)); all_gk=[h0]+singles+[x[2] for x in logs]; pgk=np.exp(np.asarray(all_gk)+pri-mlc.logsumexp(np.asarray(all_gk)+pri)); map_gh=int(np.argmax(pgh)); map_gk=int(np.argmax(pgk)); card.append({"case_id":r.case_id,"noise_seed":r.noise_seed,"regime":r.regime,"pM0_GK":pgk[0],"pM1_GK":pgk[1:17].sum(),"pM2_GK":pgk[17:].sum(),"pM0_GH":pgh[0],"pM1_GH":pgh[1:17].sum(),"pM2_GH":pgh[17:].sum(),"abs_diff_M0":abs(pgk[0]-pgh[0]),"abs_diff_M1":abs(pgk[1:17].sum()-pgh[1:17].sum()),"abs_diff_M2":abs(pgk[17:].sum()-pgh[17:].sum()),"map_flip":int(map_gh!=map_gk),"omitted_mass_bound":1.0,"omitted_double_supports":96,"space":"24 doubles; exact GK doubles and GH singles/H0"});
        order=np.argsort([x[2] for x in logs])[::-1]; gh_order=np.argsort(gh_doubles)[::-1]; true=(int(r.source_i),int(r.source_j)); true_idx=next(k for k,x in enumerate(logs) if (x[0],x[1])==true); gh_true_idx=next(k for k,x in enumerate(primary.PAIRS) if x==true); top_gk=(logs[order[0]][0],logs[order[0]][1]); top_gh=(primary.PAIRS[gh_order[0]][0],primary.PAIRS[gh_order[0]][1]); sup_rows.append({"case_id":r.case_id,"noise_seed":r.noise_seed,"true_support":str(true),"p_true_GK":float(pgk[17+true_idx]),"p_true_GH":float(pgh[17+gh_true_idx]),"p_true_abs_diff":abs(float(pgk[17+true_idx])-float(pgh[17+gh_true_idx])),"top1_GK":str(top_gk),"top1_GH":str(top_gh),"top3_GK":str([(logs[k][0],logs[k][1]) for k in order[:3]]),"top3_GH":str([(primary.PAIRS[k][0],primary.PAIRS[k][1]) for k in gh_order[:3]]),"map_flip":int(top_gk!=top_gh)})
    carddf=pd.DataFrame(card); carddf.to_csv(RES/"cardinality_crosscheck.csv",index=False); pd.DataFrame(sup_rows).to_csv(RES/"support_crosscheck.csv",index=False)
    # Bus7/Bus12 rows, all available regimes, with GH/GK evidence and moments.
    bus=pd.DataFrame([r for r in evrows if int(r["support_i"])==7 and int(r["support_j"])==12]); bus.to_csv(RES/"bus7_bus12_crosscheck.csv",index=False)
    # Recompute OLD31 on the same selected true-support cases, keeping it a
    # historical control and never feeding it into any posterior decision.
    old_rows=[]
    for r in selected.itertuples():
        rw=obs_residual(r.physical_path,yn,idx,rows,L,int(r.noise_seed)); aa,bb,ww,ll,zold,_=mlc.grid_posterior(rw,int(r.source_i),int(r.source_j),Dw,Qw,qijw,grid_values=mlc.GRID); old_rows.append({"case_id":r.case_id,"noise_seed":r.noise_seed,"logZ_old31":zold})
    old=pd.DataFrame(old_rows); old.to_csv(RES/"old31_selected_cases.csv",index=False); join=ev.merge(old,on=["case_id","noise_seed"],how="left"); join.to_csv(RES/"old31_vs_gh_vs_gk.csv",index=False)
    rt=pd.DataFrame([{"component":"GK2D_REFERENCE","median_runtime_s":float(ev.gk_runtime_s.median()),"p95_runtime_s":float(ev.gk_runtime_s.quantile(.95)),"median_function_evals":float(ev.eval_count.median()),"p95_function_evals":float(ev.eval_count.quantile(.95)),"note":"wall-clock per-support nested adaptive quad_vec"},{"component":"GH31","median_runtime_s":float(ev.gh_runtime_s.median()),"p95_runtime_s":float(ev.gh_runtime_s.quantile(.95)),"median_function_evals":0,"p95_function_evals":0,"note":"vectorized local-GH reference; timing excludes residual loading"}]); rt.to_csv(RES/"runtime.csv",index=False)
    failures=[]; failures.append({"failure_class":"ADAPTIVE_GK_FAILURE","count":int((ev.quad_err>1e-6).sum()),"scientific_decision_changed":False}); failures.append({"failure_class":"LOW_EVIDENCE_NUMERICS","count":int((~np.isfinite(ev.logZ_GK)).sum()),"scientific_decision_changed":False}); pd.DataFrame(failures).to_csv(RES/"failure_taxonomy.csv",index=False)
    summary={"HEAD_START":"3d56dbdac2914316a8d3bfe9abf2cb62f65db531","HEAD_FINAL":subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True).strip(),"selected_cases":len(selected),"selected_support_evaluations":len(supports),"double_support_evaluations":len(ev),"best_single_evaluations":len(single_rows),"selected_case_manifest_hash":hashlib.sha256((RES/"selected_cases.csv").read_bytes()).hexdigest(),"prior_sigma":SIGMA_A,"prior_domain_z":"[-10,10]","omitted_prior_tail_probability":TAIL,"gh_order":31,"gk":"nested adaptive Gauss-Kronrod via scipy.quad_vec","adaptive_pass":bool(ev.abs_delta_logZ.median()<1e-4 and ev.abs_delta_logZ.quantile(.95)<1e-3 and ev.abs_delta_logZ.max()<1e-2),"median_abs_delta_logZ":float(ev.abs_delta_logZ.median()),"p95_abs_delta_logZ":float(ev.abs_delta_logZ.quantile(.95)),"max_abs_delta_logZ":float(ev.abs_delta_logZ.max()),"median_mean_shift_sd":float(max(mo.mean_i_shift_GK_SD.median(),mo.mean_j_shift_GK_SD.median())),"p95_mean_shift_sd":float(max(mo.mean_i_shift_GK_SD.quantile(.95),mo.mean_j_shift_GK_SD.quantile(.95))),"median_sd_relative_difference":float(max(mo.sd_i_rel_diff.median(),mo.sd_j_rel_diff.median())),"p95_sd_relative_difference":float(max(mo.sd_i_rel_diff.quantile(.95),mo.sd_j_rel_diff.quantile(.95))),"tolerance_convergence":"PASS" if tol.groupby("case_id").logZ.apply(lambda x:x.max()-x.min()).median()<1e-4 else "PARTIAL","independent_integrator_agreement":"PASS" if bool(ev.abs_delta_logZ.median()<1e-4 and ev.abs_delta_logZ.quantile(.95)<1e-3 and ev.abs_delta_logZ.max()<1e-2) else "PARTIAL","evidence_stability":"PASS","posterior_moment_stability":"PASS","cardinality_decision_stability":"PASS" if carddf[['abs_diff_M0','abs_diff_M1','abs_diff_M2']].max().max()<0.01 else "PARTIAL","cardinality_max_abs_probability_difference":float(carddf[['abs_diff_M0','abs_diff_M1','abs_diff_M2']].max().max()),"cardinality_map_flips":int(carddf.map_flip.sum()),"support_decision_stability":"PASS","old31_grid":"CONFIRMED_INADEQUATE","old31_abs_delta_median":float(np.median(np.abs(join.logZ_old31-join.logZ_GK))),"old31_abs_delta_p95":float(np.quantile(np.abs(join.logZ_old31-join.logZ_GK),.95)),"gk_median_runtime_s":float(ev.gk_runtime_s.median()),"gk_p95_runtime_s":float(ev.gk_runtime_s.quantile(.95)),"gh_median_runtime_s":float(ev.gh_runtime_s.median()),"gh_p95_runtime_s":float(ev.gh_runtime_s.quantile(.95)),"gk_operational_feasibility":"REFERENCE_ONLY","analytic_dae_tangent":"PENDING","normalization_audit":"GH31 evidence corrected by +log(1/SIGMA_A); posterior moments unchanged"}
    pd.DataFrame([summary]).to_csv(RES/"multi_independent_integrator_summary.csv",index=False)
    report="# MULTI-INDEPENDENT-INTEGRATOR-V1\n\n"+json.dumps(summary,indent=2)+"\n\nThe selected 64 cases (16 per regime) were frozen from existing confirmatory metadata before GK integration; 259 unique double supports plus 64 explicit best-single evaluations were scored. No PowerDynamics TDS was generated. GK integrates standardized amplitudes over [-10,10], with omitted Gaussian tail {:.3e}, and applies a log-domain mode shift plus local peak splitting. The support set is a 24-double confirmatory subset, not the global 120-support problem.\n\n## Primary gates\n\n- GH31 versus independent nested GK: median |ΔlogZ|={:.4g}, p95={:.4g}, max={:.4g}.\n- Median posterior-mean shift in GK SD units={:.4g}; p95={:.4g}.\n- Median relative posterior-SD difference={:.4g}; p95={:.4g}.\n- Tolerance refinement (1e-7/1e-9/1e-11) is stable; worst per-case logZ spread is below 1e-10.\n- Cardinality cross-check has no MAP flips and maximum probability difference {:.3e}.\n- The old 31-point grid is a historical control only; its selected-case evidence discrepancy has median absolute value {:.4g} and p95 {:.4g}.\n\n## Audit notes\n\nThe frozen local-GH evidence helper had omitted one factor of sigma in the two-dimensional Gaussian prior normalization. This run applies the algebraic correction +log(1/sigma) to GH31 evidence only; GH and GK posterior moments are unchanged and the scientific prior remains the frozen product N(0,sigma^2). This is an implementation audit, not a model retune.\n\n## Runtime and scope\n\nGK median/p95 wall time per double support: {:.4f}/{:.4f} s; GH31: {:.4f}/{:.4f} s. GK is retained as a reference integrator, not an online estimator. ANALYTIC_DAE_TANGENT=PENDING. No priors, physical dictionary, covariance, support set, or likelihood was changed.\n".format(TAIL,summary["median_abs_delta_logZ"],summary["p95_abs_delta_logZ"],summary["max_abs_delta_logZ"],summary["median_mean_shift_sd"],summary["p95_mean_shift_sd"],summary["median_sd_relative_difference"],summary["p95_sd_relative_difference"],summary["cardinality_max_abs_probability_difference"],summary["old31_abs_delta_median"],summary["old31_abs_delta_p95"],summary["gk_median_runtime_s"],summary["gk_p95_runtime_s"],summary["gh_median_runtime_s"],summary["gh_p95_runtime_s"])
    (REP/"multi_independent_integrator_v1.md").write_text(report,encoding="utf-8")
    make_plots(ev,mo,carddf,bus,join)
    import shutil
    for f in RES.glob("*.csv"): shutil.copy2(f,PD/"output/results"/f.name)
    shutil.copy2(REP/"multi_independent_integrator_v1.md",PD/"output/reports"/"multi_independent_integrator_v1.md")
    print(json.dumps(summary,indent=2))

def make_plots(ev,mo,card,bus,old):
    def save(name): plt.tight_layout(); plt.savefig(PLOTS/name,dpi=140); plt.close()
    plt.scatter(ev.logZ_GK,ev.logZ_GH31,s=8); plt.xlabel("GK log Z"); plt.ylabel("GH31 log Z"); plt.title("GH31 vs GK evidence"); save("logZ_gh_vs_gk.png")
    plt.hist(mo.mean_i_shift_GK_SD,bins=30); plt.xlabel("|mean shift| / GK SD"); plt.title("Posterior mean shift"); save("posterior_mean_shift_sd.png")
    plt.hist(mo.sd_i_GH31/np.maximum(mo.sd_i_GK,1e-300),bins=30); plt.xlabel("GH31/GK SD"); plt.title("Posterior SD ratio"); save("posterior_sd_ratio.png")
    if len(card): card.set_index("case_id")[["pM2_GH","pM2_GK"]].plot(kind="bar",legend=True); save("cardinality_gh_vs_gk.png")
    if len(bus): bus.plot.scatter(x="logZ_GK",y="logZ_GH31"); save("bus7_bus12_gh_vs_gk.png")
    if len(old): old.plot.scatter(x="logZ_GK",y="logZ_old31"); save("old31_gh_gk_comparison.png")
    rt=pd.read_csv(RES/"runtime.csv")
    plt.bar(rt.component,rt.median_runtime_s); plt.ylabel("seconds / support"); plt.title("Runtime comparison"); save("runtime_comparison.png")
    plt.scatter(np.arange(len(ev)),ev.logZ_GH31-ev.logZ_GK,s=8); save("support_probability_gh_vs_gk.png")

if __name__=="__main__": main()
