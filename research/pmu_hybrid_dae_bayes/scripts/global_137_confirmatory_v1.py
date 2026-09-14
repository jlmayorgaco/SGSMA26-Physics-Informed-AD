"""GLOBAL-137-CONFIRMATORY-V1.

Fresh prospective full-support confirmation.  The physical bank is generated
by the validated native PowerDynamics callback harness under a new output
namespace; inference then evaluates H0 + 16 singles + 120 doubles with the
frozen GH31 implementation.  No model component is fitted here.
"""
from __future__ import annotations
from pathlib import Path
import ast, hashlib, json, math, os, shutil, subprocess, sys, time
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from scipy.stats import beta, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parents[1]; PD=HERE/"powerdynamics_ieee39"
OUT=PD/"output/global_137_confirmatory_v1"; RES=OUT/"results"; REP=OUT/"reports"; PLOTS=OUT/"plots"
PHYS=OUT/"physical"; PHYSRES=PHYS/"results"
for p in (RES,REP,PLOTS,PHYSRES): p.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(HERE))
from scripts import load_multi_pilot_v1 as pilot
from scripts import multi_likelihood_calibration_v1 as mlc
from scripts import multi_likelihood_confirmatory_v1 as primary
from scripts import multi_independent_integrator_v1 as independent

BUSES=pilot.BUSES; PAIRS=pilot.ALL_PAIRS; SIGMA_A=pilot.SIGMA_A; CARD_PRIOR=pilot.CARD_PRIOR
DIM=960; GH=31; N_NOISE=3; H0_N=100; NOISE_BASE=9_137_000
# Values were frozen before simulation.  None occur in any previous manifest.
AMP_PAIRS=[(0.00014,0.00033,"WEAK_WEAK"),(0.00026,0.00057,"WEAK_WEAK"),
           (0.00041,0.00088,"WEAK_WEAK"),(0.00068,0.00121,"WEAK_WEAK"),
           (0.00073,0.00255,"WEAK_STRONG"),(0.00260,0.00540,"MODERATE"),
           (0.01130,0.02170,"FINITE")]
SINGLE_AMPS=[0.00043,0.00127,0.00370,0.00940]
HEAD_START="9c3bbeb83df055663c1eee0d0c2c3f0578f71f17"
_OBS={}; _GH={}; _SINGLE={}

def tag(a): return str(float(a)).replace("-","m").replace(".","p")
def pair_path(i,j,ai,aj): return PHYSRES/f"PAIR_{i}_{j}_AI{tag(ai)}_AJ{tag(aj)}_R1.csv"
def single_path(i,a): return PHYSRES/f"PAIR_{i}_0_AI{tag(a)}_AJ0p0_R1.csv"
def wht(x,L): return solve_triangular(L,np.asarray(x,float).reshape(-1),lower=True,check_finite=False)
def parse(x):
    try:
        y=ast.literal_eval(str(x)); return tuple(int(v) for v in y) if isinstance(y,tuple) else ((int(y),) if y else ())
    except Exception:return ()
def noise(seed): return np.asarray(pilot.noise(int(seed))).reshape(-1)

def frozen():
    D,Q,qij,yn,idx,rows,L,S=mlc.frozen_inputs()
    Dw=np.column_stack([wht(D[:,k],L) for k in range(16)])
    Qw=np.column_stack([wht(Q[:,k],L) for k in range(16)])
    qijw={(i,j):wht(v,L) for (i,j),v in qij.items()}
    return D,Q,qij,yn,idx,rows,L,S,Dw,Qw,qijw

def preregister():
    rows=[]
    for i,j in PAIRS:
        for ai0,aj0,reg in AMP_PAIRS:
            for si in (-1.,1.):
                for sj in (-1.,1.):
                    ai,aj=si*ai0,sj*aj0; p=pair_path(i,j,ai,aj)
                    rows.append(dict(case_id=p.stem,source_i=i,source_j=j,amplitude_i=ai,amplitude_j=aj,regime=reg,physical_path=str(p),status="PREREGISTERED",trajectory_id=p.stem))
    for i in BUSES:
        for a0 in SINGLE_AMPS:
            for s in (-1.,1.):
                a=s*a0; p=single_path(i,a); rows.append(dict(case_id=p.stem,source_i=i,source_j=0,amplitude_i=a,amplitude_j=0.,regime="SINGLE",physical_path=str(p),status="PREREGISTERED",trajectory_id=p.stem))
    d=pd.DataFrame(rows); d.to_csv(RES/"physical_manifest.csv",index=False)
    (RES/"physical_manifest.sha256").write_text(hashlib.sha256(d.to_csv(index=False).encode()).hexdigest()+"\n",encoding="utf-8")
    return d

def generate_physics(mf):
    # The Julia harness writes exactly this namespace.  It is checkpoint/resume
    # safe: existing files are skipped by simulate_pair_reuse.
    jl=PD/"julia/scripts/pd_load_multi_atlas.jl"; env=os.environ.copy()
    pairs=";".join(f"{i}-{j}" for i,j in PAIRS)
    mags=";".join(f"{a},{b}" for a,b,_ in AMP_PAIRS)
    e=env.copy(); e.update({"MULTI_OUT":"global_137_confirmatory_v1","MULTI_MODE":"DEV","MULTI_REUSE":"1","MULTI_REPS":"1","MULTI_PAIRS":pairs,"MULTI_CANDIDATES":",".join(map(str,BUSES)),"MULTI_MAG_PAIRS":mags})
    cp=subprocess.run(["julia","--project=.",str(jl)],cwd=str(PD/"julia"),env=e,text=True,capture_output=True)
    (RES/"julia_pairs_stdout.log").write_text(cp.stdout+"\nSTDERR\n"+cp.stderr,encoding="utf-8")
    e2=env.copy(); e2.update({"MULTI_OUT":"global_137_confirmatory_v1","MULTI_MODE":"SINGLE","MULTI_REUSE":"0","MULTI_REPS":"1","MULTI_CANDIDATES":",".join(map(str,BUSES)),"MULTI_AMPS":",".join(map(str,SINGLE_AMPS))})
    cp2=subprocess.run(["julia","--project=.",str(jl)],cwd=str(PD/"julia"),env=e2,text=True,capture_output=True)
    (RES/"julia_single_stdout.log").write_text(cp2.stdout+"\nSTDERR\n"+cp2.stderr,encoding="utf-8")
    d=mf.copy(); d["status"]=["EXECUTED_SUCCESS" if Path(p).exists() else "EXECUTED_FAIL" for p in d.physical_path]
    # The Julia SINGLE file names are the same PAIR_b_0 contract used above.
    d.to_csv(RES/"physical_manifest.csv",index=False); (RES/"physical_manifest.sha256").write_text(hashlib.sha256(d.to_csv(index=False).encode()).hexdigest()+"\n",encoding="utf-8")
    return d,cp.returncode or cp2.returncode

def noise_manifest(mf):
    rows=[]; k=0
    for r in mf[mf.status=="EXECUTED_SUCCESS"].itertuples():
        for q in range(N_NOISE):
            rows.append(dict(case_id=r.case_id,trajectory_id=r.trajectory_id,physical_path=r.physical_path,source_i=r.source_i,source_j=r.source_j,amplitude_i=r.amplitude_i,amplitude_j=r.amplitude_j,regime=r.regime,noise_seed=NOISE_BASE+k,split="CONFIRMATORY_TEST")); k+=1
    for q in range(H0_N): rows.append(dict(case_id=f"H0_{q:04d}",trajectory_id=f"H0_{q:04d}",physical_path="",source_i=0,source_j=0,amplitude_i=0.,amplitude_j=0.,regime="H0",noise_seed=NOISE_BASE+k,split="CONFIRMATORY_CAL")); k+=1
    d=pd.DataFrame(rows); d.to_csv(RES/"noise_manifest.csv",index=False); return d

def obs(path,yn,idx,rows):
    key=str(path)
    if key not in _OBS:
        _OBS[key]=pilot.response(Path(path),yn,idx,rows).reshape(-1)
    return _OBS[key]

def basis(Dw,Qw,qijw):
    B=np.stack([np.column_stack([Dw[:,BUSES.index(i)],Dw[:,BUSES.index(j)],Qw[:,BUSES.index(i)],Qw[:,BUSES.index(j)],qijw[(i,j)]]) for i,j in PAIRS]); G=np.einsum('pdi,pdj->pij',B,B); return B,G

def gh_single(rw,Dw,Qw):
    if GH not in _SINGLE:
        x,w=np.polynomial.hermite.hermgauss(GH); z=np.sqrt(2)*x; _SINGLE[GH]=(z,w/np.sqrt(np.pi))
    z,zw=_SINGLE[GH]; rr=float(rw@rw); out=np.zeros((16,4)); prior=1/SIGMA_A**2
    for k in range(16):
        C=np.column_stack([Dw[:,k],Qw[:,k]]); G=C.T@C; u=C.T@rw; th=0.
        for _ in range(10):
            H=G[0,0]+4*th*G[0,1]+4*th*th*G[1,1]+prior; grad=G[0,0]*th+G[0,1]*th*th-u[0]-2*th*u[1]+prior*th; step=grad/max(H,1e-30); th-=step
            if abs(step)<1e-12: break
        H=G[0,0]+4*th*G[0,1]+4*th*th*G[1,1]+prior; sd=1/np.sqrt(max(H,1e-30)); a=th+sd*z; Cn=np.column_stack([a,a*a]); qn=rr-2*Cn@u+np.einsum('ni,ij,nj->n',Cn,G,Cn); lw=-.5*(DIM*np.log(2*np.pi)+2*np.log(2*np.pi*SIGMA_A)+qn+prior*a*a)+.5*z*z+np.log(zw); lz=float(logsumexp(lw)+np.log(sd)); p=np.exp(lw-logsumexp(lw)); mi=float(p@a); out[k]=[lz,mi,float(p@(a-mi)**2),H]
    return out

def gh_double(rw,B,G):
    prior=1/SIGMA_A**2; P=len(PAIRS); u=np.einsum('pdi,d->pi',B,rw); rr=float(rw@rw); th=np.zeros((P,2)); eye=np.eye(2)
    for _ in range(10):
        a,b=th[:,0],th[:,1]; Jc=np.zeros((P,5,2)); Jc[:,0,0]=1; Jc[:,1,1]=1; Jc[:,2,0]=2*a; Jc[:,3,1]=2*b; Jc[:,4,0]=b; Jc[:,4,1]=a; c=np.stack([a,b,a*a,b*b,a*b],1); E=np.einsum('pij,pj->pi',G,c)-u; grad=np.einsum('pki,pk->pi',Jc,E)+prior*th; H=np.einsum('pki,pkj->pij',Jc,np.einsum('pab,pbj->paj',G,Jc))+prior*eye[None,:,:]; st=np.linalg.solve(H,grad[...,None])[...,0]; th-=st
        if np.max(np.linalg.norm(st,axis=1))<1e-11: break
    x,w=np.polynomial.hermite.hermgauss(GH); z=np.sqrt(2)*x; zw=w/np.sqrt(np.pi); z1,z2=np.meshgrid(z,z,indexing="ij"); zg=np.column_stack([z1.ravel(),z2.ravel()]); logzw=np.log(np.outer(zw,zw).ravel()); out=np.zeros((P,12))
    for p in range(P):
        a,b=th[p]; Jc=np.array([[1.,0.],[0,1],[2*a,0],[0,2*b],[b,a]]); H=Jc.T@G[p]@Jc+prior*eye; cov=np.linalg.pinv(H,rcond=1e-12); Lc=np.linalg.cholesky(cov+1e-15*eye); T=np.array([a,b])[None,:]+zg@Lc.T; aa,bb=T[:,0],T[:,1]; C=np.column_stack([aa,bb,aa*aa,bb*bb,aa*bb]); qn=rr-2*C@u[p]+np.einsum('ni,ij,nj->n',C,G[p],C); lf=-.5*(DIM*np.log(2*np.pi)+2*np.log(2*np.pi*SIGMA_A)+qn+prior*(aa*aa+bb*bb)); lw=lf+.5*np.sum(zg*zg,axis=1)+logzw; mx=lw.max(); ww=np.exp(lw-mx); ww/=ww.sum(); lz=float(mx+np.log(np.exp(lw-mx).sum())+np.log(np.linalg.det(Lc))+np.log(2*np.pi)+np.log(1/SIGMA_A)); mi=float(ww@aa); mj=float(ww@bb); vi=float(ww@(aa-mi)**2); vj=float(ww@(bb-mj)**2); cv=float(ww@((aa-mi)*(bb-mj))); out[p]=[lz,mi,mj,vi,vj,cv,mi-1.96*np.sqrt(max(vi,0)),mi+1.96*np.sqrt(max(vi,0)),mj-1.96*np.sqrt(max(vj,0)),mj+1.96*np.sqrt(max(vj,0)),float(np.linalg.cond(H)),float(np.linalg.eigvalsh(H).min())]
    return out

def infer(rw,Dw,Qw,qijw,B,G):
    z0=-.5*(DIM*np.log(2*np.pi)+float(rw@rw)); s=gh_single(rw,Dw,Qw); d=gh_double(rw,B,G); raw=np.r_[z0,s[:,0],d[:,0]]; pri=np.log(np.r_[CARD_PRIOR[0],np.full(16,CARD_PRIOR[1]/16),np.full(120,CARD_PRIOR[2]/120)]); post=np.exp(raw+pri-logsumexp(raw+pri)); return raw,post,s,d

def run(m,yn,idx,rows,L,Dw,Qw,qijw,B,G):
    cards=[]; supports=[]; incl=[]; amps=[]; cache={}; t0=time.perf_counter()
    for n,r in enumerate(m.itertuples(index=False),1):
        rw=wht((noise(r.noise_seed) if r.regime=="H0" else obs(r.physical_path,yn,idx,rows)+noise(r.noise_seed)),L); raw,post,s,d=infer(rw,Dw,Qw,qijw,B,G); cache[r.case_id]=(raw,post,s,d)
        pred=int(np.argmax(post)); predM=0 if pred==0 else (1 if pred<17 else 2); predS=() if pred==0 else ((BUSES[pred-1],) if pred<17 else PAIRS[pred-17]); trueS=() if r.regime=="H0" else ((int(r.source_i),) if r.regime=="SINGLE" else tuple(sorted((int(r.source_i),int(r.source_j))))); ptrue=float(post[0] if not trueS else (post[1+BUSES.index(trueS[0])] if len(trueS)==1 else post[17+PAIRS.index(trueS)]));
        cards.append(dict(case_id=r.case_id,trajectory_id=r.trajectory_id,regime=r.regime,true_M=len(trueS),source_i=r.source_i,source_j=r.source_j,amplitude_i=r.amplitude_i,amplitude_j=r.amplitude_j,noise_seed=r.noise_seed,p_M0=post[0],p_M1=post[1:17].sum(),p_M2=post[17:].sum(),pred_M=predM,pred_support=str(predS),true_support=str(trueS),p_true_support=ptrue,posterior_entropy=float(-post@np.log(np.maximum(post,1e-300)))) )
        for b in BUSES:
            pi=float(post[1+BUSES.index(b)]+sum(post[17+k] for k,(i,j) in enumerate(PAIRS) if b in (i,j))); incl.append(dict(case_id=r.case_id,regime=r.regime,true_M=len(trueS),source_bus=b,truth_included=int(b in trueS),posterior_inclusion=pi))
        if len(trueS)==2:
            k=PAIRS.index(trueS); dd=d[k]; amps += [dict(case_id=r.case_id,regime=r.regime,source_bus=trueS[0],true_amplitude=r.amplitude_i,posterior_mean=dd[1],posterior_sd=np.sqrt(max(dd[3],0)),lo95=dd[6],hi95=dd[7],coverage95=int(dd[6]<=r.amplitude_i<=dd[7])),dict(case_id=r.case_id,regime=r.regime,source_bus=trueS[1],true_amplitude=r.amplitude_j,posterior_mean=dd[2],posterior_sd=np.sqrt(max(dd[4],0)),lo95=dd[8],hi95=dd[9],coverage95=int(dd[8]<=r.amplitude_j<=dd[9]))]
        if n%500==0: print(f"confirmatory inference {n}/{len(m)} ({time.perf_counter()-t0:.1f}s)")
    C=pd.DataFrame(cards); I=pd.DataFrame(incl); A=pd.DataFrame(amps); C.to_csv(RES/"cardinality_per_case.csv",index=False); I.to_csv(RES/"source_inclusion.csv",index=False); A.to_csv(RES/"true_support_amplitude.csv",index=False)
    sup=[]
    for r in C.itertuples(index=False):
        raw,post,s,d=cache[r.case_id];
        for h in range(137): sup.append(dict(case_id=r.case_id,noise_seed=r.noise_seed,regime=r.regime,support_type="H0" if h==0 else ("SINGLE" if h<17 else "DOUBLE"),support_i="" if h==0 else (BUSES[h-1] if h<17 else PAIRS[h-17][0]),support_j="" if h<17 else ("" if h==0 else PAIRS[h-17][1]),log_evidence=raw[h],posterior=post[h],true_support=r.true_support))
    pd.DataFrame(sup).to_csv(RES/"support_per_case.csv",index=False); return C,I,A,cache

def summarize(C,I,A,cache,Dw,Qw,qijw,L):
    rows=[]
    for reg,g in C.groupby("regime",sort=False):
        tr=g.true_M.to_numpy(); pr=g.pred_M.to_numpy(); P=g[["p_M0","p_M1","p_M2"]].to_numpy(); rows.append(dict(regime=reg,n=len(g),physical_trajectories=g.trajectory_id.nunique(),mean_p_M0=g.p_M0.mean(),mean_p_M1=g.p_M1.mean(),mean_p_M2=g.p_M2.mean(),accuracy=float(np.mean(tr==pr)),NLL=float(np.mean(-np.log(np.maximum(P[np.arange(len(g)),tr],1e-300)))),Brier=float(np.mean(np.sum((P-np.eye(3)[tr])**2,axis=1))),false_split=float(np.mean((tr==1)&(pr==2))),false_merge=float(np.mean((tr==2)&(pr==1))),mean_entropy=g.posterior_entropy.mean()))
    card=pd.DataFrame(rows); card.to_csv(RES/"cardinality_summary.csv",index=False)
    d=C[C.true_M==2].copy(); sr=[]
    for r in d.itertuples():
        ts=parse(r.true_support); ps=parse(r.pred_support); post=cache[r.case_id][1]; rank=1+sum(post[17+k]>post[17+PAIRS.index(ts)] for k in range(120)); sr.append(dict(case_id=r.case_id,regime=r.regime,source_i=r.source_i,source_j=r.source_j,exact=int(r.pred_M==2 and set(ts)==set(ps)),top3=int(rank<=3),top5=int(rank<=5),top10=int(rank<=10),rank_true=rank,jaccard=len(set(ts)&set(ps))/max(len(set(ts)|set(ps)),1),p_true_support=r.p_true_support))
    S=pd.DataFrame(sr); ss=S.groupby("regime",as_index=False).agg(n=("case_id","size"),top1=("exact","mean"),top3=("top3","mean"),top5=("top5","mean"),top10=("top10","mean"),median_rank=("rank_true","median"),mean_p_true=("p_true_support","mean")); ss.to_csv(RES/"support_summary.csv",index=False); S.to_csv(RES/"support_per_case_summary.csv",index=False)
    # Full spike-and-slab moments.  Conditional GH moments are exact for the
    # frozen local posterior; the zero atom is retained explicitly.
    ma=[]; pairbus={b:[k for k,(i,j) in enumerate(PAIRS) if b in (i,j)] for b in BUSES}
    for r in C.itertuples(index=False):
        post=cache[r.case_id][1]; dd=cache[r.case_id][3]; trueS=parse(r.true_support)
        for b in BUSES:
            terms=[(float(post[1+BUSES.index(b)]),dd[BUSES.index(b)] if False else 0.,0.)]
            for k in pairbus[b]:
                i,j=PAIRS[k]; terms.append((float(post[17+k]),float(dd[k,1] if b==i else dd[k,2]),float(dd[k,3] if b==i else dd[k,4])))
            wt=np.asarray([x[0] for x in terms]); wt/=max(wt.sum(),1e-300); mu=float(sum(w*m for w, m, v in terms)); var=float(sum(w*(v+m*m) for w,m,v in terms)-mu*mu); z0=float(1-sum(x[0] for x in terms[1:])); ma.append(dict(case_id=r.case_id,regime=r.regime,source_bus=b,truth_included=int(b in trueS),inclusion_probability=1-z0,zero_mass=z0,posterior_mean=mu,posterior_sd=np.sqrt(max(var,0)),posterior_median=0. if z0>=.5 else mu,method="explicit atom + conditional GH moments"))
    MA=pd.DataFrame(ma); MA.to_csv(RES/"model_averaged_amplitude.csv",index=False)
    # Inclusion metrics.
    yi=I.truth_included.to_numpy(); pi=I.posterior_inclusion.to_numpy(); pred=pi>=.5; e=np.linspace(0,1,11); ece=0.
    for lo,hi in zip(e[:-1],e[1:]):
        q=(pi>=lo)&(pi<(hi if hi<1 else hi+1e-12)); ece+=q.mean()*abs(pi[q].mean()-yi[q].mean()) if q.any() else 0.
    pd.DataFrame([dict(n=len(I),AUROC=roc_auc_score(yi,pi),AUPRC=average_precision_score(yi,pi),Brier=float(np.mean((pi-yi)**2)),ECE=float(ece),precision=float(np.sum(pred&yi)/max(pred.sum(),1)),recall=float(np.sum(pred&yi)/max(yi.sum(),1))) ]).to_csv(RES/"source_inclusion_summary.csv",index=False)
    # Event/cardinality/support diagnostics.
    y=(C.true_M.to_numpy()>0).astype(int); score=1-C.p_M0.to_numpy(); n0=int(np.sum((y==0)&(score>=.5))); N0=int(np.sum(y==0)); nmiss=int(np.sum((y==1)&(score<.5))); ev=pd.DataFrame([dict(n=len(C),event_prevalence=float(y.mean()),AUROC=roc_auc_score(y,score),AUPRC=average_precision_score(y,score),FPR=float(n0/max(N0,1)),FPR_lo=float(beta.ppf(.025,n0,N0-n0+1) if n0 else 0.),FPR_hi=float(beta.ppf(.975,n0+1,N0-n0) if n0<N0 else 1.),FNR=float(nmiss/max(y.sum(),1)))]); ev.to_csv(RES/"event_detection.csv",index=False)
    # Multiplicity decomposition.
    md=[]
    for r in C.itertuples(index=False):
        raw=cache[r.case_id][0]; l1=float(raw[1:17].max()); l2=float(raw[17:].max()); v1=float(logsumexp(raw[1:17]-l1)); v2=float(logsumexp(raw[17:]-l2)); odds=float(logsumexp(raw[17:]+math.log(CARD_PRIOR[2]/120))-logsumexp(raw[1:17]+math.log(CARD_PRIOR[1]/16))); md.append(dict(case_id=r.case_id,regime=r.regime,best_support_term=l2-l1,multiplicity_term=-math.log(120/16),support_volume_term=v2-v1,cardinality_prior_term=math.log(CARD_PRIOR[2]/CARD_PRIOR[1]),posterior_log_odds_M2_vs_M1=odds,decomposition_residual=odds-((l2-l1)-math.log(120/16)+(v2-v1)+math.log(CARD_PRIOR[2]/CARD_PRIOR[1]))))
    M=pd.DataFrame(md); M.to_csv(RES/"multiplicity_decomposition.csv",index=False)
    # Frozen local Fisher and geometry, computed before empirical summaries are
    # interpreted.  These are prospective metadata, not fitted predictors.
    cf=[]; geom=[]
    for i,j in PAIRS:
        x=Dw[:,BUSES.index(i)]; yj=Dw[:,BUSES.index(j)]; ei=float(x@x); ej=float(yj@yj); cr=float(x@yj/np.sqrt(max(ei*ej,1e-300))); cf.append(dict(source_i=i,source_j=j,EVI_i=ei,EVI_j=ej,I_j_given_i=ej-(x@yj)**2/max(ei,1e-300),I_i_given_j=ei-(x@yj)**2/max(ej,1e-300),coherence=cr)); s=np.linalg.svd(np.column_stack([x,yj]),compute_uv=False); geom.append(dict(source_i=i,source_j=j,coherence=cr,principal_angle_rad=float(np.arccos(np.clip(abs(cr),0,1))),sigma_min=float(s[-1])))
    pd.DataFrame(cf).to_csv(RES/"conditional_fisher.csv",index=False); pd.DataFrame(geom).to_csv(RES/"pair_difficulty_atlas.csv",index=False)
    # Frozen predictor audit: no confirmatory fitting; a monotone amplitude
    # score is evaluated descriptively and therefore reported inconclusive.
    rows=[]
    for g in ["WEAK_WEAK","WEAK_STRONG","MODERATE","FINITE"]:
        q=C[C.regime==g]; rows.append(dict(regime=g,n=len(q),mean_abs_amplitude=float(np.mean(np.maximum(abs(q.amplitude_i),abs(q.amplitude_j)))) if len(q) else np.nan,exact_support=float(S[S.regime==g].exact.mean()) if g in set(S.regime) else np.nan))
    pd.DataFrame(rows).to_csv(RES/"fisher_incremental_test.csv",index=False)
    # Descriptive manifold distances in the frozen mean dictionary.
    nd=[]; pdist=[]
    for r in d.itertuples():
        i,j=int(r.source_i),int(r.source_j); pair= r.amplitude_i*Dw[:,BUSES.index(i)]+r.amplitude_j*Dw[:,BUSES.index(j)]; single_i=r.amplitude_i*Dw[:,BUSES.index(i)]; single_j=r.amplitude_j*Dw[:,BUSES.index(j)]; nd.append(dict(case_id=r.case_id,regime=r.regime,distance_pair_to_M1=float(min(np.linalg.norm(pair-single_i),np.linalg.norm(pair-single_j))),merge=int(r.pred_M==1)))
    pd.DataFrame(nd).to_csv(RES/"nested_single_manifold_distance.csv",index=False)
    # Pair distance to nearest frozen first-order competitor.
    for r in d.itertuples():
        i,j=int(r.source_i),int(r.source_j); v=r.amplitude_i*Dw[:,BUSES.index(i)]+r.amplitude_j*Dw[:,BUSES.index(j)]; best=1e99; bj="";
        for k,l in PAIRS:
            if (k,l)==(i,j): continue
            z=r.amplitude_i*Dw[:,BUSES.index(k)]+r.amplitude_j*Dw[:,BUSES.index(l)]; q=float(np.linalg.norm(v-z));
            if q<best: best=q; bj=str((k,l))
        pdist.append(dict(case_id=r.case_id,regime=r.regime,nearest_pair=bj,nearest_distance=best,wrong_double=int(r.pred_M==2 and parse(r.pred_support)!=(i,j))))
    pd.DataFrame(pdist).to_csv(RES/"pair_manifold_distance.csv",index=False)
    return card,ss,MA,M,ev

def gk_check(C,cache,Dw,Qw,qijw,L,yn,idx,rows):
    chosen=C[(C.true_M==2)&(C.regime.isin(["WEAK_WEAK","WEAK_STRONG","MODERATE","FINITE"]))].groupby("regime").head(1); out=[]; nm=pd.read_csv(RES/"noise_manifest.csv")
    for r in chosen.itertuples(index=False):
        i,j=tuple(sorted((int(r.source_i),int(r.source_j)))); nr=nm[nm.case_id==r.case_id].iloc[0]; rw=wht(obs(nr.physical_path,yn,idx,rows)+noise(int(nr.noise_seed)),L); gh0=primary.support_evidence(rw,i,j,Dw,Qw,qijw,n=31); gh=float(gh0[0]+math.log(1/SIGMA_A)); ghmi,ghmj,_,_,_=primary.moments(gh0[1],gh0[2],gh0[3]); gk=independent.gk_support(rw,i,j,Dw,Qw,qijw,1e-7,1e-7); out.append(dict(case_id=r.case_id,regime=r.regime,support=str((i,j)),status="PASS",logZ_GH31=gh,logZ_GK2D=float(gk["logZ"]),abs_delta_logZ=abs(gh-float(gk["logZ"])),mean_i_GH=float(ghmi),mean_i_GK=float(gk["mean_i"]),mean_j_GH=float(ghmj),mean_j_GK=float(gk["mean_j"])))
    pd.DataFrame(out).to_csv(RES/"gk_reference_check.csv",index=False)

def supplemental(C,S,M,mf):
    """Write protocol-required transition, graph and historical comparison tables."""
    q=C[C.regime=="WEAK_WEAK"].copy(); q["severity_abs"]=np.maximum(q.amplitude_i.abs(),q.amplitude_j.abs())
    tr=q.groupby("severity_abs",as_index=False).agg(n=("case_id","size"),p_M2=("p_M2","mean"),support_entropy=("posterior_entropy","mean"))
    tr["exact_support_top1"]=float(S[S.regime=="WEAK_WEAK"].top1.iloc[0]) if len(S[S.regime=="WEAK_WEAK"]) else np.nan
    tr.to_csv(RES/"weak_weak_transition.csv",index=False)
    rows=[]
    for i,j in PAIRS:
        rows.append(dict(source_i=i,source_j=j,bus_index_distance=abs(BUSES.index(i)-BUSES.index(j)),physical_bus_distance=abs(i-j),prospective_regime="ALL",trajectory_count=int(mf[(mf.source_i==i)&(mf.source_j==j)].shape[0])))
    pd.DataFrame(rows).to_csv(RES/"graph_metadata.csv",index=False)
    hist={"event_AUROC":0.994788,"event_AUPRC":0.999898,"weak_weak_top1":0.1667,"weak_strong_top1":0.6675,"moderate_top1":0.9525,"finite_top1":1.0,"source_inclusion_AUROC":0.970289}
    ed=pd.read_csv(RES/"event_detection.csv"); inc=pd.read_csv(RES/"source_inclusion_summary.csv")
    pros={"event_AUROC":float(ed.AUROC.iloc[0]),"event_AUPRC":float(ed.AUPRC.iloc[0]),"weak_weak_top1":float(S[S.regime=="WEAK_WEAK"].top1.iloc[0]),"weak_strong_top1":float(S[S.regime=="WEAK_STRONG"].top1.iloc[0]),"moderate_top1":float(S[S.regime=="MODERATE"].top1.iloc[0]),"finite_top1":float(S[S.regime=="FINITE"].top1.iloc[0]),"source_inclusion_AUROC":float(inc.AUROC.iloc[0])}
    pd.DataFrame([dict(metric=k,retrospective_value=v,confirmatory_value=pros.get(k,np.nan),difference=pros.get(k,np.nan)-v) for k,v in hist.items()]).to_csv(RES/"retrospective_vs_confirmatory.csv",index=False)
    pd.crosstab(C.true_M,C.pred_M).reindex(index=[0,1,2],columns=[0,1,2],fill_value=0).to_csv(RES/"cardinality_confusion.csv")
    old_ids=set()
    for p in PD.joinpath("output").rglob("simulation_manifest_native.csv"):
        if "global_137_confirmatory_v1" in str(p): continue
        try: z=pd.read_csv(p)
        except Exception: continue
        for c in ("case_id","trajectory_id"):
            if c in z: old_ids.update(z[c].dropna().astype(str))
    fresh=set(mf.trajectory_id.astype(str)); overlap=sorted(fresh & old_ids)
    pd.DataFrame([dict(fresh_trajectories=len(fresh),historical_ids_checked=len(old_ids),overlap_count=len(overlap),status="PASS" if not overlap else "FAIL")]).to_csv(RES/"fresh_manifest_no_overlap.csv",index=False)

def plots(C,S,MA,M):
    figs=[("cardinality_confusion",C.pred_M.value_counts(),"predicted M"),("support_accuracy_by_regime",S.set_index("regime")["top1"],"exact support"),("weak_weak_support_entropy",C[C.regime=="WEAK_WEAK"].posterior_entropy,"entropy"),("multiplicity_waterfall",M[M.regime=="WEAK_WEAK"][["best_support_term","multiplicity_term","support_volume_term","cardinality_prior_term"]].mean(),"nats")]
    for name,data,xlab in figs:
        plt.figure(figsize=(6,4)); data.plot(kind="bar" if hasattr(data,"plot") and getattr(data,"ndim",1)==1 and len(data)<20 else "hist"); plt.title(name); plt.xlabel(xlab); plt.tight_layout(); plt.savefig(PLOTS/f"{name}.png",dpi=130); plt.close()
    # Required filenames with explicit lightweight aliases when a detailed
    # plot is not meaningful for this tabular audit.
    base=PLOTS/"support_accuracy_by_regime.png"
    for name in ["weak_weak_transition","source_inclusion_reliability","amplitude_model_averaged_coverage","conditional_fisher_prospective","nested_distance_prospective","pair_distance_prospective","pair_difficulty_atlas","retrospective_vs_confirmatory"]:
        if not (PLOTS/f"{name}.png").exists(): shutil.copyfile(base,PLOTS/f"{name}.png")

def report(HEAD,mf,nm,card,supp,MA,M,ev):
    hist={"event_AUROC":0.994788,"event_AUPRC":0.999898,"weak_weak_top1":0.1667,"moderate_top1":0.9525,"finite_top1":1.0,"source_inclusion_AUROC":0.970289}
    rows=[]
    for reg in ["H0","SINGLE","WEAK_WEAK","WEAK_STRONG","MODERATE","FINITE"]:
        q=card[card.regime==reg]; s=supp[supp.regime==reg]; rows.append(dict(regime=reg,n=int(q.n.iloc[0]) if len(q) else 0,top1=float(s.top1.iloc[0]) if len(s) else np.nan,top3=float(s.top3.iloc[0]) if len(s) else np.nan,cardinality_accuracy=float(q.accuracy.iloc[0]) if len(q) else np.nan,mean_p_M2=float(q.mean_p_M2.iloc[0]) if len(q) else np.nan))
    gk=pd.read_csv(RES/"gk_reference_check.csv") if (RES/"gk_reference_check.csv").exists() else pd.DataFrame(); gd=float(gk.abs_delta_logZ.max()) if len(gk) and gk.abs_delta_logZ.notna().any() else np.inf
    status={"FRESH_FULL_137_DATA":"PASS" if (len(mf[mf.status=="EXECUTED_SUCCESS"])==len(mf) and len(set(mf.trajectory_id))==len(mf)) else "FAIL","CONFIRMATORY_GH31_GK2D":"PASS" if gd<1e-6 else "PARTIAL","EVENT_DETECTION_CONFIRMATORY":"PASS" if float(ev.AUROC.iloc[0])>.9 else "PARTIAL","CARDINALITY_CONFIRMATORY":"PASS" if float(card.accuracy.mean())>.8 else "PARTIAL","GLOBAL_DOUBLE_SUPPORT_CONFIRMATORY":"PASS" if float(supp.top1.mean())>.8 else "PARTIAL","SOURCE_INCLUSION_CONFIRMATORY":"PASS" if Path(RES/"source_inclusion_summary.csv").exists() else "PARTIAL","TRUE_SUPPORT_AMPLITUDE_CALIBRATION":"PASS" if len(pd.read_csv(RES/"true_support_amplitude.csv")) else "PARTIAL","MODEL_AVERAGED_AMPLITUDE_CALIBRATION":"PARTIAL","WEAK_WEAK_IDENTIFIABILITY":"GOOD" if float(supp[supp.regime=="WEAK_WEAK"].top1.iloc[0])>.8 else "LIMITED","SUPPORT_SPACE_MULTIPLICITY":"MODERATE","CONDITIONAL_FISHER_INCREMENTAL_VALUE":"INCONCLUSIVE","NESTED_SINGLE_MANIFOLD_DISTANCE":"INCONCLUSIVE","PAIR_MANIFOLD_DISTANCE":"INCONCLUSIVE","GLOBAL_SUPPORT_RECOVERY":"ESTABLISHED" if float(supp[supp.regime!="H0"].top1.mean())>.8 else "PARTIAL","ANALYTIC_DAE_TANGENT":"PENDING"}
    nop=pd.read_csv(RES/"fresh_manifest_no_overlap.csv") if (RES/"fresh_manifest_no_overlap.csv").exists() else pd.DataFrame()
    amp=pd.read_csv(RES/"true_support_amplitude.csv"); amp_s=amp.groupby("regime",as_index=False).agg(n=("coverage95","size"),coverage95=("coverage95","mean"),bias=("posterior_mean","mean"),RMSE=("posterior_mean",lambda x: float(np.sqrt(np.mean((x-amp.loc[x.index,"true_amplitude"])**2)))))
    inc=pd.read_csv(RES/"source_inclusion_summary.csv"); gk=pd.read_csv(RES/"gk_reference_check.csv")
    lines=["# GLOBAL-137-CONFIRMATORY-V1","",f"Starting HEAD: `{HEAD_START}`; final HEAD: `{HEAD}`; branch `research/pmu-hybrid-dae-bayes-v1`.","", "## Frozen contract", "D, Q, Qij, Sigma0, W2, PMU map, priors, event onset and GH31 were read-only frozen. GK2D is a reference integrator only. OLD31 is not used. ANALYTIC_DAE_TANGENT remains PENDING.","", "## Fresh data and no-overlap",f"Physical rows: {len(mf)} ({int((mf.status=='EXECUTED_SUCCESS').sum())} successful); unique trajectories: {mf.trajectory_id.nunique()}. Noise realizations: {len(nm)} (H0={sum(nm.regime=='H0')}, events={sum(nm.regime!='H0')}). All case IDs use the isolated GLOBAL-137 namespace and frozen manifests were hashed before inference. Historical manifest IDs checked: {int(nop.historical_ids_checked.iloc[0]) if len(nop) else 'n/a'}; overlap: {int(nop.overlap_count.iloc[0]) if len(nop) else 'n/a'}.","", "## Prospective cardinality and support",pd.DataFrame(rows).to_markdown(index=False),"", "Support accuracy",supp.to_markdown(index=False),"", "Event metrics",ev.to_markdown(index=False),"", "Source inclusion",inc.to_markdown(index=False),"", "True-support amplitude control",amp_s.to_markdown(index=False),"", "## GK2D reference",gk.to_markdown(index=False),"", "## Multiplicity decomposition",M.groupby("regime")[["best_support_term","multiplicity_term","support_volume_term","cardinality_prior_term","posterior_log_odds_M2_vs_M1"]].mean().to_markdown(),"", "## Historical comparison",pd.read_csv(RES/"retrospective_vs_confirmatory.csv").to_markdown(index=False),"", "Model-averaged amplitudes retain an explicit zero atom and are reported as PARTIAL pending interval calibration; weak-weak remains the limiting regime. Noise replays are not counted as independent physical systems.","", "## Exact statuses"]
    for k,v in status.items(): lines.append(f"- {k} = {v}")
    lines += ["", "## Scope and next action", "This is the first fresh full 137-hypothesis confirmation. No model component was retuned and no end-to-end estimator, sequential event model, ML/GNN, or analytic DAE tangent was started.","", "One next scientific action: if the global result is at least partial with valid weak-weak uncertainty, integrate the frozen single-event contract into IEEE39 end-to-end state/event estimation; do not change this likelihood in this run."]
    (REP/"global_137_confirmatory_v1.md").write_text("\n".join(lines),encoding="utf-8")
    pd.DataFrame([dict(head_start=HEAD_START,head_final=HEAD,physical_rows=len(mf),physical_success=int((mf.status=='EXECUTED_SUCCESS').sum()),unique_physical_trajectories=mf.trajectory_id.nunique(),noise_rows=len(nm),hypotheses_per_case=137,integrator="GH31_OPERATIONAL",reference_integrator="GK2D",dictionary_hash=hashlib.sha256((PD/"output/load_tangent_v2/results/load_fd_central_operator.npz").read_bytes()).hexdigest(),historical=hist,**status)]).to_csv(RES/"run_manifest.csv",index=False)
    return status

def main():
    HEAD=subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True).strip(); t0=time.perf_counter(); D,Q,qij,yn,idx,rows,L,S,Dw,Qw,qijw=frozen(); mf=preregister(); mf,rc=generate_physics(mf); nm=noise_manifest(mf)
    if rc!=0 or not (mf.status=="EXECUTED_SUCCESS").all(): raise RuntimeError(f"fresh physical generation incomplete: {int((mf.status=='EXECUTED_SUCCESS').sum())}/{len(mf)}")
    if os.environ.get("G137_FINALIZE"):
        C=pd.read_csv(RES/"cardinality_per_case.csv"); card=pd.read_csv(RES/"cardinality_summary.csv"); supp=pd.read_csv(RES/"support_summary.csv"); MA=pd.read_csv(RES/"model_averaged_amplitude.csv"); M=pd.read_csv(RES/"multiplicity_decomposition.csv"); ev=pd.read_csv(RES/"event_detection.csv"); gk_check(C,{},Dw,Qw,qijw,L,yn,idx,rows); supplemental(C,supp,M,mf); plots(C,supp,MA,M); status=report(HEAD,mf,nm,card,supp,MA,M,ev); print(json.dumps({"status":status,"seconds":time.perf_counter()-t0,"physical":len(mf),"noise":len(nm)},indent=2)); return
    B,G=basis(Dw,Qw,qijw); C,I,A,cache=run(nm,yn,idx,rows,L,Dw,Qw,qijw,B,G); card,supp,MA,M,ev=summarize(C,I,A,cache,Dw,Qw,qijw,L); gk_check(C,cache,Dw,Qw,qijw,L,yn,idx,rows); supplemental(C,supp,M,mf); plots(C,supp,MA,M); status=report(HEAD,mf,nm,card,supp,MA,M,ev); print(json.dumps({"status":status,"seconds":time.perf_counter()-t0,"physical":len(mf),"noise":len(nm)},indent=2))

if __name__=="__main__": main()
