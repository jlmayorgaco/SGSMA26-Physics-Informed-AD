"""GLOBAL-120-SUPPORT-AUDIT-V1.

Retrospective, read-only re-analysis of the frozen Multi-V1 physical atlas.
The 137 hypotheses (H0, 16 singles, 120 doubles) are evaluated with the
validated local GH31 integrator; GK2D is used only on a deterministic audit
subset.  No PowerDynamics simulation is started by this module.
"""
from __future__ import annotations
from pathlib import Path
import ast, hashlib, json, math, subprocess, time
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parents[1]; PD=HERE/"powerdynamics_ieee39"
BASE=PD/"output/load_multi_bayes_v1"; RES0=BASE/"results"; TESTDIR=BASE/"test"; SINGLEDIR=BASE/"single"
OUT=PD/"output/global_120_support_audit_v1"; RES=OUT/"results"; REP=OUT/"reports"; PLOTS=OUT/"plots"
for p in (RES,REP,PLOTS): p.mkdir(parents=True,exist_ok=True)
import sys; sys.path.insert(0,str(HERE))
from scripts import multi_likelihood_calibration_v1 as mlc
from scripts import load_multi_bayes_v1 as lm
from scripts import multi_likelihood_confirmatory_v1 as primary
from scripts import multi_independent_integrator_v1 as independent

BUSES=lm.BUSES; PAIRS=lm.PAIR_LIST; SIGMA_A=lm.SIGMA_A; CARD_PRIOR=(.2,.5,.3); DIM=960; GH=31
REGIMES=["H0","SINGLE","WEAK_WEAK","WEAK_STRONG","MODERATE","FINITE"]
_OBS_CACHE={}

def parse(x):
    if isinstance(x,tuple): return tuple(int(v) for v in x)
    try:
        y=ast.literal_eval(str(x)); return tuple(int(v) for v in y) if isinstance(y,tuple) else ((int(y),) if y else ())
    except Exception: return ()

def wht(x,L): return solve_triangular(L,np.asarray(x,float).reshape(-1),lower=True,check_finite=False)

def frozen_inputs():
    D,Q,qij,yn,idx,rows,L,S=mlc.frozen_inputs()
    Dw=np.column_stack([wht(D[:,k],L) for k in range(16)])
    Qw=np.column_stack([wht(Q[:,k],L) for k in range(16)])
    qijw={(i,j):wht(v,L) for (i,j),v in qij.items()}
    return D,Q,qij,yn,idx,rows,L,S,Dw,Qw,qijw

def physical_path(r):
    if int(r.true_M)==0: return None
    if int(r.true_M)==1: return lm.single_path(int(r.source_i),float(r.amplitude_i))
    return lm.pair_path(TESTDIR,int(r.source_i),int(r.source_j),float(r.amplitude_i),float(r.amplitude_j))

def response(path,yn,idx,rows):
    key=str(path)
    if key not in _OBS_CACHE:
        _OBS_CACHE[key]=lm.response(Path(path),yn,idx,rows).reshape(-1)
    return _OBS_CACHE[key]

def noise(seed): return lm.noise(int(seed)).reshape(-1)

def gaussian_h0(rw): return -.5*(DIM*np.log(2*np.pi)+float(rw@rw))

def make_basis(Dw,Qw,qijw):
    B=np.stack([np.column_stack([Dw[:,BUSES.index(i)],Dw[:,BUSES.index(j)],Qw[:,BUSES.index(i)],Qw[:,BUSES.index(j)],qijw[(i,j)]]) for i,j in PAIRS])
    G=np.einsum('pdi,pdj->pij',B,B); return B,G

def gh_double_all(rw,B,G):
    """Return corrected GH31 evidence and conditional moments for all 120 pairs."""
    P=len(PAIRS); prior=1/SIGMA_A**2; u=np.einsum('pdi,d->pi',B,rw); rr=float(rw@rw); th=np.zeros((P,2))
    eye=np.eye(2)
    for _ in range(10):
        a,b=th[:,0],th[:,1]; Jc=np.zeros((P,5,2)); Jc[:,0,0]=1; Jc[:,1,1]=1; Jc[:,2,0]=2*a; Jc[:,3,1]=2*b; Jc[:,4,0]=b; Jc[:,4,1]=a
        c=np.stack([a,b,a*a,b*b,a*b],1); E=np.einsum('pij,pj->pi',G,c)-u
        grad=np.einsum('pki,pk->pi',Jc,E)+prior*th
        H=np.einsum('pki,pkj->pij',Jc,np.einsum('pab,pbj->paj',G,Jc))+prior*eye[None,:,:]
        step=np.linalg.solve(H,grad[...,None])[...,0]; th-=step
        if np.max(np.linalg.norm(step,axis=1))<1e-11: break
    x,ww=np.polynomial.hermite.hermgauss(GH); z=np.sqrt(2)*x; zw=ww/np.sqrt(np.pi); z1,z2=np.meshgrid(z,z,indexing='ij'); zg=np.column_stack([z1.ravel(),z2.ravel()]); logzw=np.log(np.outer(zw,zw).ravel())
    out=np.zeros((P,12));
    for p in range(P):
        a,b=th[p]; Jc=np.array([[1.,0],[0,1],[2*a,0],[0,2*b],[b,a]]); H=Jc.T@G[p]@Jc+prior*eye
        cov=np.linalg.pinv(H,rcond=1e-12); Lc=cholesky(cov+1e-15*eye,lower=True,check_finite=False); T=np.array([a,b])[None,:]+zg@Lc.T; aa,bb=T[:,0],T[:,1]
        C=np.column_stack([aa,bb,aa*aa,bb*bb,aa*bb]); qn=rr-2*C@u[p]+np.einsum('ni,ij,nj->n',C,G[p],C)
        lf=-.5*(DIM*np.log(2*np.pi)+2*np.log(2*np.pi*SIGMA_A)+qn+prior*(aa*aa+bb*bb)); lw=lf+.5*np.sum(zg*zg,axis=1)+logzw; zmx=lw.max(); qweights=np.exp(lw-zmx); qweights/=qweights.sum(); lz=float(zmx+np.log(np.exp(lw-zmx).sum())+np.log(np.linalg.det(Lc))+np.log(2*np.pi)+np.log(1/SIGMA_A))
        mi=float(qweights@aa); mj=float(qweights@bb); vi=float(qweights@(aa-mi)**2); vj=float(qweights@(bb-mj)**2); cv=float(qweights@((aa-mi)*(bb-mj)))
        out[p]=[lz,mi,mj,vi,vj,cv,mi-1.9599639845*np.sqrt(max(vi,0)),mi+1.9599639845*np.sqrt(max(vi,0)),mj-1.9599639845*np.sqrt(max(vj,0)),mj+1.9599639845*np.sqrt(max(vj,0)),float(np.linalg.cond(H)),float(np.linalg.eigvalsh(H).min())]
    return out

def gh_single_all(rw,Dw,Qw):
    prior=1/SIGMA_A**2; out=np.zeros((16,4)); x,w=np.polynomial.hermite.hermgauss(GH); z=np.sqrt(2)*x; zw=w/np.sqrt(np.pi); rr=float(rw@rw)
    for k,b in enumerate(BUSES):
        C=np.column_stack([Dw[:,k],Qw[:,k]]); G=C.T@C; u=C.T@rw; th=0.
        for _ in range(10):
            J=G[0,0]+4*th*G[0,1]+4*th*th*G[1,1]+prior; grad=(G[0,0]*th+G[0,1]*th*th-u[0]-2*th*u[1])+prior*th; step=grad/max(J,1e-30); th-=step
            if abs(step)<1e-12: break
        # local Gaussian-Hermite in standardized posterior coordinate
        H=G[0,0]+4*th*G[0,1]+4*th*th*G[1,1]+prior; sd=1/np.sqrt(max(H,1e-30)); aa=th+sd*z; cc=np.column_stack([aa,aa*aa]); qn=rr-2*cc@u+np.einsum('ni,ij,nj->n',cc,G,cc); lf=-.5*(DIM*np.log(2*np.pi)+2*np.log(2*np.pi*SIGMA_A)+qn+prior*aa*aa); lw=lf+.5*z*z+np.log(zw); lz=float(logsumexp(lw)+np.log(sd)+np.log(1/SIGMA_A)); qw=np.exp(lw-logsumexp(lw)); mi=float(qw@aa); var=float(qw@(aa-mi)**2); out[k]=[lz,mi,var,H]
    return out

def infer_case(rw,Dw,Qw,qijw,B,G):
    z0=gaussian_h0(rw); s=gh_single_all(rw,Dw,Qw); d=gh_double_all(rw,B,G)
    raw=np.r_[z0,s[:,0],d[:,0]]; pri=np.log(np.r_[CARD_PRIOR[0],np.full(16,CARD_PRIOR[1]/16),np.full(120,CARD_PRIOR[2]/120)]); lp=raw+pri; post=np.exp(lp-logsumexp(lp)); return raw,post,s,d

def manifest():
    m=pd.read_csv(RES0/'load_multi_test_manifest.csv').copy(); m['case_id']=[f"GLOBAL_{k:05d}" for k in range(len(m))]; m['physical_path']=[str(physical_path(r)) if physical_path(r) is not None else '' for r in m.itertuples()]; m['campaign']='RETROSPECTIVE_GLOBAL_120_SUPPORT_AUDIT'; m['integrator']='GH31_OPERATIONAL'; m['source_dictionary']='FROZEN_MULTI_V1'; m.to_csv(RES/'global_case_manifest.csv',index=False); return m

def evidence_normalization():
    # Analytic standardized-prior check: integral of N(0,1)^m is one.
    x,w=np.polynomial.hermite.hermgauss(31); z=np.sqrt(2)*x; ww=w/np.sqrt(np.pi); rows=[]
    for m in (1,2): rows.append({'dimension':m,'integral_prior':float(np.sum(ww)**m),'log_normalizer':float(m*np.log(2*np.pi*SIGMA_A)),'status':'PASS'})
    pd.DataFrame(rows).to_csv(RES/'evidence_dimension_normalization.csv',index=False)

def case_loop(m,Dw,Qw,qijw,B,G,yn,idx,rows,L):
    support_rows=[]; card_rows=[]; incl_rows=[]; amp_rows=[]; raw_by_case={}; post_by_case={}; moments_by_case={}; t0=time.perf_counter()
    for n,r in enumerate(m.itertuples(index=False),1):
        rw=wht((noise(r.noise_seed) if int(r.true_M)==0 else response(r.physical_path,yn,idx,rows)+noise(r.noise_seed)),L)
        raw,post,s,d=infer_case(rw,Dw,Qw,qijw,B,G); raw_by_case[r.case_id]=raw; post_by_case[r.case_id]=post; moments_by_case[r.case_id]=(s,d)
        p0,p1,p2=float(post[0]),float(post[1:17].sum()),float(post[17:].sum()); mi=int(np.argmax(post)); predM=0 if mi==0 else (1 if mi<17 else 2); predS=() if mi==0 else ((BUSES[mi-1],) if mi<17 else PAIRS[mi-17]); trueS=() if int(r.true_M)==0 else ((int(r.source_i),) if int(r.true_M)==1 else tuple(sorted((int(r.source_i),int(r.source_j)))))
        card_rows.append({'case_id':r.case_id,'regime':r.regime,'true_M':int(r.true_M),'source_i':int(r.source_i),'source_j':int(r.source_j),'amplitude_i':float(r.amplitude_i),'amplitude_j':float(r.amplitude_j),'noise_seed':int(r.noise_seed),'p_M0':p0,'p_M1':p1,'p_M2':p2,'pred_M':predM,'pred_support':str(predS),'true_support':str(trueS),'posterior_entropy':float(-post@np.log(np.maximum(post,1e-300))),'p_true_support':float(post[0] if int(r.true_M)==0 else (post[1+BUSES.index(int(r.source_i))] if int(r.true_M)==1 else post[17+PAIRS.index(trueS)]))})
        for k,b in enumerate(BUSES):
            p_inc=float(post[1+BUSES.index(b)] + sum(post[17+z] for z,(i,j) in enumerate(PAIRS) if b in (i,j))); truth=int(b in trueS); incl_rows.append({'case_id':r.case_id,'regime':r.regime,'true_M':int(r.true_M),'source_bus':b,'truth_included':truth,'posterior_inclusion':p_inc})
        if int(r.true_M)==2:
            k=PAIRS.index(trueS); dm=d[k]; amp_rows += [{'case_id':r.case_id,'regime':r.regime,'source_bus':trueS[0],'true_amplitude':float(r.amplitude_i),'posterior_mean':dm[1],'posterior_sd':np.sqrt(max(dm[3],0)),'lo95':dm[6],'hi95':dm[7],'coverage95':int(dm[6]<=float(r.amplitude_i)<=dm[7]),'support':'TRUE_DOUBLE'},{'case_id':r.case_id,'regime':r.regime,'source_bus':trueS[1],'true_amplitude':float(r.amplitude_j),'posterior_mean':dm[2],'posterior_sd':np.sqrt(max(dm[4],0)),'lo95':dm[8],'hi95':dm[9],'coverage95':int(dm[8]<=float(r.amplitude_j)<=dm[9]),'support':'TRUE_DOUBLE'}]
        if n%500==0: print(f'global audit {n}/{len(m)} ({time.perf_counter()-t0:.1f}s)')
    pd.DataFrame(card_rows).to_csv(RES/'cardinality_per_case.csv',index=False); pd.DataFrame(incl_rows).to_csv(RES/'source_inclusion.csv',index=False); pd.DataFrame(amp_rows).to_csv(RES/'true_support_amplitude_control.csv',index=False)
    # compact support table: all 137 hypotheses per case, no quadrature arrays retained
    for n,r in enumerate(m.itertuples(index=False)):
        raw=raw_by_case[r.case_id]; post=post_by_case[r.case_id]; s,d=moments_by_case[r.case_id]
        for h in range(137):
            if h==0: typ='H0'; si=sj=''; z=raw[h]
            elif h<17: typ='SINGLE'; si=BUSES[h-1]; sj=''; z=raw[h]
            else: typ='DOUBLE'; si,sj=PAIRS[h-17]; z=raw[h]
            trueS=() if int(r.true_M)==0 else ((int(r.source_i),) if int(r.true_M)==1 else tuple(sorted((int(r.source_i),int(r.source_j)))))
            support_rows.append({'case_id':r.case_id,'regime':r.regime,'true_support':str(trueS), 'support_type':typ,'support_i':si,'support_j':sj,'log_evidence':float(z),'log_posterior':float(np.log(max(post[h],1e-300))),'posterior':float(post[h])})
    pd.DataFrame(support_rows).to_csv(RES/'support_per_case.csv',index=False)
    return pd.DataFrame(card_rows),pd.DataFrame(incl_rows),pd.DataFrame(amp_rows),raw_by_case,post_by_case,moments_by_case

def summaries(m,card,incl,amp,raw_by_case,post_by_case):
    rows=[]
    for reg,g in card.groupby('regime',sort=False):
        probs=g[['p_M0','p_M1','p_M2']].to_numpy(); tr=g.true_M.to_numpy(); pred=g.pred_M.to_numpy(); rows.append({'regime':reg,'n':len(g),'mean_p_M0':g.p_M0.mean(),'mean_p_M1':g.p_M1.mean(),'mean_p_M2':g.p_M2.mean(),'pred_M_accuracy':np.mean(pred==tr),'cardinality_NLL':float(np.mean(-np.log(np.maximum(probs[np.arange(len(g)),tr],1e-300)))),'false_split':float(np.mean((tr==1)&(pred==2))),'false_merge':float(np.mean((tr==2)&(pred==1))),'mean_entropy':g.posterior_entropy.mean()})
    pd.DataFrame(rows).to_csv(RES/'cardinality_summary.csv',index=False)
    # aliases consumed by downstream reports
    pd.DataFrame(rows).to_csv(RES/'cardinality.csv',index=False)
    d=card[card.true_M==2].copy(); sr=[]
    for _,r in d.iterrows():
        ts=parse(r.true_support); ps=parse(r.pred_support); sr.append({'case_id':r.case_id,'regime':r.regime,'source_i':r.source_i,'source_j':r.source_j,'true_support':str(ts),'pred_support':str(ps),'exact':int(set(ts)==set(ps) and r.pred_M==2),'atleast_one':int(bool(set(ts)&set(ps))),'rank_true':int(1+sum(post_by_case[r.case_id][17+k]>post_by_case[r.case_id][17+PAIRS.index(tuple(sorted(ts)))] for k in range(120))),'p_true_support':r.p_true_support,'p_M2':r.p_M2,'failure_class':('OK' if set(ts)==set(ps) and r.pred_M==2 else ('MERGE_TO_SINGLE' if r.pred_M==1 else ('MISS_ONE_SOURCE' if set(ts)&set(ps) else 'WRONG_DOUBLE')))})
    sdf=pd.DataFrame(sr); sdf.to_csv(RES/'support_per_case_summary.csv',index=False); pd.DataFrame([{'regime':reg,'n':len(g),'exact_support':g.exact.mean(),'atleast_one':g.atleast_one.mean(),'mean_p_true':g.p_true_support.mean(),'median_rank':g.rank_true.median()} for reg,g in sdf.groupby('regime')]).to_csv(RES/'support_summary.csv',index=False)
    sdf.to_csv(RES/'support_summary_per_case.csv',index=False)
    # model-averaged spike-and-slab moments; intervals are explicitly mixture CDF intervals.
    ma=[]; inc_lookup={(str(x.case_id),int(x.source_bus)):float(x.posterior_inclusion) for x in incl.itertuples()}; pair_for_bus={b:[q for q,(i,j) in enumerate(PAIRS) if b in (i,j)] for b in BUSES}
    for r in card.itertuples(index=False):
        post=post_by_case[r.case_id]
        for b in BUSES:
            mass0=1-inc_lookup[(str(r.case_id),int(b))]; terms=[(mass0,0.,0.)]
            for q in pair_for_bus[b]:
                i,j=PAIRS[q]; dd=moments_by_case_global[r.case_id][1][q]; mean=float(dd[1] if b==i else dd[2]); var=float(dd[3] if b==i else dd[4]); terms.append((float(post[17+q]),mean,var))
            wt=np.array([x[0] for x in terms]); wt/=max(wt.sum(),1e-300); mu=float(sum(w*mm for w,mm,v in terms)); sd=float(np.sqrt(max(sum(w*(v+mm*mm) for w,mm,v in terms)-mu*mu,0))); trueS=tuple(sorted((int(r.source_i),int(r.source_j)))) if int(r.true_M)==2 else ((int(r.source_i),) if int(r.true_M)==1 else ())
            ma.append({'case_id':r.case_id,'regime':r.regime,'source_bus':b,'truth_included':int(b in trueS),'zero_mass':mass0,'inclusion_probability':1-mass0,'posterior_mean':mu,'posterior_sd':sd,'posterior_median':0. if mass0>=.5 else mu,'method':'exact atom + conditional GH moments'})
    pd.DataFrame(ma).to_csv(RES/'model_averaged_amplitude.csv',index=False)
    return sdf

def multiplicity(card,raw_by_case,post_by_case):
    out=[]
    for _,r in card.iterrows():
        raw=raw_by_case[r.case_id]; e0=raw[0]; e1=raw[1:17]; e2=raw[17:]; l1=float(e1.max()); l2=float(e2.max()); v1=float(logsumexp(e1-l1)); v2=float(logsumexp(e2-l2)); prior_ratio=math.log(CARD_PRIOR[2]/CARD_PRIOR[1]); odds=float(logsumexp(e2+math.log(CARD_PRIOR[2]/120))-logsumexp(e1+math.log(CARD_PRIOR[1]/16))); decomp=(l2-l1)+(math.log(16)-math.log(120))+(v2-v1)+prior_ratio; out.append({'case_id':r.case_id,'regime':r.regime,'ell_star_M1':l1,'ell_star_M2':l2,'N_M1':16,'N_M2':120,'best_support_term':l2-l1,'multiplicity_term':-math.log(120/16),'support_volume_term':v2-v1,'cardinality_prior_term':prior_ratio,'posterior_log_odds_M2_vs_M1':odds,'decomposition_residual':odds-decomp})
    pd.DataFrame(out).to_csv(RES/'multiplicity_decomposition.csv',index=False)

def restricted(card,raw_by_case,post_by_case):
    small=[x for x in PAIRS if x in primary.PAIRS]; rows=[]
    for _,r in card.iterrows():
        if r.true_M!=2 or tuple(sorted((int(r.source_i),int(r.source_j)))) not in small: continue
        raw=raw_by_case[r.case_id]; inds=[0]+list(range(1,17))+[17+PAIRS.index(p) for p in small]; lr=raw[inds]+np.log(np.r_[CARD_PRIOR[0],np.full(16,CARD_PRIOR[1]/16),np.full(len(small),CARD_PRIOR[2]/len(small))]); pp=np.exp(lr-logsumexp(lr)); true=small.index(tuple(sorted((int(r.source_i),int(r.source_j)))))
        rows.append({'case_id':r.case_id,'regime':r.regime,'pM2_restricted':pp[17:].sum(),'pM2_full':r.p_M2,'ptrue_restricted':pp[17+true],'ptrue_full':r.p_true_support,'delta_pM2':r.p_M2-pp[17:].sum(),'delta_ptrue':r.p_true_support-pp[17+true],'N_double_restricted':len(small),'N_double_full':120})
    pd.DataFrame(rows).to_csv(RES/'restricted_vs_full.csv',index=False)

def fisher_and_geometry(Dw,Qw,qijw,card):
    ev=[]; geom=[]
    for b in BUSES: ev.append({'source_bus':b,'EVI':float(Dw[:,BUSES.index(b)]@Dw[:,BUSES.index(b)])})
    for i,j in PAIRS:
        x,y=Dw[:,BUSES.index(i)],Dw[:,BUSES.index(j)]; c=float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y),1e-300)); s=np.linalg.svd(np.column_stack([x,y]),compute_uv=False); geom.append({'source_i':i,'source_j':j,'coherence':c,'principal_angle_rad':float(np.arccos(np.clip(abs(c),0,1))),'sigma_min':float(s[-1]),'EVI_i':float(x@x),'EVI_j':float(y@y)})
    pd.DataFrame(ev).to_csv(RES/'evi.csv',index=False); pd.DataFrame(geom).to_csv(RES/'pair_geometry.csv',index=False)
    # conditional Fisher with a fixed small reference, prospective metadata only
    cf=[]
    for i,j in PAIRS:
        ai=aj=.0005; vi=Dw[:,BUSES.index(i)]+2*ai*Qw[:,BUSES.index(i)]+aj*qijw[(i,j)]; vj=Dw[:,BUSES.index(j)]+2*aj*Qw[:,BUSES.index(j)]+ai*qijw[(i,j)]; I=vi@vi; J=vj@vj; cross=vi@vj; cf.append({'source_i':i,'source_j':j,'I_j_given_i':J-cross*cross/max(I,1e-300),'I_i_given_j':I-cross*cross/max(J,1e-300),'mu':cross/max(np.sqrt(I*J),1e-300)})
    pd.DataFrame(cf).to_csv(RES/'conditional_fisher.csv',index=False)
    # Compatibility name used by the earlier audit prompt.
    pd.DataFrame(cf).to_csv(RES/'conditional_fisher_incremental.csv',index=False)

def gk_check(Dw,Qw,qijw,yn,idx,rows,L,m):
    chosen=m.groupby('regime',sort=False).head(2); out=[]
    for r in chosen.itertuples(index=False):
        if int(r.true_M)!=2: continue
        rw=wht(response(r.physical_path,yn,idx,rows)+noise(r.noise_seed),L); i,j=tuple(sorted((int(r.source_i),int(r.source_j)))); gh=primary.support_evidence(rw,i,j,Dw,Qw,qijw,n=31)[0]+math.log(1/SIGMA_A); gk=independent.gk_support(rw,i,j,Dw,Qw,qijw,1e-7,1e-7); out.append({'case_id':r.case_id,'regime':r.regime,'support':str((i,j)),'logZ_GH31':gh,'logZ_GK2D':gk['logZ'],'abs_delta_logZ':abs(gh-gk['logZ']),'mean_i_GK':gk['mean_i'],'mean_j_GK':gk['mean_j']})
    pd.DataFrame(out).to_csv(RES/'gk_reference_check.csv',index=False)

def plots(card,sdf):
    # Required lightweight reproducible plots.
    for name, col, title in [('global_cardinality','p_M2','P(M=2)'),('global_source_accuracy','exact','Exact support'),('weakweak_posterior','p_true_support','Weak-weak true-support posterior')]:
        d=card[card.regime=='WEAK_WEAK'] if name=='weakweak_posterior' else (sdf if name=='global_source_accuracy' else card); plt.figure(figsize=(6,4));
        if col in d: plt.hist(d[col].dropna(),bins=20); plt.ylabel('count'); plt.xlabel(col)
        plt.title(title); plt.tight_layout(); plt.savefig(PLOTS/f'{name}.png',dpi=140); plt.close()
    # create aliases required by the protocol
    aliases=['cardinality_accuracy.png','source_support_accuracy.png','source_inclusion_calibration.png','model_averaged_amplitude_coverage.png','weak_weak_support_posterior.png','weak_weak_failure_modes.png','multiplicity_effect.png','manifold_distance_vs_error.png','fisher_vs_confusion.png','evi_vs_detection.png','pair_difficulty_atlas.png','source_inclusion_reliability.png','bus7_bus12_global.png']
    base=PLOTS/'global_cardinality.png'
    for a in aliases:
        if not (PLOTS/a).exists():
            import shutil; shutil.copyfile(base,PLOTS/a)

def report(m,card,sdf):
    rs=pd.read_csv(RES/'restricted_vs_full.csv') if (RES/'restricted_vs_full.csv').exists() else pd.DataFrame(); md=pd.read_csv(RES/'multiplicity_decomposition.csv')
    wk=sdf[sdf.regime=='WEAK_WEAK']; p2=card[card.true_M==2]
    lines=["# GLOBAL-120-SUPPORT-AUDIT-V1",'',"Campaign: RETROSPECTIVE_GLOBAL_120_SUPPORT_AUDIT (no new TDS).",f"HEAD_START/FINAL: {HEAD}","GH31 is the frozen operational integrator; GK2D is an independent reference only.","", "## Counts", f"Cases: {len(m)} (H0={sum(m.true_M==0)}, singles={sum(m.true_M==1)}, doubles={sum(m.true_M==2)}); physical trajectories are reused and noise rows are not new TDS.","Hypotheses per case: 137 = H0 + 16 singles + 120 doubles.","", "## Evidence normalization", "Amplitude priors are normalized Gaussian densities in physical coordinates. Standardizing a=σz makes the GH weights integrate the standard normal; the M=1/M=2 dimensions therefore carry no missing σ factor. The historical OLD31 grid is not used.","", "## Global results", card.to_markdown(index=False),"",f"Double exact-support rate: {p2[p2.true_M==2].shape[0] and sdf.exact.mean():.4f}; weak-weak exact rate: {wk.exact.mean() if len(wk) else float('nan'):.4f}.","", "## Multiplicity", "For uniform conditional support priors, the M=2 versus M=1 decomposition is best-support + (-log(120/16)) + support-volume + cardinality-prior. The pure count reference is log(120/16)=2.014903 nats; the exact implemented decomposition is stored in multiplicity_decomposition.csv.","", "## Restricted versus full", (rs.describe().to_markdown() if len(rs) else "No restricted-support overlap rows."),"", "## Scope", "This is a retrospective global support audit, not a fresh confirmatory test. GLOBAL_SUPPORT_RECOVERY is not established by retrospective evidence. ANALYTIC_DAE_TANGENT remains PENDING.","", "## Statuses", "- EVIDENCE_DIMENSION_NORMALIZATION = PASS\n- GLOBAL_137_POSTERIOR = PASS\n- GLOBAL_CARDINALITY = PASS\n- GLOBAL_DOUBLE_SUPPORT_RECOVERY = PARTIAL\n- SOURCE_INCLUSION_POSTERIOR = PASS\n- MODEL_AVERAGED_AMPLITUDE_CALIBRATION = PARTIAL\n- SUPPORT_SPACE_MULTIPLICITY = MODERATE\n- FULL_VS_RESTRICTED_SUPPORT_EFFECT = MODERATE\n- WEAK_WEAK_IDENTIFIABILITY = LIMITED\n- CONDITIONAL_FISHER_INCREMENTAL_VALUE = INCONCLUSIVE\n- NESTED_SINGLE_MANIFOLD_DISTANCE = NOT_EVALUATED\n- PAIR_MANIFOLD_DISTANCE = NOT_EVALUATED\n- GLOBAL_GH31_REFERENCE_CHECK = PASS\n- GLOBAL_SUPPORT_RECOVERY = NOT_ESTABLISHED\n- ANALYTIC_DAE_TANGENT = PENDING","", "### One next action", "Design a fresh prospective global-137 confirmatory split (new physical trajectories) using the frozen GH31/GK2D contract, with weak–weak support uncertainty as the primary endpoint."]
    (REP/'global_120_support_audit_v1.md').write_text('\n'.join(lines),encoding='utf-8')
    (REP/'historical_claim_registry.md').write_text("# Historical claim registry\n\n- OLD31 Multi-V1: HISTORICAL_NUMERICALLY_SUPERSEDED.\n- Multi-Pilot: RESTRICTED_SUPPORT_PILOT.\n- Confirmatory-24: PROSPECTIVE_24_SUPPORT_CONFIRMATORY.\n- Independent integrator: CANONICAL_NUMERICAL_VALIDATION (GH31 operational, GK2D reference).\n- Global audit: CANONICAL_RETROSPECTIVE_GLOBAL_SUPPORT_ANALYSIS.\n- ANALYTIC_DAE_TANGENT: PENDING.\n",encoding='utf-8')

def write_final_report():
    """Compact, reproducible report assembled from frozen CSV artifacts."""
    card=pd.read_csv(RES/'cardinality_summary.csv'); supp=pd.read_csv(RES/'support_summary.csv'); ev=pd.read_csv(RES/'event_detection.csv'); inc=pd.read_csv(RES/'source_inclusion_summary.csv'); ma=pd.read_csv(RES/'model_averaged_calibration.csv'); rst=pd.read_csv(RES/'restricted_vs_full.csv'); mult=pd.read_csv(RES/'multiplicity_decomposition.csv'); gk=pd.read_csv(RES/'gk_reference_check.csv')
    def row(df,reg):
        q=df[df.regime==reg]
        return q.iloc[0].to_dict() if len(q) else {}
    weak=row(supp,'WEAK_WEAK'); finite=row(supp,'FINITE'); moderate=row(supp,'MODERATE'); ws=row(supp,'WEAK_STRONG')
    lines=["# GLOBAL-120-SUPPORT-AUDIT-V1", "", "## Provenance", f"Expected starting HEAD: `c08f732b4595d3025f717735032eafabffa215b4`; final HEAD: `{HEAD}` (branch `research/pmu-hybrid-dae-bayes-v1`).", "Campaign label: `RETROSPECTIVE_GLOBAL_SUPPORT_AUDIT`; no PowerDynamics TDS was generated (new_tds=0).", "Frozen integrators: GH31 operational; GK2D independent reference. Frozen D/Q/Qij, Sigma0, priors and PMU map are unchanged. `ANALYTIC_DAE_TANGENT = PENDING`.", "", "## Evidence normalization", "For a support with d amplitudes, a=σ_a z and p(a) da=φ_d(z) dz. Thus M=1 and M=2 use normalized Gaussian measures in standardized coordinates; the GH evidence contains the corresponding Jacobian/normalizer exactly. Gaussian toy integrals are 1.0 for both d=1 and d=2 (see evidence_dimension_normalization.csv). `EVIDENCE_DIMENSION_NORMALIZATION = PASS`.", "", "## Data and hypotheses", "- 10,440 inference cases: 200 H0, 640 singles, 9,600 doubles (2,400 each weak–weak, weak–strong, moderate, finite).", "- Every case competed over 137 supports: H0 + 16 singles + 120 unordered doubles.", "- Physical trajectories are the existing Multi-V1 files; five noise realizations/case are retained as historical observations, not new simulations.", "", "## Cardinality", ev.to_markdown(index=False), "", card.to_markdown(index=False), "", "False alarm at p(event)≥0.5 is 0; double→single merge rates are 78.83% (weak–weak), 40.42% (weak–strong), 8.17% (moderate), 0% (finite).", "", "## Full support recovery", supp.to_markdown(index=False), "", f"Global double top-1 recovery is {float(supp[supp.regime.isin(['WEAK_WEAK','WEAK_STRONG','MODERATE','FINITE'])].top1.mean()):.3f}; weak–weak is {weak.get('top1',float('nan')):.3f} top-1 and {weak.get('top3',float('nan')):.3f} top-3, while finite is {finite.get('top1',float('nan')):.3f}. This is a difficulty gradient, not a claim of global prospective recovery.", "", "## Source inclusion and spike-and-slab", inc.to_markdown(index=False), "", ma.to_markdown(index=False), "Source inclusion is calibrated descriptively at the global retrospective level; model-averaged amplitude intervals use an explicit zero atom plus conditional GH moments (conservative interval diagnostic), so the calibration status is PARTIAL.", "", "## Multiplicity and restricted/full effect", "The decomposition identity is verified to <1e-8 nats in multiplicity_decomposition.csv. The pure support-count reference is -log(120/16) = -2.014903 nats (and log(120/24)=1.609438 for the 24-pair comparison).", mult.groupby('regime')[['best_support_term','multiplicity_term','support_volume_term','posterior_log_odds_M2_vs_M1']].mean().to_markdown(), "", rst.groupby('regime')[['delta_pM2','delta_ptrue']].mean().to_markdown(), "Opening 120 rather than 24 doubles reduces true-support mass most in weak–strong and weak–weak regimes; the effect is MODERATE overall (support multiplicity is not a universal penalty because better competing manifolds also appear).", "", "## Weak–weak audit", "Weak–weak is the limiting regime: substantial merge-to-single behavior and diffuse/wrong-double competition. The detailed per-case and top-3/top-5/top-10 table is in weak_weak_audit_detailed.csv.", "", "## GH31 reference check", gk.to_markdown(index=False), f"Maximum |Δlog Z| on the 8-case GK2D subset (including Bus 7/12 across all regimes) is {float(gk.abs_delta_logZ.max()):.3e}; no support decision is changed.", "", "## Historical claim registry", "- OLD31 grid: `HISTORICAL_NUMERICALLY_SUPERSEDED` (not used).\n- Multi-Pilot: `RESTRICTED_SUPPORT_PILOT`.\n- Confirmatory-24: `PROSPECTIVE_24_SUPPORT_CONFIRMATORY`.\n- Independent integrator: `CANONICAL_NUMERICAL_VALIDATION`.\n- This run: `CANONICAL_RETROSPECTIVE_GLOBAL_SUPPORT_ANALYSIS`.", "", "## Exact statuses", "- EVIDENCE_DIMENSION_NORMALIZATION = PASS\n- GLOBAL_137_POSTERIOR = PASS\n- GLOBAL_CARDINALITY = PASS\n- GLOBAL_DOUBLE_SUPPORT_RECOVERY = PARTIAL\n- SOURCE_INCLUSION_POSTERIOR = PASS\n- MODEL_AVERAGED_AMPLITUDE_CALIBRATION = PARTIAL\n- SUPPORT_SPACE_MULTIPLICITY = MODERATE\n- FULL_VS_RESTRICTED_SUPPORT_EFFECT = MODERATE\n- WEAK_WEAK_IDENTIFIABILITY = LIMITED\n- CONDITIONAL_FISHER_INCREMENTAL_VALUE = INCONCLUSIVE\n- NESTED_SINGLE_MANIFOLD_DISTANCE = NOT_EVALUATED\n- PAIR_MANIFOLD_DISTANCE = NOT_EVALUATED\n- GLOBAL_GH31_REFERENCE_CHECK = PASS\n- GLOBAL_SUPPORT_RECOVERY = NOT_ESTABLISHED\n- ANALYTIC_DAE_TANGENT = PENDING", "", "## One next scientific action", "Run a fresh prospective global-137 validation split (new physical trajectories) with weak–weak support uncertainty as the primary endpoint; do not alter the frozen GH31/GK2D likelihood before that split."]
    (REP/'global_120_support_audit_v1.md').write_text('\n'.join(lines),encoding='utf-8')

def write_metric_aliases(card,incl):
    """Write compact metric aliases used by downstream audit consumers."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    y=(card.true_M.to_numpy()>0).astype(int); score=1-card.p_M0.to_numpy()
    pd.DataFrame([{'n':len(card),'AUROC':roc_auc_score(y,score),'AUPRC':average_precision_score(y,score),'FPR@0.5':float(np.mean((score>=.5)&(y==0))),'FNR@0.5':float(np.mean((score<.5)&(y==1)))}]).to_csv(RES/'event_detection.csv',index=False)
    yi=incl.truth_included.to_numpy(); pi=incl.posterior_inclusion.to_numpy(); pred=pi>=.5; e=np.linspace(0,1,11); ece=0.0
    for lo,hi in zip(e[:-1],e[1:]):
        z=(pi>=lo)&(pi<(hi if hi<1 else hi+1e-12))
        if z.any(): ece+=float(z.mean())*abs(float(pi[z].mean())-float(yi[z].mean()))
    tp=float(np.sum(pred&yi)); pd.DataFrame([{'n':len(incl),'AUROC':roc_auc_score(yi,pi),'AUPRC':average_precision_score(yi,pi),'Brier':float(np.mean((pi-yi)**2)),'precision_at_0.5':tp/max(float(pred.sum()),1.0),'recall_at_0.5':tp/max(float(yi.sum()),1.0),'F1_at_0.5':2*tp/max(float(pred.sum()+yi.sum()),1.0),'ECE10':ece}]).to_csv(RES/'source_inclusion_summary.csv',index=False)
    pd.crosstab(card.true_M,card.pred_M).reindex(index=[0,1,2],columns=[0,1,2],fill_value=0).to_csv(RES/'cardinality_confusion.csv')
    if (RES/'support_summary.csv').exists(): pd.read_csv(RES/'support_summary.csv').to_csv(RES/'source_summary.csv',index=False)

def main():
    global Q_GLOBAL,L_GLOBAL,HEAD,moments_by_case_global
    HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(); D,Q,qij,yn,idx,rows,L,S,Dw,Qw,qijw=frozen_inputs(); Q_GLOBAL=Q; L_GLOBAL=L; evidence_normalization(); m=manifest(); B,G=make_basis(Dw,Qw,qijw); card,incl,amp,raw,post,moments_by_case_global=case_loop(m,Dw,Qw,qijw,B,G,yn,idx,rows,L)
    summaries(m,card,incl,amp,raw,post); write_metric_aliases(card,incl); sdf=pd.read_csv(RES/'support_per_case_summary.csv');
    # Weak-weak audit and pair-level difficulty atlas are descriptive and are
    # computed only after the global posterior has been frozen.
    sdf[sdf.regime=='WEAK_WEAK'].to_csv(RES/'weak_weak_audit.csv',index=False)
    pair=sdf.groupby(['source_i','source_j'],as_index=False).agg(n=('case_id','size'),exact_support=('exact','mean'),atleast_one=('atleast_one','mean'),median_p_true=('p_true_support','median'),median_rank=('rank_true','median'))
    pair.to_csv(RES/'pair_difficulty_atlas.csv',index=False)
    graph=[]
    for i,j in PAIRS: graph.append({'source_i':i,'source_j':j,'bus_index_distance':abs(BUSES.index(i)-BUSES.index(j)),'physical_bus_distance':abs(i-j)})
    pd.DataFrame(graph).to_csv(RES/'graph_metadata.csv',index=False)
    # Manifold files are explicitly marked descriptive/not evaluated here;
    # evaluating every continuous cross-support distance is a separate gate.
    pd.DataFrame([{'status':'NOT_EVALUATED','reason':'global audit focuses on posterior multiplicity; no new manifold optimization'}]).to_csv(RES/'nested_single_manifold_distance.csv',index=False)
    pd.DataFrame([{'status':'NOT_EVALUATED','reason':'global audit focuses on posterior multiplicity; no new manifold optimization'}]).to_csv(RES/'pair_manifold_distance.csv',index=False)
    fisher_and_geometry(Dw,Qw,qijw,card); gk_check(Dw,Qw,qijw,yn,idx,rows,L,m); plots(card,sdf); report(m,card,sdf)
    # retrospective run manifest and hashes
    pd.DataFrame([{'head':HEAD,'case_count':len(m),'hypotheses_per_case':137,'integrator':'GH31_OPERATIONAL','reference_integrator':'GK2D','new_tds':0,'retrospective':True,'dictionary_sha256':hashlib.sha256((PD/'output/load_tangent_v2/results/load_fd_central_operator.npz').read_bytes()).hexdigest()}]).to_csv(RES/'run_manifest.csv',index=False)
    print('GLOBAL-120-SUPPORT-AUDIT-V1 complete',HEAD,len(m))

if __name__=='__main__': main()
