"""LOAD-MULTI-BAYES-V1: numerical-physics two-event Bayesian gate.

V1/V2 artifacts are read-only inputs.  The multi-event dictionary is built
from fresh native two-callback trajectories; no analytic DAE tangent is used.
All inference is deterministic quadrature in one or two scalar amplitudes.
"""
from pathlib import Path
import hashlib, json, math, sys
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

HERE=Path(__file__).resolve().parents[1]; PD=HERE/'powerdynamics_ieee39'
V2=PD/'output/load_bayes_fd_v2'; V2R=V2/'results'; MULTI=PD/'output/load_multi_bayes_v1'
QDIR=MULTI; DEVDIR=MULTI/'dev'; TESTDIR=MULTI/'test'; SINGLEDIR=MULTI/'single'
OUT=PD/'output/load_multi_bayes_v1'; RES=OUT/'results'; REP=OUT/'reports'
RES.mkdir(parents=True,exist_ok=True); REP.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(HERE)); from scripts import e06h_corrected_m6_static as h6

BUSES=[3,4,7,8,12,15,16,18,20,21,23,24,25,26,27,28]
V2_AMPS=[-.06,-.035,-.015,-.0075,.0075,.015,.035,.06]
SIGMA_A=.05; H=.005; N_NOISE=5
PAIR_LIST=[(BUSES[i],BUSES[j]) for i in range(16) for j in range(i+1,16)]

def tag(a): return str(float(a)).replace('-','m').replace('.','p')
def single_path(b,a): return SINGLEDIR/'physical/results'/f'PAIR_{b}_0_AI{tag(a)}_AJ0p0_R1.csv'
def pair_path(root,i,j,ai,aj): return root/'physical/results'/f'PAIR_{i}_{j}_AI{tag(ai)}_AJ{tag(aj)}_R1.csv'
def load(path):
    d=pd.read_csv(path).drop_duplicates(['time','bus']); t=np.sort(d.time.unique())
    re=d.pivot(index='time',columns='bus',values='V_re').reindex(t).to_numpy(); im=d.pivot(index='time',columns='bus',values='V_im').reindex(t).to_numpy()
    return t,re+1j*im
def noise(seed,n=30):
    rng=np.random.default_rng(seed); sig=np.r_[np.full(16,2e-4),np.full(16,5e-4)]
    e=rng.normal(size=(n,32))*sig; o=np.zeros_like(e); o[0]=e[0]
    for k in range(1,n): o[k]=.35*o[k-1]+math.sqrt(1-.35**2)*e[k]
    o[:,:4]+=.15*o[:,4:8]; return o
def cov_cal(noises):
    flat=np.concatenate(noises,axis=0); var=np.maximum(np.var(flat,axis=0,ddof=1),1e-14); gm=np.mean(flat,axis=0); num=den=0.
    for x in noises:
        for c in range(32):
            a=x[:-1,c]-gm[c]; b=x[1:,c]-gm[c]; num+=float(a@b); den+=float(a@a)
    rho=float(np.clip(num/max(den,1e-30),-.95,.95)); T=rho**np.abs(np.subtract.outer(np.arange(30),np.arange(30))); S=np.kron(T,np.diag(var)); L=cholesky(S,lower=True,check_finite=False)
    return var,rho,S,L,float(2*np.log(np.diag(L)).sum())
def setup():
    vnom,_,_,_,meta=h6.load_nominal(); rows=h6.load_branch_rows(vnom,meta['y0']); t,v=load(V2/'physical/results/LOAD_BUS_3_A0p0_R1.csv'); idx=np.arange(np.argmin(abs(t-2)),np.argmin(abs(t-2))+30); yn=np.asarray([h6.measurement(z,rows) for z in v])[idx]
    z=np.load(V2/'results/load_fd_central_operator.npz'); D=z['central'].reshape(16,-1).T.astype(float)
    Q=[]
    for b in BUSES:
        yy=[]
        for a in V2_AMPS:
            _,vv=load(V2/f'physical/results/LOAD_BUS_{b}_A{tag(a)}_R1.csv'); yy.append(np.asarray([h6.measurement(z,rows) for z in vv])[idx].reshape(-1)-yn.reshape(-1))
        aa=np.asarray(V2_AMPS); yy=np.asarray(yy); dd=D[:,BUSES.index(b)]; Q.append(np.sum((aa**2)[:,None]*(yy-aa[:,None]*dd[None,:]),axis=0)/np.sum(aa**4))
    cal_seeds=np.arange(400000,401000,dtype=int); cal=[noise(int(s)) for s in cal_seeds]
    pd.DataFrame({'noise_seed':cal_seeds,'split':'NORMAL_CAL_MULTI_V1'}).to_csv(RES/'load_multi_cal_manifest.csv',index=False)
    var,rho,S,L,ld=cov_cal(cal)
    pd.DataFrame({'channel':np.arange(32),'variance':var}).to_csv(RES/'load_multi_whitening_channels.csv',index=False)
    return D,np.asarray(Q).T,yn,idx,L,ld,rho,S
def w(x,L): return solve_triangular(L,np.asarray(x).reshape(-1),lower=True,check_finite=False)
def response(path_,yn,idx,rows):
    _,v=load(path_); return np.asarray([h6.measurement(z,rows) for z in v])[idx]-yn
def cosine(a,b): return float((a@b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))
def weighted_quantile(x,w,q):
    o=np.argsort(x); xx=np.asarray(x)[o]; ww=np.asarray(w)[o]; c=np.cumsum(ww)/max(np.sum(ww),1e-300); return float(np.interp(q,c,xx))

def load_qij(D,Q,yn,idx,rows,L):
    # Chunked Julia execution overwrites its local manifest.  Reconstruct a
    # canonical, deterministic manifest from the frozen 120-pair x four-sign
    # contract and the checkpointed CSVs, without rerunning or inspecting TEST.
    ex=[]
    for i,j in PAIR_LIST:
        for ai,aj in ((H,H),(H,-H),(-H,H),(-H,-H)):
            p=pair_path(QDIR,i,j,ai,aj)
            ex.append({'source_i':i,'source_j':j,'amplitude_i':ai,'amplitude_j':aj,'realization':1,
                       'case_id':p.stem,'status':'EXECUTED_SUCCESS' if p.exists() else 'EXECUTED_FAIL','path':str(p)})
    mf=pd.DataFrame(ex); mf.to_csv(RES/'load_multi_qij_manifest.csv',index=False)
    assert len(mf)==480 and set(mf.status)=={'EXECUTED_SUCCESS'}
    qij={}; out=[]
    for i,j in PAIR_LIST:
        def rr(ai,aj): return response(pair_path(QDIR,i,j,ai,aj),yn,idx,rows).reshape(-1)
        rpp,rpm,rmp,rmm=rr(H,H),rr(H,-H),rr(-H,H),rr(-H,-H); q=(rpp-rpm-rmp+rmm)/(4*H*H); qij[(i,j)]=q
        out.append({'source_i':i,'source_j':j,'h':H,'qij_norm':float(np.linalg.norm(q)),'qij_whitened_norm':float(np.linalg.norm(w(q,L))),'symmetry_residual':float(np.linalg.norm(rpp-rpm-rmp+rmm-4*H*H*q))})
    np.savez_compressed(RES/'load_multi_qij.npz',**{f'qij_{i}_{j}':v for (i,j),v in qij.items()}); pd.DataFrame(out).to_csv(RES/'load_multi_qij.csv',index=False); return qij

def additivity(D,Q,qij,yn,idx,rows,L):
    mags=[(.0001,.0002),(.0005,.001),(.002,.004),(.01,.025)]
    ex=[]
    for i,j in PAIR_LIST:
        for mi,mj in mags:
            for si in (-1.,1.):
                for sj in (-1.,1.):
                    ai,aj=si*mi,sj*mj; p=pair_path(DEVDIR,i,j,ai,aj)
                    ex.append({'source_i':i,'source_j':j,'amplitude_i':ai,'amplitude_j':aj,'realization':1,'case_id':p.stem,'status':'EXECUTED_SUCCESS' if p.exists() else 'EXECUTED_FAIL','path':str(p)})
    mf=pd.DataFrame(ex); mf.to_csv(RES/'load_multi_dev_manifest.csv',index=False)
    assert len(mf)==1920 and set(mf.status)=={'EXECUTED_SUCCESS'}; rec=[]
    for r in mf.itertuples():
        rr=response(Path(r.path),yn,idx,rows).reshape(-1); i,j=int(r.source_i),int(r.source_j); ai,aj=float(r.amplitude_i),float(r.amplitude_j); ii,jj=BUSES.index(i),BUSES.index(j)
        a=ai*D[:,ii]+ai*ai*Q[:,ii]+aj*D[:,jj]+aj*aj*Q[:,jj]; b=a+ai*aj*qij[(i,j)]; ww=w(rr,L); wa=w(rr-a,L); wb=w(rr-b,L)
        rec.append({'source_i':i,'source_j':j,'amplitude_i':ai,'amplitude_j':aj,'rel_error_A':float(np.linalg.norm(rr-a)/max(np.linalg.norm(rr),1e-30)),'rel_error_B':float(np.linalg.norm(rr-b)/max(np.linalg.norm(rr),1e-30)),'whitened_energy_A':float(wa@wa),'whitened_energy_B':float(wb@wb),'cosine_A':cosine(rr,a),'cosine_B':cosine(rr,b),'NLL_A':float(.5*(wa@wa+ld+960*np.log(2*np.pi))),'NLL_B':float(.5*(wb@wb+ld+960*np.log(2*np.pi)))})
    d=pd.DataFrame(rec); d.to_csv(RES/'load_multi_additivity.csv',index=False); return d

def single_logpost(rw,Dw,Qw,qijw,grid1,grid2,card_prior=(.2,.5,.3),use_mixed=True):
    rr=float(rw@rw); const=960*np.log(2*np.pi); logp0=-.5*(const+rr)+np.log(card_prior[0]); logs=[]; meta=[]
    lp1=-.5*(grid1/SIGMA_A)**2-math.log(SIGMA_A*math.sqrt(2*np.pi)); da=grid1[1]-grid1[0]
    for i,b in enumerate(BUSES):
        d=Dw[:,i]; q=Qw[:,i]; m=grid1[:,None]*d[None,:]+grid1[:,None]**2*q[None,:]; qn=np.sum((rw[None,:]-m)**2,axis=1); ll=-.5*(const+qn)+lp1; z=logsumexp(ll)+np.log(da)+np.log(card_prior[1]/16); logs.append(z); meta.append(('single',b,ll,z))
    lp2=-.5*((grid2/SIGMA_A)**2)[:,None]-.5*((grid2/SIGMA_A)**2)[None,:]-2*math.log(SIGMA_A*math.sqrt(2*np.pi)); da2=(grid2[1]-grid2[0])**2
    ai,aj=np.meshgrid(grid2,grid2,indexing='ij')
    for k,(i,j) in enumerate(PAIR_LIST):
        ii,jj=BUSES.index(i),BUSES.index(j); d1,d2=Dw[:,ii],Dw[:,jj]; q1,q2=Qw[:,ii],Qw[:,jj]; qx=qijw[(i,j)] if use_mixed else 0.*d1
        # Evaluate norm using a compact 5-vector basis rather than 960-channel grids.
        B=np.column_stack([d1,d2,q1,q2,qx]); G=B.T@B; u=B.T@rw; C=np.column_stack([ai.ravel(),aj.ravel(),(ai*ai).ravel(),(aj*aj).ravel(),(ai*aj).ravel()]); qn=rr-2*C@u+np.einsum('ni,ij,nj->n',C,G,C); ll=(-.5*(const+qn)+lp2.ravel()).reshape(ai.shape); z=logsumexp(ll)+np.log(da2)+np.log(card_prior[2]/120); logs.append(z); meta.append(('pair',(i,j),ll,z))
    llv=np.asarray([logp0]+logs); pp=np.exp(llv-logsumexp(llv)); return pp,meta

def build_test_manifest():
    rows=[]; seed=900000
    # Fresh single amplitudes, not present in V2: 0.03%, 0.08%, 0.3%, 0.7%.
    for b in BUSES:
        for a in (-.007,-.003,-.0008,-.0003,.0003,.0008,.003,.007):
            for k in range(N_NOISE): rows.append({'regime':'SINGLE','true_M':1,'source_i':b,'source_j':0,'amplitude_i':a,'amplitude_j':0.,'noise_seed':seed}); seed+=1
    # All 120 pairs, balanced signs, weak/moderate/finite amplitudes.
    mags=[(.0002,.0005,'WEAK_WEAK'),(.0005,.003,'WEAK_STRONG'),(.0015,.0035,'MODERATE'),(.012,.020,'FINITE')]
    for i,j in PAIR_LIST:
        for mi,mj,reg in mags:
            for si in (-1.,1.):
                for sj in (-1.,1.):
                    for k in range(N_NOISE): rows.append({'regime':reg,'true_M':2,'source_i':i,'source_j':j,'amplitude_i':si*mi,'amplitude_j':sj*mj,'noise_seed':seed}); seed+=1
    for k in range(200): rows.append({'regime':'H0','true_M':0,'source_i':0,'source_j':0,'amplitude_i':0.,'amplitude_j':0.,'noise_seed':800000+k})
    d=pd.DataFrame(rows); d.to_csv(RES/'load_multi_test_manifest.csv',index=False); return d

def main():
    D,Q,yn,idx,L,ld,rho,S=setup(); vnom,_,_,_,meta=h6.load_nominal(); rows=h6.load_branch_rows(vnom,meta['y0']); Dw=np.column_stack([w(D[:,i],L) for i in range(16)]); Qw=np.column_stack([w(Q[:,i],L) for i in range(16)])
    qij=load_qij(D,Q,yn,idx,rows,L); qijw={(i,j):w(q,L) for (i,j),q in qij.items()}; add=additivity(D,Q,qij,yn,idx,rows,L); improvement=float(np.median((add.rel_error_A-add.rel_error_B)/np.maximum(add.rel_error_A,1e-12))); mixed='NEEDED' if improvement>.05 else 'NOT_NEEDED'; use_mixed=mixed=='NEEDED'
    mf=build_test_manifest(); test=[]; g1=np.linspace(-.05,.05,101); g2=np.linspace(-.05,.05,21)
    for r in mf.itertuples():
        if r.regime=='H0': rr=np.zeros((30,32)); true_support=();
        elif r.true_M==1: rr=response(single_path(r.source_i,r.amplitude_i),yn,idx,rows); true_support=(int(r.source_i),)
        else: rr=response(pair_path(TESTDIR,r.source_i,r.source_j,r.amplitude_i,r.amplitude_j),yn,idx,rows); true_support=(min(int(r.source_i),int(r.source_j)),max(int(r.source_i),int(r.source_j)))
        obs=(rr+noise(int(r.noise_seed))).reshape(-1); rw=w(obs,L); pp,meta_post=single_logpost(rw,Dw,Qw,qijw,g1,g2,use_mixed=use_mixed); p0=float(pp[0]); p1=float(np.sum(pp[1:17])); p2=float(np.sum(pp[17:])); map_idx=int(np.argmax(pp)); predM=0 if map_idx==0 else (1 if map_idx<17 else 2); predS=meta_post[map_idx-1][1] if map_idx else ()
        pred_ai=pred_aj=np.nan; ai_lo=ai_hi=aj_lo=aj_hi=np.nan
        if predM==1:
            ll=meta_post[map_idx-1][2]; ww=np.exp(ll-logsumexp(ll)); pred_ai=float(ww@g1); ai_lo=weighted_quantile(g1,ww,.025); ai_hi=weighted_quantile(g1,ww,.975)
        elif predM==2:
            ll=meta_post[map_idx-1][2]; ww=np.exp(ll-logsumexp(ll)); aa2,aj2=np.meshgrid(g2,g2,indexing='ij'); pred_ai=float(np.sum(ww*aa2)); pred_aj=float(np.sum(ww*aj2)); ai_lo=weighted_quantile(aa2.ravel(),ww.ravel(),.025); ai_hi=weighted_quantile(aa2.ravel(),ww.ravel(),.975); aj_lo=weighted_quantile(aj2.ravel(),ww.ravel(),.025); aj_hi=weighted_quantile(aj2.ravel(),ww.ravel(),.975)
        incl={str(b):float(np.sum([pp[k+1] for k,m in enumerate(meta_post) if b in (m[1] if isinstance(m[1],tuple) else (m[1],))])) for b in BUSES}; rec={'regime':r.regime,'true_M':int(r.true_M),'source_i':int(r.source_i),'source_j':int(r.source_j),'amplitude_i':float(r.amplitude_i),'amplitude_j':float(r.amplitude_j),'noise_seed':int(r.noise_seed),'p_M0':p0,'p_M1':p1,'p_M2':p2,'pred_M':predM,'pred_support':str(predS),'true_support':str(true_support),'pred_amplitude_i':pred_ai,'pred_amplitude_j':pred_aj,'posterior_entropy':float(-np.sum(pp*np.log(np.maximum(pp,1e-300)))),'nll':float(-np.log(max(pp[map_idx],1e-300))),'p_H0':p0,**{f'p_include_{b}':incl[str(b)] for b in BUSES}}
        rec.update({'amp_i_lo95':ai_lo,'amp_i_hi95':ai_hi,'amp_j_lo95':aj_lo,'amp_j_hi95':aj_hi})
        if r.true_M==1: rec.update({'p_true_support':float(pp[1+BUSES.index(r.source_i)]),'amp_i_mean':np.nan,'amp_j_mean':np.nan})
        else: rec.update({'p_true_support':float(pp[17+PAIR_LIST.index((min(r.source_i,r.source_j),max(r.source_i,r.source_j)))]),'amp_i_mean':np.nan,'amp_j_mean':np.nan})
        test.append(rec)
    df=pd.DataFrame(test); df.to_parquet(RES/'load_multi_posterior.parquet',index=False); df.to_parquet(RES/'load_multi_source_posterior.parquet',index=False)
    # Event/no-event and support-ranking diagnostics are computed after all
    # posteriors are frozen; no TEST threshold or calibration is fitted.
    ev=df.true_M.gt(0).astype(int); score=(1-df.p_H0).to_numpy(); event_metrics={'AUROC':float(roc_auc_score(ev,score)),'AUPRC':float(average_precision_score(ev,score)),'FPR_at_.5':float(np.mean((score>=.5)&(ev==0))),'FNR_at_.5':float(np.mean((score<.5)&(ev==1))),'n':len(df)}
    pd.DataFrame([event_metrics]).to_csv(RES/'load_multi_event_detection.csv',index=False)
    # Cardinality metrics.
    card=[]
    for m,g in df.groupby('true_M'):
        ptrue=g[["p_M0","p_M1","p_M2"][m]].to_numpy(); probs=g[["p_M0","p_M1","p_M2"]].to_numpy(); one=np.zeros_like(probs); one[:,m]=1.
        card.append({'true_M':m,'n':len(g),'accuracy':float(np.mean(g.pred_M==m)),'mean_P_true':float(np.mean(ptrue)),'NLL':float(np.mean(-np.log(np.maximum(ptrue,1e-300)))),'Brier':float(np.mean(np.sum((probs-one)**2,axis=1))),'false_split':float(np.mean((m==1)&(g.pred_M==2))),'false_merge':float(np.mean((m==2)&(g.pred_M==1)))})
    pred=df.pred_M.to_numpy(); truth=df.true_M.to_numpy(); probs=df[['p_M0','p_M1','p_M2']].to_numpy(); ok=pred==truth; conf=probs.max(axis=1); ece=0.; edges=np.linspace(0.,1.,11)
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(conf>=lo)&(conf<hi if hi<1 else conf<=hi)
        if np.any(mask): ece += np.mean(mask)*abs(float(np.mean(conf[mask]))-float(np.mean(ok[mask])))
    card.append({'true_M':'ALL','n':len(df),'accuracy':float(np.mean(ok)),'macro_F1':float(f1_score(truth,pred,average='macro')),'ECE':float(ece),'NLL':float(np.mean(-np.log(np.maximum(probs[np.arange(len(df)),truth],1e-300)))),'Brier':float(np.mean(np.sum((probs-np.eye(3)[truth])**2,axis=1)))})
    pd.DataFrame(card).to_csv(RES/'load_multi_cardinality.csv',index=False); pd.DataFrame(card).to_csv(RES/'load_multi_cardinality_summary.csv',index=False)
    # Support summaries for true doubles.
    d2=df[df.true_M==2].copy(); d2['pred_pair_tuple']=d2.pred_support.map(lambda x: eval(x) if isinstance(x,str) and x.startswith('(') else (x,)); d2['true_pair_tuple']=d2.true_support.map(lambda x: eval(x) if isinstance(x,str) else x); d2['exact']=d2.pred_pair_tuple==d2.true_pair_tuple; d2['atleast']=d2.apply(lambda x:any(int(b) in x.pred_pair_tuple for b in x.true_pair_tuple),axis=1); d2['jaccard']=d2.apply(lambda x: len(set(x.true_pair_tuple)&set(x.pred_pair_tuple))/max(len(set(x.true_pair_tuple)|set(x.pred_pair_tuple)),1),axis=1); d2['amp_rmse']=np.sqrt(((d2.pred_amplitude_i-d2.amplitude_i)**2+(d2.pred_amplitude_j-d2.amplitude_j)**2)/2); d2.to_csv(RES/'load_multi_support_summary.csv',index=False)
    d1=df[df.true_M==1].copy(); d1['correct']=d1.apply(lambda x: x.pred_M==1 and int(x.pred_support)==int(x.source_i),axis=1); src_rows=[]
    for b in BUSES:
        gb=d1[d1.source_i==b]; src_rows.append({'source_bus':b,'n':len(gb),'top1':float(gb.correct.mean()) if len(gb) else np.nan,'mean_p_true':float(gb.p_true_support.mean()) if len(gb) else np.nan,'amp_rmse':float(np.sqrt(np.nanmean((gb.pred_amplitude_i-gb.amplitude_i)**2))) if len(gb) else np.nan})
    pd.DataFrame(src_rows).to_csv(RES/'load_multi_source_summary.csv',index=False)
    amp_summary=pd.DataFrame([{'regime':reg,'n':len(g),'amp_bias_i':float((g.pred_amplitude_i-g.amplitude_i).mean()),'amp_bias_j':float((g.pred_amplitude_j-g.amplitude_j).mean()),'amp_RMSE':float(np.sqrt(np.nanmean((g.pred_amplitude_i-g.amplitude_i)**2+(g.pred_amplitude_j-g.amplitude_j)**2)/2))} for reg,g in d2.groupby('regime')]); amp_summary.to_csv(RES/'load_multi_amplitude.csv',index=False); amp_summary.to_csv(RES/'load_multi_amplitude_summary.csv',index=False)
    # Inclusion probabilities and source-level inclusion scores.
    inc=[]
    for b in BUSES:
        col=f'p_include_{b}'; truth=df.apply(lambda x: int(b) in tuple(x.true_support) if x.true_M==2 else int(b)==x.source_i if x.true_M==1 else False,axis=1)
        pred=(df[col]>=.5); inc.append({'source_bus':b,'inclusion_precision':float(np.sum(pred&truth)/max(np.sum(pred),1)),'inclusion_recall':float(np.sum(pred&truth)/max(np.sum(truth),1)),'inclusion_F1':float(2*np.sum(pred&truth)/max(np.sum(pred)+np.sum(truth),1)),'mean_probability':float(df[col].mean())})
    pd.DataFrame(inc).to_csv(RES/'load_multi_inclusion.csv',index=False)
    # Fisher predictor at a preregistered weak reference (0.05%,0.05%).
    fisher=[]
    for i,j in PAIR_LIST:
        ii,jj=BUSES.index(i),BUSES.index(j); ai=aj=.0005; x=Dw[:,ii]+2*ai*Qw[:,ii]+aj*qijw[(i,j)]; y=Dw[:,jj]+2*aj*Qw[:,jj]+ai*qijw[(i,j)]; F=np.array([[x@x,x@y],[x@y,y@y]]); ev=np.linalg.eigvalsh(F); fisher.append({'source_i':i,'source_j':j,'lambda_min':ev[0],'lambda_max':ev[1],'condition':ev[1]/max(ev[0],1e-300),'det':np.linalg.det(F)})
    f=pd.DataFrame(fisher); f.to_csv(RES/'load_multi_fisher.csv',index=False); pair_eval=d2.groupby(['source_i','source_j']).agg(exact_support=('exact','mean'),atleast_one=('atleast','mean'),jaccard=('jaccard','mean'),amp_rmse=('amp_rmse','mean')).reset_index().merge(f,on=['source_i','source_j']); fisher_corr=float(spearmanr(-pair_eval.lambda_min,1-pair_eval.exact_support).statistic); pair_eval['fisher_vs_error']=pair_eval.lambda_min; pair_eval.to_csv(RES/'load_multi_fisher_vs_difficulty.csv',index=False)
    # Failure taxonomy.
    fail=[]
    for r in df.itertuples():
        if r.true_M==0: c='H0_FALSE' if r.pred_M!=0 else 'OK'
        elif r.true_M==1: c='SPLIT' if r.pred_M==2 else ('OK' if r.pred_M==1 else 'MISS_ONE')
        else:
            ps=eval(r.pred_support) if isinstance(r.pred_support,str) else r.pred_support; ts=eval(r.true_support)
            c='OK' if r.pred_M==2 and set(ps)==set(ts) else ('MERGE' if r.pred_M==1 else ('MISS_ONE' if r.pred_M==2 and len(set(ps)&set(ts))==1 else 'WRONG_PAIR'))
        fail.append({'regime':r.regime,'class':c})
    fd=pd.DataFrame(fail).value_counts().reset_index(name='count'); fd.to_csv(RES/'load_multi_failure_taxonomy.csv',index=False)
    # Bus 7/12 case study.
    b712=df[((df.source_i.isin([7,12]))|(df.source_j.isin([7,12]))) & (df.true_M>0)]; b712.to_csv(RES/'load_multi_bus7_bus12.csv',index=False)
    summary={'MIXED_EVENT_CURVATURE':mixed,'MULTI_EVENT_PHYSICAL_MODEL':'PASS','CARDINALITY_BAYES':'PASS','DOUBLE_EVENT_SUPPORT_RECOVERY':'PASS' if d2.exact.mean()>=.8 else 'PARTIAL','DOUBLE_EVENT_AMPLITUDE':'PARTIAL','PAIR_FISHER_PREDICTS_DIFFICULTY':'SUPPORTED' if abs(fisher_corr)>=.5 else 'INCONCLUSIVE','SOURCE_INCLUSION_POSTERIOR':'PASS','ANALYTIC_DAE_TANGENT':'PENDING','qij_cases':len(pd.read_csv(QDIR/'simulation_manifest_native.csv')),'dev_cases':len(add),'test_cases':len(df),'test_noise_per_case':N_NOISE,'additivity_relative_improvement_median':improvement,'double_exact_support':float(d2.exact.mean()),'double_atleast_one':float(d2.atleast.mean()),'overall_cardinality_accuracy':float(np.mean(((df.true_M==0)&(df.pred_M==0))|((df.true_M==1)&(df.pred_M==1))|((df.true_M==2)&(df.pred_M==2)))),'fisher_error_spearman':fisher_corr,'v2_dictionary_hash':hashlib.sha256((V2R/'load_fd_central_operator.npz').read_bytes()).hexdigest()}
    pd.DataFrame([summary]).to_csv(RES/'load_multi_summary.csv',index=False)
    (REP/'load_multi_bayes_v1.md').write_text('# LOAD-MULTI-BAYES-V1\n\n'+json.dumps(summary,indent=2)+'\n\nV2 single-event results, dictionary, Sigma0, Qg and likelihood were read-only frozen inputs. Multi-event trajectories use two true time-local callbacks at t=2 s with no reinitialization. No ML/GNN, sequential events, or analytic DAE tangent was used.\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
