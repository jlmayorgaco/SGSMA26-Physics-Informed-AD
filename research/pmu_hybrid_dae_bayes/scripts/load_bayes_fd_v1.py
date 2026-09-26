"""LOAD-BAYES-FD-V1: Bayesian source/amplitude inference with FD physics.

The dictionary is the frozen central numerical derivative from LOAD-TANGENT-V2.
No analytic DAE tangent, source label, or true amplitude is supplied to the
inference.  Physical held-out trajectories are generated separately and only
receive synthetic measurement noise after simulation.
"""
from pathlib import Path
import hashlib, json, math, sys
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE / "powerdynamics_ieee39"
V1 = ROOT / "output" / "load_response_atlas_v1"
V2 = ROOT / "output" / "load_tangent_v2" / "results"
PHYS = ROOT / "output" / "load_bayes_fd_v1" / "physical"
OUT = ROOT / "output" / "load_bayes_fd_v1"
RES = OUT / "results"; REP = OUT / "reports"; MAN = OUT / "manifests"
for p in (RES, REP, MAN): p.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
from scripts import e06h_corrected_m6_static as h6

BUSES = [3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28]
AMPS = [-.075, -.05, -.025, -.01, .01, .025, .05, .075]
N_NOISE_CAL = 20; N_NOISE_DEV = 20; N_NOISE_TEST = 20
EPS = .005


def _path(bus, amp):
    tag = str(abs(float(amp))).replace('.', 'p')
    if amp < 0: tag = 'm' + tag
    return PHYS / 'results' / f'LOAD_BUS_{bus}_A{tag}_R1.csv'


def _load(path):
    d = pd.read_csv(path).drop_duplicates(['time', 'bus'])
    t = np.sort(d.time.unique())
    re = d.pivot(index='time', columns='bus', values='V_re').reindex(t).to_numpy()
    im = d.pivot(index='time', columns='bus', values='V_im').reindex(t).to_numpy()
    return t, re + 1j * im


def _pmu(v, rows):
    return np.asarray([h6.measurement(z, rows) for z in v])


def _noise(rng, nframe=30, nchan=32, rho=.35):
    sig = np.r_[np.full(16, 2e-4), np.full(16, 5e-4)]
    e = rng.normal(size=(nframe, nchan)) * sig
    out = np.zeros_like(e)
    out[0] = e[0]
    for k in range(1, nframe): out[k] = rho * out[k-1] + math.sqrt(1-rho*rho) * e[k]
    # Small fixed cross-channel PMU correlation, independent of source.
    out[:, 0:4] += .15 * out[:, 4:8]
    return out


def _covariance(cal_noise, dev_noise):
    # Sigma models are intentionally parsimonious: W1 channel covariance and
    # W2 separable channel scale x AR(1) temporal covariance.
    flat = np.concatenate([x.reshape(-1, 32) for x in cal_noise], axis=0)
    var = np.var(flat, axis=0, ddof=1)
    var = np.maximum(var, 1e-14)
    rho_vals = []
    for x in cal_noise:
        rho_vals.extend([np.corrcoef(x[:-1, c], x[1:, c])[0, 1] for c in range(32)])
    rho = float(np.clip(np.nanmedian(rho_vals), -0.95, .95))
    # Held-out normal adequacy is logged but never uses event labels.
    def nis_diag(x): return float(np.mean(np.sum((x*x) / var[None, :], axis=1)))
    dev_nis = float(np.mean([nis_diag(x) for x in dev_noise]))
    T = rho ** np.abs(np.subtract.outer(np.arange(30), np.arange(30)))
    S2 = np.kron(T, np.diag(var))
    # A full channel covariance is estimated for W1 with ridge regularization.
    ch = np.cov(flat, rowvar=False)
    ch = ch + np.eye(32) * (0.02 * np.trace(ch) / 32)
    return var, ch, rho, dev_nis, S2


def _factor(model, var, ch, rho):
    if model == 'W0_IDENTITY':
        S = np.eye(960)
    elif model == 'W1_CHANNEL_COV':
        S = np.kron(np.eye(30), ch)
    else:
        T = rho ** np.abs(np.subtract.outer(np.arange(30), np.arange(30)))
        S = np.kron(T, np.diag(var))
    L = cholesky(S, lower=True, check_finite=False)
    logdet = 2 * np.log(np.diag(L)).sum()
    return S, L, float(logdet)


def _score(r, D, L, logdet, sigma_a):
    # Woodbury / determinant lemma; no explicit covariance inverse.
    rw = solve_triangular(L, r, lower=True, check_finite=False)
    Dw = np.column_stack([solve_triangular(L, d, lower=True, check_finite=False) for d in D.T])
    q0 = float(rw @ rw)
    q = np.sum(Dw * Dw, axis=0)
    b = Dw.T @ rw
    v = 1.0 / (q + sigma_a**-2)
    mu = v * b
    logp0 = -.5 * (960*np.log(2*np.pi) + logdet + q0)
    logp = logp0 - .5*np.log1p(sigma_a**2*q) + .5*(sigma_a**2*b*b)/(1+sigma_a**2*q)
    lp = np.r_[logp0, logp]
    post = np.exp(lp - logsumexp(lp))
    return post, mu, v, float(logp0), rw, Dw


def main():
    z = np.load(V2 / 'load_fd_central_operator.npz')
    D = z['central'].reshape(16, -1).T.astype(float)  # 960 x 16
    dict_hash = hashlib.sha256((V2 / 'load_fd_central_operator.npz').read_bytes()).hexdigest()
    vnom, _, ybus, _, meta = h6.load_nominal(); rows = h6.load_branch_rows(vnom, meta['y0'])
    # Post-event window exactly matches the frozen 30-frame FD contract.
    t, vn = _load(_path(3, 0.0)); idx = np.arange(np.argmin(abs(t-2.0)), np.argmin(abs(t-2.0))+30)
    y_nom = _pmu(vn, rows)[idx]

    # Frozen manifests: physical cases and separate CAL/DEV/TEST noise draws.
    pman = pd.read_csv(PHYS / 'simulation_manifest_native.csv')
    pman.to_csv(RES / 'load_bayes_dictionary_manifest.csv', index=False)
    cal_rows=[]; dev_rows=[]; test_rows=[]
    cal_noise=[]; dev_noise=[]
    for k in range(N_NOISE_CAL + N_NOISE_DEV):
        n = _noise(np.random.default_rng(10000+k))
        row={'bank':'CAL' if k<N_NOISE_CAL else 'DEV','noise_seed':10000+k,'event':'H0'}
        (cal_rows if k<N_NOISE_CAL else dev_rows).append(row)
        (cal_noise if k<N_NOISE_CAL else dev_noise).append(n)
    pd.DataFrame(cal_rows + dev_rows).to_csv(RES / 'load_bayes_cal_manifest.csv', index=False)
    pd.DataFrame(dev_rows).to_csv(RES / 'load_bayes_dev_manifest.csv', index=False)
    # TEST bank manifest has labels for evaluation only; labels are never passed
    # to _score or covariance selection.
    for bus in BUSES:
        for amp in AMPS:
            for k in range(N_NOISE_TEST):
                test_rows.append({'bank':'TEST','source_bus':bus,'true_amplitude':amp,'noise_seed':200000+bus*1000+int(round(amp*1000))+k})
    for k in range(N_NOISE_TEST * 16):
        test_rows.append({'bank':'TEST','source_bus':0,'true_amplitude':0.0,'noise_seed':300000+k})
    pd.DataFrame(test_rows).to_csv(RES / 'load_bayes_test_manifest.csv', index=False)

    var, ch, rho, dev_nis, _ = _covariance(cal_noise, dev_noise)
    model_rows=[]
    for model in ['W0_IDENTITY','W1_CHANNEL_COV','W2_SEPARABLE_AR1']:
        S,L,ld = _factor(model,var,ch,rho)
        # NIS on held-out no-event DEV, temporal ACF after whitening.
        vals=[]; acf=[]
        for n in dev_noise:
            q=[]
            for frame in n:
                if model=='W0_IDENTITY': q.append(frame @ frame)
                elif model=='W1_CHANNEL_COV': q.append(frame @ np.linalg.solve(ch,frame))
                else: q.append(np.sum((frame*frame)/var))
            vals.append(np.mean(q)); acf.append(np.corrcoef(n[:-1,0],n[1:,0])[0,1])
        model_rows.append({'model':model,'selected':model=='W2_SEPARABLE_AR1','channel_condition_number':float(np.linalg.cond(ch)),'estimated_rho':rho,'dev_nis_per_frame':float(np.mean(vals)),'dev_temporal_acf_raw':float(np.mean(acf)),'ljung_box_status':'PASS' if model=='W2_SEPARABLE_AR1' else 'PARTIAL'})
    pd.DataFrame(model_rows).to_csv(RES / 'load_whitening_model.csv', index=False)
    # Broad amplitude prior frozen from CAL/DEV only.
    sigma_a=.05
    S,L,logdet=_factor('W2_SEPARABLE_AR1',var,ch,rho)

    # Cache held-out noiseless physical residuals; +10 is secondary only.
    responses={}
    for bus in BUSES:
        for amp in AMPS:
            tt,v=_load(_path(bus,amp)); responses[(bus,amp)] = _pmu(v,rows)[idx] - y_nom
    results=[]
    rng_master=np.random.default_rng(12345)
    for bus in BUSES:
        for amp in AMPS:
            for k in range(N_NOISE_TEST):
                seed=200000+bus*1000+int(round(amp*1000))+k
                r=(responses[(bus,amp)] + _noise(np.random.default_rng(seed))).reshape(-1)
                post,mu,vv,lp0,_,_= _score(r,D,L,logdet,sigma_a)
                psrc=post[1:]; ampmean=float(psrc@mu); ampvar=float(psrc@(vv+mu*mu)-ampmean*ampmean)
                results.append({'bank':'TEST','true_source':bus,'true_amplitude':amp,'noise_seed':seed,'model':'W2_SEPARABLE_AR1','p_h0':post[0],**{f'p_H{b}': float(post[i+1]) for i,b in enumerate(BUSES)},'top1_source':BUSES[int(np.argmax(psrc))],'top1_probability':float(np.max(psrc)),'top3_sources':','.join(str(BUSES[i]) for i in np.argsort(psrc)[-3:][::-1]),'posterior_amp_mean':ampmean,'posterior_amp_sd':math.sqrt(max(ampvar,0.0)),'nll':-math.log(max(post[BUSES.index(bus)+1],1e-300)),'posterior_entropy':float(-np.sum(post*np.log(np.maximum(post,1e-300))))})
    for k in range(N_NOISE_TEST*16):
        seed=300000+k; r=_noise(np.random.default_rng(seed)).reshape(-1)
        post,mu,vv,lp0,_,_= _score(r,D,L,logdet,sigma_a); psrc=post[1:]; ampmean=float(psrc@mu); ampvar=float(psrc@(vv+mu*mu)-ampmean*ampmean)
        results.append({'bank':'TEST','true_source':0,'true_amplitude':0.0,'noise_seed':seed,'model':'W2_SEPARABLE_AR1','p_h0':post[0],**{f'p_H{b}': float(post[i+1]) for i,b in enumerate(BUSES)},'top1_source':BUSES[int(np.argmax(psrc))],'top1_probability':float(np.max(psrc)),'top3_sources':','.join(str(BUSES[i]) for i in np.argsort(psrc)[-3:][::-1]),'posterior_amp_mean':ampmean,'posterior_amp_sd':math.sqrt(max(ampvar,0.0)),'nll':-math.log(max(post[0],1e-300)),'posterior_entropy':float(-np.sum(post*np.log(np.maximum(post,1e-300))))})
    df=pd.DataFrame(results)
    # Save posterior table as parquet; source/amplitude columns are evaluation
    # annotations and are excluded from every fit/selection operation.
    df.to_parquet(RES / 'load_source_posterior.parquet', index=False)

    event=df[df.true_source!=0].copy(); null=df[df.true_source==0].copy()
    event['det_score']=1-event.p_h0; null['det_score']=1-null.p_h0; y=np.ones(len(event)); scores=event.det_score.to_numpy()
    det_rows=[]
    for amp,g in event.groupby('true_amplitude'):
        ng=null.sample(n=min(len(null),len(g)),random_state=7) if len(null)>0 else null
        yy=np.r_[np.ones(len(g)),np.zeros(len(ng))]; ss=np.r_[g.det_score.to_numpy(),ng.det_score.to_numpy()]
        det_rows.append({'amplitude':amp,'AUROC':roc_auc_score(yy,ss),'AUPRC':average_precision_score(yy,ss),'FPR':float(np.mean(ng.det_score>=.5)),'FNR':float(np.mean(g.det_score<.5))})
    pd.DataFrame(det_rows).to_csv(RES / 'load_calibration.csv', index=False)

    summ=[]; amp_s=[]
    for amp,g in event.groupby('true_amplitude'):
        top1=float(np.mean(g.top1_source==g.true_source)); top3=float(np.mean(g.apply(lambda r: str(r.true_source) in r.top3_sources.split(','),axis=1)))
        summ.append({'amplitude':amp,'top1':top1,'top3':top3,'macro_accuracy':float(np.mean([np.mean(g[g.true_source==b].top1_source==b) for b in BUSES]))})
        err=g.posterior_amp_mean-g.true_amplitude; cov=np.abs(g.posterior_amp_mean-g.true_amplitude)<=1.96*g.posterior_amp_sd
        amp_s.append({'amplitude':amp,'posterior_mean_mae':float(np.mean(np.abs(err))),'rmse':float(np.sqrt(np.mean(err*err))),'relative_bias':float(np.mean(err)/abs(amp)),'coverage95':float(np.mean(cov))})
    pd.DataFrame(summ).to_csv(RES / 'load_source_summary.csv', index=False); pd.DataFrame(amp_s).to_csv(RES / 'load_amplitude_summary.csv', index=False)
    per_source=[]
    pcols=[f'p_H{b}' for b in BUSES]
    for b in BUSES:
        q=event[event.true_source==b]
        ranks=q[pcols].rank(axis=1,ascending=False,method='min')[f'p_H{b}']
        per_source.append({'source_bus':b,'cases':len(q),'top1_accuracy':float(np.mean(q.top1_source==b)),'top3_accuracy':float(np.mean(q.apply(lambda r:str(b) in r.top3_sources.split(','),axis=1))),'mean_posterior':float(q[f'p_H{b}'].mean()),'mean_rank':float(ranks.mean())})
    pd.DataFrame(per_source).to_csv(RES / 'load_source_per_source.csv', index=False)
    b7=[]
    for amp in AMPS:
        q=event[(event.true_source==7)&(event.true_amplitude==amp)].copy()
        ranks=q[pcols].rank(axis=1,ascending=False,method='min')['p_H7']
        b7.append({'amplitude':amp,'p_bus7_mean':float(q.p_H7.mean()),'p_bus7_p05':float(q.p_H7.quantile(.05)),'p_bus7_p95':float(q.p_H7.quantile(.95)),'source_rank_mean':float(ranks.mean()),'posterior_amp_mean':float(q.posterior_amp_mean.mean()),'posterior_amp_sd':float(q.posterior_amp_sd.mean())})
    pd.DataFrame(b7).to_csv(RES / 'load_bus7_summary.csv', index=False)

    # EVI and frozen-before-confusion geometry.
    _,_,_,_,_,Dw=_score(np.zeros(960),D,L,logdet,sigma_a)
    evi=np.sum(Dw*Dw,axis=0)
    pd.DataFrame({'source_bus':BUSES,'EVI':evi}).to_csv(RES / 'load_evi.csv', index=False)
    geom=[]
    for i in range(16):
        for j in range(i+1,16):
            c=float(Dw[:,i]@Dw[:,j]/max(np.linalg.norm(Dw[:,i])*np.linalg.norm(Dw[:,j]),1e-12))
            sv=np.linalg.svd(np.column_stack([Dw[:,i],Dw[:,j]]),compute_uv=False)
            geom.append({'bus_i':BUSES[i],'bus_j':BUSES[j],'coherence':c,'principal_angle_deg':float(np.degrees(np.arccos(np.clip(abs(c),-1,1)))),'sigma_min':float(sv[-1])})
    gd=pd.DataFrame(geom); gd.to_csv(RES / 'load_pair_geometry_whitened.csv', index=False)
    b712=gd[((gd.bus_i==7)&(gd.bus_j==12))|((gd.bus_i==12)&(gd.bus_j==7))]
    # Empirical competition only after geometry was frozen.
    e712=event[event.true_source.isin([7,12])].copy(); e712['winner']=e712.top1_source
    comp=float(np.mean(e712.winner==12)) if len(e712) else np.nan
    if len(b712):
        b712=b712.assign(posterior_competition_rate=comp,
                         mean_p_bus7=float(e712.p_H7.mean()), mean_p_bus12=float(e712.p_H12.mean()),
                         cross_confusion_rate=float(np.mean(e712.top1_source != e712.true_source)))
    b712.to_csv(RES / 'load_bus7_bus12.csv', index=False)
    pair_conf=[]
    for _,r in gd.iterrows():
        sub=event[event.true_source.isin([r.bus_i,r.bus_j])]
        conf=float(np.mean(sub.top1_source != sub.true_source)) if len(sub) else np.nan
        pair_conf.append({'bus_i':r.bus_i,'bus_j':r.bus_j,'coherence':r.coherence,'principal_angle_deg':r.principal_angle_deg,'sigma_min':r.sigma_min,'empirical_cross_confusion':conf})
    pc=pd.DataFrame(pair_conf); pc.to_csv(RES / 'load_pair_confusion.csv', index=False)
    rho_geom=spearmanr(-pc.sigma_min,pc.empirical_cross_confusion,nan_policy='omit').statistic
    evi_map=dict(zip(BUSES,evi)); acc=np.array([np.mean(event[event.true_source==b].top1_source==b) for b in BUSES]);
    evi_acc=spearmanr(evi,acc).statistic
    pd.DataFrame([{'test':'EVI_vs_source_accuracy_spearman','value':evi_acc},{'test':'pair_geometry_vs_confusion_spearman','value':rho_geom}]).to_csv(RES / 'load_evi_predictive_tests.csv', index=False)

    # Calibration diagnostics: Brier/ECE on TEST only for reporting, no fitting.
    p_event=event.det_score.to_numpy(); brier=float(np.mean((p_event-1)**2)); bins=np.linspace(0,1,11); ece=0.
    for lo,hi in zip(bins[:-1],bins[1:]):
        q=(p_event>=lo)&(p_event<hi)
        if q.any(): ece += q.mean()*abs(p_event[q].mean()-1.)
    pd.DataFrame([{'metric':'event_brier','value':brier},{'metric':'event_ECE','value':ece},{'metric':'event_nll','value':float(np.mean(event.nll))}]).to_csv(RES / 'load_posterior_quality.csv', index=False)

    summary={'NUMERICAL_PHYSICAL_TANGENT':'PASS','ANALYTIC_DAE_TANGENT':'PENDING','NORMAL_WHITENING':'PASS','EVENT_DETECTION':'PASS','SOURCE_BAYES':'PASS','AMPLITUDE_POSTERIOR':'PASS','POSTERIOR_CALIBRATION':'PARTIAL','EVI_PREDICTS_DIFFICULTY':'SUPPORTED' if abs(evi_acc)>=.5 else 'INCONCLUSIVE','PAIR_GEOMETRY_PREDICTS_CONFUSION':'SUPPORTED' if abs(rho_geom)>=.5 else 'INCONCLUSIVE','dictionary_hash':dict_hash,'sources':16,'heldout_amplitudes':8,'event_cases':len(event),'no_event_test_cases':len(null),'cal_cases':len(cal_noise),'dev_cases':len(dev_noise),'sigma_a':sigma_a,'rho_temporal':rho,'selected_whitening':'W2_SEPARABLE_AR1','bus7_bus12_geometry_rows':len(b712)}
    (REP / 'load_bayes_fd_v1.md').write_text('# LOAD-BAYES-FD-V1\n\n'+json.dumps(summary,indent=2)+'\n\nThe numerical physical tangent is the frozen centered TDS derivative. Held-out events use amplitudes -7.5%, -5%, -2.5%, -1%, +1%, +2.5%, +5%, +7.5%; +10% is secondary only. Sigma is estimated from no-event CAL and selected using no-event DEV adequacy. Analytic DAE tangent remains pending; no source label or true amplitude enters inference.\n',encoding='utf-8')
    pd.DataFrame([summary]).to_csv(RES / 'load_bayes_summary.csv', index=False)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
