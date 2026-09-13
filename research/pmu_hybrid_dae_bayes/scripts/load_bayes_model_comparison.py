"""Compare identity, channel-covariance and selected AR whitening on held-out data."""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(HERE))
from scripts.load_bayes_fd_v1 import (BUSES, AMPS, N_NOISE_TEST, EPS, ROOT, V1, V2, PHYS,
    _path, _load, _pmu, _noise, _factor, _score, _covariance, h6)
OUT=ROOT/'output/load_bayes_fd_v1/results'
def main():
 z=np.load(V2/'load_fd_central_operator.npz'); D=z['central'].reshape(16,-1).T.astype(float)
 vnom,_,_,_,meta=h6.load_nominal(); rows=h6.load_branch_rows(vnom,meta['y0']); t,v0=_load(_path(3,0.0)); idx=np.arange(np.argmin(abs(t-2.0)),np.argmin(abs(t-2.0))+30); yn=_pmu(v0,rows)[idx]
 cal=[_noise(np.random.default_rng(10000+k)) for k in range(20)]; dev=[_noise(np.random.default_rng(10020+k)) for k in range(20)]; var,ch,rho,_,_=_covariance(cal,dev)
 responses={(b,a):_pmu(_load(PHYS/'results'/f'LOAD_BUS_{b}_A{("m" if a<0 else "")}{str(abs(a)).replace(".","p")}_R1.csv')[1],rows)[idx]-yn for b in BUSES for a in AMPS}
 rows_out=[]
 for model in ['W0_IDENTITY','W1_CHANNEL_COV','W2_SEPARABLE_AR1']:
  _,L,ld=_factor(model,var,ch,rho); ev_scores=[]; null_scores=[]; top=[]
  for b in BUSES:
   for a in AMPS:
    for k in range(N_NOISE_TEST):
     r=(responses[(b,a)]+_noise(np.random.default_rng(200000+b*1000+int(round(a*1000))+k))).reshape(-1); p,mu,v,_,_,_=_score(r,D,L,ld,.05); ev_scores.append(1-p[0]); top.append(int(BUSES[np.argmax(p[1:])])==b)
  for k in range(N_NOISE_TEST*16):
   r=_noise(np.random.default_rng(300000+k)).reshape(-1); p,_,_,_,_,_=_score(r,D,L,ld,.05); null_scores.append(1-p[0])
  yy=np.r_[np.ones(len(ev_scores)),np.zeros(len(null_scores))]; ss=np.r_[ev_scores,null_scores]
  rows_out.append({'model':model,'event_AUROC':roc_auc_score(yy,ss),'event_FPR_at_.5':float(np.mean(np.asarray(null_scores)>=.5)),'event_FNR_at_.5':float(np.mean(np.asarray(ev_scores)<.5)),'source_top1':float(np.mean(top)),'event_cases':len(ev_scores),'no_event_cases':len(null_scores)})
 pd.DataFrame(rows_out).to_csv(OUT/'load_bayes_model_comparison.csv',index=False); print(pd.DataFrame(rows_out).to_string(index=False))
if __name__=='__main__': main()
