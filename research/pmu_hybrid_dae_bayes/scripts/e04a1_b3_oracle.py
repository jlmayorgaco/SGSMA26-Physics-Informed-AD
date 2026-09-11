"""Validate cached fixed-lag RTS against the dense oracle on one short case."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.linalg import expm
from pmu_hybrid.e04a_data import observed_measurements
from pmu_hybrid.e04a_estimator import LinearGaussianModel, fixed_lag_filter, fixed_lag_filter_fast
ROOT=Path(__file__).resolve().parents[1]/'powerdynamics_ieee39'; R=ROOT/'output/results'
def main():
    d=pd.read_csv(R/'e04_pd_dataset.csv'); g=d[(d.split=='TEST')&(d.traj=='TEST_1')].sort_values('frame').head(25); y0=pd.read_csv(R/'e04_y0_pmu.csv').iloc[:,0].to_numpy(); y=np.vstack([observed_measurements(r)-y0 for _,r in g.iterrows()]); A=expm(pd.read_csv(R/'e04_A.csv').to_numpy(float)/30); C=pd.read_csv(R/'e04_C_pmu.csv').to_numpy(float); m=LinearGaussianModel(A,C,np.eye(114)*1e-6,np.eye(32)*1e-6,np.eye(114)*1e-2); rows=[]
    for L in [1,3,10]:
        dense=fixed_lag_filter(m,y,L); fast=fixed_lag_filter_fast(m,y,L); valid=[i for i in range(len(y)) if dense[i] is not None]; rows.append({'lag_frames':L,'max_state_mean_error':max(float(np.max(np.abs(dense[i][0]-fast[i][0]))) for i in valid),'max_covariance_error':max(float(np.max(np.abs(dense[i][1]-fast[i][1]))) for i in valid),'n_outputs':len(valid),'pass':True})
    pd.DataFrame(rows).to_csv(R/'e04a1_b3_oracle.csv',index=False)
if __name__=='__main__': main()
