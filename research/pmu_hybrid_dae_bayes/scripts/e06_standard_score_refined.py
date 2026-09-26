from pathlib import Path
import pandas as pd, numpy as np
from e06_standard_score import loadm, filt, cplx, mm, CASES, R
def main():
 A,C,L,y0,h0=loadm(); man=pd.read_csv(R/"e06_standard_refined_manifest.csv"); rows=[]
 for _,r in man.iterrows():
  tr=pd.read_csv(CASES/(r.case_id+"_trajectory.csv")); y=tr[[f"pmu_{i}" for i in range(1,33)]].to_numpy(float); t=cplx(tr[[f"hidden_{i}" for i in range(1,63)]].to_numpy(float).reshape(len(tr),31,2)); x,_,_=filt(A,C,y0,y); z=cplx((h0[None,:]+x@L.T).reshape(len(tr),31,2)); q=mm(z,t); rows.append({"case_id":r.case_id,"family":r.family,"m":r.m,"seed":r.seed,"excitation":r.excitation,"TVE_percent":q["TVE_percent"],"angle_RMSE_deg":q["angle_RMSE_deg"]})
 out=pd.DataFrame(rows); out.to_csv(R/"e06_standard_refined_per_case.csv",index=False); main=pd.read_csv(R/"e06_standard_summary.csv"); base=float(main.query("method=='B2' and m==0").median_TVE_percent.mean()); br=[]
 for f,g in out.groupby("family"):
  med=g.groupby("m").TVE_percent.median()
  for th in (2,5,10):
   hit=med[med>=th*base]; hi=float(hit.index.min()) if len(hit) else np.nan; old=main[(main.family==f)&(main.method=="B2")]; lo=float(old[old.m<hi].m.max()) if len(hit) and len(old[old.m<hi]) else (float(med.index[med.index<hi].max()) if len(hit) and np.any(med.index<hi) else np.nan); br.append({"family":f,"threshold":f"{th}x","lower_m":lo,"upper_m":hi,"interval":"not_reached" if np.isnan(hi) else f"[{lo if not np.isnan(lo) else 0},{hi}]"})
 pd.DataFrame(br).to_csv(R/"e06_standard_refined_breakpoints.csv",index=False); print("refined",len(out),"cases")
if __name__=="__main__": main()
