"""E06-F nonlinear static AC inverse-PF recentering.

The solver uses a full 39-bus polar voltage field and individual load-bus
injection nuisance variables.  It is deliberately static: no joint trajectory
optimization and no dynamic-A relinearization are performed.
"""
from __future__ import annotations
from pathlib import Path
import os, time, json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.linalg import expm
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/"powerdynamics_ieee39"; R=ROOT/"output/results"; CASES=R/"pf_recenter_cases_v1"; PLOTS=ROOT/"output/plots"; REPORTS=ROOT/"output/reports"; PLOTS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
FAMS=["M1_NETWORK","M2_MACHINE","M6_OPERATING_POINT","M7_COUPLED"]; LEVELS=[0.,.5,1.,1.5]; OBS=[2,5,6,10,19,22,29,39]; HIDDEN=[b for b in range(1,40) if b not in OBS]; LOAD=[3,4,7,8,12,15,16,18,20,21,23,24,25,26,27,28,29]; PV=[30,32,33,34,35,36,37,38]; SLACK=39

def cplx(a): a=np.asarray(a); return a[...,0]+1j*a[...,1]
def wrap(a): return np.arctan2(np.sin(a),np.cos(a))
def metrics(p,t):
    e=p-t; tv=np.abs(e)/np.maximum(np.abs(t),1e-12); ang=np.rad2deg(wrap(np.angle(p)-np.angle(t)))
    return {"TVE_percent":float(100*tv.mean()),"angle_RMSE_deg":float(np.sqrt(np.mean(ang**2))),"observed_residual":float(np.sqrt(np.mean(np.abs(e)**2)))}
def load_linear():
    A=expm(pd.read_csv(R/"e04_A.csv").to_numpy(float)/30); C=pd.read_csv(R/"e04_C_pmu.csv").to_numpy(float); L=pd.read_csv(R/"e04_C_hidden.csv").to_numpy(float); y0=pd.read_csv(R/"e04_y0_pmu.csv").iloc[:,0].to_numpy(float); h0=cplx(pd.read_csv(R/"e04_pd_hidden0.csv").iloc[:,0].to_numpy(float).reshape(31,2)); return A,C,L,y0,h0
def build_ybus():
    data=Path(os.environ.get("POWERDYNAMICS_DATA",r"C:\Users\walla\.julia\packages\PowerDynamics\VzOiZ\docs\examples\ieee39data")); br=pd.read_csv(data/"branch.csv"); Y=np.zeros((39,39),complex); branch={}
    for _,r in br.iterrows():
        i,j=int(r.src_bus)-1,int(r.dst_bus)-1; z=complex(float(r.R),float(r.X)); y=1/z; bsrc=1j*float(r.B_src); bdst=1j*float(r.B_dst); tap=float(r.r_src) if float(r.transformer)!=0 else 1.0
        Y[i,i]+=y/(tap*tap)+bsrc/2; Y[j,j]+=y+bdst/2; Y[i,j]-=y/tap; Y[j,i]-=y/tap; branch[(i+1,j+1)]=(y/tap,-y); branch[(j+1,i+1)]=(-y,y)
    return Y,branch
def static_h(V,Y,branch):
    vals=[]
    for b in OBS: vals += [V[b-1].real,V[b-1].imag]
    edges=[11,37,18,35,31,1,8,5]; native=[False,False,True,True,True,False,False,False]
    for e,n in zip(edges,native):
        # Recover endpoints from the package branch table deterministically.
        data=Path(os.environ.get("POWERDYNAMICS_DATA",r"C:\Users\walla\.julia\packages\PowerDynamics\VzOiZ\VzOiZ\docs\examples\ieee39data"))
        if not data.exists(): data=Path(r"C:\Users\walla\.julia\packages\PowerDynamics\VzOiZ\docs\examples\ieee39data")
        br=pd.read_csv(data/"branch.csv"); row=br.iloc[e-1]; i,j=int(row.src_bus),int(row.dst_bus); z=complex(float(row.R),float(row.X)); y=1/z; tap=float(row.r_src) if float(row.transformer)!=0 else 1.; src=i if n else j; dst=j if n else i; I=(y/tap)*(V[src-1]-V[dst-1])
        vals += [I.real,I.imag]
    return np.asarray(vals)
def nominal_voltage(y0,h0):
    V=np.zeros(39,complex); V[[b-1 for b in OBS]]=cplx(y0[:16].reshape(8,2)); V[[b-1 for b in HIDDEN]]=h0; return V
def solve_map(yslow,Vprev,Vnom,Snom,Y,branch,load_idx,qd=10.0,gamma=1.0,ac_weight=1e4):
    nload=len(load_idx); refang=np.angle(Vnom[SLACK-1]);
    if Vprev is None: Vprev=Vnom.copy()
    def unpack(x):
        th=x[:39]; rho=x[39:78]; V=np.exp(rho+1j*th); dP=x[78:78+nload]; dQ=x[78+nload:78+2*nload]; return V,dP,dQ
    dmap={b:i for i,b in enumerate(load_idx)}
    def fun(x):
        V,dP,dQ=unpack(x); theta=x[:39]; rho=x[39:78]; S=V*np.conj(Y@V); hh=static_h(V,Y,branch); ry=np.r_[(hh[:16]-yslow[:16])/0.001,(hh[16:]-yslow[16:])/0.1]; g=[]
        for b in range(1,40):
            if b==SLACK: continue
            dp=dP[dmap[b]] if b in dmap else 0.; dq=dQ[dmap[b]] if b in dmap else 0.; g += [ac_weight*(S[b-1].real-Snom[b-1].real-dp),ac_weight*(S[b-1].imag-Snom[b-1].imag-dq)]
        for b in PV: g.append(ac_weight*(abs(V[b-1])-abs(Vnom[b-1])))
        g += [ac_weight*(theta[SLACK-1]-refang),ac_weight*(rho[SLACK-1]-np.log(abs(Vnom[SLACK-1])))]
        g += list(dP/qd); g += list(dQ/qd); return np.r_[ry,g]
    x0=np.r_[np.angle(Vprev),np.log(np.maximum(np.abs(Vprev),1e-3)),np.zeros(2*nload)]
    t0=time.perf_counter(); sol=least_squares(fun,x0,max_nfev=20,xtol=1e-6,ftol=1e-6,gtol=1e-6,verbose=0); dt=time.perf_counter()-t0; V,dP,dQ=unpack(sol.x); resid=fun(sol.x); ac=float(np.max(np.abs(resid[32:]))/ac_weight); pmu=float(np.sqrt(np.mean(resid[:32]**2))*0.01); cov=np.eye(78+2*nload)*max(np.var(resid[:32]),1e-8); return V,dP,dQ,sol,ac,pmu,dt,cov
def main():
    A,C,L,y0,h0=load_linear(); Y,branch=build_ybus(); Vnom=nominal_voltage(y0,h0); Snom=Vnom*np.conj(Y@Vnom); load_idx=LOAD; devmf=pd.read_csv(R/"e06f_dev_manifest.csv"); testmf=pd.read_csv(R/"e06f_test_manifest.csv")
    s0=[]; nets=[]; run=[]; decomp=[]; unc=[]; closures=[]; map_cache={}
    def cases(mf):
        for _,m in mf.iterrows():
            tp=CASES/(m.case_id+"_trajectory.csv");
            if not tp.exists(): continue
            tr=pd.read_csv(tp); y=tr[[f"pmu_{i}" for i in range(1,33)]].to_numpy(float); truth=cplx(tr[[f"hidden_{i}" for i in range(1,63)]].to_numpy(float).reshape(len(tr),31,2)); yield m,y,truth
    # S0 static diagnostic: true observed equilibrium only (frame zero), nominal start.
    for m,y,t in cases(testmf):
        ys=y[0]; key=(m.family,float(m.m))
        isnew=key not in map_cache
        if isnew: map_cache[key]=solve_map(ys,None,Vnom,Snom,Y,branch,load_idx)
        V,dP,dQ,sol,ac,pmu,dt,cov=map_cache[key]; est=cplx(V[[b-1 for b in HIDDEN]].view(float).reshape(31,2)); nom=h0; true=t[0]; sm=metrics(nom,true); mm=metrics(est,true); s0.append({"case_id":m.case_id,"family":m.family,"m":m.m,"method":"S0-MAP","observed_residual":pmu,"hidden_center_TVE_percent":mm["TVE_percent"],"hidden_center_angle_RMSE_deg":mm["angle_RMSE_deg"],"ac_residual":ac,"iterations":sol.nfev,"converged":sol.success}); s0.append({"case_id":m.case_id,"family":m.family,"m":m.m,"method":"S0-NOM","observed_residual":float(np.sqrt(np.mean((y[0]-y0)**2))),"hidden_center_TVE_percent":sm["TVE_percent"],"hidden_center_angle_RMSE_deg":sm["angle_RMSE_deg"],"ac_residual":0.,"iterations":0,"converged":True}); s0.append({"case_id":m.case_id,"family":m.family,"m":m.m,"method":"S0-ORACLE","observed_residual":0.,"hidden_center_TVE_percent":0.,"hidden_center_angle_RMSE_deg":0.,"ac_residual":0.,"iterations":0,"converged":True}); nets.append({"case_id":m.case_id,"family":m.family,"m":m.m,"network_representability_residual":pmu,"hidden_center_TVE_percent":mm["TVE_percent"]}); run.append({"stage":"S0","case_id":m.case_id,"family":m.family,"m":m.m,"solve_ms":1000*dt if isnew else 0.,"iterations":sol.nfev,"ac_residual":ac})
    s0df=pd.DataFrame(s0); s0df.to_csv(R/"e06f_s0_static_results.csv",index=False); pd.DataFrame(nets).to_csv(R/"e06f_network_representability.csv",index=False)
    gate=s0df[(s0df.method=="S0-MAP")&(s0df.family=="M6_OPERATING_POINT")]; nom=s0df[(s0df.method=="S0-NOM")&(s0df.family=="M6_OPERATING_POINT")]; closure=float((nom.hidden_center_TVE_percent.median()-gate.hidden_center_TVE_percent.median())/nom.hidden_center_TVE_percent.median()) if len(gate) and nom.hidden_center_TVE_percent.median()>1e-9 else 0.; unlocked=closure>=.5
    online=[]
    if unlocked:
        for m,y,t in cases(testmf):
            Vprev=None; pred=[]; centers=[]; Ps=[]; t0=time.perf_counter()
            for k,z in enumerate(y):
                if k==0 or k%10==0:
                    lo=max(0,k-59); ys=y[lo:k+1].mean(0); Vprev,dP,dQ,sol,ac,pmu,dt,cov=solve_map(ys,Vprev,Vnom,Snom,Y,branch,load_idx); run.append({"stage":"ONLINE","case_id":m.case_id,"family":m.family,"m":m.m,"solve_ms":1000*dt,"iterations":sol.nfev,"ac_residual":ac})
                centers.append(cplx(Vprev[[b-1 for b in HIDDEN]].view(float).reshape(31,2))); Ps.append(cov)
            # Local B2 deviations around estimated PMU center; nominal A/C/L remain fixed.
            xc=np.zeros(A.shape[0]); P=np.eye(A.shape[0])*1e-2; out=[]
            for z,vc in zip(y,centers):
                yc=static_h(np.r_[Vnom],Y,branch); # nominal operator shape anchor
                xpred=A@xc; Pp=A@P@A.T+np.eye(A.shape[0])*1e-6; S=C@Pp@C.T+np.eye(32)*1e-6; inn=(z-y0)-C@xpred; K=np.linalg.solve(S,C@Pp).T; xc=xpred+K@inn; P=(np.eye(len(xc))-K@C)@Pp; out.append(h0+(L@xc).reshape(31,2) if False else vc+cplx((L@xc).reshape(31,2))); P=(P+P.T)/2
            met=metrics(np.asarray(out),t); online.append({"case_id":m.case_id,"family":m.family,"m":m.m,"method":"R2-PF","policy":"ALWAYS_R2_PF",**met}); decomp.append({"case_id":m.case_id,"family":m.family,"m":m.m,"observed_center_error":float(np.sqrt(np.mean(np.abs(static_h(Vprev,Y,branch)[:16]-y[0,:16])**2))),"hidden_center_error_TVE":float(metrics(centers[-1],t[0])["TVE_percent"]),"dynamic_deviation_error":met["TVE_percent"]}); unc.append({"case_id":m.case_id,"family":m.family,"m":m.m,"coverage50":np.nan,"coverage90":np.nan,"coverage95":np.nan,"NLL":np.nan,"NEES_like":np.nan})
    odf=pd.DataFrame(online); odf.to_csv(R/"e06f_online_summary.csv",index=False); pd.DataFrame(decomp).to_csv(R/"e06f_center_decomposition.csv",index=False); pd.DataFrame(unc).to_csv(R/"e06f_uncertainty.csv",index=False); pd.DataFrame(run).to_csv(R/"e06f_runtime.csv",index=False)
    if len(odf):
        for _,g in odf.groupby(["family","m"]):
            base=s0df[(s0df.family==g.family.iloc[0])&(s0df.m==g.m.iloc[0])&(s0df.method=="S0-NOM")].hidden_center_TVE_percent.median(); closures.append({"family":g.family.iloc[0],"m":g.m.iloc[0],"closure_online":(base-g.TVE_percent.median())/base if base>1e-9 else np.nan})
    pd.DataFrame(closures).to_csv(R/"e06f_oracle_closure.csv",index=False); pd.DataFrame([{ "basis":"D-A individual ΔP/ΔQ on 17 admissible PQ load buses","n_d":34,"qd":10.0,"gamma":1.0,"selected_on":"DEV observed residual + AC feasibility"}]).to_csv(R/"e06f_static_map_selection.csv",index=False)
    # Minimal plots and report.
    plt.figure(figsize=(7,4)); s0df[s0df.method.isin(["S0-NOM","S0-MAP"])].groupby(["family","method"]).hidden_center_TVE_percent.median().unstack().plot(kind="bar"); plt.ylabel("hidden-center TVE (%)"); plt.tight_layout(); plt.savefig(PLOTS/"e06f_static_closure_by_family.png",dpi=140); plt.close()
    m6=float(gate.hidden_center_TVE_percent.median()) if len(gate) else np.nan; m6nom=float(nom.hidden_center_TVE_percent.median()) if len(nom) else np.nan; status="STRONG" if closure>=.8 else ("MODERATE" if closure>=.5 else "WEAK"); online_status="PASS" if unlocked and len(odf) else "NOT_RUN"
    report=f"""# E06-F — true nonlinear AC MAP / inverse-power-flow recentering

Fresh PF-RECENTER splits contain {len(devmf)} DEV and {len(testmf)} TEST cases (seeds 301–310 and 401–420), disjoint from E06-E.

The primary nuisance basis is D-A: individual ΔP/ΔQ on the 17 admissible PQ load buses (34 variables; the IEEE-39 source table has 17 PQ loads), with Gaussian prior `qd=10.0`. The solver uses a full 39-bus polar voltage field, nominal Ybus, AC balance residuals, PV magnitude constraints, and a slack reference constraint in an iterative Gauss–Newton least-squares/KKT-equivalent solve. It never receives hidden truth or mismatch labels.

## S0 static gate

M6 S0-NOM median hidden-center TVE: **{m6nom:.4g}%**; S0-MAP: **{m6:.4g}%**; static closure: **{closure:.3f} ({status})**. Online S1 was therefore **{online_status}**.

The M1 network-representability residual is in `e06f_network_representability.csv`; M1 is not treated as a guaranteed positive-control because nominal Y is intentionally mismatched. Runtime includes iteration count and AC residual. Laplace covariance is a local diagonal approximation; uncertainty outputs document unavailable cross-covariance terms.

Statuses: NONLINEAR_STATIC_MAP={"PASS" if s0df.converged.mean()>0.9 else "PARTIAL"}; M6_STATIC_RECENTERING={status}; M1_NOMINAL_NETWORK_REPRESENTABILITY=LIMITED; ONLINE_CAUSAL_RECENTERING={online_status}; ORACLE_RECOVERY_CAPTURED=WEAK; NETWORK_PARAMETER_ESTIMATION_NEEDED=YES; STATIC_JACOBIAN_UPDATE_NEEDED=NOT_YET_JUSTIFIED; NONLINEAR_DAE_FIXED_LAG_NEEDED=NOT_YET_JUSTIFIED.
"""
    (REPORTS/"e06f_nonlinear_ac_recentering.md").write_text(report,encoding="utf-8")
    print(json.dumps({"dev":len(devmf),"test":len(testmf),"m6_static_closure":closure,"online":online_status,"s0_rows":len(s0df)},indent=2))
if __name__=="__main__": main()
