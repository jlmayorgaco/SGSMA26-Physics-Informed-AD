"""Finalize confirmatory artifacts without generating additional TDS trajectories."""
from pathlib import Path
import json, math, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parents[1]; PD=HERE/"powerdynamics_ieee39"
OUT=PD/"output/multi_likelihood_confirmatory_v1"; RES=OUT/"results"; PLOTS=OUT/"plots"; PLOTS.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(HERE))
from scripts import multi_likelihood_confirmatory_v1 as c
from scripts import multi_likelihood_calibration_v1 as mlc

def old_grid_comparison():
    D,Q,qij,yn,idx,rows,L,S,Dw,Qw,qijw=c.frozen(); nm=pd.read_csv(RES/"noise_manifest.csv"); rows_out=[]
    for rr in nm[nm.regime!="H0"].head(48).itertuples():
        rw=c.wht(c.load_obs(rr.physical_path,yn,idx,rows)+c.noise(rr.noise_seed),L)
        aa,bb,ww,ll,zold,_=mlc.grid_posterior(rw,int(rr.source_i),int(rr.source_j),Dw,Qw,qijw,grid_values=mlc.GRID)
        zn,An,Bn,Wn,_=c.support_evidence(rw,int(rr.source_i),int(rr.source_j),Dw,Qw,qijw,n=11)
        mo, mj, vi, vj, _ = c.moments(An,Bn,Wn)
        old_m=float((ww*aa).sum()); old_v=float((ww*(aa-old_m)**2).sum());
        rows_out.append({"case_id":rr.case_id,"noise_seed":rr.noise_seed,"logZ_old31":zold,"logZ_adaptive":zn,"logZ_diff":zn-zold,"mean_i_old":old_m,"mean_i_adaptive":mo,"mean_diff_i":mo-old_m,"sd_old":math.sqrt(max(old_v,0)),"sd_adaptive":math.sqrt(max(vi,0)),"mean_diff_over_sd":abs(mo-old_m)/max(math.sqrt(vi),1e-12),"label":"SECONDARY_RETROSPECTIVE_DIAGNOSTIC"})
    pd.DataFrame(rows_out).to_csv(RES/"old_grid_vs_adaptive.csv",index=False)

def plots():
    a=pd.read_csv(RES/"true_support_calibration.csv"); cdf=pd.read_csv(RES/"full_bayes_cardinality.csv"); s=pd.read_csv(RES/"full_bayes_support.csv"); q=pd.read_csv(RES/"quadrature_stability.csv"); cf=pd.read_csv(RES/"conditional_fisher.csv"); nd=pd.read_csv(RES/"nested_manifold_distance.csv")
    def save(name): plt.tight_layout(); plt.savefig(PLOTS/name,dpi=140); plt.close()
    g=a.groupby("regime")[["cover95_i","cover95_j"]].mean(); g.plot(kind="bar",ylim=(0,1.05),title="Level A true-support 95% coverage"); plt.axhline(.95,color="k",ls="--"); save("true_support_coverage.png")
    inc=s.groupby("bus").inclusion_probability.mean(); inc.plot(kind="bar",title="Model-averaged inclusion probability"); save("model_averaged_coverage.png")
    pd.crosstab(cdf.pred_M,2).plot(kind="bar",legend=False,title="Predicted cardinality (confirmatory subset)"); save("cardinality_confusion.png")
    a.assign(abs_amp=np.hypot(a.amplitude_i,a.amplitude_j)).plot.scatter(x="abs_amp",y="mean_i",title="Support amplitude vs posterior mean"); save("support_vs_amplitude.png")
    q.set_index("case_id")[["logZ_order15","logZ_order21","logZ_order31"]].head(24).plot(title="Quadrature order refinement"); save("old_vs_adaptive_evidence.png")
    q.set_index("case_id")[["sd_diff_21_31"]].plot(title="Adaptive posterior-SD refinement"); save("old_vs_adaptive_posterior_sd.png")
    cf.plot.scatter(x="I_j_given_i",y="coherence",title="Conditional information (descriptive)"); save("conditional_information_vs_merge.png")
    nd.plot.scatter(x="distance_pair_to_M1",y="merge_label",title="Nested-manifold distance (descriptive)"); save("nested_distance_vs_merge.png")
    plt.bar(["best","multiplicity","volume"],[0,-math.log(24),0]); plt.title("24-support multiplicity reference"); save("multiplicity_decomposition.png")

def main():
    old_grid_comparison(); plots()
    # Repair the human-readable report after the execution script completed.
    summary=pd.read_csv(RES/"multi_likelihood_confirmatory_summary.csv").iloc[0].to_dict() if (RES/"multi_likelihood_confirmatory_summary.csv").exists() else {}
    a=pd.read_csv(RES/"true_support_calibration.csv"); q=pd.read_csv(RES/"quadrature_stability.csv"); old=pd.read_csv(RES/"old_grid_vs_adaptive.csv")
    old_line="Median secondary old-grid/adaptive log-evidence difference: {:.4g}; median mean-shift in adaptive SD units: {:.4g}.".format(float(old.logZ_diff.median()),float(old.mean_diff_over_sd.median()))
    lines=["# MULTI-LIKELIHOOD-CONFIRMATORY-V1","",json.dumps(summary,indent=2),"","## Contract","","384 fresh physical trajectories (24 frozen supports x 4 regimes x 4 signs), each reused for 20 noise seeds (7,680 rows); 200 independent H0 CAL rows. Levels A/B/C use a deterministic five-seed-per-trajectory analysis subset to bound runtime; the complete noise manifest remains frozen.","","Integration uses local adaptive Gauss-Hermite centered at a damped Gauss-Newton posterior mode, with log-domain evidence and the frozen unbounded Gaussian prior. Order refinement and the secondary 31-point grid comparison are diagnostic only. No V1/V2 TEST row entered fitting or selection.","","## Level A coverage by regime","",a.groupby('regime')[['cover95_i','cover95_j']].mean().to_markdown(),"","## Quadrature","",q.describe().to_markdown(),"",old_line,"","## Scope","","The prospective hypothesis space contains 24 of 120 doubles; GLOBAL_SUPPORT_RECOVERY=NOT_ESTABLISHED. ANALYTIC_DAE_TANGENT remains PENDING. No multi-event beyond simultaneous doubles, ML, or future action was executed."]
    (OUT/"reports/multi_likelihood_confirmatory_v1.md").write_text("\n".join(lines),encoding="utf-8")
    import shutil
    for f in RES.glob("*.csv"): shutil.copy2(f,PD/"output/results"/f.name)
    shutil.copy2(OUT/"reports/multi_likelihood_confirmatory_v1.md",PD/"output/reports"/"multi_likelihood_confirmatory_v1.md")
if __name__=="__main__": main()
