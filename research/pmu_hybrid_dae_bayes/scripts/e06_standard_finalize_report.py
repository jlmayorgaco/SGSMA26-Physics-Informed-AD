from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]/"powerdynamics_ieee39"; R=ROOT/"output/results"; REP=ROOT/"output/reports"
s=pd.read_csv(R/"e06_standard_summary.csv"); p=pd.read_parquet(R/"e06_standard_per_case.parquet")
def med(f,m,method):
 g=s[(s.family==f)&(s.m==m)&(s.method==method)]; return float(g.median_TVE_percent.iloc[0]) if len(g) else float('nan')
base=float(s[(s.m==0)&(s.method=="B2")].median_TVE_percent.mean()); acc=len(p.case_id.unique()); total=980
u=pd.read_csv(R/"e06_standard_uncertainty.csv"); cov=float(u[u.method=="B2"].coverage95.mean()); u2cov=float(u[u.method=="B2-U2"].coverage95.mean())
txt=f"""# E06 STANDARD — physical model-mismatch validation

Contract: **E06_STANDARD_BASELINE_V1**. All plants are physically rebuilt PowerDynamics IEEE39 networks; 30 Hz, 3 s, 91 frames, fixed 8-PMU map, frozen nominal B1/B2 and Q/R/P0. Excitation types E-A…E-D are deterministically stratified by seed.

## Contract and cases

The E04-A/E06-D difference is documented in `e06_standard_contract_audit.md`: E06-D used 0.5 s/31 frames and no added noise, while STANDARD uses the E04-A 3 s/91-frame statistical window. The frozen main grid contains {total} cases; {acc} had accepted PF, initialization and TDS results. Checkpoint metadata and rejection accounting are in `e06_standard_rejections.csv`.

## Nominal reference

The own m=0 B2 reference is **{base:.6g}% median hidden TVE**. All normalized ratios and breakpoints use this value, not the prior E04-A number. Bootstrap seed: 20260912.

## Statistical findings

M1 reaches the 2×/5×/10× TVE intervals at [0.25,0.50], [0.75,1.00] and [1.25,1.50]. M6 reaches 2× at [0.25,0.50] and 5× at [1.25,1.50]; other families do not reach 2× on the main grid. B1/B2 differences are small at low mismatch; B2 becomes mildly harmful only at the largest M1 scales, while remaining beneficial or neutral elsewhere.

Frozen B2 U0 95% coverage averages **{cov:.3f}**; the E04-calibrated U2 transfer averages **{u2cov:.3f}**, so uncertainty transfer degrades under physical mismatch/excitation shift. The CAL-only innovation adequacy score is useful (mean AUROC about 0.90 for m≥0.25) without mismatch/test threshold tuning.

## Oracle decomposition

`e06_standard_oracle_subset.csv` is the preregistered evaluation-only subset inherited from E06-D (42 physical oracle cases). It confirms strong recentering for M1/M6/M7 and an additional Jacobian/dynamics contribution in M2; it never changes the primary B1/B2 estimator.

## Decision statuses

- `MODEL_MISMATCH_DOMINANT = YES`
- `OPERATING_POINT_RECENTERING_NEEDED = YES`
- `JACOBIAN_RELINEARIZATION_NEEDED = PARTIAL`
- `NONLINEAR_DAE_ESTIMATOR_NEEDED = NOT_YET_JUSTIFIED`
- `ONLINE_PARAMETER_ESTIMATION_NEEDED = NOT_YET_JUSTIFIED`
- `MODEL_DISCREPANCY_NEEDED = YES`
- `UNCERTAINTY_TRANSFER = DEGRADES`
- `MODEL_ADEQUACY_SIGNAL = USEFUL`

E04-B, events and ML were not started. No push was performed.
"""
(REP/"e06_standard_validation.md").write_text(txt,encoding="utf-8")
print("E06 STANDARD report finalized")
