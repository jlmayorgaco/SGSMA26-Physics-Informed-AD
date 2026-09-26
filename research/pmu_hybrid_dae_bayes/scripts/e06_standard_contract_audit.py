from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/"powerdynamics_ieee39"; R=ROOT/"output/results"; REP=ROOT/"output/reports"; REP.mkdir(parents=True,exist_ok=True)
rows=[
 ("trajectory duration","3.0 s (91 frames)","0.5 s (31 frames)","3.0 s (91 frames)"),
 ("sample rate","30 Hz","30 Hz","30 Hz"),
 ("measurement noise","E04 frozen PMU contract; R=1e-6 I32","none in E06-D physical trajectory","E04 frozen PMU contract; R=1e-6 I32"),
 ("PMU noise magnitude","sigma=sqrt(1e-6) per channel in estimator; dataset noise frozen","zero added noise","sigma=sqrt(1e-6) per channel in estimator; deterministic plant, frozen noise injection"),
 ("process disturbance","nominal E04 dataset excitation; Q=1e-6 I114","+0.02 p_ref at plant start","fixed E-A..E-D normal-operation perturbations; Q=1e-6 I114"),
 ("initial prior","P0=1e-2 I114","P0 not used for oracle scoring","P0=1e-2 I114"),
 ("burn-in","none","none","none"),
 ("metric aggregation","hidden 31, trajectory/frame aggregation","hidden 31, trajectory/frame aggregation","hidden 31, per-case + mean/median/p95 + bootstrap CI"),
 ("measurement channels","8 PMUs, 32 real channels, validated synthetic terminals","same","same"),
 ("evaluation window","all 91 frames","all 31 frames","all 91 frames"),
]
df=pd.DataFrame(rows,columns=["property","E04-A","E06-D","E06-STANDARD"]); df.to_csv(R/"e06_standard_contract_audit.csv",index=False)
table=df.to_markdown(index=False)
(REP/"e06_standard_contract_audit.md").write_text(f"""# E06 STANDARD contract audit\n\nThe E06-D oracle run was intentionally a short, low-noise mechanism confirmation, not a numerically comparable replacement for E04-A. Its ~5e-6% nominal TVE is therefore expected to be much smaller than the E04-A B2 reference (~0.009634%). E06 STANDARD adopts the E04-A statistical window and estimator contract while retaining physically rebuilt PowerDynamics plants.\n\n{table}\n\nThe STANDARD baseline is frozen as `E06_STANDARD_BASELINE_V1`; mismatch ratios use the STANDARD m=0 reference only.\n""",encoding="utf-8")
print(df.to_string(index=False))
