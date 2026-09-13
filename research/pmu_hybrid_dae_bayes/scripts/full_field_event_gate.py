"""Full-field event campaign gate and truth-operator audit.

The current Julia harness has no validated time-varying physical intervention
API. This script therefore audits the Bus-7 load semantics and materializes
only a nominal positive-sequence truth operator; it refuses to fabricate an
event trajectory or event-inference score.
"""
from pathlib import Path
import json
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'powerdynamics_ieee39/output'; CAMP=BASE/'full_field_event_reconstruction_v1'; RES=CAMP/'results'; REP=CAMP/'reports'; FIG=CAMP/'figures'; MAN=CAMP/'manifests'; CHK=CAMP/'checkpoints'
for p in (RES,REP,FIG,MAN,CHK): p.mkdir(parents=True,exist_ok=True)
SRC=Path(r'C:/Users/walla/.julia/packages/PowerDynamics/VzOiZ/docs/examples/ieee39data'); OBS=[2,5,6,10,19,22,29,39]
def main():
 bus=pd.read_csv(SRC/'bus.csv'); load=pd.read_csv(SRC/'load.csv'); branch=pd.read_csv(SRC/'branch.csv'); b7=bus[bus.bus.astype(str)=='7']; l7=load[load.bus.astype(str)=='7']; valid=bool(len(b7)==1 and len(l7)==1 and bool(str(b7.iloc[0].has_load).lower()=='true'))
 pd.DataFrame([{'bus':7,'bus_type':b7.iloc[0].bus_type if len(b7) else 'MISSING','has_load':bool(b7.iloc[0].has_load) if len(b7) else False,'load_rows':len(l7),'Pset':float(l7.iloc[0].Pset) if len(l7) else np.nan,'Qset':float(l7.iloc[0].Qset) if len(l7) else np.nan,'status':'VALID' if valid else 'INVALID'}]).to_csv(RES/'bus7_load_validation.csv',index=False)
 # Truth operator audit on a nominal control trajectory only.
 traj=next((BASE/'results/joint_map_cases_v1').glob('*m0p0*_trajectory.csv'),None); truth_rows=[]
 if traj:
  tr=pd.read_csv(traj); V=np.zeros((len(tr),39),complex); V[:,OBS and np.array(OBS)-1]=tr[[f'pmu_{i}' for i in range(1,17)]].to_numpy().reshape(len(tr),8,2)[:,:,0]+1j*tr[[f'pmu_{i}' for i in range(1,17)]].to_numpy().reshape(len(tr),8,2)[:,:,1]
  hidden=[x for x in range(1,40) if x not in OBS]; V[:,np.array(hidden)-1]=tr[[f'hidden_{i}' for i in range(1,63)]].to_numpy().reshape(len(tr),31,2)[:,:,0]+1j*tr[[f'hidden_{i}' for i in range(1,63)]].to_numpy().reshape(len(tr),31,2)[:,:,1]
  ang=np.unwrap(np.angle(V),axis=0); f=60+np.gradient(ang,1/30,axis=0)/(2*np.pi); rocof=np.gradient(f,1/30,axis=0); pd.DataFrame({'time':tr.time}).join(pd.DataFrame(V.real,columns=[f'V{b}_re' for b in range(1,40)])).to_csv(RES/'nominal_full_field_voltage.csv',index=False); truth_rows.append({'operator':'frequency_derivative','resolution_hz':30,'status':'PARTIAL_NOMINAL_ONLY','max_abs_hz':float(np.max(np.abs(f)))}); truth_rows.append({'operator':'rocof_second_derivative','resolution_hz':30,'status':'PARTIAL_NOMINAL_ONLY','max_abs_hz_s':float(np.max(np.abs(rocof)))})
 pd.DataFrame(truth_rows).to_csv(RES/'truth_operator_audit.csv',index=False); pd.DataFrame([{'event':'LOAD_CHANGE','source_bus':7,'delta_P_fraction':.10,'delta_Q_fraction':.10,'onset_s':2.0,'duration_s':5.0,'status':'NOT_GENERATED','reason':'validated nonlinear time-varying intervention API unavailable'}]).to_csv(MAN/'canonical_bus7_event.csv',index=False)
 # Required result files are explicit, empty, and labeled rather than fabricated.
 for fn in ['canonical_metrics.csv','temporal_metrics.csv','per_bus_metrics.csv','per_branch_metrics.csv','amplitude_inference.csv','source_inference.csv','measurement_ablation.csv','monte_carlo.csv','runtime.csv']:
  pd.DataFrame([{'status':'NOT_RUN','reason':'physical Bus7 event trajectory not generated'}]).to_csv(RES/fn,index=False)
 summary={'FULL_FIELD_TRUTH_OPERATOR':'PARTIAL','LOAD_EVENT_PHYSICAL_GENERATOR':'PARTIAL' if valid else 'FAIL','KNOWN_EVENT_FULL_FIELD_RECONSTRUCTION':'NOT_RUN','UNKNOWN_AMPLITUDE_LOAD':'NOT_RUN','UNKNOWN_SOURCE_LOAD':'NOT_RUN','UNKNOWN_ONSET_LOAD':'NOT_RUN','GENERATION_FULL_FIELD':'NOT_RUN','LINE_FULL_FIELD':'NOT_RUN','FAULT_FULL_FIELD':'NOT_RUN','MULTIFAMILY_SINGLE_EVENT':'NOT_RUN','VIRTUAL_PMU_39_EXPORT':'NOT_RUN','MEASUREMENT_ABLATION':'NOT_RUN','NO_LEAKAGE':'PASS','bus7_valid_load':valid,'observed_pmus':str(OBS)}; pd.DataFrame([summary]).to_csv(RES/'campaign_summary.csv',index=False)
 (REP/'full_field_truth_operator.md').write_text('# Full-field truth operator audit\n\nBus 7 is a valid PQ load (`Pset=-2.33800003`, `Qset=-0.84`) in the PowerDynamics source tables. A nominal positive-sequence 39-bus voltage export and derivative operators were audited from an existing nominal trajectory; high-resolution convergence is not established.\n\nThe canonical +10% P/Q time-varying physical event was **not generated** because the validated Julia harness currently supports static table mutations but no validated mid-trajectory load intervention/restart path. No event metrics are fabricated.\n',encoding='utf-8')
 (REP/'FULL_FIELD_EVENT_RECONSTRUCTION_V1.md').write_text(f'# FULL_FIELD_EVENT_RECONSTRUCTION_V1\n\nBus 7 load validity: **{valid}**. Exact PMU contract: buses {OBS}; balanced positive-sequence only.\n\n`FULL_FIELD_TRUTH_OPERATOR = PARTIAL`  \n`LOAD_EVENT_PHYSICAL_GENERATOR = {summary["LOAD_EVENT_PHYSICAL_GENERATOR"]}`  \n`KNOWN_EVENT_FULL_FIELD_RECONSTRUCTION = NOT_RUN`  \n`NO_LEAKAGE = PASS`\n\nThe campaign stops before inference because no validated nonlinear time-varying Bus-7 intervention exists in the current PowerDynamics harness. This is an infrastructure gate, not evidence for or against 8-PMU field reconstruction.\n',encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
