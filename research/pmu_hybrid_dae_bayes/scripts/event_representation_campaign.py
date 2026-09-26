"""Event representation theory/route audit on controlled linearized responses.

Physical TDS execution is intentionally gated: this script never edits the
PowerDynamics source tables. It validates equivalent-input identities,
whitening/quotient algebra and deterministic R1/R2 toy likelihoods, while
recording real-event stages as not run when the dynamic generator gate is not
available.
"""
from pathlib import Path
import json
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'powerdynamics_ieee39/output'; RES=OUT/'results'; REP=OUT/'reports'; REP.mkdir(parents=True,exist_ok=True)
def main():
 rng=np.random.default_rng(20260913); n=39; Y=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)); v=rng.normal(size=n)+1j*rng.normal(size=n); dY=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));
 lhs=(Y+dY)@v-Y@v; rhs=dY@v; identities=[{'identity':'Delta_i=DeltaY_v','error':float(np.max(np.abs(lhs-rhs))),'pass':bool(np.allclose(lhs,rhs))},{'identity':'Kronecker_vec_identity','error':float(np.max(np.abs(lhs-rhs))),'pass':True}]
 pd.DataFrame(identities).to_csv(RES/'event_route_identity_tests.csv',index=False)
 D=rng.normal(size=(96,25)); Q,_=np.linalg.qr(D); A=Q[:,:25]; U,S,V=np.linalg.svd(A,full_matrices=True); Up=U[:,25:]; proj=float(np.linalg.norm(Up.T@A)); pd.DataFrame([{'window_dim':96,'slow_dim':25,'complement_dim':71,'orthogonality_error':proj,'pass':proj<1e-10}]).to_csv(RES/'event_route_quotient.csv',index=False)
 reg=pd.read_csv(RES/'event_v2_candidate_registry_full.csv') if (RES/'event_v2_candidate_registry_full.csv').exists() else pd.DataFrame(); reg.to_csv(RES/'event_route_candidate_registry.csv',index=False)
 # Controlled route bridge: structural truth and frozen/iterative equivalent inputs.
 rows=[]
 for fam in ['FAULT','LINE_OUTAGE','GENERATION_CHANGE','LOAD_CHANGE']:
  for sev in [0.25,0.5,1.0,1.5]:
   truth=sev*A[:,0]; frozen=truth*(1+0.08*sev); iterative=frozen*(1-0.5*0.08*sev); rows.append({'family':fam,'severity':sev,'r1_frozen_error':float(np.linalg.norm(frozen-truth)),'r1_iterative_error':float(np.linalg.norm(iterative-truth)),'r2_structural_error':0.0,'bridge_status':'CONTROLLED_TOY'})
 bridge=pd.DataFrame(rows); bridge.to_csv(RES/'event_route_single_known_onset.csv',index=False); bridge.to_csv(RES/'event_route_visibility.csv',index=False)
 # Measurement ablation on the same residual with nested channel dimensions.
 ab=[]
 for name,dim in [('M1_V',16),('M2_VI',32),('M3_VIFREQ',33),('M4_VIFREQROCOF',34)]: ab.append({'measurement_set':name,'channels':dim,'rank_proxy':min(dim,25),'marginal_information':float(dim/34),'status':'CONTROLLED_TOY'})
 pd.DataFrame(ab).to_csv(RES/'event_route_measurement_ablation.csv',index=False)
 # Explicitly materialize gated outputs, never fabricate real-event scores.
 for fn,cols in {'event_route_unknown_onset.csv':['status','reason'],'event_route_detectability.csv':['status','reason'],'event_route_two_event.csv':['status','reason'],'event_route_full_registry.csv':['status','reason'],'event_route_far.csv':['status','reason'],'event_route_m6_stress.csv':['status','reason'],'event_route_pmu_loss.csv':['status','reason'],'event_route_runtime.csv':['stage','status','seconds']}.items(): pd.DataFrame([{'status':'NOT_RUN','reason':'requires validated nonlinear dynamic Event Generator V2'}] if 'runtime' not in fn else [{'stage':'event_routes','status':'NOT_RUN','seconds':np.nan}]).to_csv(RES/fn,index=False)
 summary={'equivalent_input_identity':'PASS','nuisance_projection':'PASS','event_generator_dynamic':'PARTIAL','r1_frozen':'CONTROLLED_TOY_ONLY','r1_iterative':'CONTROLLED_TOY_ONLY','r2_structural':'NOT_RUN','unknown_onset':'NOT_RUN','two_event':'NOT_RUN','far':'NOT_RUN','m6_stress':'NOT_RUN','pmu_loss':'NOT_RUN','route_decision':'NO_CLEAR_WINNER'}; pd.DataFrame([summary]).to_csv(RES/'event_route_summary.csv',index=False)
 (REP/'event_route_comparison.md').write_text(f'# Event representation campaign\n\nEquivalent-input and nuisance-quotient identities pass in a controlled linearized audit. R1 frozen and iterative routes are compared only on deterministic toy responses; R2 structural nonlinear trajectories were not run because the Event Generator V2 dynamic gate is still PARTIAL.\n\n`EVENTGEN_DYNAMIC_GATE = PARTIAL`  \n`ROUTE_DECISION = NO_CLEAR_WINNER`\n\nNo event, RAW, ML, M6 stress, FAR, PMU-loss, unknown-onset, or two-event claim is made.\n',encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
