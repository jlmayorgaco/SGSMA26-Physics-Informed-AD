"""Build a reusable physical IEEE-39 event bank from PowerDynamics source tables.

This is a registry/smoke gate: candidates are derived from the validated model
tables, never from estimator labels or the historical ANDES event list.
"""
from pathlib import Path
import json, pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'powerdynamics_ieee39/output'; RES=OUT/'results'; REP=OUT/'reports'; REP.mkdir(parents=True,exist_ok=True)
SRC=Path(r'C:/Users/walla/.julia/packages/PowerDynamics/VzOiZ/docs/examples/ieee39data')
def main():
 bus=pd.read_csv(SRC/'bus.csv'); branch=pd.read_csv(SRC/'branch.csv'); machine=pd.read_csv(SRC/'machine.csv')
 pq=bus[bus.bus_type=='PQ']; pv=bus[bus.bus_type=='PV'];
 rows=[]
 for fam,items in [('NORMAL',[('system','baseline')]),('FAULT',[(f'BUS{int(x)}','shunt_fault') for x in bus.bus]),('LINE_OUTAGE',[(f'{int(a)}-{int(b)}','open_branch') for a,b in zip(branch.src_bus,branch.dst_bus)]),('GENERATION_CHANGE',[(f'GEN_BUS{int(x)}','delta_p') for x in machine.bus]),('LOAD_CHANGE',[(f'LOAD_BUS{int(x)}','delta_pq') for x in pq.bus])]:
  for j,(target,sem) in enumerate(items): rows.append({'family':fam,'candidate_id':f'EV2_{fam}_{j+1:02d}','target':target,'physical_semantics':sem,'source_bus_table':str(SRC/'bus.csv'),'source_branch_table':str(SRC/'branch.csv'),'source_machine_table':str(SRC/'machine.csv'),'smoke_required':5 if fam!='NORMAL' else 1,'status':'REGISTERED'})
 df=pd.DataFrame(rows); df.to_csv(RES/'event_v2_registry.csv',index=False)
 smoke=[]
 for fam,g in df.groupby('family'):
  n=len(g); smoke.append({'family':fam,'registered':n,'required':5 if fam!='NORMAL' else 1,'physical_table_backed':True,'status':'PASS' if n>=(5 if fam!='NORMAL' else 1) else 'FAIL'})
 sd=pd.DataFrame(smoke); sd.to_csv(RES/'event_v2_smoke.csv',index=False); df.to_csv(RES/'event_v2_candidate_registry_full.csv',index=False)
 manifest=df[df.family!='NORMAL'].copy(); manifest['seed']=np.arange(len(manifest))+7001; manifest['onset_s']=1.0+(manifest.seed%16)/10; manifest['duration_s']=np.where(manifest.family=='FAULT',0.1+(manifest.seed%5)*0.05,np.nan); manifest.to_csv(RES/'event_v2_manifest.csv',index=False)
 rej=pd.DataFrame([{'family':f,'reason':'dynamic intervention harness not executed in bounded campaign','count':len(g)} for f,g in manifest.groupby('family')]); rej.to_csv(RES/'event_v2_rejections.csv',index=False)
 stats=df.groupby('family').size().rename('candidate_count').reset_index(); stats.to_csv(RES/'event_v2_basic_statistics.csv',index=False)
 status='PARTIAL'
 (REP/'event_v2_dynamic_validation.md').write_text(f'# Event Generator V2 dynamic validation\n\nRegistry derived from PowerDynamics IEEE-39 source tables at `{SRC}`. All table-backed candidates are enumerated ({len(df)} including NORMAL).\n\n{sd.to_markdown(index=False)}\n\n`EVENT_GENERATOR_V2 = {status}`. Candidate coverage passes; nonlinear TDS intervention/replay validation is explicitly pending and all such cases are listed in `event_v2_rejections.csv`.\n',encoding='utf-8')
 print(json.dumps({'EVENT_GENERATOR_V2':status,'families':smoke},indent=2))
if __name__=='__main__': main()
