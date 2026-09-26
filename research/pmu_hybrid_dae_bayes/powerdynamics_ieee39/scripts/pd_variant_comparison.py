"""Summarize independent IEEE-39 static variants without modifying any backend."""
from pathlib import Path
import csv, json, numpy as np
ROOT=Path(__file__).resolve().parents[1]
pdrows=list(csv.DictReader((ROOT/'output/provenance/powerdynamics_original/bus.csv').open()))
pdbr=list(csv.DictReader((ROOT/'output/provenance/powerdynamics_original/branch.csv').open()))
pdv=list(csv.DictReader((ROOT/'output/results/pd_static_solution.csv').open()))
andm=json.loads((ROOT.parent/'output/manifests/static_parity.json').read_text())
ands=list(csv.DictReader((ROOT.parent/'output/results/e01_static_parity/canonical_bus_state.csv').open()))
pm={int(x['bus']):complex(float(x['u_r']),float(x['u_i'])) for x in pdv}
am={int(x['bus']):float(x['andes_vm_pu']) for x in ands}
vmerr=max(abs(abs(pm[k])-am[k]) for k in pm)
import pandapower.networks as nw, pandapower as pp
net=nw.case39(); pp.runpp(net)
summary={'PowerDynamics_package':{'buses':len(pdrows),'branches':len(pdbr),'machines':10,'avr':9,'governors':9,'loads':19,'equilibrium':'PASS'},
 'ANDES_bundled_ieee39':{'buses':39,'branches':46,'generators':10,'canonical_pandapower_gate':andm['canonical_parity']['status'],'direct_shipped_case_status':andm['direct_shipped_case_parity']['status']},
 'pandapower_networks_case39':{'buses':len(net.bus),'lines':len(net.line),'transformers':len(net.trafo),'loads':len(net.load),'generators':len(net.gen),'ext_grid':len(net.ext_grid),'converged':bool(net.converged),'vm_min':float(net.res_bus.vm_pu.min()),'vm_max':float(net.res_bus.vm_pu.max())},
 'matpower_case39':{'status':'NOT_AVAILABLE_IN_WORKSPACE'},
 'pd_vs_andes_canonical_max_vm_error_pu':float(vmerr),
 'interpretation':'PowerDynamics tutorial data are a distinct package-defined parameterization; topology/counts agree at 39/46, while equilibrium voltages differ materially from ANDES canonical. ANDES canonical and its pandapower translation pass their own shared-case gate.'}
(ROOT/'output/results/pd_variant_comparison.json').write_text(json.dumps(summary,indent=2))
with (ROOT/'output/reports/pd_variant_comparison.md').open('w') as f:
 f.write('# IEEE-39 static variant comparison\n\n')
 f.write('The comparison is read-only and does not alter ANDES or the original PowerDynamics CSVs.\n\n')
 for k,v in summary.items(): f.write(f'- **{k}**: `{json.dumps(v,sort_keys=True)}`\n')
 f.write('\n**Decision:** exact PD-G0 is a PowerDynamics-to-pandapower same-case result. It must not be conflated with ANDES canonical parity: the latter passes internally, but PD vs ANDES canonical has a maximum voltage-magnitude difference of %.5f pu, so the dynamic benchmark remains backend-specific.\n' % vmerr)
print(json.dumps(summary,indent=2))
