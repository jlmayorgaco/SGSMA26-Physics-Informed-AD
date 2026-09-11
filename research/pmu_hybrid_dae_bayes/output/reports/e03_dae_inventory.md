# E03 ANDES DAE inventory

The installed ANDES runtime after `setup → PFlow → TDS.init` reports `nx=220` dynamic states and `nz=479` algebraic variables. Ordering is exactly the persisted `e03_dynamic_states.csv` and `e03_algebraic_states.csv`; no documentation-only vector was assumed.

Dynamic device counts: `{'GENROU': 10, 'TGOV1N': 10, 'IEEEX1': 10, 'IEEEST': 10, 'BusFreq': 10}`. The CSVs include global index, model, device, state name, bus, unit and physical meaning.
