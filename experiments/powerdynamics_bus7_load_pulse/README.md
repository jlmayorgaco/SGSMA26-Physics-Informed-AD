# IEEE39 Bus-7 PowerDynamics load pulse

This self-contained Julia experiment runs the current PowerDynamics IEEE39 model with a deterministic controlled load event:

- simulation interval: `0.0:15.0` s;
- physical export grid: exact `k / 30`, `k = 0..450` (451 samples);
- event: Bus 7 `ZIPLoad` `Pset` and `Qset` multiplied by `1.10` at exactly `t = 5.0` s;
- restoration: the exact initialized nominal `Pset`/`Qset` values are restored at exactly `t = 10.0` s;
- solver: `Rodas5P`, `rtol = atol = 1e-9`, `dtmax = 1/60` s;
- output: one RAW0001-like CSV for each of the 39 buses, with no artificial measurement noise.

The model is supplied by the official `JuliaEnergy/IEEE39.jl` repository and pinned in `Project.toml` to commit `d47ef9c655e215e62c7ca484dbe4b31786caabf7`. The simulation uses the package's canonical power-flow and dynamic initialization path. Since ZIP-load initialization makes the dynamic `Pset`/`Qset` power-flow consistent, the audit records both the raw `load.csv` values and the initialized nominal values used by the callback.

## Run

From the repository root:

```powershell
julia --project=experiments/powerdynamics_bus7_load_pulse -e "using Pkg; Pkg.instantiate()"
$env:PD39_BUS7_PLOTS = "true"
julia --project=experiments/powerdynamics_bus7_load_pulse experiments/powerdynamics_bus7_load_pulse/scripts/run_bus7_load_pulse.jl
```

Set `PD39_BUS7_PLOTS=false` for a faster export-only run. The default output directory is:

`output/SIM_PD39_LOAD_BUS7_PLUS10_RETURN_V1/`

Generated outputs are intentionally ignored by Git. The output tree contains:

```text
00_manifest/   scenario and environment manifests
01_model_audit/ model audit and per-bus export manifest
02_raw_csv/     Bus1..Bus39_Competition_Data_nanmask.csv
03_event/       event_command.csv
04_plots/       per-bus and system plots
05_validation/  format comparison and validation metrics
06_review/      FINAL_REPORT.md
```

## Semantics

The physical solution is evaluated at exact `k/30` values. The CSV `TIMESTAMP` column follows the supplied RAW0001 convention by truncating those values to milliseconds (`0.033`, `0.066`, `0.100`, ...). Voltages and currents are exported as balanced synthetic phase quantities from the native positive-sequence BusBar observables. Frequency is the 60 Hz network base plus the derivative of the unwrapped positive-sequence voltage angle; ROCOF is the derivative of that exported frequency. These fallbacks and units are recorded in `model_audit.json`.

## Tests

The fast unit checks validate the time grid, RAW0001 column contract, timestamp projection, and event command semantics:

```powershell
julia --project=experiments/powerdynamics_bus7_load_pulse experiments/powerdynamics_bus7_load_pulse/test/runtests.jl
```

Set `PD39_RUN_INTEGRATION_TEST=1` to make the test script also execute the full 15-second IEEE39 simulation in a temporary output directory.

## Next stage: sparse-PMU E0 estimator

The Python E0 pipeline consumes this experiment as a controlled reference and
withholds 31 buses from inference. Only Bus2, Bus5, Bus6, Bus10, Bus19, Bus22,
Bus29, and Bus39 are copied to `input_sparse/` with the `Event` label removed; all 39 buses are later used
by the evaluator in `ground_truth/`. It produces 39 virtual-PMU estimates,
REF-vs-EST comparisons, windowed metrics, uncertainty coverage, electrical
distance analysis, and plots. See
[`V5/docs/e0_sparse_pmu_experiment.md`](../../V5/docs/e0_sparse_pmu_experiment.md)
for the command and leakage contract.
