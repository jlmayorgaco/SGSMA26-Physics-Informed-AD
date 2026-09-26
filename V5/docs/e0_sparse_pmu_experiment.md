# E0 — sparse-PMU full-state reconstruction

This experiment is the first estimator unit test on the clean PowerDynamics
IEEE-39 Bus-7 load pulse. It uses only the eight observed buses:

```text
BUS2, BUS5, BUS6, BUS10, BUS19, BUS22, BUS29, BUS39
```

The other 31 buses are withheld until evaluation. The pipeline creates four
explicit data products:

```text
ground_truth/   all 39 reference files; evaluator only
input_sparse/   the eight files visible to the estimator; Event is removed
estimated/      39 virtual-PMU CSVs reconstructed from the sparse input
comparison/     REF vs EST tables created after inference
```

It also writes uncertainty CSVs, windowed signal metrics, circular angle
errors, complex voltage/current phasor errors, 95% coverage, electrical
distance metadata, event scores, latent-load `alpha` trajectories, an
alpha heatmap, 39 per-bus figures, a 39-by-6 NRMSE heatmap, and a final
report.

## Run

First generate the clean PowerDynamics reference case:

```powershell
$env:PD39_BUS7_PLOTS = "false"
julia --project=experiments/powerdynamics_bus7_load_pulse experiments/powerdynamics_bus7_load_pulse/scripts/run_bus7_load_pulse.jl
```

Then run E0 from the repository root:

```powershell
python V5/pipelines/e0_sparse_pmu_estimation.py `
  --reference-dir output/SIM_PD39_LOAD_BUS7_PLUS10_RETURN_V1/02_raw_csv `
  --output-dir output/SIM_PD39_SPARSE_PMU_E0
```

The generated output is ignored by Git. The estimator records the leakage
contract in `manifest.json` and `estimator_diagnostics.csv`:

- `ground_truth_used_during_estimation = false`;
- `event_label_used_during_estimation = false`;
- `REF_*` columns are created only in the evaluator's comparison stage.

The current implementation separates two gates:

- **E0-R reconstruction:** the oracle supplies only the event class `load`.
  The estimator receives eight positive-sequence voltage PMUs and the model
  audit, but no hidden bus files, `Event` column, event bus, magnitude, or
  interval. Static loads use latent power-factor-preserving multipliers,
  `P=(1+alpha)P0` and `Q=(1+alpha)Q0`, with nominal values acting as sparse
  priors rather than hard constraints.
- **E0-I inference:** the bus, multiplier, and interval are read from the
  inferred alpha trajectory. A voltage-only change gate is derived from the
  sparse PMUs to relax alpha sparsity during the observed transient; it is
  not the hidden event label.

The core signals are voltage phasor, frequency, and ROCOF: sparse PMU
frequency/ROCOF constrain observed phase increments while the voltage phasor
drives the network solve. Current is derived from `Ybus @ V` and is explicitly
reported as out-of-model validation because the current export is a fallback
semantic. Results remain per signal and per time window; voltage, current,
frequency, and ROCOF are not collapsed into one mixed-unit score. If alpha
does not pass its localization threshold, the report marks the inference
`needs_review` rather than claiming a bus.

Run the unit checks with:

```powershell
python -m pytest V5/tests/unit/test_e0_sparse_pmu.py -q
```
