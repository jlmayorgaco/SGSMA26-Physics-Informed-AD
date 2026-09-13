# IEEE-39 sparse-PMU Hybrid-DAE Bayesian campaign

This is a standalone, physics-first validation campaign. It is intentionally
separate from the competition submission and does not import its trained models,
RAW calibration code, or event labels.

The observed PMU set is fixed to `{2, 5, 6, 10, 19, 22, 29, 39}`. The intended
estimator infers a posterior over dynamic and algebraic state, physical
configuration jumps, parameters, and discrepancy; it is not an event-label
classifier.

## Data and leakage contract

- `data/external_holdout/` is reserved for RAW0001, RAW0002, and any supplied
  external traces. This campaign never reads it during simulation, fitting, or
  design.
- Splits are scenario-, operating-point-, and model-seed-level. Adjacent
  windows from one trajectory cannot cross splits.
- Simulator truth is serialized only as evaluation data. Estimator inputs are
  observations and a declared physical model.
- The competition current-terminal metadata are not asserted by this folder.
  Until authoritative terminal metadata are provided, PMU currents use an
  explicitly declared, deterministic synthetic terminal map. Every output is
  labeled `CONTROLLED_SYNTHETIC_TERMINAL_MAP`.

## Phases implemented in this initial campaign

1. Deterministic environment audit, manifest and experiment registry.
2. ANDES-versus-pandapower static AC parity with an explicit gate.
3. PMU measurement operator, branch-terminal convention, causal frequency and
   ROCOF operator, and mapping audit.
4. Deterministic scenario, configuration-jump, integrity-noise, rejection, and
   resume API.

Later estimator, observability, support-oracle, discrepancy, calibration and
holdout phases are deliberately pending. The campaign must not report event
localization accuracy before the static-parity and measurement gates pass.

## E06-H/E06-I operating-point gate

The corrected E06-H static gate uses the physically valid 43-coordinate M6
nuisance basis (bus 31 Slack, bus 39 PV) and demonstrates strong hidden-voltage
functional recovery from eight PMUs (global median static closure 0.981),
despite a median local rank of 25/43. E06-I then evaluates the causal streaming
extension on fresh M6 trajectories. A raw causal Huber trailing mean selected
on DEV (60 frames, updates every 30 frames) is not safe by itself: TEST online
closure is negative, convergence is 0.812, and nominal TVE increases materially.
The static MAP result is therefore retained, while the next estimator design
must separate dynamic deviations before forming an online center. Details and
artifacts are in `powerdynamics_ieee39/output/reports/e06h_corrected_m6_static_recentering.md`
and `powerdynamics_ieee39/output/reports/e06i_online_causal_recentering.md`.

## Commands

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = 'research/pmu_hybrid_dae_bayes/src'
python -m pmu_hybrid.experiments.smoke --root research/pmu_hybrid_dae_bayes
python -m pmu_hybrid.experiments.e00_environment --root research/pmu_hybrid_dae_bayes
python -m pmu_hybrid.experiments.e01_static_parity --root research/pmu_hybrid_dae_bayes
python -m pmu_hybrid.experiments.e02_measurement_audit --root research/pmu_hybrid_dae_bayes
python -m pmu_hybrid.experiments.e03_scenario_smoke --root research/pmu_hybrid_dae_bayes
python -m pytest research/pmu_hybrid_dae_bayes/tests -q
```

The first two commands may take longer on a fresh machine because ANDES and
pandapower initialize numerical dependencies. Outputs are written only below
this folder's `output/` directory and are resumable through their manifests.
