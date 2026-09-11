# E03 descriptor / functional observability

## A. Git / commit

Generated on the active research branch; no push performed.

## B. DAE dimensions

`nx=220`, `nz=479`. Dynamic model counts: `{'GENROU': 10, 'TGOV1N': 10, 'IEEEX1': 10, 'IEEEST': 10, 'BusFreq': 10}`.

## C. Gauge treatment

ANDES fixes the reference through the network reference/slack treatment; the first 39 algebraic entries are bus angles. `G_z` was retained in full, rank=479/479, smallest singular value=0.00209934, condition=562870. Reduction used linear solves, never `pinv` or an explicit inverse.

## D. Jacobian audit

Analytic ANDES blocks were exported to `output/results/e03_matrices/`; selected VI finite-difference columns are in `e03_jacobian_checks.csv`.

## E. Eigenvalue / discretization

Continuous spectral abscissa=0.0371275; expm discretization at Δt=1/30 s gives spectral radius=1.00124. No eigenvalue clipping was applied.

## F-G. Observability and functional observability

See `e03_observability_vs_horizon.csv` and `e03_functional_observability.csv`. Results are reported for VI_ONLY and VI_F_R_SYNTHETIC. At 60 frames VI_ONLY reaches rank 136/220 at the 1e-10 threshold and global hidden-voltage residual 0.008072; buses with per-bus residual >1e-3 are: 34, 20, 33.

## H. Frequency ablation

VI_F_R_SYNTHETIC is explicitly marked history-only: no instantaneous native BusFreq output is invented at seven PMUs. Consequently its instantaneous descriptor Jacobian equals VI_ONLY; a future dynamic filter state must be added before claiming an added rank.

## I. PMU-loss structural ranking

All single and pair removals are recorded in `e03_pmu_loss_observability.csv`; this is structural sensitivity, not placement optimization. At 60 frames, the largest single-removal residuals were: PMU 39 (residual 0.0558), PMU 2 (residual 0.02301), PMU 5 (residual 0.023).

## J. Operating-point robustness

The five G1 AC-feasible points remain the static source set. Full dynamic re-linearization at redispatched points is not silently approximated and is listed as a limitation.

## K. Nonlinear TDS cross-check

Native TDS initialization is verified, but a controlled input-to-linearized-output perturbation harness is not exposed by this ANDES build; no nonlinear agreement claim is made.

## L. Limitations

No estimator, localization, Bayesian inference, ML, or Monte Carlo was run. Frequency/ROCOF are not instantaneous algebraic outputs for this descriptor. TDS API smoke is separate from PMU dynamic parity.

## M. Gate

**E03 = FAIL** — inventory/Jacobian infrastructure is present, but the required locally validated nonlinear TDS cross-check and dynamic frequency augmentation are not yet complete; no favorable threshold was invented.

## N. One next step

Implement a minimal ANDES perturbation-input bridge that records nonlinear TDS outputs and compares them against the expm descriptor at 30 Hz, then reassess E03 without changing the PMU map.
