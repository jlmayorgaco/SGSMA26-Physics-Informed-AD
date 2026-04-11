# ANDES IEEE-39 Fault Reconstruction Experiment

## Scenario

- Simulator: ANDES IEEE-39 full dynamic case.
- Single event: three-phase fault at non-PMU Bus 7.
- Time horizon: 10.000 s.
- Fault applied at t_final / 2 = 5.000 s.
- Fault cleared at 5.083 s.
- Observed PMU buses: [2, 5, 6, 10, 19, 22, 29, 39].
- Hidden non-PMU buses compared against truth: [7, 1, 3, 4, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38].

## Estimator

The estimator receives only the 8 PMU voltage phasors. For each time step,
it solves a Ybus-weighted harmonic extension problem over all 39 buses.
This is a Kirchhoff/topology-constrained state proxy, not a full dynamic
Kalman estimator. PMU buses are Dirichlet boundary conditions; non-PMU
states are inferred from the IEEE-39 admittance graph.

## Accuracy Summary

- Hidden-bus mean voltage RMSE: 0.020394 p.u.
- Hidden-bus median voltage RMSE: 0.013799 p.u.
- Hidden-bus mean angle RMSE: 3.5193 deg.
- Hidden-bus median angle RMSE: 3.6850 deg.
- Post-fault mean voltage RMSE: 0.021867 p.u.
- Post-fault mean angle RMSE: 3.5316 deg.
- Fault Bus 7 post-fault voltage RMSE: 0.054102 p.u.
- Fault Bus 7 post-fault angle RMSE: 1.6281 deg.

## Worst Hidden Buses By Post-Fault Voltage RMSE

| bus | post_fault_vm_rmse_pu | post_fault_va_rmse_deg |
|---:|---:|---:|
| 31 | 0.069133 | 7.0466 |
| 20 | 0.061939 | 1.4095 |
| 32 | 0.061763 | 8.3434 |
| 7 | 0.054102 | 1.6281 |
| 33 | 0.049895 | 5.1723 |
| 34 | 0.044497 | 3.7062 |
| 8 | 0.037638 | 1.9670 |
| 37 | 0.034946 | 3.6157 |

## Plots

- fault_bus_vm: [figures/bus7_vm_reconstruction.png](figures/bus7_vm_reconstruction.png)
- fault_bus_angle: [figures/bus7_angle_reconstruction.png](figures/bus7_angle_reconstruction.png)
- post_fault_rmse: [figures/hidden_bus_post_fault_vm_rmse.png](figures/hidden_bus_post_fault_vm_rmse.png)

## Files

- `truth_all_buses.csv`: ANDES all-bus simulated voltage magnitude and angle truth.
- `pmu_observed.csv`: only the 8 PMU buses used by the estimator.
- `non_pmu_truth.csv`: hidden non-PMU truth used for scoring.
- `full_state_estimate.csv`: reconstructed all-39-bus voltage state.
- `non_pmu_errors.csv`: hidden-bus pointwise errors.
- `metrics.csv`: per-hidden-bus RMSE/MAE metrics.
- `metadata.json`: scenario configuration and bus partitions.

## Interpretation

The Ybus harmonic estimator is intentionally lightweight and defensible:
it enforces network smoothness and Kirchhoff/topological coupling while
using only the PMU phasors. It should capture the spatial footprint of
the non-PMU fault, but it cannot perfectly reproduce the severe local
voltage collapse at an unobserved faulted bus. That residual is useful
for localization features and is the main motivation for keeping a later
full dynamic Kalman/DAE estimator as an advanced branch.
