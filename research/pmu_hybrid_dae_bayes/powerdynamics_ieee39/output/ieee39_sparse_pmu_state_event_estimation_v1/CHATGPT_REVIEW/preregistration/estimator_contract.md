# IEEE39-SPARSE-PMU-STATE-EVENT-ESTIMATION-V1 — preregistered contract

This is a matched-model integration experiment, not prospective V3 and not a
general unknown-initial-state estimator. The estimator sees exactly the eight
SGSMA PMUs and competes over the complete H0 + 16 singles + 120 doubles
hypothesis space.

- Start HEAD: `8b3784cf803b92a38237f0f7604341bb6412242d`
- Branch: `research/pmu-hybrid-dae-bayes-v1`
- Plant: PowerDynamics IEEE39, op_m085 (bus 3 P/Q multiplied by 1+0.03*0.85)
- Native state: 192 coordinates (114 differential, 78 algebraic)
- PMUs: buses 39,29,10,22,19,2,5,6 in repository canonical order
- Event: ZIP load Bus 7, true-time-local callback at 2.0 s, +0.0033 fraction
- Sampling: 30 Hz; one physical trajectory per scenario; post-event prefixes
  5,10,20,30,45,60,90,120 from one simulation
- Likelihood: frozen corrected OP-conditioned D/Q/Qij, W2 separable AR(1),
  rho=0.3512083596353588, diagonal Omega, Gaussian amplitude prior sigma=.05,
  cardinality prior (.20,.50,.30), uniform support conditional on cardinality,
  operational GH31 order 31.
- Truth is separate from estimator input. The estimator input has only
  `time_s`, `pmu_32`, and an immutable contract hash.
- Reconstructions: NOMINAL, ORACLE, MAP, BMA. BMA intervals use 4096 draws
  with preregistered seed 260915120.
- Full-state normalization is `max(abs(u0),1e-3)`, frozen before scoring.

The event-induced delta reconstruction is primary for electrical outputs:
`Delta V = V_event - V_nominal` and wrapped `Delta theta`. Oracle, MAP and
BMA may use the frozen physical manifold; truth is loaded only after inference
artifacts are written and hashed. This pilot cannot establish calibration or
population performance.
