# Paper 2 — sparse-PMU functional state reconstruction

This directory is the independent manuscript for the PowerDynamics IEEE-39
state-estimation campaign. It is deliberately separate from
`paper_journal/revision_20260910`, which remains the ExtraTrees/typed-routing
event-diagnosis paper.

## Current scope

- Eight observed PMUs: buses 2, 5, 6, 10, 19, 22, 29, and 39.
- Hidden voltage field: the remaining 31 buses.
- Validated static nonlinear AC MAP under M6 operating-point mismatch (E06-H).
- Causal streaming low-pass test (E06-I), which failed the nominal-safety gate.
- No claim of calibrated Bayesian uncertainty and no nonlinear fixed-lag DAE
  claim yet.

`main.tex` is a structured working draft, not a submission-ready manuscript.
`claims.md` is the claim/evidence ledger. Numerical artifacts remain in
`research/pmu_hybrid_dae_bayes/powerdynamics_ieee39/output/`.
