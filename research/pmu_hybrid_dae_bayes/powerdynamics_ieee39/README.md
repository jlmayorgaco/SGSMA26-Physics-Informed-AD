# Independent PowerDynamics IEEE-39 validation

This directory is isolated from the ANDES campaign. The Julia project uses the
official PowerDynamics tutorial data and records package/environment provenance
before any translated or corrected model is introduced.

The current campaign is diagnostic only: it does not modify the installed
tutorial CSVs, Python estimator code, or the existing E03 artifacts.

Reproduction commands (from the repository root):

```powershell
$env:JULIAUP_CHANNEL='1.11.9'
C:\Users\walla\AppData\Local\Programs\Julia\julia.exe --project=research/pmu_hybrid_dae_bayes/powerdynamics_ieee39/julia research/pmu_hybrid_dae_bayes/powerdynamics_ieee39/julia/scripts/pd39_official_tutorial.jl
C:\Users\walla\AppData\Local\Programs\Julia\julia.exe --project=research/pmu_hybrid_dae_bayes/powerdynamics_ieee39/julia research/pmu_hybrid_dae_bayes/powerdynamics_ieee39/julia/scripts/pd39_data_audit.jl
C:\Users\walla\AppData\Local\Programs\Julia\julia.exe --project=research/pmu_hybrid_dae_bayes/powerdynamics_ieee39/julia -e 'include("research/pmu_hybrid_dae_bayes/powerdynamics_ieee39/julia/test/runtests.jl")'
```

Post-reproduction audits are run with `julia/scripts/pd39_descriptor_eigen.jl`,
`pd39_nonlinear_probe.jl`, `pd39_linear_validation.jl`,
`pd39_dynamic_sanity.jl`, and `pd39_observability.jl`; the read-only Python
audits are `scripts/pd_static_parity.py`, `pd_pmugate.py`, and
`pd_variant_comparison.py`.

The qualification decision and the complete A–P handoff are in
`output/reports/FINAL_VALIDATION_REPORT.md`.
