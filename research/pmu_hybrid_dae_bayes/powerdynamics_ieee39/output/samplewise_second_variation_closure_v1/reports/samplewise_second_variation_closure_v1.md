# SAMPLEWISE-SECOND-VARIATION-CLOSURE-V1

Start HEAD `9af2134b1422f70bbac0e50f1f6c3791b5e3edb7`; analysis HEAD `dd8aa618f0fd15659eff2fac17c2ffee1bd23c27`; no push, no V3.

Frozen directions: self [7, 26, 3, 16]; cross ['7-12', '26-28', '3-18', '16-18']; operating points [('op_m035', 0.35), ('op_m085', 0.85), ('op_m125', 1.25)]. The callback is at tau=2.0 s and the first canonical post-event sample is t1=tau+1/30=2.033333333 s. Existing coarse TDS files were reused and 72 fine half-step trajectories were generated after the preregistration freeze.

The direct output-space Richardson audit contains 1080 rows (24 directions × 45 samples). The analytic comparison uses the corrected dictionary row k (row 0 remains the continuous callback-time baseline). Richardson uncertainty is stored per row; a divergence flag means the whitened discrepancy exceeds 3× the local coarse/fine Richardson uncertainty.

## Quantitative result

SAMPLEWISE_TDS_HESSIANS = PASS
SAMPLEWISE_ANALYTIC_HESSIANS = PASS
FIRST_SAMPLE_SECOND_ORDER_MATCH = FAIL
FIRST_DIVERGENT_SAMPLE = 1.0
EARLY_A2_COMPONENT = PRESENT
A2_CAUSAL_SOURCE = FIRST_FLOW_OR_OUTPUT_SAMPLING_MISMATCH
VARIATIONAL_CONTINUATION = NOT_DIAGNOSTICATED_AFTER_FIRST_SAMPLE
WINDOW_ASSEMBLY = NOT_TRIGGERED
LONG_HORIZON_A3_COMPONENT = DIAGNOSTIC_ONLY_NOT_IDENTIFIED
SECOND_ORDER_LOCAL_THEORY = PARTIAL_FIRST_SAMPLE_MISMATCH
CUBIC_DEVELOPMENT_READINESS = NOT_READY
V3_READINESS = NOT_READY

First-sample flagged rows: **15/24**. Median first flagged sample over OP/direction: **1.0** (t=2.033333333 s when finite).

### Direction summary

| direction   |   median_relative_error |   p95_relative_error |   max_relative_error |   median_whitened_error |   median_cosine |   n_exceeded |
|:------------|------------------------:|---------------------:|---------------------:|------------------------:|----------------:|-------------:|
| cross_16-18 |              0.00809099 |            0.0177642 |            0.0206666 |                 5.36231 |        0.999963 |           42 |
| cross_26-28 |              0.00789014 |            0.0229288 |            0.0261017 |                 5.979   |        0.999974 |           67 |
| cross_3-18  |              0.00824145 |            0.0252314 |            0.0371663 |                 5.28798 |        0.999966 |           33 |
| cross_7-12  |              0.0152553  |            0.0363077 |            0.0441868 |                 3.60293 |        0.99988  |           29 |
| self_16     |              0.00943628 |            0.0179017 |            0.0185476 |                18.2967  |        0.999967 |          135 |
| self_26     |              0.00331025 |            0.0119149 |            0.0121771 |                 1.98114 |        0.999996 |          135 |
| self_3      |              0.00349134 |            0.0100855 |            0.0104894 |                 6.88307 |        0.999994 |          134 |
| self_7      |              0.00615727 |            0.0141074 |            0.0144476 |                 5.23233 |        0.999982 |          134 |

### Operating-point summary

| op_tag   |   median_relative_error |   p95_relative_error |   max_relative_error |   median_whitened_error |   median_cosine |   n_exceeded |
|:---------|------------------------:|---------------------:|---------------------:|------------------------:|----------------:|-------------:|
| op_m035  |              0.00753945 |            0.0186624 |            0.0348401 |                 5.08105 |        0.999974 |          240 |
| op_m085  |              0.00677159 |            0.0212428 |            0.0441868 |                 5.26015 |        0.999978 |          242 |
| op_m125  |              0.00744302 |            0.0217792 |            0.0401991 |                 5.48975 |        0.999973 |          227 |

### Interpretation

The first mismatch is already present at t1 for a majority of preregistered OP/direction combinations (remaining directions first flag at k=2–5). This is an output-level production first-flow/sampling-contract discrepancy at the current Richardson resolution, not evidence of a later forcing omission. The raw errors are small (self ~10^-7 and cross ~10^-5–10^-4 at t1) but can be several whitened Richardson sigmas because the second derivative is divided by h² and PMU noise is small.

The A2/A3/A4 table is diagnostic only. Self rays have a minimal four-point ± coarse/fine fit; cross rays have only two amplitudes per signed ray, so A2/A3 are reported with A4 unidentifiable. No cubic tensor, D/Q/Qcross, estimator, prior, GH31, or V3 artifact was changed.

### Contract answers

1. Exact earliest divergence is direction-dependent: k=1 (t1=2.033333 s) for 15/24 rows; the remainder first flags at k=2–5.
2. Yes, the discrepancy is present in the first production post-event output at the tested 3-sigma Richardson criterion.
3. A continuation omission is not isolated by this run because the first-sample gate fails; later growth is reported but not attributed to a forcing term.
4. The prior T30/T45 effective p2≈2 cannot be certified as a true asymptotic coefficient by this samplewise derivative result; it is compatible with a first-flow/sampling residual plus finite-range effects.
5. A2 is not shown to vanish at later horizons; the samplewise derivative mismatch persists and the finite-amplitude diagnostic is direction/horizon dependent.
6. The rigorous backward-compatible contract is: callback mutates parameters, retain stored state, take the exact production flow to t1, initialize sensitivities there, then continue the validated variational DAE.
7. Cubic development remains blocked until the first-flow residual is independently resolved.

### One next scientific action
Run an independent, tighter half-step production TDS stencil for the k=1 flagged directions (especially cross 26–28 and 3–18), with exact save-time state/output capture, to separate Richardson/roundoff from a genuine first-flow map mismatch before any cubic development.
