# IEEE39-END2END-SINGLE-V1

START_HEAD: `c9d4a1404950e06acc55d2da39c532eb0f89d348`
FINAL_HEAD: `PENDING_RESULTS_COMMIT`

## Scope
This is a frozen integration/scientific-sanity pilot, not prospective V3 and not a general unknown-initial-state DAE estimator. It performs event-conditioned, posterior-model-averaged reconstruction around the known pre-event op_m085 operating point from exactly eight PMUs.

## Event inference
| case_id   | evaluation_scenario           |   horizon |   p_event |         p_M0 |     p_M1 |      p_M2 |   p_support_7 |   p_include_7 |   rank_support_7 | map_support   |   map_probability | top_wrong_support   |   top_wrong_probability |   support_entropy |   cardinality_entropy |   credible_support_set_size | support_7_in_C95   |   severity_mean_7 |   severity_map_7 |   severity_lo95_7 |   severity_hi95_7 |   runtime_s |
|:----------|:------------------------------|----------:|----------:|-------------:|---------:|----------:|--------------:|--------------:|-----------------:|:--------------|------------------:|:--------------------|------------------------:|------------------:|----------------------:|----------------------------:|:-------------------|------------------:|-----------------:|------------------:|------------------:|------------:|
| case_c    | BUS7_MODERATE_CANONICAL_NOISE |        30 |         1 | 2.01076e-286 | 0.983862 | 0.0161382 |      0.983862 |             1 |                1 | (7,)          |          0.983862 | (7, 12)             |              0.00262213 |          0.124006 |             0.0826027 |                           1 | True               |        0.00329098 |       0.00329098 |        0.00308987 |        0.00344089 |   0.0463135 |
| case_c    | BUS7_MODERATE_CANONICAL_NOISE |        60 |         1 | 0            | 0.984916 | 0.0150845 |      0.984916 |             1 |                1 | (7,)          |          0.984916 | (7, 8)              |              0.00394467 |          0.11274  |             0.0782357 |                           1 | True               |        0.00332069 |       0.00332069 |        0.00325571 |        0.00336912 |   0.0634434 |
| case_c    | BUS7_MODERATE_CANONICAL_NOISE |       120 |         1 | 0            | 0.982049 | 0.0179508 |      0.982049 |             1 |                1 | (7,)          |          0.982049 | (7, 8)              |              0.00380102 |          0.134211 |             0.0899532 |                           1 | True               |        0.00330365 |       0.00330365 |        0.0032908  |        0.00331323 |   0.0938869 |

## Reconstruction summary
| method   | quantity                     |      median |           p90 |         p95 |     maximum |          rmse |
|:---------|:-----------------------------|------------:|--------------:|------------:|------------:|--------------:|
| ORACLE   | differential_state_nrmse     | 1.53939e-10 |   4.96206e-09 | 2.11964e-08 | 4.81378e-07 | nan           |
| ORACLE   | algebraic_state_nrmse        | 3.61642e-08 |   3.07955e-07 | 6.91951e-07 | 1.26959e-05 | nan           |
| ORACLE   | unobserved_voltage_magnitude | 4.12865e-09 | nan           | 4.41627e-09 | 4.53007e-09 |   4.14525e-09 |
| ORACLE   | unobserved_voltage_angle_rad | 1.25442e-08 | nan           | 1.31307e-08 | 1.33612e-08 |   1.26565e-08 |
| MAP      | differential_state_nrmse     | 4.53324e-08 |   3.67123e-06 | 5.59153e-06 | 0.000772962 | nan           |
| MAP      | algebraic_state_nrmse        | 8.89576e-06 |   7.7325e-05  | 0.000176511 | 0.00316808  | nan           |
| MAP      | unobserved_voltage_magnitude | 2.78008e-08 | nan           | 7.03959e-08 | 8.69235e-08 |   3.87331e-08 |
| MAP      | unobserved_voltage_angle_rad | 3.16792e-06 | nan           | 3.24619e-06 | 3.27569e-06 |   3.18233e-06 |
| BMA      | differential_state_nrmse     | 4.41208e-08 |   3.54086e-06 | 5.04559e-06 | 0.000776394 | nan           |
| BMA      | algebraic_state_nrmse        | 8.88325e-06 |   7.77041e-05 | 0.000177905 | 0.0031738   | nan           |
| BMA      | unobserved_voltage_magnitude | 2.38945e-08 | nan           | 5.35637e-08 | 7.03989e-08 |   3.07717e-08 |
| BMA      | unobserved_voltage_angle_rad | 3.18931e-06 | nan           | 3.24732e-06 | 3.276e-06   |   3.2e-06     |

## Best reconstructed native coordinates
| state_name                          | kind         |       nrmse |
|:------------------------------------|:-------------|------------:|
| VIndex(33, :ctrld_gen₊machine₊E′_d) | differential | 4.33504e-09 |
| VIndex(39, :machine₊E′_q)           | differential | 5.04981e-09 |
| VIndex(34, :ctrld_gen₊machine₊ω)    | differential | 5.08726e-09 |
| VIndex(36, :ctrld_gen₊machine₊ω)    | differential | 5.08938e-09 |
| VIndex(38, :ctrld_gen₊machine₊ω)    | differential | 5.10161e-09 |
| VIndex(30, :ctrld_gen₊machine₊ω)    | differential | 5.11822e-09 |
| VIndex(33, :ctrld_gen₊machine₊ω)    | differential | 5.12639e-09 |
| VIndex(35, :ctrld_gen₊machine₊ω)    | differential | 5.13054e-09 |
| VIndex(37, :ctrld_gen₊machine₊ω)    | differential | 5.13809e-09 |
| VIndex(32, :ctrld_gen₊machine₊ω)    | differential | 5.13883e-09 |

## Worst reconstructed native coordinates
| state_name              | kind         |       nrmse |
|:------------------------|:-------------|------------:|
| VIndex(31, :busbar₊u_i) | algebraic    | 0.0031738   |
| VIndex(39, :machine₊δ)  | differential | 0.000776394 |
| VIndex(23, :busbar₊u_i) | algebraic    | 0.000675419 |
| VIndex(22, :busbar₊u_i) | algebraic    | 0.000389252 |
| VIndex(29, :busbar₊u_i) | algebraic    | 0.000342581 |
| VIndex(19, :busbar₊u_i) | algebraic    | 0.000148844 |
| VIndex(37, :busbar₊u_i) | algebraic    | 8.26779e-05 |
| VIndex(20, :busbar₊u_i) | algebraic    | 8.20682e-05 |
| VIndex(28, :busbar₊u_i) | algebraic    | 8.15434e-05 |
| VIndex(32, :busbar₊u_i) | algebraic    | 7.60586e-05 |

## Runtime
| stage                          |   horizon |     seconds |
|:-------------------------------|----------:|------------:|
| physical_dictionary_load_build |       120 | 259.485     |
| PowerDynamics_truth_simulation |       120 | 164.18      |
| GH31_estimator                 |         5 |   0.0437285 |
| GH31_estimator                 |        10 |   0.0445305 |
| GH31_estimator                 |        20 |   0.0537934 |
| GH31_estimator                 |        30 |   0.0573758 |
| GH31_estimator                 |        45 |   0.0677452 |
| GH31_estimator                 |        60 |   0.078893  |
| GH31_estimator                 |        90 |   0.116736  |
| GH31_estimator                 |       120 |   0.117896  |
| BMA_full_state_reconstruction  |       120 |   1.0744    |
| total_pipeline                 |       120 | 435.52      |

## Leakage
No inference or BMA stage consumed hidden truth. The separate oracle uses true Bus 7/+0.0033 solely to measure the physical-manifold ceiling.

## Direct scientific answers
1. Event detection: YES in this integration case; P(event) was 1 at T5 and 1 at T120, while H0 retained P(H0)=0.99993.
2. Cardinality/localization: YES; Bus 7 was rank 1 from T5, P(S={7}) rose from 0.638828 to 0.982049, and P(K=1) reached 0.982049.
3. Evidence accumulation: the exact support exceeded 0.95 by T10 and its 95% credible support set was a singleton from T10 onward.
4. Severity: posterior mean 0.00330365 versus truth 0.00330000; the 95% interval [0.00329080,0.00331323] contains truth.
5. Hidden-bus electrical reconstruction: T120 BMA unobserved-bus |V| RMSE=3.077e-08; wrapped-angle RMSE=3.200e-06 rad.
6. Physical versus inference error: ORACLE is the physical-manifold ceiling; MAP/BMA minus ORACLE quantifies the additional inference contribution. At this small moderate event, inference uncertainty dominates the residual electrical-output error.
7. Weakest hidden coordinates: the largest BMA NRMSE is bus 31 imaginary voltage, followed by machine angle at bus 39; the report tables preserve the full best/worst lists.
8. Leakage: none detected. Only scoring loaded truth; estimator NPZ keys were time_s, pmu_32, and contract_sha256.
9. Runtime: T120 GH31=0.1179s; BMA=1.0744s; full cold pipeline=435.52s.
10. V3 readiness: software/integration readiness is demonstrated, but one case does not establish prospective performance or calibration.

## Interpretation
A single case demonstrates executable integration only; it cannot establish population accuracy or calibration. Reconstruction error is separated into the oracle physical-manifold ceiling and additional MAP/BMA event-inference uncertainty.

## Exact status block
```text
START_HEAD = c9d4a1404950e06acc55d2da39c532eb0f89d348
FINAL_HEAD = PENDING_RESULTS_COMMIT
COMMITS = 3_LOCAL_COMMITS
PUSH = NO

ESTIMATOR_CONTRACT_AUDIT = PASS
PMU_OBSERVATION_CONTRACT = PASS
BUS7_UNOBSERVED = PASS
BUS7_CANDIDATE_SOURCE = PASS
OP_CONDITIONING = PASS
TRUTH_SIMULATION = PASS
INFORMATION_LEAKAGE_GUARD = PASS
H0_CONTROL = PASS
BUS7_NOISELESS = PASS
BUS7_CANONICAL_NOISE = PASS
HYPOTHESIS_COUNT = PASS_137
GH31_INFERENCE = PASS
AR1_DENSE_REGRESSION = PASS
POSTERIOR_NORMALIZATION = PASS
BUS7_EVENT_DETECTION = PASS
BUS7_CARDINALITY = PASS
BUS7_LOCALIZATION = PASS
BUS7_SEVERITY = PASS
CREDIBLE_SUPPORT_SET = PASS
FULL_STATE_MANIFOLD = PASS
ORACLE_RECONSTRUCTION = PASS
MAP_RECONSTRUCTION = PASS
BMA_RECONSTRUCTION = PASS
UNOBSERVED_BUS_VOLTAGE_RECONSTRUCTION = PASS
UNOBSERVED_BUS_ANGLE_RECONSTRUCTION = PASS
DIFFERENTIAL_STATE_RECONSTRUCTION = PASS
ALGEBRAIC_STATE_RECONSTRUCTION = PASS
POSTERIOR_PREDICTIVE_INTERVALS = SINGLE_CASE_DIAGNOSTIC_ONLY
RUNTIME_PROFILE = PASS
V3_EXCLUSION = PASS
TESTS = PASS_TARGETED_PREEXISTING_SUITE_FAILURE_1
CHATGPT_REVIEW_ZIP = PASS
```

## One next scientific action
Run the already planned prospective V3 only after freezing this end-to-end software contract, using a new exclusion-clean physical/noise split and no parameter changes.
