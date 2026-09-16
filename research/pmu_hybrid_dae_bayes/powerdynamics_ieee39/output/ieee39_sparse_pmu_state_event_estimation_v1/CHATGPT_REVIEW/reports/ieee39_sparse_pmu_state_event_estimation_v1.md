# IEEE39-SPARSE-PMU-STATE-EVENT-ESTIMATION-V1

START_HEAD: `8b3784cf803b92a38237f0f7604341bb6412242d`
FINAL_HEAD: `PENDING_RESULTS_COMMIT`

## Frozen scope
Matched nonlinear IEEE-39 PowerDynamics integration at op_m085 with exactly eight PMUs and one hidden Bus-7 event. This is an integration/control experiment, not prospective V3 or a population claim.

## Counts and contract
Three physical trajectories (H0, noiseless Bus 7, canonical-noise Bus 7), 120 nested post-event frames, 32 channels, 137 hypotheses, GH31 and frozen AR(1) W2. Bus 7 is not observed.

## Event posterior
| case_id   | evaluation_scenario           |   horizon |   p_event |         p_M0 |     p_M1 |      p_M2 |   p_support_7 |   p_include_7 |   rank_support_7 | map_support   |   map_probability | top_wrong_support   |   top_wrong_probability |   support_entropy |   cardinality_entropy |   credible_support_set_size | support_7_in_C95   |   severity_mean_7 |   severity_map_7 |   severity_lo95_7 |   severity_hi95_7 |   runtime_s |
|:----------|:------------------------------|----------:|----------:|-------------:|---------:|----------:|--------------:|--------------:|-----------------:|:--------------|------------------:|:--------------------|------------------------:|------------------:|----------------------:|----------------------------:|:-------------------|------------------:|-----------------:|------------------:|------------------:|------------:|
| case_c    | BUS7_MODERATE_CANONICAL_NOISE |         5 |         1 | 2.51577e-36  | 0.649691 | 0.350309  |      0.638828 |      0.986163 |                1 | (7,)          |          0.638828 | (3, 7)              |              0.120671   |         1.41697   |             0.647637  |                           8 | True               |        0.00365349 |       0.00365348 |        0.00304681 |        0.00410572 |   0.0368368 |
| case_c    | BUS7_MODERATE_CANONICAL_NOISE |        10 |         1 | 1.13353e-65  | 0.955224 | 0.0447757 |      0.95512  |      0.99989  |                1 | (7,)          |          0.95512  | (7, 12)             |              0.0101812  |         0.29583   |             0.182835  |                           1 | True               |        0.00338289 |       0.00338289 |        0.00295894 |        0.00369892 |   0.0321851 |
| case_c    | BUS7_MODERATE_CANONICAL_NOISE |        20 |         1 | 1.97977e-138 | 0.969163 | 0.0308366 |      0.969163 |      1        |                1 | (7,)          |          0.969163 | (7, 12)             |              0.00635478 |         0.215176  |             0.137638  |                           1 | True               |        0.0032013  |       0.0032013  |        0.00292163 |        0.00340977 |   0.0370466 |
| case_c    | BUS7_MODERATE_CANONICAL_NOISE |        30 |         1 | 2.01076e-286 | 0.983862 | 0.0161382 |      0.983862 |      1        |                1 | (7,)          |          0.983862 | (7, 12)             |              0.00262213 |         0.124006  |             0.0826027 |                           1 | True               |        0.00329098 |       0.00329098 |        0.00308987 |        0.00344089 |   0.0506608 |
| case_c    | BUS7_MODERATE_CANONICAL_NOISE |        45 |         1 | 0            | 0.988486 | 0.0115145 |      0.988486 |      1        |                1 | (7,)          |          0.988486 | (7, 12)             |              0.001841   |         0.0922377 |             0.0628502 |                           1 | True               |        0.00325589 |       0.00325589 |        0.00314098 |        0.00334155 |   0.0644833 |
| case_c    | BUS7_MODERATE_CANONICAL_NOISE |        60 |         1 | 0            | 0.984916 | 0.0150845 |      0.984916 |      1        |                1 | (7,)          |          0.984916 | (7, 8)              |              0.00394467 |         0.11274   |             0.0782357 |                           1 | True               |        0.00332069 |       0.00332069 |        0.00325571 |        0.00336912 |   0.0839318 |
| case_c    | BUS7_MODERATE_CANONICAL_NOISE |        90 |         1 | 0            | 0.980311 | 0.0196889 |      0.980311 |      1        |                1 | (7,)          |          0.980311 | (7, 12)             |              0.006495   |         0.142531  |             0.0968258 |                           1 | True               |        0.00330493 |       0.00330493 |        0.00327944 |        0.00332392 |   0.0990148 |
| case_c    | BUS7_MODERATE_CANONICAL_NOISE |       120 |         1 | 0            | 0.982049 | 0.0179508 |      0.982049 |      1        |                1 | (7,)          |          0.982049 | (7, 8)              |              0.00380102 |         0.134211  |             0.0899532 |                           1 | True               |        0.00330365 |       0.00330365 |        0.0032908  |        0.00331323 |   0.115065  |

H0 control at T120: P(H0)=0.99992953, P(event)=7.0472444e-05, max non-null={12} (p=1.5513891e-05).

Bus-7 noisy event at T120: P(S={7})=0.98204918, P(7 in S)=1, rank=1, severity mean=0.0033036518, 95% CI=[0.003290796,0.0033132349].

## Primary event-induced reconstruction
| method   | window    |        rmse |   normalized_rmse |   relative_l2 |   cosine |   explained_variance |   n_buses |
|:---------|:----------|------------:|------------------:|--------------:|---------:|---------------------:|----------:|
| NOMINAL  | FULL_POST | 3.4295e-05  |       1           |   1           |        0 |             0        |        39 |
| ORACLE   | FULL_POST | 4.16199e-09 |       0.000121359 |   0.000121359 |        1 |             1        |        39 |
| MAP      | FULL_POST | 3.97848e-08 |       0.00116008  |   0.00116008  |        1 |             0.999999 |        39 |
| BMA      | FULL_POST | 3.15881e-08 |       0.00092107  |   0.00092107  |        1 |             0.999999 |        39 |

The NOMINAL row is the zero-event-effect predictor. ORACLE isolates the physical manifold ceiling; MAP/BMA include event-inference uncertainty.

## Native 192-state reconstruction
BMA differential states: median NRMSE=4.412e-08, p95=5.046e-06. Algebraic: median=8.883e-06, p95=1.779e-04.

### Ten hardest BMA coordinates
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
| stage                          |   horizon |    seconds | mode               |
|:-------------------------------|----------:|-----------:|:-------------------|
| PowerDynamics_truth_simulation |       120 | 23.867     | OFFLINE            |
| GH31_estimator                 |         5 |  0.0368368 | ONLINE_BATCH       |
| GH31_estimator                 |        10 |  0.0321851 | ONLINE_BATCH       |
| GH31_estimator                 |        20 |  0.0370466 | ONLINE_BATCH       |
| GH31_estimator                 |        30 |  0.0506608 | ONLINE_BATCH       |
| GH31_estimator                 |        45 |  0.0644833 | ONLINE_BATCH       |
| GH31_estimator                 |        60 |  0.0839318 | ONLINE_BATCH       |
| GH31_estimator                 |        90 |  0.0990148 | ONLINE_BATCH       |
| GH31_estimator                 |       120 |  0.115065  | ONLINE_BATCH       |
| BMA_full_state_reconstruction  |       120 |  3.15153   | ONLINE_BATCH       |
| total_pipeline                 |       120 |  5.70325   | OFFLINE_PLUS_BATCH |

## Leakage and exclusion
Estimator inputs were hashed NPZ files containing only time_s, pmu_32, and contract_sha256. Truth is loaded only after inference artifacts are written. All three trajectories and both noise seeds are permanently excluded from future V3.

## Direct answers
1. Eight PMUs detect a hidden Bus-7 event in this matched case: yes, with posterior event probability near one by T120.
2. Localization: yes in this integration case; this is not population accuracy.
3. Severity: the conditional posterior recovers the frozen +0.33% event with a narrow interval.
4. Hidden reconstruction: BMA event-delta metrics are primary; absolute metrics are supplementary.
5. Error sources: ORACLE is physical-manifold error; MAP/BMA minus ORACLE is inference uncertainty; sparsity is represented by the observed/unobserved split and sensitivity table.
6. Recoverability: differential and algebraic coordinates are reported separately; the worst native coordinates are retained without collapsing scales.
7. Nominal safety: H0 retains essentially all posterior mass and does not manufacture an event.
8. Prospective status: not V3; no general calibration claim.

## Exact status block
```text
ESTIMATOR_CONTRACT_AUDIT = PASS
PMU_OBSERVATION_CONTRACT = PASS
BUS7_UNOBSERVED = PASS
BUS7_CANDIDATE_SOURCE = PASS
OP_M085_MATCH = PASS
TRUTH_TDS = PASS
THREE_SCENARIOS = PASS
HYPOTHESES_137 = PASS
GH31 = PASS
AR1_DENSE_REGRESSION = PASS
POSTERIOR_NORMALIZATION = PASS
FULL_STATE_MANIFOLD = PASS
NOMINAL_ORACLE_MAP_BMA = PASS
EVENT_DELTA_METRICS = PASS
DIFFERENTIAL_ALGEBRAIC_SPLIT = PASS
LEAKAGE_AUDIT = PASS
V3_EXCLUSION = PASS
ANALYTIC_DAE_TANGENT = PENDING
PROSPECTIVE_V3 = NOT_RUN
PUSH = NO
```

## One next scientific action
Freeze this integration contract and only then run the preregistered prospective V3 on an exclusion-clean split; do not change the estimator semantics.
