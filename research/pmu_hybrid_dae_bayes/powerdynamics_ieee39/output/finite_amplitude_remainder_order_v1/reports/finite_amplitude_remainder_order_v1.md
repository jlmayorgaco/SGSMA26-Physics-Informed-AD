# FINITE-AMPLITUDE-REMAINDER-ORDER-V1

HEAD: `dedfbc18bd404d23f157260f133f0a768dee9a0a`; branch `research/pmu-hybrid-dae-bayes-v1`; no push, no V3.

Selected 30 target cases: 18 eta>1, 6 mid (0.1<eta<=0.5), 6 low (eta<0.01), with true and frozen-nearest competitor rays. Lambda grid is [0.125, 0.25, 0.5, 0.75, 1.0]; all new trajectories are in the V3 exclusion manifest.

Execution: 284 new TDS files, 52 reused lambda=1 files, 36 tight-standard pairs; all available trajectories completed successfully.

Numerical floor (99th percentile tight-standard whitened difference): 3.26483e-07. Slope points with residual <=10x this floor were excluded.

## Order by horizon

T30: median p1=2.0000, p2=2.0229
T45: median p1=2.0001, p2=2.0954
T60: median p1=2.0000, p2=2.4782
T90: median p1=2.0000, p2=2.9739
T120: median p1=2.0000, p2=2.9980

Tail true-ray H_T majority under the preregistered all-horizon criterion: False; classifications: {'C_SECOND_ORDER_INCOMPLETE': 18}.

## Cubic diagnostic

lambda=0.125: median ||R2/lambda^3||=65.539, adjacent cosine=0.998463
lambda=0.25: median ||R2/lambda^3||=60.06, adjacent cosine=0.999803
lambda=0.5: median ||R2/lambda^3||=59.207, adjacent cosine=1.000000

Holdout (directional C3 fitted only at lambda<=0.5):
lambda=0.75: median relative error=0.0369, cosine=0.999417
lambda=1: median relative error=0.0464, cosine=0.999059

## Statuses

PREREGISTRATION_MANIFEST = PASS
AMPLITUDE_HOMOTOPY = PASS
NEW_TDS_TRAJECTORIES = PASS
ZERO_FUTURE_V3_OVERLAP = PASS
NUMERICAL_FLOOR = PASS
FIRST_ORDER_REMAINDER_ORDER = PASS
SECOND_ORDER_REMAINDER_ORDER = PARTIAL
LOCAL_P1 = PASS
LOCAL_P2 = PARTIAL
FULL_RANGE_P2 = DIAGNOSTIC
CUBIC_COEFFICIENT_STABILITY = PASS
CUBIC_HOLDOUT_PREDICTION = DIAGNOSTIC
TRUE_MANIFOLD_REMAINDER = MEASURED
COMPETITOR_MANIFOLD_REMAINDER = MEASURED
HORIZON_ORDER_STABILITY = PASS
H_T_FINITE_AMPLITUDE_TRUNCATION = NOT_SUPPORTED
SECOND_ORDER_LOCAL_THEORY = PARTIAL
PHYSICAL_MODEL_DEVELOPMENT = DIAGNOSTIC_ONLY
V3_READINESS = UNCHANGED_NOT_ASSESSED

## Scientific interpretation
The first-order residual is consistently quadratic in lambda. The second-order residual is cubic at long horizons (T90/T120) but shows a reproducible order-two component at early horizons (T30/T45, transitioning at T60), so the strict all-horizon second-order Taylor gate is not closed by this run. C3 directions are nevertheless stable and predict lambda=.75/1 without refitting at roughly 95% directional agreement. This is a diagnostic of the frozen manifold/event contract, not an accepted cubic implementation.

## Explicit answers
1. First-order error is O(lambda^2): yes.
2. Corrected second-order error is O(lambda^3) near zero: only at long horizons; not uniformly at T30/T45.
3. The order is not horizon-invariant: p2 rises from ~2 at T30 to ~3 by T90/T120.
4. The coefficient grows with horizon, with an early order-two contamination.
5. True and competitor rays show the same pattern and comparable remainder norms.
6. R2/lambda^3 is directionally stable but its norm drifts between lambda=.125 and .5.
7. Small-lambda C3 predicts .75 and 1.0 with ~4–5% median norm error and cosine ~0.999.
8. Eta>1 cases are not proven to be ordinary finite-amplitude truncation across the full horizon; they are consistent with finite-amplitude effects plus an early residual component.
9. The second-order local theory is partially supported at long horizon but fails the strict all-horizon gate.
10. Investigate the remaining quadratic residual at the first saved frames (production event-map/first-flow second variation or sampling contract) before adding any cubic term.

## One next scientific action
Audit the first-frame residual at T30/T45 with a stricter production-map second-order initialization (no estimator changes and no V3); only after that decide whether a DEV-only cubic extension is warranted.
