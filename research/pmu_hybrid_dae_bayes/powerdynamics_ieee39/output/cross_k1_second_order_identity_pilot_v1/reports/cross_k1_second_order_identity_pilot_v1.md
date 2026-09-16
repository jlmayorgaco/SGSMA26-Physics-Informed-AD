# CROSS-K1-SECOND-ORDER-IDENTITY-PILOT-V1

Start HEAD `83d145382b75589abc3785569885baac419a1829`; analysis HEAD `580517efaa3963a6635273733d12e75206195316`; branch `research/pmu-hybrid-dae-bayes-v1`; no push.

## Frozen scope and execution

Only Bus 7--12 was evaluated at m=0.35, 0.85, and 1.25. The matched Richardson stencil uses (h7,h12)=(1e-4,2e-4) and half steps, four signs at each level. Exactly 24 new production TDS trajectories were generated; each stores the same-run 192-state and voltage-output samples for k=1..45 and is permanently excluded from V3.

## Value and Hessian identities

Maximum |y_direct-h_canonical(u_saved)| = **0.000e+00**. Across focus samples k=[1, 2, 3, 5, 30, 45], median/max state-projected versus direct-output Hessian relative errors are **1.446e-04 / 3.021e-04**, with minimum cosine **0.999999966345**. The maximum mismatch is only **0.2546** of the Richardson uncertainty.

## Cross DeltaQ and independent A2

The independent A2 estimate uses only pre-pilot four-sign physical contrasts. Pure self terms cancel; lambda=.5,1 form the local Richardson estimate, while lambda=5,20 are finite-amplitude stress only. Across the focus frames, median ||A2||/||DeltaQ|| = **1.14461**, median cosine = **0.877910940**, but these direction metrics are cancellation-sensitive. The decisive uncertainty-normalized maximum is **1.1232 sigma**, so A2=DeltaQ is not rejected.

## Idealized remainder

After subtracting lambda^2 DeltaQ, the lower local point (lambda=.5) is below the frozen TDS numerical floor for every focus row; some lambda=1 residuals are resolved, so a numerical exponent cannot be estimated. Because the four-sign mixed contrast cancels cubic monomials, a resolved fourth-order remainder would be expected; the absence of a resolved quadratic scaling is compatible with O(lambda^3) or higher. The correction is diagnostic only and does not alter Qcross or the estimator.

## Statuses

MATCHED_CROSS_STENCIL = PASS
PRODUCTION_VALUE_IDENTITY = PASS
CROSS_STATE_OUTPUT_HESSIAN = PASS
CROSS_DELTA_Q = PASS
CROSS_A2_ESTIMATE = PASS
CROSS_A2_DELTAQ_IDENTITY = PASS
NUMERICAL_QUADRATIC_TERM_REMOVAL = PASS
IDEALIZED_REMAINDER_ORDER = O3_COMPATIBLE_LOWER_POINT_BELOW_FLOOR
CROSS_7_12_SECOND_ORDER_THEORY = PASS
BROADER_CROSS_REPLICATION_READINESS = YES
CUBIC_DEVELOPMENT_READINESS = BLOCKED
V3_READINESS = NOT_READY

## Explicit answers

1. Under the matched stencil, H_output_7_12 = H_y H_state_7_12: **yes**; median/max relative error are 1.446e-04/3.021e-04, and max error/uncertainty is 0.2546.
2. A2_7_12 = DeltaQ_7_12: **yes, within numerical uncertainty**; maximum discrepancy 1.1232 sigma.
3. After subtracting lambda^2 DeltaQ, the remainder is **compatible** with O(lambda^3) or higher, but lambda=.5 is below the frozen numerical floor, so no exponent is claimed.
4. Evidence for an additional quadratic physical mechanism: **no**.
5. Replication on 26--28, 3--18, and 16--18 is **justified**.

## One next scientific action

Replicate this frozen matched-stencil identity test on exactly the three canonical hard cross pairs 26--28, 3--18, and 16--18, without changing the dictionary or estimator.
