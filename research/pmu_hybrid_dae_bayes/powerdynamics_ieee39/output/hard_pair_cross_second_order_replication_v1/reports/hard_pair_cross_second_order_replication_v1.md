# HARD-PAIR-CROSS-SECOND-ORDER-REPLICATION-V1

Start HEAD `218fdfda69ea94c15664fa1f2e79096997432ee2`; analysis HEAD `4506f9ff2d4adf3a6acbcd0407b86d09666d4dee`; no push.

Exactly 72 new TDS trajectories were generated: three pairs, three operating points, two Richardson levels, and four signs. Full 192-state and production-output samples k=1..45 come from the same solve; all are permanently excluded from V3.

## Pair outcomes

| pair   | value_pass   | hessian_pass   | a2_deltaq_pass   | quadratic_removal_pass   | classification             |   value_max |   hessian_rel_max |   hessian_cos_min |   hessian_uncertainty_ratio_max |   a2_sigma_max |   n_order_resolved |
|:-------|:-------------|:---------------|:-----------------|:-------------------------|:---------------------------|------------:|------------------:|------------------:|--------------------------------:|---------------:|-------------------:|
| 26-28  | True         | True           | True             | True                     | PASS_SECOND_ORDER_IDENTITY |           0 |       8.02321e-05 |                 1 |                        0.456516 |        1.04881 |                  0 |
| 3-18   | True         | True           | True             | True                     | PASS_SECOND_ORDER_IDENTITY |           0 |       0.000132342 |                 1 |                        0.495964 |        1.31111 |                  0 |
| 16-18  | True         | True           | True             | True                     | PASS_SECOND_ORDER_IDENTITY |           0 |       0.000133399 |                 1 |                        0.272201 |        1.22888 |                  0 |

## Statuses

PREREGISTRATION_MANIFEST = PASS
NEW_TDS_TRAJECTORIES = PASS
ZERO_FUTURE_V3_OVERLAP = PASS
PAIR_26_28_VALUE_IDENTITY = PASS
PAIR_26_28_HESSIAN_IDENTITY = PASS
PAIR_26_28_A2_DELTAQ = PASS
PAIR_26_28_QUADRATIC_REMOVAL = O3_OR_HIGHER_COMPATIBLE
PAIR_3_18_VALUE_IDENTITY = PASS
PAIR_3_18_HESSIAN_IDENTITY = PASS
PAIR_3_18_A2_DELTAQ = PASS
PAIR_3_18_QUADRATIC_REMOVAL = O3_OR_HIGHER_COMPATIBLE
PAIR_16_18_VALUE_IDENTITY = PASS
PAIR_16_18_HESSIAN_IDENTITY = PASS
PAIR_16_18_A2_DELTAQ = PASS
PAIR_16_18_QUADRATIC_REMOVAL = O3_OR_HIGHER_COMPATIBLE
CROSS_SECOND_ORDER_THEORY = PASS_BROAD
ADDITIONAL_QUADRATIC_MECHANISM = NOT_DETECTED
SECOND_ORDER_LOCAL_THEORY = CLOSED_FOR_TESTED_DOMAIN
CUBIC_DEVELOPMENT_READINESS = BLOCKED
V3_READINESS = NOT_READY

## Explicit answers

1. 26-28 A2=DeltaQ: yes; max 1.0488 sigma.
2. 3-18 A2=DeltaQ: yes; max 1.3111 sigma.
3. 16-18 A2=DeltaQ: yes; max 1.2289 sigma.
4. Resolvable lambda^2 term after correction: none.
5. Production-output/measurement contract problem: none.
6. Additional physical quadratic mechanism: no evidence.
7. Cross theory closure over tested hard directions: yes.
8. Cubic development is neither necessary nor justified by this replication.
9. The one remaining pre-V3 physics question is TARGETED_NONLINEAR_MARGIN_CLOSURE.

## One next scientific action

Execute TARGETED_NONLINEAR_MARGIN_CLOSURE under its already frozen research-plan contract; do not alter the second-order dictionary or begin V3 in the same run.
