# K1-SECOND-ORDER-IDENTITY-CLOSURE-V1

This is an audit-only replay. No new PowerDynamics TDS was generated; D, Q, Qcross, the estimator, and the historical contracts were not modified.

Source HEAD: `a69a3406480eaf12ec4cdf371c9573504e426253`.

## Exact statuses

- PRODUCTION_OUTPUT_VALUE_IDENTITY = **PASS_OUTPUT_LEVEL**
- MEASUREMENT_CHAIN_RULE = **PASS_AFFINE_NO_FEEDTHROUGH**
- SAME_STENCIL_STATE_OUTPUT_HESSIAN = **PARTIAL_SELF_PASS_CROSS_UNAVAILABLE**
- SELF_HALF_FACTOR_CONTRACT = **PASS**
- DELTA_Q_CONSTRUCTION = **PASS**
- HOMOTOPY_A2_ESTIMATE = **PASS_EXISTING_TRAJECTORIES**
- A2_DELTAQ_IDENTITY_K1 = **PASS_SELF_PARTIAL_CROSS**
- A2_DELTAQ_IDENTITY_T45 = **PASS_SELF_PARTIAL_CROSS**
- EARLY_LAMBDA2_CAUSAL_SOURCE = **SUPPORTED_SELF_CROSS_UNRESOLVED**
- STRICTER_STENCIL_NEEDED = **YES_FOR_CROSS_STATE_OUTPUT_CLOSURE**
- SECOND_ORDER_LOCAL_THEORY = **PASS_WITH_OUTPUT_FIRST_FLOW_CAVEAT**
- CUBIC_DEVELOPMENT_READINESS = **BLOCKED**
- V3_READINESS = **NOT_READY**

## Production output value identity

State-map/output rows: 48 (48 self rows with an independently stored production output). Maximum `C u - h(state)` absolute difference is 1.421e-13; maximum independent response difference is 5.437e-12. The canonical output is therefore identical at value level to numerical precision.

## Measurement derivative contract

`C_pmu_frozen` is a fixed linear map in the native state coordinates. The fixed PiLine output has no event-parameter argument, so h_p=h_pp=h_up=h_pu=h_uu=0 analytically. The approximately 3e-6 centered-FD second difference is cancellation roundoff after division by eps^2, not curvature.

## Same-stencil state/output Hessian

For the strict self set (12 OP×direction cases), the state-map Richardson Hessian projected by C versus the independently stored output Hessian has median relative error 3.756e-06, maximum 9.311e-06, and median cosine 0.999999999994. Cross state maps are not stored at the direct output stencil (.005/.0025 versus .0001/.0002), so cross identity is not claimed.

## Factor convention and A2 test

The self convention is Q=0.5 H_self; the cross convention is Qcross=H_cross. The independently fitted homotopy A2 coefficient agrees with DeltaQ for self rays: k=1 median norm ratio 1.000000, cosine 1.000000000000; k=45 ratio 1.000000, cosine 1.000000000000. Cross A2 remains unresolved by the two-level tiny-amplitude output fit (k=1 ratio 5.636, cosine 0.285375) and is not promoted to PASS.

## Causal answer

The value-level production output identity and the affine measurement chain rule pass. The early O(lambda^2) remainder is causally explained for the self terms by the measured first-flow second-order coefficient mismatch: the fitted A2 is the same DeltaQ coefficient. A complete self+cross closure is not established because the frozen artifacts lack a matching 192-state cross stencil; this is an evidence-coverage limitation, not an analytic claim.

## Explicit answers

1. Yes at value level; C applied to each saved 192-state map reproduces the canonical PMU output to 1.4e-13, and independent saved output agrees to 5.5e-12.
2. Yes, the production measurement map is affine with no direct parameter feedthrough.
3. Yes for the strict self set; cross is not assessed under identical state/output perturbation points.
4. Yes, the 1/2 self and raw mixed-cross conventions are consistent.
5. Yes for self at k=1 and k=45; cross is unresolved by the stored two-level tiny stencil.
6. No value-level mismatch remains; the remaining uncertainty enters through the missing cross same-stencil state/output artifact and second-difference sensitivity.
7. No finer stencil is needed for value/self closure; a matching cross state/output stencil is needed for a full closure.
8. The early O(lambda^2) source is explained for self terms, but not yet closed for cross terms.
9. Yes, cubic development remains blocked.

## Exactly one next scientific action

Generate one matching 192-state/output cross stencil for Bus7-12 at the existing .0001/.0002 perturbations, then repeat only this identity comparison; do not change the estimator or start cubic development.
