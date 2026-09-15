# TARGETED-NONLINEAR-MARGIN-CLOSURE-V1

Start HEAD: `f1d5fceed3d5b31320cbf2d748f63a89486129d8`; final HEAD: `adbe3be061f2b2774310d4b0ec4a667809e49dbb`; branch `research/pmu-hybrid-dae-bayes-v1`; no push and no V3.

## Frozen preregistration and data

The preregistration SHA-256 is `fdb9e91f7490af379904589f72070ecafadeec995fb0fc26c07fb2523c69c842`.  It contains 48 targets (18 with eta>1, 6 with 0.5<eta<=1, 12 mid controls and 12 eta<0.01 controls), 48 de-duplicated Stage-A competitor points and an immutable Stage-B stencil authorization.  New TDS files are listed in `v3_exclusion_manifest_additions.csv` and are permanently excluded from future V3.

## Stage A and triangle certificate

All 48 Stage-A points completed; triangle certificates pass for 240/240 horizon rows.  The pointwise interval is `[max(0,d_analytic-e_S-e_R), d_analytic+e_S+e_R]`; it is not a continuous nonlinear optimum.

## Nonlinear local margins

Stage B was activated for 30 targets and evaluated on the immutable local stencil.  Classifications are: `{'D_RESIDUAL_MANIFOLD_ERROR': 30, 'E_INCONCLUSIVE_NEEDS_TARGETED_EXTENSION': 16, 'A_GENUINE_PHYSICAL_SMALL_MARGIN_SUPPORTED': 2}`.  Eta>1 classification counts are `{'D_RESIDUAL_MANIFOLD_ERROR': 18}`.  Hard-pair details are in `hard_pair_physical_validation.csv`; controls are separated in `control_case_margin_validation.csv`.

## Interpretation and limits

The nonlinear grid is a local physical reference only (no global-margin claim).  Finite-horizon small margins are compared with the historical positive equilibrium gamma4 values in `finite_vs_equilibrium_resolvability.csv`.  The canonical D/Q/Qij, event map, GH31, AR1 and priors were not modified.

## Statuses

PREREGISTRATION_MANIFEST = PASS  
TARGET_CASES = PASS (48; mandatory strata and hard-pair coverage)  
NEW_TDS_TRAJECTORIES = PASS (306 files; all successful)  
ZERO_FUTURE_V3_OVERLAP = PASS  
STAGE_A_CENTER_CHECK = PASS  
TRIANGLE_INEQUALITY_CERTIFICATES = PASS  
STAGE_B_LOCAL_PROFILES = PASS  
ETA_GT1_PHYSICAL_MARGIN_CLASSIFICATION = PASS  
H_M_SMALL_MARGIN_HYPOTHESIS = NOT_SUPPORTED  
CONTROL_CASE_MARGIN_FIDELITY = PARTIAL  
HARD_PAIR_NONLINEAR_DIFFICULTY = PARTIAL  
FINITE_VS_EQUILIBRIUM_RESOLVABILITY = PASS  
NONLINEAR_MARGIN_CLOSURE = FAIL  
T120_PHYSICAL_CONTRACT = OPEN_MARGIN_QUESTION  
PHYSICAL_MODEL_DEVELOPMENT = TARGETED_EXTENSION_REMAINING  
V3_READINESS = NOT_READY

## One next scientific action

Run the minimum targeted extension for the residual-manifold-error tail cases; do not start V3.
