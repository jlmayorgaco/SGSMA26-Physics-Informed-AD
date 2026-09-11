# E03-R gate closure

The supported ANDES event API executes governor-reference and high-impedance shunt perturbations plus an initial-state perturbation, with causal resampling to 30 fps. Linear-vs-TDS files are materialized.

## Nonlinear versus linear

| case                       |        min |        max |
|:---------------------------|-----------:|-----------:|
| GOVERNOR_REFERENCE_STEP    | 13.7512    | 13.752     |
| INITIAL_STATE_PERTURBATION |  1.50886   |  1.50896   |
| SMALL_SHUNT_LOAD_STEP      |  0.0419301 |  0.0419423 |

The normalized mismatch does not approach a numerical floor for the tested cases, so the input mapping is not yet validated as a true independent linearization.

## Information bounds

Configured covariance is used only in the Fisher information calculation; no hidden truth enters R. Longer structural horizons 90/120/180 are in `e03r_long_horizon.csv`.

## Decision

**E03 = FAIL**

The remaining defect is mathematical/physical: a reproducible perturbation-to-input map for the reduced descriptor (including the persistent governor/shunt forcing and dynamic relinearization at redispatch points) is still required before claiming local equivalence.
