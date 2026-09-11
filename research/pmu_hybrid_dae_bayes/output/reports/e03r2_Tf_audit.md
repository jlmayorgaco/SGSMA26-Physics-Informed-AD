# E03-R2 Tf audit

After setup/PFlow/TDS.init, `nx=220`, `Tf_min=0`, `Tf_max=1199`, `zero_Tf=50`, unique count `44`.

## Convention proven

andes/core/var.py: State.t_const is collected to dae.Tf; time constants are applied to the left-hand side and not e_str; andes/routines/eig.py: A_s=T^{-1}(f_x-f_y g_y^{-1}g_x), zero-Tf states are folded into algebraic equations. TGOV1N inherits tgbase.py wref=e.g. wref0-wref. Therefore ANDES exposes `T x_dot=f`, not `x_dot=f`; the old E03 `A_raw` was missing `T^{-1}`. The corrected matrix uses ANDES' own EIG path, including zero-Tf folding, dead algebraic elimination and state-constraint projection; corrected dimension is `160`.

## Inputs and feedthrough

TGOV1N.wref0 and Shunt.b derivatives are centered finite differences of actual fixed-state `f,g` residuals; see `e03r2_input_derivative_checks.csv` and summary. `D_r=-H_y solve(G_y,G_u)` is computed and not assumed zero.

## Initial condition / native timestamps

The requested `dy0=-solve(Gy,Gx dx0)` construction and native `dae.ts.t` comparison are materialized. The residual and six-epsilon slope are recorded in `e03r2_native_timestamp_response.csv`; the 30-fps causal resampling remains secondary.

Corrected spectral abscissa=1.0327798; positive-mode classification is CHANGED relative to the historical unnormalized value. Native initial-condition slope p=1.0014.

## Gate

**E03-R2 = FAIL**: the Tf convention and corrected reduction are proven, but the full governor/shunt native-timestamp first-order convergence gate is not passed by this audit. E04 is not started.

## Recorded numerical outcomes

Governor: `Fu_l2=0`, `Gu_l2=1`, `B_r_l2=4160`, `D_r_l2=0`. Shunt: `Fu_l2=0`, `Gu_l2=1.10839`, `B_r_l2=0.302008`, `D_r_l2=0.480071`.

Consistent-IC residuals are in `e03r2_consistent_ic.csv`; at epsilon `1e-4`, `||g||=1.60868e-07` and `||g||/epsilon^2=16.0868`. Corrected VI_ONLY ranks at horizons 0/3/10/30/60/120 are `18/72/105/119/125/126`.
