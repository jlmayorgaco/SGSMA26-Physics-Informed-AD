# G1 phase and representation contract

The canonical internal row is `[Re(V), Im(V), Re(I_terminal), Im(I_terminal), f, ROCOF, DATA_PRESENT]` and uses one positive-sequence complex phasor per PMU. The official competition CSV is three-phase A/B/C magnitude-angle data; no positive-sequence claim is made for that external file.

`phase_export: balanced_compatibility_only` means a three-phase view may be synthesized from a balanced positive-sequence phasor for compatibility checks only. It is not treated as measured A/B/C truth. Missing samples remain an explicit boolean mask; absent values are represented as NaN only at the I/O/noise boundary, never by silently replacing them with zero.
