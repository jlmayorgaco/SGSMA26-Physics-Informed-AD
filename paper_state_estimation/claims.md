# Paper 2 claim ledger

Status labels: `SUPPORTED`, `QUALIFIED`, `PENDING`, `WITHDRAWN`.

| ID | Claim | Status | Evidence / limitation |
|---|---|---|---|
| C1 | Eight PMUs can functionally reconstruct the hidden M6 voltage field with a physically constrained nonlinear AC MAP. | SUPPORTED | E06-H: 60 TEST cases, 100% valid convergence, median static closure 0.9809. |
| C2 | Nuisance injections need not be parameter-identifiable for hidden-voltage functional reconstruction. | SUPPORTED | E06-H median local rank 25/43 and nullspace 18, while closure is 0.974/0.981/0.983 for m=0.5/1.0/1.5. |
| C3 | The corrected physical nuisance semantics are required. | SUPPORTED | E06-G current 17-PQ basis failed; corrected basis passed to numerical tolerance. |
| C4 | Static mismatch degradation is largely stale operating-point physics rather than loss of observability. | QUALIFIED | E06-D oracle recentering and E06-H support this for M6; transfer to M1/M7 belongs to a separate frozen study. |
| C5 | A raw causal low-pass of streaming PMU measurements is a safe online center estimator. | WITHDRAWN | E06-I: online closure negative, convergence 0.8125, and nominal TVE rose to 5.69%. |
| C6 | The uncertainty output is calibrated. | PENDING | E06-H/E06-I uncertainty is `PARTIAL`; NEES-like diagnostics remain poor. |
| C7 | A nonlinear fixed-lag DAE estimator is necessary. | WITHDRAWN | E06-H static MAP captures the M6 recovery; E06-I failure instead motivates causal dynamic-deviation separation. |

## Required follow-up before submission

1. Develop and preregister a causal innovation/deviation-compensated center
   extractor, then evaluate it on a fresh M6 split.
2. Close the joint covariance/coverage problem without tuning on TEST.
3. Add a frozen transfer check to M1/M7 only after the M6 causal gate passes.
4. Verify every bibliographic entry against a primary source before populating
   `references.bib`.
