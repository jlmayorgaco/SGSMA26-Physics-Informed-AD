# Round 1 independent reviews

## Reviewer 1 — methodology and statistics

**Recommendation: Reject. Confidence: 5/5.**

The simulation holdout was inspected during model development, and RAW0001 labels
selected the representation. Neither result is an unbiased final test. The 8/12
RAW0001 interval ignores selection over 16 variants. The paper should report counts,
state the conditional nature of the Wilson intervals, and avoid inferential language
for differences of one or two windows. A new frozen test set and paired,
selection-aware uncertainty are required for confirmatory claims.

## Reviewer 2 — power-system domain

**Recommendation: Reject. Confidence: 5/5.**

The frozen artifact does not implement the effective-impedance candidate evidence
described in the paper. Runtime graph summaries use fixed observed-PMU pairs and
default unit distances. The model also relies on raw-unit rule thresholds and an
assumed first 3-s reference segment. These choices may work in the competition
configuration but do not establish physics-based transfer. The method section must
describe the executable artifact, not a nearby experimental branch.

## Reviewer 3 — scientific argument

**Recommendation: Major revision bordering on reject. Confidence: 4/5.**

The narrative infers a monotone mechanism from a non-monotone feature sweep. The
selected representation scores 8/12, but a Hilbert variant ties it and the full
combination scores 7/12. The strongest target performance is almost entirely due to
missing/bad-data localization. The contribution should be framed as a typed
hierarchical implementation and an exploratory sensitivity study, not a demonstration
of target-domain transfer.

## Reviewer 4 — editor-in-chief synthesis

**Recommendation: Reject and invite resubmission. Confidence: 5/5.**

The original text conflicts with the artifact in feature semantics, tree counts,
and placement assumptions. The repository supports CPU inference, but not exact
training reconstruction because generated feature tables and the original corpus are
not present. The paper should make exact, auditable claims and separate what can be
reproduced now from what depends on restricted or missing inputs.

## Reviewer 5 — operations and deployment

**Recommendation: Reject and encourage resubmission after new validation.
Confidence: 5/5.**

The reported 823.67 s for 89.65 min of input establishes batch throughput, not online
alarm latency. Complete 30-s windows are loaded before a decision is assigned. The
system has no streaming buffer, onset detector, calibration study, reject option, or
operational loss model. It should be described as computationally feasible batch
analysis for one fixed eight-PMU placement, not deployable real-time diagnosis.

## Common required experiments for a journal version

- Freeze the representation before opening a multi-operating-point target test set.
- Report physical, integrity, and mixed events separately, with multiple examples per
  class and location.
- Match hierarchy and estimator capacity when comparing feature families; add strong
  current CNN/GNN and observer-based baselines.
- Evaluate PMU relocation/dropout, topology changes, timing faults, packet loss, and
  multiple simultaneous events.
- Add calibration, abstention, selective risk, Top-3, and electrical-distance error.
- Measure causal onset-to-alarm delay and p50/p95/p99 compute latency in a streaming
  implementation.
