# Round 2 assessment

**SGSMA conference decision: Minor revision / weak accept.**

**Journal decision: Reject pending new experiments.**

The re-review found that the fatal representation errors and overclaims from round 1
were removed. The revised text now identifies RAW0001 as a representation-selection
set, reports 7/7 integrity versus 1/5 physical/mixed localization, removes the absent
effective-impedance method, discloses the fixed PMU schema, and treats the 16-variant
experiment as sensitivity rather than causal ablation.

Four conference-level corrections were requested and applied:

1. Rule gates are described as post-processing ExtraTrees predictions and
   confidences, matching the executable order.
2. Rule thresholds are described as specified or fitted without RAW0001 labels;
   the text no longer says that every threshold was learned.
3. PI-HED is expanded as PMU-Integrity-Aware Hierarchical Event Diagnosis, which
   matches the typed integrity/physical hierarchy without reviving unsupported
   topology or physics-transfer claims.
4. The uninstrumented local runtime number was removed. The paper now states only
   the batch execution boundary and makes no real-time claim.

Journal acceptance still requires an untouched efficacy test, repeated seeds,
capacity-matched feature controls, tuned contemporary baselines, placement-shift
tests, calibration/abstention, and complete retraining provenance.
