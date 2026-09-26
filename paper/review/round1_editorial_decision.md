# Round 1 editorial decision

**Decision: Reject and invite a substantially revised resubmission.**

Five independent reviews examined scientific validity, power-system substance,
methodology, argument quality, and deployment claims. Their recommendations were
consistent: the software artifact is credible enough for a conference case study,
but the original manuscript made claims that the available experiment cannot
support at journal standard.

## Decisive issues

1. RAW0001 is a 21-window validation set, not an independent target-domain test.
   Its 12 abnormal labels were inspected while selecting one of 16 feature
   representations. The nominal 95% interval therefore describes binomial sampling
   conditional on the selected representation and does not account for selection.
2. The target result is physically unbalanced. The reported 8/12 exact locations
   consist of 7/7 integrity events and only 1/5 physical or mixed events.
3. The manuscript's effective-impedance candidate kernel is absent from the frozen
   selected schema. The 714 selected `GRAPH__` variables are cross-PMU summaries;
   production inference does not pass a topology-distance matrix, so its distance
   lookup defaults to one.
4. The learned schema is tied to buses 2, 5, 6, 10, 19, 22, 29, and 39. Dynamic
   file discovery prevents parser-order dependence, but it does not establish PMU
   placement invariance.
5. The staged “ablation” was not a causal ablation. All 16 combinations were
   inspected on the same 12 RAW0001 labels, two variants tied at 8/12, and adding
   every block reduced the score to 7/12.
6. The generic 200-tree baselines do not match the hierarchical model's routing,
   capacity, or training support. They are diagnostic controls, not evidence of
   superiority over published methods.
7. The runtime experiment measures batch throughput on complete 30-s windows. It
   does not measure detection delay, streaming latency, calibration, abstention, or
   behavior under late and out-of-order synchrophasor frames.

## Editorial path

The paper can be defensible after a full rewrite as a fixed-deployment,
development-set case study. It must remove topology-transfer and deployment-ready
claims, disclose the representation-selection protocol, report the integrity versus
physical split, and distinguish inference replication from training reproducibility.
Journal-level acceptance would still require a frozen independent test set, matched
strong baselines, placement-shift experiments, calibrated uncertainty, and a causal
streaming evaluation.
