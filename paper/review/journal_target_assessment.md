# Journal-target assessment

Review date: 2026-08-12

## Editorial verdict

The six-page manuscript now has a journal-facing argument: partial observability motivates a physics-informed representation, hierarchical event inference narrows the origin space, and a controlled benchmark tests the resulting accuracy--efficiency trade-off. The slides supply the method narrative and diagrams; the executable and evidence files set the claim boundary.

As written, it is submission-ready for a six-page IEEE conference track. It is not yet a complete journal article. Expanding the prose alone would not close that gap.

## Experiments needed for a journal version

1. Hold the candidate-ranker architecture and sampling fixed while adding effective distance, candidate masks, and $Y_{\mathrm{bus}}/Z_{\mathrm{bus}}$ response signatures one at a time.
2. Repeat the comparison with leave-bus-out and leave-line-out splits. Report both target-covered and unseen-target results.
3. Add the deterministic presence gate suggested by the missing-only failure and measure its effect on false alarms and delay.
4. Test at least one contemporary temporal or graph baseline under the same past-only protocol and candidate universe.
5. Vary PMU placement, operating condition, event onset, event severity, and normal duration. Include a learning curve and repeated partitions.
6. Report end-to-end latency, including feature extraction, and repeat timing after warm-up.

## Reviewer-facing risk register

- **Physics-informed claim:** supported as a representation and candidate-prior claim, not as a governing-equation-constrained learner.
- **Topology claim:** deliberately inconclusive because candidate pooling and localizer structure change with the distance input.
- **Generalization:** limited to new trajectories at known assets.
- **Field validity:** absent; RAW0001 calibrates the simulator and was used during development.
- **Integrity events:** missing-only detection is unresolved.
- **Reproducibility:** inference, ontology checks, figures, and the revised benchmark are reproducible; original training is not.

## Recommendation

Submit the current artifact as the concise conference paper. Use the fixed-architecture physics ablation and unseen-target protocol as the experimental center of a later journal manuscript; do not relabel the present six-page result as a finished journal study.
