# External-review adjustment

Review incorporated: 2026-08-12

The two long `Dictamen general` attachments are byte-identical (SHA-256
`2dc1e527...50d863`). They were treated as one review. The separate
`Physics-Informed` proposal was evaluated against the frozen artifact, saved
predictions, and current six-page conference scope.

## Claim and action matrix

| Review point | Evidence in the repository | Manuscript action | Status |
|---|---|---|---|
| The old title defends a hierarchy that loses row-level accuracy | Table III and paired bootstrap | Replaced it with a design-to-evaluation title that names the task and protocol | Resolved |
| `causal` may be read as causal inference | Features only satisfy temporal availability | Replaced scientific uses with `temporally non-anticipative` or `past-only` | Resolved |
| Streaming requirement was stated too categorically | Guide requires timestamp outputs and evaluates delay/FAR, but its window length is advisory | Framed the condition as the paper's streaming-valid interpretation and removed `task-breaking` language | Resolved |
| Candidate coverage needs formal treatment | Frozen line model has 24/34 valid labels and 10 invalid transformers | Added explicit coverage and invalid-set definitions and reported both counts | Resolved |
| Row-level localization overweights persistent events | Saved predictions contain complete scenario identifiers | Added deterministic scenario-level Top-1 and paired bootstrap intervals | Resolved |
| Event 8 is not open-set | All replicas use one supervised mixed-event recipe | Renamed it `mixed` and explicitly excluded open-set claims | Resolved |
| RAW0001 calibration is not neutral field validation | Normal labels set scale, noise, and sampling rate | Labeled it target-domain calibration and excluded field-validation language | Resolved |
| `balanced_subsample` with default bootstrap requires verification | scikit-learn 1.7.2 resolves it to global balanced weights when bootstrap is false | Stated the effective behavior and exact library version | Resolved |
| Tree scores are not calibrated probabilities | No sigmoid/isotonic calibration exists | Replaced probability language with class-vote or abnormal score | Resolved |
| Model runtime is not end-to-end | Recorded timing starts after feature computation and has one pass per seed | Kept the result but labeled it model-only and stated the timing limitations | Resolved |
| The ranker does not isolate electrical distance | It changes candidates, rows, features, and localizer architecture | Kept it as a topology-conditioned composite variant; no distance attribution | Resolved |
| `Physics-Informed` must mean network information reaches inference | Archived extractor received no distance matrix; ranker receives distance but is confounded | The title names the intended design; the body states exactly where distance enters and keeps its effect unassigned | Resolved for current evidence |
| Missing-only failure needs a deterministic integrity path | Typed recall is zero; flat recall is approximately 0.006 | Failure remains explicit; a presence gate is not claimed or inserted post hoc | Needs new experiment |
| Architecture effects require a 2x2 factorial | Current flat/typed comparison changes both event and localizer structure | Confounding is explicit; routing is not isolated | Needs new experiment |
| Physics needs an isolated ablation | No fixed-architecture candidate-mask/distance/signature sequence exists | Predeclared as the next campaign | Needs new experiment |
| Physics value should be tested under spatial shift | Every target occurs in all current partitions | Current claim is known-asset interpolation; leave-target-out is predeclared | Needs new experiment |

## Added scenario-level evidence

The physical scenario decision is the modal row-level Top-1 candidate over
target-active rows. Empty output participates as `NONE`; ties go to the
candidate that first reached Top-1. Across seeds, Flat scores
`0.590 +/- 0.017`, Typed `0.614 +/- 0.029`, and Typed+ranker
`0.581 +/- 0.029`. The paired Typed-minus-Flat interval is
`[-0.0551, 0.1019]`; the change in ranking is not statistically established.
This result is descriptive and uses fixed saved predictions. It did not alter
features, thresholds, model selection, or fitted weights.

## Predeclared journal extension

The current conference paper does not claim the following as completed. A
journal version should freeze the protocol before running it:

1. presence-based integrity gate with a traced Event 5 decision path;
2. event-head by localizer 2x2 architecture factorial;
3. fixed-architecture sequence: data only, candidate masks, effective
   distance, candidate-conditioned physical signatures, and full method;
4. known-target and leave-target-out protocols;
5. broader operating conditions and PMU placements;
6. temporal and boosting baselines;
7. repeated end-to-end latency and memory measurements.

Until those experiments exist, the performance effect of network information
remains open. The present paper reconstructs the intended design, identifies
the inputs that reached the executable, and states a controlled next test.

## Voice revision, 2026-08-12

The manuscript now follows a four-part arc: sparse-PMU diagnosis as a partially
observed spatial task; the hierarchical design presented at the hackathon; a
reconstruction of its executable behavior; and a matched non-anticipative
comparison. Repeated `audit`, `frozen`, `does not`, and `not evidence` formulas
were removed from the title, abstract, introduction, section headings, figure
and table captions, discussion, and conclusion. Limitations now follow the
result they qualify instead of leading the argument.

Major claim-to-evidence links remain unchanged:

| Claim | Evidence | Status |
|---|---|---|
| The archived hierarchy has 45,162 variables and typed localizers | Frozen schema and serialized model inventory; Table I | Supported |
| Complete-segment broadcasting and incomplete line coverage affect interpretation | Executable inference path and candidate record; Fig. 1 and Table II | Supported |
| Flat improves detection and row-level localization | Saved predictions and paired bootstrap; Table III | Supported |
| Event conditioning reduces serialized size and model-only time | Recorded model metadata and benchmark timing; Table III | Supported |
| The scenario-level localization ordering is uncertain | Scenario modal scores and paired interval | Supported |
| Electrical distance has an isolated positive effect | No fixed-architecture ablation | Removed from manuscript claims |

## Five-dimension self-review

- **Contribution: pass for a six-page conference paper.** The new knowledge is
  the executable mismatch between the stated method and runtime, plus the
  change in conclusions when temporal validity, candidate space, thresholds,
  tree budget, and aggregation unit are made explicit.
- **Writing clarity: pass.** `Non-anticipative`, `spatial-temporal`,
  `topology-conditioned`, `row-level`, and `scenario-level` have one meaning
  throughout. Abstract and Introduction claims map to Tables I--III and the
  executable reconstruction.
- **Experimental strength: bounded.** The matched comparison and paired
  scenario bootstrap are reproducible, but they are not a SOTA study. The paper
  says so directly.
- **Evaluation completeness: pass for the reconstruction and matched benchmark.** Missing-only,
  fixed architecture, isolated distance effects, unseen targets, and
  end-to-end efficiency still need experiments before a positive journal claim.
- **Method soundness: pass for the reported benchmark.** Future samples never
  affect earlier features, all candidate truths are valid, splits are by whole
  scenario, and model selection uses validation only. The archived runtime's
  mismatches remain explicit findings rather than hidden assumptions.
