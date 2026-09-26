# Peer Review Report — Methodology and Statistical Evaluation

## Protocol note

The installed reviewer skill refers to a frozen v3.6.2 sprint contract, but that contract is not available in this review context. Per the review assignment, this report therefore uses the legacy independent-review format. No sprint-contract failure condition or panel-relative rule has been inferred or applied.

## Manuscript information

- **Title:** *Physics-Informed Hierarchical Tree Ensembles for Sparse-PMU Event Detection and Localization*
- **Target journal:** *IEEE Transactions on Power Systems*
- **Review date:** 2026-08-16
- **Review round:** Journal-readiness review of the current conference-format manuscript
- **Primary manuscript reviewed:** `paper/main.tex`
- **Supporting evidence checked:** the target-complete protocol, scenario manifest, per-seed and per-event metrics, saved paired-bootstrap results, benchmark source, and reproducibility README under `paper/evidence/revised/` and `paper/experiments/`

## Reviewer information

### Reviewer role

Peer Reviewer 1 — Methodology and statistical evaluation.

### Reviewer identity

A power-system machine-learning evaluation specialist with expertise in grouped validation, trajectory and asset holdout, distribution shift, uncertainty intervals, ablation design, operational timing, and computational reproducibility.

### Review focus

This review is restricted to whether the research questions are answered by the experimental design: target versus trajectory generalization, identification of the hierarchy and topology effects, data generation, nested dependence, threshold selection, bootstrap units, uncertainty, baselines, missingness, runtime measurement, and reproducibility. I do not assess the detailed physical fidelity of the ANDES models or the completeness of the power-systems literature.

## Overall assessment

### Recommendation

- [ ] **Accept**
- [ ] **Minor Revision**
- [x] **Major Revision**
- [ ] **Reject**

**Journal-readiness judgment:** Not ready for submission to *IEEE Transactions on Power Systems* in its current form. The required revision is experimental, not primarily editorial.

### Confidence score

**5/5.** The evaluation-design questions are directly within my expertise. The confidence applies to the methodological and statistical assessment, not to a specialist judgment on the physical fidelity of each simulated disturbance model.

### Summary assessment

The manuscript reconstructs a hierarchical ExtraTrees system and evaluates flat, typed, and typed+topology variants on 690 simulated IEEE 39-bus trajectories. It makes several unusually strong methodological choices for a conference paper: complete scenarios stay within one split, thresholds are selected on validation rather than test data, causal features are directly tested for non-anticipation, candidate coverage is audited, and the paper explicitly limits its claim to new trajectories at known targets. The authors also report both row- and scenario-level localization, per-event heterogeneity, and paired scenario-block intervals.

Those safeguards do not yet support the central journal-level questions. The topology comparison changes the localizer architecture, candidate-row construction, and training representation simultaneously with electrical-distance features, so it does not identify a topology effect; the manuscript correctly admits this at lines 49, 229, 236, and 242. The single target-covered 3/1/1 split tests only a narrow interpolation setting with fixed event onset, one network, one PMU placement, and limited operating/severity values. Three forest seeds on that fixed split quantify estimator randomness, not dataset, target, or shift uncertainty. Thresholds optimize row-weighted F1, while the operational claims concern false-alarm episodes, delay, and scenario decisions. Model-only batch timing is measured once per seed and excludes the dominant online pipeline stages. These limitations are sufficiently central that new, pre-specified experiments are needed before the work is journal-ready.

## Dimension scores

Scores use the skill's 0–100 quality rubric and are calibrated to the target journal. They are ordinal review aids, not estimated probabilities of acceptance.

| Dimension | Score | Descriptor | Methodology-specific basis |
|---|---:|---|---|
| Originality (20%) | 67 | Adequate | The hierarchy/topology combination is potentially useful, but the distinctive topology contribution is not isolated. |
| Methodological rigor (25%) | 48 | Weak | Good leakage controls coexist with a single target-covered split, compound ablations, incomplete operational timing, and insufficient treatment of dependence. |
| Evidence sufficiency (25%) | 52 | Weak | One simulated network/PMU placement and three model seeds do not support transfer or journal-scale robustness claims. |
| Argument coherence (15%) | 82 | Strong | The paper distinguishes prototype, controlled evaluation, and unsupported claims with commendable clarity. |
| Writing quality (15%) | 84 | Strong | Methods and limitations are concise and generally precise; several statistical definitions still need fuller specification. |
| **Weighted average** | **63.3** | **Major Revision** | New experiments are required to identify the claimed effects and quantify generalization. |

## Strengths

### S1. The generalization target is stated honestly

The abstract states that every target appears in training and that the results concern “new trajectories at known assets” (lines 34–35). Section IV repeats that the 3/1/1 within-target partition is target-covered (lines 168–170), and the Discussion identifies asset-signature memorization as an alternative explanation (line 236). This distinction prevents an otherwise serious overclaim from being hidden.

### S2. The primary split blocks complete trajectories

The use of whole-scenario train/validation/test partitions (lines 168–170) avoids direct row leakage across splits, which is essential because adjacent 30-Hz rows from the same five-second simulation are strongly dependent. Threshold selection is confined to validation, and test trajectories are excluded from fitting and threshold selection (line 170). The saved manifest corroborates the stated 414/138/138 partition.

### S3. Temporal non-anticipation is specified and tested

The prediction contract is written explicitly as a function of samples available through time (t) (lines 121–127). The causal feature code uses trailing windows, and a prefix-invariance unit test is described at lines 179–180 and 187–188. This is a meaningful correction to the archived 30-s label-replication executable.

### S4. The comparison controls several important nuisance factors

The variants share trajectories, model seeds, a total 360-tree budget, and a common validation grid (lines 49, 175, and 182). The manuscript also discloses the effective `balanced_subsample` behavior with bootstrap disabled and the exact scikit-learn version (line 182). These controls make the reported system-level comparison more interpretable, even though they do not isolate every component.

### S5. Aggregate reversals and failure cases are not concealed

The paper reports that flat wins row-level physical Top-1, while typed has a higher but uncertain scenario-level point estimate (lines 227 and 234). Per-event results expose the domination of row-level support by line outages and very poor load localization (lines 229–231). Missing-only failure is stated directly (line 225). This reduces the risk of selective success reporting.

### S6. The paired resampling begins from the correct row-dependence unit

The reported intervals resample complete scenarios within event type and preserve pairing between methods and seeds (line 185). The saved bootstrap script confirms that rows are never sampled independently. This is materially better than treating thousands of time rows as independent observations, although the resampling still omits important higher-level sources of uncertainty discussed below.

## Weaknesses

### W1. The topology effect is not identified

**Problem:** The stated question is whether a candidate ranker “supplied with electrical distance improves localization” (line 49), but typed+topology replaces class heads with candidate-conditioned rankers and simultaneously changes candidate pooling, negative-sampled training rows, and the feature representation (lines 182 and 229). The manuscript explicitly concedes that this comparison cannot attribute the result to electrical distance (lines 49, 229, 236, and 242).

**Why it matters:** The title foregrounds “physics-informed” modeling, while the only explicit network input in the controlled benchmark is the treatment that is not isolated. A null or negative difference in this compound comparison is neither evidence that topology is ineffective nor evidence that the candidate ranker is inferior. Under the target-covered split, candidate identifiers and asset-specific signatures can also substitute for topology.

**Required fix:** Fit the same candidate-ranker architecture twice on identical candidate rows and identical negative samples: once with all distance/affinity columns masked and once with them present. Keep the detector, event router, tree allocation, training rows, hyperparameters, seeds, and threshold fixed. Evaluate both target-covered and grouped leave-bus/leave-line-out folds. Report oracle-route localization separately from end-to-end routed localization. Add a candidate-ID permutation/placebo check to determine whether the ranker is memorizing numeric bus/endpoint identity.

**Severity:** Critical for the paper's central contribution.

### W2. The split addresses trajectory leakage but not the intended localization challenge

**Problem:** Each target has three training, one validation, and one test trajectory (lines 168–170). This is one fixed within-target split, with all assets represented during fitting. It therefore measures interpolation around known targets. There is no repeated split, asset holdout, topology perturbation, PMU-placement shift, or external network.

**Why it matters:** Sparse-PMU localization is scientifically most interesting when asset signatures, operating conditions, or measurement configurations change. A class-based localizer is structurally incapable of emitting unseen classes; a candidate ranker is specifically valuable only if evaluated inductively. Model-seed variation cannot stand in for this missing data-level experiment.

**Required fix:** Separate the estimands explicitly: (i) known-target/new-trajectory performance; (ii) unseen-bus and unseen-line performance; and (iii) distribution-shift performance. Rotate the five replicas through repeated grouped train/validation/test assignments for the first estimand. For the second, use grouped target folds with all trajectories for a held-out asset excluded from fitting and threshold tuning. For the third, hold out ranges of onset time, severity, operating point, noise/missingness regime, PMU dropout/placement, and, ideally, a second network or network contingency.

**Severity:** Major.

### W3. The research questions lack estimands, primary endpoints, and decision margins

**Problem:** The experiments ask whether typed heads offer a “useful accuracy–size trade-off” and whether electrical distance improves localization (line 49), but “useful” is not operationalized. The paper reports many row-, scenario-, event-, integrity-, delay-, false-alarm-, size-, and runtime metrics without a pre-declared primary endpoint or multiplicity hierarchy. Typed is significantly worse for row-level F1 and physical Top-1 under the reported paired intervals, while its scenario Top-1 advantage is uncertain (lines 191 and 227).

**Why it matters:** A resource–accuracy trade-off cannot be accepted on preference alone. It requires an application-defined noninferiority margin or utility function. Similarly, multiple endpoints permit a favorable narrative to be chosen after observing results, even when individual numbers are correctly reported.

**Required fix:** State testable research questions before the experiment. For the typed system, pre-specify acceptable losses Δ for detection F1 and localization, together with storage and end-to-end latency constraints; demonstrate that the paired confidence bound satisfies those margins. For topology, designate scenario/target-level Top-1 or distance error as the primary localization endpoint and identify the exact evaluation population. Treat remaining endpoints as secondary or apply a hierarchical testing/reporting plan.

**Severity:** Major.

### W4. Current uncertainty intervals do not cover the main sources of sampling variation

**Problem:** Table III reports mean ± sample SD across only three forest seeds (lines 193–206). Those runs share the same generated trajectories and split, so the SD reflects algorithm randomness only. The paired bootstrap resamples 138 test scenarios within event type and resamples three paired model seeds, but it conditions on one training/validation split, one selected threshold per model, and one held-out trajectory per target. It cannot estimate within-target trajectory variability, split variability, simulation-design uncertainty, or calibration uncertainty. Only six pairwise contrasts receive intervals; absolute performance, per-event performance, delay, and runtime do not.

**Why it matters:** The effective inferential structure is rows within trajectories, trajectories within targets, targets within event types, crossed with model seeds. The paper's row counts (for example, 7,915 physical rows) greatly overstate the number of independent test units. Line outages alone contribute 3,128 of those rows (line 231), so pooled row metrics are also duration/support weighted.

**Required fix:** Generate or rotate multiple held-out trajectories per target and repeat grouped splits. Use a hierarchical or multiway paired analysis whose highest relevant sampling unit is the target for target-general claims and the trajectory within target for known-target claims. Propagate threshold selection by repeating it inside each outer split. Report 95% intervals for pre-specified absolute metrics and paired differences, along with `n` at every level. With only three model seeds, treat seed variation descriptively or increase the number of seeds; do not present a three-seed bootstrap as stable algorithm-population inference.

**Severity:** Major.

### W5. The data-generating design is narrow and partially confounded

**Problem:** The manuscript states only that replicas vary severity, operating scale, and simulator seed (line 168). The saved manifest shows four severity values (0.06, 0.09, 0.12, 0.15), five operating scales (0.97–1.03), a fixed 2-s onset, and fixed event durations. The benchmark source pairs the ten generator targets with eight PMU integrity sites deterministically for events 6 and 8 rather than crossing them. Event 8 additionally perturbs a randomly chosen load, but the saved protocol scores only the declared generator as the physical location. These details are not fully represented in the manuscript.

**Why it matters:** Fixed onset and limited deterministic grids make delay and robustness unusually easy to learn and do not represent deployment shift. Fixed physical/integrity pairs allow one signature to reveal the other target and can inflate composite-event localization. Scoring only the generator in a mixed generation+load event does not evaluate recovery of all physical origins and can reward incomplete localization.

**Required fix:** Publish the complete factorial design and justify ranges. Randomize or systematically hold out onset times, event durations, operating states, noise levels, and event severity. Cross physical and integrity targets using a balanced design or hold out unseen pairs. For mixed events, either use multi-label physical ground truth and appropriate exact/partial metrics or justify a pre-declared primary target while reporting recovery of every constituent target separately.

**Severity:** Major.

### W6. Thresholding and probability calibration are not aligned with operational claims

**Problem:** Flat and typed thresholds are independently selected to maximize row-level abnormal F1 over a 33-value validation grid (lines 182, 188, and 191). No score-calibration analysis is reported. The selected thresholds are then treated as fixed in the test bootstrap. False-alarm episodes and delay are operationally important, but the threshold objective does not constrain either. The validation rule's tie breaking is not stated in the manuscript.

**Why it matters:** Row-level F1 weights scenarios by duration and class occupancy. It may choose a threshold inappropriate for an alarm system with a specified false-alarm budget. Comparing FAR after independently optimizing F1 also does not compare methods at a common operating point. Threshold instability from using one validation trajectory per target is omitted from the intervals.

**Required fix:** Pre-specify the operational selection rule, such as maximizing scenario detection subject to a validation FAR limit, or report paired performance curves across thresholds. Add reliability diagrams and Brier score/ECE if probabilities are interpreted or thresholded across regimes. Repeat threshold selection inside each outer split and report its distribution. State grid, tie rule, confidence level, and whether the chosen point is interpolated or restricted to the grid.

**Severity:** Major.

### W7. Baselines do not isolate the hierarchy and do not establish an operational reference

**Problem:** Flat, typed, and typed+topology all use ExtraTrees and a fixed tree total (line 182). The flat-to-typed change simultaneously alters detection decomposition, event routing, and localizer partitioning. The topology treatment lacks a no-distance candidate-ranker control. There is no deterministic presence gate despite missingness being directly observable, no simple distance/nearest-PMU localizer, and no temporal detector under the same past-only inputs. Equal tree counts also do not imply equal compute because candidate expansion changes prediction work dramatically.

**Why it matters:** The present results compare three systems but do not identify which component produces a benefit or failure. The question is not which model family a reviewer prefers; it is whether the claimed hierarchy and topology effects exceed transparent alternatives under identical information and operating constraints.

**Required fix:** Add the minimally diagnostic controls: shared event stage + global versus typed location heads; identical candidate ranker without versus with distance; a deterministic missingness gate; a simple physical-distance/nearest-observed-PMU ranking heuristic; and a standard temporal/change-detection baseline using the same causal measurements. If a broad state-of-the-art claim is retained, include a competitive sequence or graph baseline under the same splits and inputs. Compare accuracy–resource frontiers across tree/compute budgets rather than one arbitrary 360-tree point.

**Severity:** Major.

### W8. Timing evidence does not support deployment latency or a statistically established speed advantage

**Problem:** Prediction time excludes feature computation (line 185), and the evidence code times one vectorized prediction call over the entire concatenated test table. Each seed is timed once, without warm-up, using `n_jobs=-1` on a shared multicore Windows system. Thus `s/min` is model-only batch throughput, not per-timestamp online latency. No repeated-run distribution or paired timing interval is reported.

**Why it matters:** The typed system's storage/timing benefit is one of the paper's supported positive conclusions (lines 35 and 234). Online sparse-PMU diagnosis also incurs alignment, missing-data handling, feature extraction, routing, candidate expansion, and output costs. A cold single batch can be dominated by scheduling and parallel startup.

**Required fix:** Measure end-to-end streaming latency and throughput, including ingestion/alignment, feature update, all heads/rankers, and output. Report batch size, thread affinity, warm-up, repeated runs, median and tail latency (p95/p99), and paired intervals on a fixed machine. Retain the current number only as “model-only vectorized batch throughput.” Report single-thread results or controlled thread counts in addition to `n_jobs=-1`.

**Severity:** Major because timing is part of the main trade-off claim.

### W9. Reproducibility is strong internally but incomplete for an external reader

**Problem:** Lines 187–188 state that manifests, hashes, seeds, and predictions are stored, and the supporting repository indeed contains those artifacts. The manuscript itself does not give a public archive/DOI, commit hash, executable command, environment lock, ANDES version, split/base/model seeds, negative-sampling rule, or full parameterization of the data generator. The original training corpus for the archived prototype was not retained (line 164).

**Why it matters:** A local artifact tree is not yet a reproducible journal supplement. Exact software and simulation versions can change trajectories, and the ranker's random negative sampling affects its training distribution.

**Required fix:** Archive code, manifests, generated scenario metadata or regeneration scripts, calibration values, predictions, and analysis outputs under a persistent release. Provide commit and data hashes, environment lock/container, ANDES and solver versions, deterministic commands, all random seeds, and expected checksums. Clearly separate what is reproducible for the reconstructed benchmark from what cannot be reproduced for the historical prototype.

**Severity:** Major for journal readiness; the prototype limitation itself can remain if accurately scoped.

## Detailed comments

### Research questions and hypotheses

The two questions at line 49 are understandable but not yet falsifiable in journal terms. “Useful trade-off” needs a pre-specified loss margin and resource constraint. “Distance improves localization” needs a one-factor ablation. The manuscript should define the population, unit, estimand, endpoint, direction, and decision criterion for each question. Suggested estimands are:

1. Paired difference in known-target scenario-level localization on a new trajectory, averaged equally over targets.
2. Paired difference in unseen-target localization for held-out buses/lines, averaged over target folds.
3. Detection at a validation-fixed FAR budget, with delay analyzed jointly rather than conditionally among detected events.
4. End-to-end latency and storage difference subject to a pre-declared noninferiority margin in predictive performance.

### Research design and target/trajectory splits

The current within-target split is valid for the explicitly narrow known-asset estimand. It should remain as an interpolation experiment, not be replaced. A journal article needs a second inductive target-holdout experiment and a third shift/robustness experiment. The class-based flat and typed localizers are ineligible for unseen-label prediction by construction; that is not a reason to avoid the test. It means the unseen-target claim should be assessed with candidate-conditioned methods, while class-based methods are reported as “not applicable” or zero-coverage controls.

The manuscript should also clarify that the five replicas are distinct designed conditions rather than draws demonstrated to be i.i.d. A repeated grouped rotation of these five replicas would use the existing simulation budget more efficiently and expose split sensitivity, but broader randomization is still needed for a population-level claim.

### Identification of hierarchy and topology effects

Two factorial questions are currently mixed:

| Factor | Required matched control | What it identifies |
|---|---|---|
| Event hierarchy | Same features/localizer with global versus routed event stage | Whether decomposed detection/classification and routing help |
| Typed location heads | Same event predictions with global versus event-typed localizers | Whether restricting the candidate space helps |
| Candidate ranking | Same event stage with class head versus candidate-conditioned no-topology ranker | Effect of candidate representation |
| Electrical distance | Identical candidate ranker and rows with distance columns off versus on | Incremental topology effect |
| Target holdout | Same paired rankers on held-out asset groups | Inductive value rather than target memorization |

The negative samples must be paired between the topology-on/off rankers. Tree count, depth, features unrelated to topology, tuning effort, and candidate evaluation count should be held fixed or explicitly reported as part of a resource frontier.

### Data collection and simulation design

The scenario universe is comprehensive in target coverage for the fixed IEEE 39-bus case, but not in disturbance-condition coverage. The paper should report exact ranges/distributions for severity and operating scale, how simulator seeds affect initial conditions/noise, all event mechanisms and durations, and whether cases share the same calibrated measurement model. The current fixed 2-s onset should be randomized over a range that leaves enough pre-event and post-event exposure. Normal records should include long event-free scenarios rather than relying predominantly on pre-event portions of event simulations (line 238).

For events 6 and 8, generator and PMU sites should be independently varied or balanced. For event 8, the randomly selected load is a second physical source and must enter the target definition or a clearly justified secondary-target analysis. These are design and labeling issues, not merely missing prose.

### Analysis methods and nested dependence

Rows are appropriate for evaluating per-timestamp behavior, but they are not independent replicates. Report row metrics descriptively and base uncertainty on scenario/target clusters. The row/scenario reversal at line 227 is an empirical warning that the aggregation unit changes the conclusion. Equal-target and macro-event summaries should accompany duration-weighted row scores.

The current bootstrap is paired and scenario-blocked, which is good, but its inferential scope must be described as conditional on the generated target set, single split, selected thresholds, and fixed simulation/calibration design. A nested repeated-split analysis should refit models and reselect thresholds in each outer repeat. If the intended population is the finite set of 39 buses/34 lines, report finite-population, equal-target summaries. If the intended population includes future trajectories or grids, sample those levels explicitly.

The six reported intervals should be labeled 95% percentile intervals if that is the method used. Report the bootstrap estimand, number of independent units per stratum, and interval construction. There is no need to add p-values if effect estimates and well-designed intervals answer the questions.

### Threshold selection and calibration

Maximizing validation row-F1 is defensible only if row-F1 is the deployment objective. Otherwise, thresholds should be chosen using a utility or constraint that corresponds to false-alarm tolerance and detection delay. A comparison at common recall and a comparison at common FAR would be more interpretable than comparing independently F1-optimal thresholds alone. Threshold performance curves and calibration plots would show whether 0.75 versus 0.45 reflects genuine score quality or merely different scales.

### Results presentation

Table III should state the independent units directly: 138 test scenarios, 121 physical-target scenarios, 36 integrity-target scenarios, and three model seeds. The table's ± values are sample SD over seeds, not uncertainty over future trajectories. Primary absolute metrics and paired differences need 95% intervals. Per-event supports should appear in the manuscript rather than only the figure/evidence files.

Scenario Top-1 is calculated using the modal prediction over ground-truth active rows (line 185). This is a valid offline diagnostic but not an implementable online rule because active rows are known from labels. Add an online aggregation rule that starts from a predicted alarm and report time to first correct/stable location. The first-arrival tie break should be pre-specified and tested for sensitivity.

Detection delay is reported among detected scenarios (line 191). Because misses are excluded, it is vulnerable to survivorship bias: a conservative model can appear fast by missing hard events. Continue reporting detection count, but add a censored time-to-detection curve or a miss-penalized delay/utility metric and paired uncertainty.

### Missingness and integrity events

The benchmark makes missing-only records a target event but provides little benign or background missingness. A deterministic presence gate could therefore look perfect without being operationally safe. Evaluate it against a learned detector under varied burst lengths, asynchronous dropout, isolated packet loss, channel-specific loss, PMU-wide loss, timestamp gaps, and normal communication failures. Report false alarms caused by non-event missingness. The composite events should use unseen physical–integrity pair tests to prevent one target from revealing the other.

### Runtime and resource analysis

The disclosed hardware and runtime scope are helpful, but Table III should relabel `s/min` as vectorized, model-only throughput. A candidate ranker evaluates every candidate and is therefore not compute-matched to a class head despite equal tree totals. Report number of candidate evaluations, CPU time, wall time, peak memory, and end-to-end online latency. Size should be measured from the released serialized artifact using a documented format/compression level, in addition to in-memory pickle byte counts if those are retained.

### Reproducibility

The saved protocol, manifest, predictions, input hashes, self-check, and benchmark scripts constitute a strong foundation. Journal readiness requires making them external and immutable. The paper should include a code/data availability statement and a minimal reproduction table covering simulator/solver version, operating system, Python/library lock, input case hashes, all seeds, split construction, threshold rule, negative sampling, command line, expected outputs, and license. The statement at line 188 that checks reject “non-finite trajectories” should distinguish finite full-state ground truth from intentionally missing PMU observations.

## Statistical reporting completeness

### Overall rating

**52/100 — Needs Improvement.** The manuscript reports useful descriptive metrics and selected paired intervals, but its inferential reporting does not yet match the hierarchical experiment or the main operational claims.

| Component | Weight | Score earned | Assessment |
|---|---:|---:|---|
| Descriptive statistics | 15 | 12 | Split sizes, row supports, seed means/SDs, and exposure are partly reported; target/scenario counts and distributions should be in the main tables. |
| Effect estimates | 20 | 13 | Raw paired metric differences are interpretable effect sizes, but only selected contrasts are covered and trade-off margins are absent. |
| Confidence intervals | 15 | 7 | Six paired intervals are reported; absolute, per-event, delay, calibration, size, and timing intervals are missing. |
| Dependence/assumption handling | 15 | 5 | Scenario blocking is appropriate, but target nesting, fixed-split training variation, threshold selection, and three-seed limitations are not propagated. |
| Sample-size/power or precision rationale | 10 | 0 | Five replicas and three seeds are not justified by detectable effect or desired interval width. A simulation-based precision analysis is preferable here to a generic classical power calculation. |
| Missing-data reporting | 10 | 5 | Missingness is modeled as an event and coverage is reported, but background missingness mechanisms and robustness are not characterized. |
| Statistical format/specification | 10 | 7 | Metric notation is clear; interval level/type, bootstrap details, units, and the meaning of ± should be more explicit. |
| Red-flag control | 5 | 3 | Test isolation and negative results are disclosed, but many endpoints lack a primary hierarchy and model-development/test chronology is not frozen in an external record. |
| **Total** | **100** | **52** | **Needs Improvement** |

Classical normality or homoscedasticity tests are not the relevant standard for these bounded performance metrics. The relevant assumptions are exchangeability of resampled clusters, independence between the highest-level sampling units, stability across split/design choices, and alignment between the resampled population and the claimed population.

## Methodological fallacy and validity audit

| Risk | Finding | Status | Corrective evidence |
|---|---|---|---|
| Row-level pseudoreplication | Thousands of rows are dependent within 138 test trajectories. The scenario bootstrap avoids the worst form, but Table III's row metrics can still look more precise than the independent-unit count warrants. | Present, partially mitigated | Clustered intervals and explicit `n` at row/scenario/target/event levels. |
| Overfitting / asset memorization | All targets occur in training; class heads and candidate identity fields can learn asset signatures. | Present and acknowledged | Leave-bus/leave-line-out folds plus ID permutation/placebo tests. |
| Confounding / compound treatment | Topology, ranker architecture, row construction, and candidate pooling change together. Flat-to-typed also changes several stages. | Present | Factorial one-change-at-a-time ablations. |
| Simpson's paradox / aggregation reversal | Flat wins row Top-1 while typed's scenario point estimate is higher; event supports are highly unequal. | Empirically visible and responsibly disclosed | Pre-specified primary unit plus macro-event/equal-target sensitivity analyses. |
| Survivorship bias | Mean delay excludes missed scenarios. | Present | Censored or miss-penalized time-to-detection analysis. |
| Selection / multiplicity | Numerous endpoints are reported without a primary hierarchy; threshold selection is validation-based, but the architecture-development chronology is not pre-registered. | Risk; no evidence of p-hacking | Freeze analysis plan, archive chronology, designate primary endpoints, and reserve an untouched final holdout. |
| Distribution-shift neglect | One network, one PMU placement, fixed onset, narrow operating/severity grid, RAW-derived calibration. | Present | Pre-declared operating, timing, noise, missingness, topology, and network shift suite. |
| Composite-target confounding | Events 6/8 use fixed generator–PMU pairs; event 8 includes an unscored load target. | Present | Balanced/crossed pairs and multi-target scoring. |
| Reverse causation / endogeneity | Not applicable to this controlled predictive simulation study. | Not detected | No action. |
| Selective reporting of failures | Missing-only failure, null topology result, and per-event losses are reported. | Not detected | Retain this transparency. |
| Causal-language overreach | “Causal” is used for temporal non-anticipation, not causal treatment inference; the paper denies identification of the distance effect. | Largely controlled | Prefer “non-anticipative” for features and reserve “effect” for matched ablations. |

## Evidence needed to resolve model-choice claims

No model should be preferred because it is flat, hierarchical, graph-based, temporal, or fashionable. The following evidence would make the choice empirical.

| Claim | Minimally sufficient evidence | Decision rule |
|---|---|---|
| Typed heads offer a useful trade-off | Repeated grouped paired comparison, pre-specified noninferiority margins for detection/localization, and repeated end-to-end latency/storage measurements | Prefer typed only if predictive lower bounds satisfy the margins and resource gains satisfy declared constraints. |
| Electrical distance contributes | Identical candidate ranker with topology columns off/on, paired candidate rows/negative samples, target-covered and leave-target-out folds | Attribute a gain to distance only if the paired interval and robustness analysis favor topology under the pre-specified primary endpoint. |
| Candidate ranking generalizes | Held-out bus/line folds with all trajectories of each held-out target excluded from fitting/tuning | Require nontrivial coverage and accuracy above simple distance/nearest-PMU controls. |
| Detector is operationally usable | Long normal exposure, varied unseen onset/missingness/noise, threshold fixed by a validation FAR constraint, delay including misses | Require the stated FAR budget, scenario detection target, and latency bound to be met together. |
| Missingness handling is adequate | Presence-gate and learned controls under event and benign dropout mechanisms | Prefer the simpler gate only if it retains event recall without unacceptable benign-dropout FAR. |
| 360-tree allocation is efficient | Accuracy–latency–memory curves over several budgets with identical tuning opportunity | Compare Pareto frontiers, not a single budget point. |

## Prioritized actionable revision plan

### Required before journal submission

1. **Pre-specify questions and endpoints.** Define known-target, unseen-target, and shift estimands; choose primary endpoints; set noninferiority or operational margins.
2. **Run a fixed-architecture topology ablation.** Same candidate ranker, training rows, negatives, seeds, and routing; toggle only distance/affinity features. Include an ID-placebo control.
3. **Add grouped target holdout and repeated trajectory splits.** Leave buses and lines out of all fitting/tuning; rotate or regenerate trajectories so each target has multiple held-out realizations.
4. **Broaden and deconfound simulation conditions.** Randomize onset/duration and broaden operating/noise/severity regimes; cross physical and integrity targets; score every origin in mixed events.
5. **Refit uncertainty analysis to the design.** Repeat training and threshold selection in outer grouped splits; use target/trajectory-aware paired intervals and report absolute as well as difference intervals.
6. **Align thresholding with deployment.** Select on a validation FAR/utility constraint, show threshold curves and calibration, and propagate threshold selection uncertainty.
7. **Add diagnostic baselines.** At minimum: no-topology candidate ranker, deterministic missingness gate, simple distance heuristic, and causal temporal detector under identical information.
8. **Replace the timing claim with an end-to-end benchmark.** Use warm-up and repeated controlled runs; report median/p95/p99 latency, throughput, memory, and paired intervals.
9. **Release an immutable reproduction package.** Include environment, simulator version, seeds, manifests, input hashes, commands, predictions, and expected checksums.

### Strongly recommended analyses

1. Report oracle-route and end-to-end localization to separate event-routing errors from localizer errors.
2. Add equal-target, macro-event, and duration-weighted results side by side.
3. Use online, prediction-defined scenario aggregation and a censored or miss-penalized detection-delay analysis.
4. Provide learning curves over trajectories per target and resource curves over tree/compute budgets.
5. Reserve an untouched final evaluation set after the revised protocol and ablations are frozen.

## Questions for the authors

1. Is the intended population the finite set of known IEEE 39-bus targets, unseen trajectories at those targets, unseen assets on the same grid, or new grids? Which one is the primary journal claim?
2. Can the authors run an otherwise identical candidate ranker with the electrical-distance and affinity columns removed, using exactly the same negative candidate rows, and evaluate it under leave-bus/leave-line-out folds?
3. How were the 136-feature map, depth, tree allocation, negative-sampling ratio, scenario aggregation rule, and selected comparisons frozen relative to inspection of the current test predictions? Is an untouched final holdout available?
4. Why are event-6/event-8 generator and integrity sites deterministically paired, and how should localization be scored when event 8 also changes a load that is not included in the reported physical target?
5. What false-alarm budget and end-to-end latency constraint define an operationally acceptable detector, and would the threshold be chosen differently under those constraints?

## Minor and reporting issues

- **Lines 185 and 191–229:** Label every interval as 95% and state percentile/basic/BCa construction. State that model seeds are resampled as paired blocks if that remains the implementation.
- **Table III, lines 193–206:** Expand “mean ± sample standard deviation across three model seeds”; do not let readers interpret the bars as trajectory-level uncertainty.
- **Line 185:** Rename runtime to “model-only vectorized batch throughput”; reserve “latency” for end-to-end per-decision timing.
- **Line 188:** Clarify that the finite-value check concerns latent/full-state trajectories and that intended missing PMU observations are allowed.
- **Lines 168–170:** Report the exact severity/operating grids, split seed, base simulation seed, and model seeds, or point to a permanent protocol table.
- **Line 185:** State the number of test scenarios and targets contributing to each endpoint, not only the number of rows.
- **Lines 225 and 238:** A presence gate should be evaluated, not only proposed, and must be tested against benign missingness to quantify false alarms.
- **Lines 227–231:** Add confidence intervals to per-event and absolute scenario metrics; the three-seed SD alone is not sufficient.
- **Line 236:** Retain the explicit statement that a class localizer can memorize asset signatures; it is an important scope condition.

## Final recommendation rationale

The manuscript has a sound methodological instinct: it detects and discloses its own strongest limitations, avoids row leakage across partitions, and resists claiming a topology advantage that the current comparison cannot identify. That transparency makes a strong journal revision feasible. It does not substitute for the missing evidence. For *IEEE Transactions on Power Systems*, the decisive additions are a fixed-architecture topology ablation, asset-level holdout, repeated grouped uncertainty with threshold refitting, deconfounded composite-event generation, operational threshold/timing evaluation, and a permanent reproduction package. Until those experiments are completed, the work remains a credible target-covered conference benchmark rather than a journal-complete demonstration of physics-informed sparse-PMU localization.
