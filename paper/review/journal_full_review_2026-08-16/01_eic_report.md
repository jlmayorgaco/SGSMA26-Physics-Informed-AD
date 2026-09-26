# EIC Review Report

## Review status and limitation

This is a **legacy-format, independent journal-readiness review of the existing six-page manuscript**. It is **not a sprint-contract-validated review** because the installed reviewer package does not contain the required frozen contract for a Phase 1 pre-commitment and Phase 2 contract check. The assessment is read-only and is limited to the Editor-in-Chief perspective requested here: journal fit, originality, systems-level significance, structural coherence, and the title/abstract/conclusion. Numerical scores are therefore best interpreted ordinally, not as calibrated acceptance probabilities.

## Manuscript information

- **Title:** *Physics-Informed Hierarchical Tree Ensembles for Sparse-PMU Event Detection and Localization*
- **Manuscript ID:** Not provided
- **Target journal:** *IEEE Transactions on Power Systems* (TPWRS)
- **Review date:** 2026-08-16
- **Review round:** Initial journal-readiness review
- **Manuscript form reviewed:** Six-page IEEE conference-format manuscript

## Reviewer information

### Reviewer role

Editor-in-Chief / journal-fit reviewer

### Reviewer identity

A senior TPWRS editor working in power-system dynamic performance and control-center analytics.

### Review focus

I assess whether the manuscript offers a systems-level contribution of lasting interest to the TPWRS readership, whether its originality is distinguishable from competition-specific implementation, and whether its title-to-conclusion argument is coherent. I do not attempt a line-by-line methodological or statistical audit.

## Overall assessment

### Recommendation

**Reject in the present form; encourage a substantially redeveloped new submission.**

The recommendation is not based on topic mismatch. The work is within TPWRS scope, but the current six-page study does not yet supply the originality and system-level evidence expected of a Transactions paper. A new submission could become competitive if it establishes the physics/topology contribution through a matched ablation and broadens the evaluation beyond target-covered trajectories on one simulated network.

### Confidence score

**4/5 — High confidence.** The journal-fit, contribution, and systems-significance questions are squarely within the stated editorial expertise. Confidence is lower on fine-grained statistical adequacy, which belongs with the methodology reviewer.

### First-impression score

**7/10.** The abstract and conclusion are unusually candid and quantitatively precise, but the title promises a physics-informed localization advance that the paper's own controlled results do not demonstrate.

### Summary assessment

The manuscript reconstructs a sparse-PMU event-diagnosis system and compares flat, typed, and typed-plus-topology ExtraTrees variants on 690 simulations of the IEEE 39-bus system. It makes useful distinctions between timestamp-level non-anticipation, physical versus integrity events, BUS/LINE/PMU candidate spaces, and trajectory holdout versus target holdout. The controlled results are reported with commendable restraint: flat performs better in detection and row-level localization; typed is substantially smaller and faster; scenario-level localization differences are uncertain; effective electrical distance adds no demonstrated gain; and missing-only events remain unresolved. These are credible engineering observations. However, the manuscript currently reads as a careful post-hoc audit and benchmark of a hackathon artifact rather than a journal-complete power-systems contribution. The evaluated topology variant changes several factors simultaneously, every target is present in training, the evidence comes from one network and one PMU placement, and the strongest positive contribution is a model-size/runtime trade-off rather than an advance in physics-informed localization. The archive reconstruction and the 136-feature causal benchmark also form two related but not fully unified technical objects. A TPWRS paper should turn this work into a clear systems-level claim and test that claim across assets, operating conditions, topology, placement, and realistic monitoring exposure. In its current form, I would not send it forward as a regular Transactions paper.

## Journal fit

The topical fit is strong. The current official TPWRS scope emphasizes a **systems viewpoint** and explicitly includes power-system dynamic measurements, intelligent and computing applications, transmission-system operation and security, energy control centers, and dynamic security assessment. Sparse-PMU event detection and localization can therefore serve the journal's readership directly ([official TPWRS scope](https://ieee-pes.org/publications/transactions-on-power-systems/)).

The contribution fit is weaker. TPWRS states that it seeks research results of lasting value, while the May 2026 PES author guidance says a Transactions paper must make a definite contribution to technical knowledge and supports that contribution with an appropriate application or practical example ([PES preparation and submission guidance](https://ieee-pes.org/publications/authors-kit/preparation-and-submission-of-transactions-papers/)). The manuscript's SGSMA framing, single IEEE 39-bus study, fixed PMU placement, fixed 2-s event onset, and target-covered split do not yet show that the conclusions survive as power-system knowledge beyond the task. Six pages are below the current ten-page first-submission maximum, so length itself is not a violation; rather, the present conference-scale compression leaves the journal contribution underdeveloped.

## Strengths

### S1: Unusually transparent boundary between the submitted artifact and the controlled evaluation

Section III-D and Table II (`tab:runtime`) clearly separate the presented design, the prototype realization, and the controlled evaluation. The manuscript discloses that the archived executable uses complete 30-s segments, copies a single label to every row, and has a line-candidate mismatch, whereas the benchmark uses a trailing 3-s window and valid candidate universes. This transparency prevents the reader from mistaking development scores for causal timestamp-level evidence and is a sound basis for a reproducibility or benchmarking contribution.

### S2: The paper reports negative and uncertain results without overclaiming them

The abstract, Results, Discussion, and Conclusion consistently state that the typed model's scenario-level localization point estimate (0.614 versus 0.590) has an interval including zero and that typed-plus-topology does not improve localization under the evaluated configuration. Section V further acknowledges that candidate pooling, training-row construction, and localizer architecture change together. This is editorially valuable: the principal claims generally follow the reported results rather than being rescued through selective emphasis.

### S3: Candidate ontology is treated as a systems problem rather than a generic label-space detail

Section III-C distinguishes 39 fault buses, 34 lines, 10 generator reporting buses, 18 load buses, and eight integrity sites. Section III-D then identifies the prototype line coverage of 24/34 valid transmission lines plus 10 transformer labels, and the load-label inclusion of BUS31. That BUS/LINE/PMU separation is directly relevant to control-center interpretation and gives the hierarchy a defensible physical rationale.

### S4: The practical trade-off is concrete

Table III (`tab:results`) reports that typed heads reduce serialized model size from 98.3 MB to 28.1 MB and model-only prediction time from 0.051 to 0.025 s per input minute, while also showing the associated losses in detection F1 and row-level physical Top-1. This trade-off is more informative than presenting accuracy alone and could matter for resource-constrained analytics if the end-to-end setting is established.

## Weaknesses

### W1: The systems-level evidence is not yet commensurate with a TPWRS article

**Problem:** Section IV evaluates 690 five-second ANDES simulations, all on the IEEE 39-bus system, with the same eight-PMU placement and event onset at 2 s. Every target key appears in training, validation, and test. Section VI correctly notes that RAW0001 only calibrates the simulations and does not provide field validation, that only 7.15 minutes are normal-labeled, and that transfer across networks, placements, event schedules, and targets is untested.

**Why it matters:** TPWRS evaluates the power system from a systems viewpoint. The current experiment shows discrimination among new simulated trajectories at known assets, not a general localization capability under partial observability. A high detection F1 on this design cannot by itself establish usefulness for transmission monitoring or control-center deployment.

**Suggestion:** Build the journal paper around a broader validation matrix: at minimum, leave-bus-out and leave-line-out tests; multiple operating points and randomized event onsets; PMU-placement sensitivity; topology and parameter perturbations; longer no-event exposure; and a second network of materially different size/structure. Include replayed or field synchrophasor data if available, even if only for detection or qualitative transfer. State in advance which results would support a system-level claim.

**Severity:** Critical

### W2: The title's “physics-informed” originality is not identified by the experiments

**Problem:** Section III-A provides physically interpretable departures, residuals, cross-PMU contrasts, and effective-distance equations. Yet it also states that the submitted extractor left pair weights at one. In the controlled benchmark, explicit electrical distance appears only in typed-plus-topology, which simultaneously replaces location heads with candidate rankers and changes the training representation. Table III then shows lower physical row-level Top-1 for typed-plus-topology (0.462) than typed (0.496), and Section V explicitly says the comparison cannot assign this difference to electrical distance.

**Why it matters:** The main title positions physics information as the source of the contribution, but the current experiment does not isolate or demonstrate a gain from that information. Hierarchical ExtraTrees with physically motivated features may be useful, but that is an incremental and different claim from a validated topology-informed localizer.

**Suggestion:** Hold architecture, training rows, tree budget, inputs, and candidate treatment fixed while adding/removing each physics component. Compare against a no-topology ranker and credible temporal/graph baselines under identical causal inputs and candidate sets. If physics does not improve accuracy but improves sample efficiency, unseen-target ranking, robustness, or calibration, make that the tested claim. Otherwise retitle the paper around non-anticipative evaluation and typed hierarchical diagnosis rather than “physics-informed” localization.

**Severity:** Critical

### W3: The manuscript has two technical centers that are not yet unified

**Problem:** Table I (`tab:featuremap`) describes a reconstructed 45,162-variable, 30-s archived representation, whereas the controlled study uses a new 136-variable, past-only 3-s representation (Section IV-B and Table II). The archived system cannot be retrained because its original corpus and feature tables were not retained (Section III-D); the reported journal evidence therefore pertains mainly to a different benchmark implementation.

**Why it matters:** The reader is left to decide whether the paper's contribution is artifact forensics, a corrected evaluation protocol, or a proposed hierarchical localizer. The title suggests the third, much of Sections III-A and III-D supports the first, and the main results support the second. This split weakens originality and makes the conclusion narrower than the method section.

**Suggestion:** Choose one primary contribution and subordinate the others. For a method paper, specify one reproducible end-to-end algorithm and evaluate that same representation throughout; move the archive audit to a motivating case study or supplement. For a benchmark/audit paper, foreground non-anticipation, ontology, and evaluation failure modes in the title and contribution statement, and demonstrate their generality on more than one model family or dataset.

**Severity:** Major

### W4: Operational significance is not yet established

**Problem:** Table III reports 3.22 alarm episodes per normal-labeled minute for flat and 4.01 for typed, while Section VI notes that normal exposure is only 7.15 minutes and is mostly pre-event. Runtime excludes feature computation. Section V also reports that typed sends every missing-only test row to normal, while flat recalls only 0.006±0.006 for that class; load localization remains below 0.05 for all methods.

**Why it matters:** These failure modes directly affect control-center usability and are more consequential than the modest model-only timing advantage. A monitoring claim needs false-alarm evidence over realistic durations, end-to-end latency, and a clear statement of which event/location outputs are operationally supported.

**Suggestion:** Evaluate hours rather than minutes of normal and perturbed operation; report end-to-end feature-plus-model latency and resource usage; separate detection, event identification, and localization service levels; and test the proposed presence gate rather than leaving the deterministic missing-only failure as future work. Frame unresolved load localization as a boundary on the proposed system, not a minor residual error.

**Severity:** Major

### W5: The title, abstract, and conclusion do not yet express a journal-scale contribution

**Problem:** The title overweights “physics-informed” despite the null topology result. The 156-word abstract is commendably quantitative, but it opens with the SGSMA task and names flat/typed variants before explaining the lasting power-system question. It does not define PMU, and “row-level physical Top-1” and “scenario-level localization estimate” require task-specific context. The conclusion accurately summarizes the benchmark but ends with a single next ablation; it does not state a control-center implication, a general technical lesson, or the conditions under which the approach should be used.

**Why it matters:** For a broad TPWRS readership, the title-to-conclusion chain should make the systems contribution visible without requiring familiarity with SGSMA. At present, the most defensible general result is an evaluation lesson—causal windows and candidate coverage materially change how sparse-PMU localization should be assessed—not a demonstrated physics-informed performance gain.

**Suggestion:** After the experimental center is strengthened, rewrite the title around the supported contribution. Recast the abstract as problem → contribution → evaluation design → decisive results → scope limitation, defining phasor measurement unit on first use. Make the conclusion answer what a system operator or researcher should do differently, and distinguish verified findings from the next hypotheses.

**Severity:** Major

## Detailed comments

### Title and abstract

The title is concise and searchable, and “Sparse-PMU Event Detection and Localization” accurately names the application. The difficulty is attribution: “Physics-Informed” is the leading qualifier even though explicit effective-distance information is not beneficial in the present comparison and was not active in the archived pair weighting. A defensible interim title would be closer to *Non-Anticipative Evaluation of Hierarchical Tree Ensembles for Sparse-PMU Event Diagnosis*, unless a fixed-architecture ablation later supports the physics-informed claim.

The abstract is within the PES 150–200-word guidance and contains the most decision-relevant numerical results. Its candor about the paired interval, missing-only failure, and known-target limitation is excellent. For a journal version, reduce competition-specific phrasing, define phasor measurement unit before “PMU,” identify the system-level gap, and state the contribution rather than primarily cataloguing three model variants. The phrase “halves model-only prediction time” should be paired with the actual timing values or removed, because excluding feature computation substantially limits its operational meaning.

### Introduction and structural coherence

The Introduction presents a coherent chain from sparse observability to propagation evidence, typed origin spaces, and non-anticipative evaluation. The final paragraph is also appropriately cautious that the topology comparison does not isolate electrical distance. What is missing is a short, explicit contribution list that distinguishes (i) archive/candidate audit, (ii) causal benchmark protocol, and (iii) model contribution. The paper should also state which of these is claimed as original relative to the cited PMU localization and graph-learning literature.

Across the manuscript, the archive reconstruction and the controlled benchmark compete for ownership of the narrative. Table II is valuable precisely because it reveals their differences, but it cannot substitute for a unified research object. A journal expansion should reorganize around one central hypothesis and use the artifact reconstruction as evidence for why the evaluation protocol is needed.

### Results and systems-level significance

Table III is well designed as a trade-off table, and the discussion correctly avoids claiming that 0.614 is superior to 0.590 when the paired interval includes zero. Nevertheless, no proposed model dominates: flat is better for detection and row-level localization; typed is smaller and faster; topology offers no overall localization gain and is much slower in the model-only timing; missing-only and load cases remain weak. A Transactions contribution can be a negative result or an evaluation framework, but the paper must explicitly establish why the result changes general power-system research or practice. At present, the evidence supports a local design lesson for this benchmark.

### Conclusion

The Conclusion is accurate, conservative, and aligned with the Results and Discussion. It does not over-infer from the point estimates, which is a substantial strength. It is also too narrow for TPWRS: the reader leaves with the next experiment (a fixed-architecture, leave-target-out ablation) rather than a supported systems-level conclusion. The journal version should state the demonstrated lesson, the domain of validity, the operational consequence, and the conditions that would invalidate deployment.

## Questions for the authors

1. What single contribution should a TPWRS reader cite this paper for: a new hierarchical method, a reconstruction/audit of the SGSMA artifact, or a non-anticipative evaluation protocol? Which result uniquely validates that contribution?
2. Can the authors perform a fixed-architecture experiment in which electrical distance is the only changed input and targets are held out by bus or line, so that the central physics-informed claim is identifiable?
3. What false-alarm burden and end-to-end latency would be acceptable in the intended control-center workflow, and how do 3.22–4.01 alarm episodes per normal-labeled minute and feature-computation time compare with that requirement?
4. Which conclusions do the authors expect to survive changes in network, PMU placement, topology, operating point, event onset, and measurement noise, and what minimum cross-system evidence can be added to test those expectations?

## Minor issues

- The source uses `\documentclass[conference]{IEEEtran}`. A TPWRS submission should use the current Transactions template and journal formatting; the current six-page format is useful for judging the existing manuscript but is not submission-ready.
- Define “phasor measurement unit (PMU)” in the abstract. The abstract should be self-contained under current PES guidance.
- “FAR” in Table III is episodes per normal-labeled minute, not the conventional probability/rate denominator many readers will assume. Retain the explicit unit in every occurrence or rename the measure “alarm episodes/min.”
- Define “row-level physical Top-1” in more general monitoring language in the abstract, because “row” is an artifact of the supplied task representation.
- The Related Work section is too compressed to establish precisely what is original about typed hierarchy, candidate ranking, and non-anticipative PMU localization. This is an editorial positioning issue rather than a request for indiscriminate citation expansion.

## Dimension scores

Scores use the seven-dimension legacy framework and are calibrated to first-submission TPWRS expectations. The weighted average uses the framework weights: originality 15%, methodological rigor 25%, evidence sufficiency 20%, argument coherence 15%, writing quality 10%, literature integration 10%, and significance 5%.

| Dimension | Score (0–100) | Descriptor | EIC rationale |
|---|---:|---|---|
| Originality | 56 | Weak | The ontology/non-anticipation audit is useful, but the hierarchy is incremental and the physics contribution is not isolated. |
| Methodological rigor | 61 | Adequate | Whole-scenario splits, past-only features, common budgets, and paired intervals are positive; the central topology comparison changes multiple factors. |
| Evidence sufficiency | 48 | Weak | One simulated network, one placement, fixed timing, known assets, short normal exposure, and no field/replay validation are insufficient for the systems claim. |
| Argument coherence | 76 | Strong | Claims are conservative and sections generally align, but the archive reconstruction and controlled benchmark are not one unified contribution. |
| Writing quality | 80 | Strong | The prose is precise, dense, and professional; terminology in the abstract remains too task-specific for a broad readership. |
| Literature integration | 58 | Weak | The 18-reference, two-paragraph Related Work section identifies relevant streams but does not establish the novelty boundary in depth. |
| Significance and impact | 57 | Weak | The problem is important, but present results do not yet show transfer or an operationally viable localization service. |
| **Weighted average** | **61.3** | **Major-revision band** | Numerically in the legacy major-revision band; the EIC recommendation is reject/resubmit because the required remedy is a journal-scale new experimental study, not manuscript-only revision. |

## Recommendation to peer reviewers

If the manuscript proceeds despite the editorial recommendation, I would ask the methodology reviewer to determine whether the topology comparison and paired resampling support the stated inferences; the synchrophasor reviewer to assess realism under PMU placement, noise, latency, and topology change; and the physics-informed-learning reviewer to evaluate whether a matched no-distance ranker and leave-target-out protocol can establish inductive localization. Those reviews should not revisit the journal-fit conclusion but should test whether the proposed redevelopment is technically feasible.

