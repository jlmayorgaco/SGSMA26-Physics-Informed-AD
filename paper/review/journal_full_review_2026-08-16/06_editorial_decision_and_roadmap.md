# Editorial Decision and Executable Revision Roadmap

## Protocol status

This is a **legacy editorial synthesis, not a sprint-contract-validated decision**. The installed `academic-paper-reviewer` package describes a frozen v3.6.2 sprint-contract process, but the required frozen contract was not available to the independent reviewers or to this synthesis. No contract dimensions, panel-relative failure conditions, or mechanical contract decision have therefore been inferred. Reviewer scores are inventoried as ordinal aids only; they are not averaged into an acceptance probability.

This synthesis uses the Editor-in-Chief (EIC), Reviewer 1 (methodology), Reviewer 2 (synchrophasor domain), and Reviewer 3 (graph/physics-informed learning) as the four-member consensus panel. The Devil's Advocate (DA) is handled separately and does not enter the four-reviewer consensus count. No issue below is introduced independently of the five reports.

## Manuscript information

- **Title reviewed:** *Physics-Informed Hierarchical Tree Ensembles for Sparse-PMU Event Detection and Localization*
- **Target journal:** *IEEE Transactions on Power Systems* (TPWRS)
- **Decision date:** 2026-08-16
- **Review stage:** Pre-submission journal-readiness review of the existing six-page conference-format manuscript
- **Manuscript ID:** Not provided

## Decision at a glance

### Decision on the current manuscript: Reject — Premature; a new submission is encouraged

The present manuscript should **not be submitted to TPWRS as a regular article**. This is not a finding that the research direction is unpublishable. It is an editorial judgment that the remedy is a new journal-scale experimental study and a newly unified research object, not a bounded revision of the present six-page paper.

### Roadmap for a new journal version

A new TPWRS submission is encouraged if it completes the required experimental gates below: an identifiable physics/topology intervention, asset-disjoint candidate ranking, broader physical and measurement conditions, system/placement/topology transfer, design-aligned uncertainty, operational alarm evaluation, and an immutable reproducibility package. The archive reconstruction should become motivation or an audit case study; it should not be presented as the system validated by the new benchmark.

### Recommended resubmission target

The primary target remains **IEEE Transactions on Power Systems**, but only after all required-revision acceptance criteria are met. If the project remains primarily an applied benchmark/audit on limited systems rather than a systems-level transferable contribution, the field analysis supports **IEEE Open Access Journal of Power and Energy** or **International Journal of Electrical Power & Energy Systems** as stronger alternatives. The current PMU/WAMS framing should not be redirected to *IEEE Transactions on Smart Grid*, which the field analysis identified as unsuitable for this transmission-system application.

---

# Part I — Editorial Decision Letter

Dear Authors,

Thank you for presenting *Physics-Informed Hierarchical Tree Ensembles for Sparse-PMU Event Detection and Localization* for a pre-submission assessment against the standards of *IEEE Transactions on Power Systems*. The manuscript has clear topical fit and several notable strengths: whole-scenario splitting, an explicit past-only prediction contract, an audit of invalid and incomplete candidate universes, unusually candid reporting of negative results, and a concrete model-size/model-only-throughput trade-off. The four panel reviewers also recognized the care with which the paper distinguishes the archived prototype from the corrected benchmark.

The editorial decision on the current manuscript is **Reject — Premature; a new submission is encouraged**. All four panel members agree that the manuscript is not ready for TPWRS and requires substantial new experiments. The three peer reviewers describe the remedy as Major Revision; the EIC recommends rejection in the present form because the necessary work creates a new journal-scale study rather than revising the existing conference paper. I adopt the EIC's procedural assessment on journal fit and manuscript maturity while retaining the peer reviewers' conclusion that the research direction is technically salvageable.

The central reason is that the title-level physics-informed localization claim is not identified. Electrical distance is introduced only when the localizer architecture, candidate-row representation, and computation also change, and the resulting variant does not improve aggregate localization. At the same time, every target occurs in training, so asset-template recognition remains a more parsimonious explanation than transferable spatial inference. One grid, one prescribed PMU placement, fixed event timing, short normal exposure, and model-only timing do not establish a systems-level or operator-facing capability. Finally, the controlled 136-feature, 360-tree benchmark is not the archived 45,162-variable, 3,490-tree submission, so it cannot validate that historical system.

These issues are serious but addressable through a newly designed study. The roadmap below specifies the evidence needed for a competitive new TPWRS submission. The resubmitted manuscript should make one research object and one contribution primary, pre-specify its estimands and decision criteria, and let the resulting evidence determine whether “physics-informed” remains in the title.

Sincerely,  
Associate Editor / Editorial Synthesizer

---

# Part II — Reviewer Inventory

## Recommendations, confidence, and score inventory

| Reviewer | Role | Recommendation on current form | Confidence | Reported overall score | Editorial interpretation |
|---|---|---|---:|---:|---|
| EIC | TPWRS journal fit and systems significance | **Reject in present form; encourage substantially redeveloped new submission** | 4/5 | 61.3/100 weighted; first impression 7/10 | Journal-fit judgment carries greatest weight on whether the work is a revision or a new submission. |
| R1 | Methodology and statistical evaluation | **Major Revision; not journal-ready** | 5/5 | 63.3/100 weighted; statistical-reporting completeness 52/100 | Highest weight on estimands, grouped validation, ablation identification, uncertainty, thresholds, timing, and reproducibility. |
| R2 | Synchrophasor/WAMS domain | **Major Revision; near major-revision versus reject/resubmit boundary** | 5/5 | 64.1/100 weighted | Highest weight on PMU measurement realism, event physics, electrical-distance validity, observability/identifiability, and operational ontology. |
| R3 | Graph/physics-informed ML and candidate ranking | **Major Revision** | 4/5 | 64.3/100 weighted | Highest weight on physics-information taxonomy, matched graph/temporal/ranking controls, and levels of inductive generalization. |
| DA | Utility-facing situational awareness and model risk | No editorial recommendation or score assigned by design | Not assigned | 3 CRITICAL, 6 MAJOR, 3 MINOR issues | Independently challenges the core thesis; handled outside the four-reviewer vote. |

The four weighted scores should not be averaged: the reports used different dimension sets and weights, and all explicitly state that the legacy scores are uncalibrated. Their common range nevertheless supports the same ordinal conclusion: promising work with evidence and positioning below journal readiness.

## Dimension-score inventory

| Dimension | EIC | R1 | R2 | R3 |
|---|---:|---:|---:|---:|
| Originality | 56 | 67 | 56 | 72 |
| Methodological rigor | 61 | 48 | 58 | 52 |
| Evidence sufficiency | 48 | 52 | 54 | 48 |
| Argument coherence | 76 | 82 | 82 | 82 |
| Writing quality | 80 | 84 | 84 | 84 |
| Literature integration | 58 | Not scored | 60 | 63 (optional) |
| Significance and impact | 57 | Not scored | 68 | 76 (optional) |
| **Reported weighted average** | **61.3** | **63.3** | **64.1** | **64.3** |

## Report-content inventory

| Reviewer | Strengths | Main weaknesses | Author questions | Explicit minor/reporting issues |
|---|---:|---:|---:|---:|
| EIC | 4 | 5 (2 Critical, 3 Major) | 4 | 5 |
| R1 | 6 | 9 (1 Critical, 8 Major) | 5 | 9 |
| R2 | 5 | 5 (3 Critical, 2 Major) | 12 | 7, plus terminology and reference recommendations |
| R3 | 4 | 5 (1 Critical, 4 Major) | 4 | 6 |
| DA | Observations recorded separately | 3 Critical, 6 Major, 3 Minor | Not framed as author questions | 3 |

---

# Part III — Consensus Analysis and Arbitration

## Points of agreement

1. **[CONSENSUS-4] The current paper is not ready for TPWRS and needs new experimental evidence.** EIC recommends reject/resubmit; R1, R2, and R3 recommend major revision, but all four state that the remedy is experimental rather than editorial. Sources: EIC Overall Assessment and W1–W4; R1 Overall Assessment and W1–W9; R2 Overall Assessment and Required Domain Experiments; R3 Overall Assessment and Priority Revision Package.

2. **[CONSENSUS-4] The physics/topology claim is not isolated.** All four reviewers require either a one-factor physical/topology ablation or a narrower title and claim. The common minimum is the same candidate ranker, same rows and negatives, same routing and model budget, with the distance/physical variables toggled only. Sources: EIC-W2; R1-W1; R2-W1–W2; R3-W1–W2.

3. **[CONSENSUS-4] Target-covered trajectory holdout cannot establish transferable localization.** All four reviewers distinguish known-asset trajectories from asset-disjoint localization and require leave-bus-out/leave-line-out evaluation for a candidate-conditioned scorer. Sources: EIC-W1–W2; R1-W2; R2-W4; R3-W3.

4. **[CONSENSUS-4] The evidence base is too narrow for the present systems-level claim.** The reports converge on broader operating conditions, randomized event timing, topology and PMU-placement variation, and evaluation beyond the single fixed IEEE 39-bus setup. Sources: EIC-W1; R1-W2 and W5; R2-W3–W4; R3-W3–W4.

5. **[CONSENSUS-4] Operational usefulness is not established.** All four identify the short normal exposure, high alarm-episode burden, missing-only failure, and/or incomplete end-to-end timing as barriers. They require a validation-selected alarm policy, realistic benign data-quality disturbances, long normal streams, online episode aggregation, and feature-inclusive latency. Sources: EIC-W4; R1-W6–W8; R2-W3 and W5; R3-W5.

6. **[CONSENSUS-4] The new paper must distinguish the archived artifact from the evaluated successor.** All four accept the audit as valuable, but none treats the 136-feature controlled benchmark as validation of the unavailable 45,162-variable archived training pipeline. Sources: EIC-W3; R1-W9 and Reproducibility; R2 Summary/Bottom Line; R3 Summary/Introduction.

7. **[CONSENSUS-4] Matched controls are needed to determine what the hierarchy, ranker, topology, and temporal summaries contribute.** The reviewers differ in the exact comparator emphasized, but all require controls under the same causal inputs and candidate universe. Sources: EIC-W2; R1-W7; R2-W1–W2; R3-W2.

8. **[CONSENSUS-3] Journal positioning and literature integration need substantial expansion.** EIC, R2, and R3 require a clearer novelty boundary, a contribution/claim taxonomy, and deeper positioning on sparse-PMU identifiability, electrical-distance alternatives, synthetic PMU realism, transfer, graph structure, and temporal/ranking methods. R1 did not oppose this; literature coverage was outside R1's declared remit and was not scored. Sources: EIC-W5 and Minor Issues; R2 Related Work and Missing Key References; R3 Related Work and Cross-Disciplinary Reading Recommendations.

9. **[CONSENSUS-3] Measurement and event conditions must be documented and broadened.** EIC, R1, and R2 explicitly require realistic measurement conditions, randomized event schedules, and broader noise/missingness/operating regimes. R3 does not dispute this but concentrates on topology and sensor-layout shift rather than PMU-domain measurement fidelity. R2's 5/5 domain expertise governs the detailed acceptance criteria. Sources: EIC-W1; R1-W2 and W5; R2-W3.

## Points of disagreement and editorial resolutions

### [SPLIT] D1 — Procedural decision: Major Revision versus Reject/Resubmit

- **EIC view:** Reject the current paper and encourage a substantially redeveloped new submission because a journal-scale experimental center and unified research object are missing.
- **R1/R2/R3 view:** Major Revision because the methodological foundation is credible and the required experiments are identifiable.
- **Disagreement type:** Severity/procedural disagreement, not a disagreement about scientific deficiencies.
- **Editor's resolution:** **Reject — Premature; new submission encouraged.** The EIC has the most relevant expertise on TPWRS fit and manuscript maturity. The peer reviewers' Major Revision recommendations are preserved as evidence that the direction is salvageable, but the scale and dependency structure of the required work exceed a conventional revision of the six-page manuscript.

### [SPLIT] D2 — Primary contribution route

- **Method-development route:** R1 emphasizes an identifiable candidate-ranker/topology experiment; R2 allows retention of “physics-informed” only after a defensible physical mechanism and matched test; R3 asks for a claim-to-mechanism taxonomy and matched graph/temporal/ranking controls.
- **Audit/benchmark route:** EIC, R2, and R3 all identify non-anticipation, candidate ontology, archive reconstruction, and negative-result reporting as the strongest presently supported contribution.
- **Disagreement type:** Direction disagreement about how to build the next paper, not about the weakness of the current title.
- **Editor's resolution:** For a new TPWRS submission, make the **reproducible causal successor and its identifiable physics-guided candidate ranking** the primary research object; use the archive audit only to motivate the prediction contract and ontology. If the authors cannot complete that experimental route, retitle and reposition the work as an audit/benchmark paper and reconsider the venue.

### [SPLIT] D3 — How much cross-system evidence is mandatory

- **EIC/R2 view:** A second, materially different transmission network is needed for a Transactions-level systems claim; R2 additionally requests field, public field-derived, or cross-simulator evidence for a compatible subtask.
- **R1 view:** A second network is strongly preferred, while pre-declared distribution shifts and independent regimes are the methodological minimum.
- **R3 view:** True new-graph transfer is mandatory only if the paper makes a graph-inductive claim; otherwise the limitation can be stated, although topology and sensor-layout stress are still required.
- **Disagreement type:** Severity/scope disagreement.
- **Editor's resolution:** Because the recommended target is TPWRS and the title/application makes a systems-level sparse-PMU localization claim, **a second network is required for the new journal version**. Field-event localization is not made an absolute gate because no reviewer establishes that compatible labels/data are available; however, a field-derived or public benchmark should be used for detection/measurement realism when feasible, with task incompatibilities stated.

## Complementary priorities, not disagreements

R1 prioritizes asset holdout and inferential design, while R2 emphasizes operating-point/topology transfer and PMU measurement realism. R3 separates known-asset, asset-disjoint, topology/layout-shift, and new-graph claims. These are complementary levels and should be implemented as an evaluation ladder rather than treated as competing protocols.

---

# Part IV — Independent Devil's Advocate Critical-Issue Disposition

Every DA CRITICAL is assessed independently below. The DA does not enter the CONSENSUS-4/3/SPLIT count.

| DA issue | DA argument | Corroboration outside DA | EIC assessment | Required author response |
|---|---|---|---|---|
| **DA-C1: Physics-informed thesis is not experimentally identified** | Network distance appears only in a compound architecture/representation change, the archived pair weights were one, and the topology variant provides no gain. | **Corroborated by all four:** EIC-W2; R1-W1; R2-W1–W2; R3-W1–W2. | **Valid and decision-determinative.** The EIC independently calls this a Critical mismatch between title-level originality and the experiment. | Complete R03 and R05 below. If the physical intervention has no identifiable value, remove or narrow “physics-informed” and report the negative result without attributing performance to grid physics. |
| **DA-C2: Asset-signature lookup is more parsimonious than transferable localization** | Every target appears in every partition; forests may recognize stable eight-PMU fingerprints rather than learn a transferable candidate relation. | **Corroborated by all four:** EIC-W1–W2; R1-W2 and validity audit; R2-W4; R3-W3. | **Valid and decision-determinative.** The EIC independently requires leave-bus-out/leave-line-out tests and states that the current design shows new trajectories at known assets only. | Complete R04. Use a shared candidate scorer, exclude held-out targets from all fitting/tuning, disclose whether candidates appeared as negatives, add nearest-signature retrieval and candidate-ID placebo controls, and keep known-asset results separately labeled. |
| **DA-C3: The benchmark is not the submitted system** | The archived and controlled systems differ in information horizon, features, tree count, candidate universe, and available training corpus; the benchmark cannot validate the archive. | **Corroborated:** EIC-W3 directly; R1-W9/Reproducibility; R2 Summary and Bottom Line; R3 Summary and Introduction framing. | **Valid.** The EIC independently identifies the two technical centers as a Major coherence problem. The audit remains useful, but validation claims about the historical system are unsupported without its corpus and an executable-equivalent comparison. | Complete R01 and R11. Choose the causal successor as the evaluated research object, state that the archive cannot be retrained, and confine historical claims to auditable properties of the executable and candidate set. |

Because DA-C1 through DA-C3 are all valid and heavily corroborated, an Accept decision is prohibited by the reviewer protocol and would be substantively untenable.

---

# Part V — Required Revisions for a New TPWRS Version

Effort is stated relatively because the reports do not establish available compute, simulator support, access to compatible field data, or the maturity of a second-system implementation. **Moderate** means mainly specification/reanalysis using current infrastructure; **High** means substantial implementation and repeated experiments; **Very high** means new data-generation, measurement-chain, or cross-system infrastructure. These are planning bands, not calendar promises.

| ID | Required revision | Source reviewers | Acceptance criteria | Effort and dependency |
|---|---|---|---|---|
| **R01** | Choose and unify the research object and primary contribution. | EIC-W3/W5; R1-W9; R2-W1/Bottom Line; R3-W1/Introduction; DA-C3 | The title, contribution list, methods, experiments, and conclusion refer to one reproducible causal successor. The archive is explicitly a motivating audit/case study. The paper makes no performance-validation claim about the historical model whose training corpus is unavailable. | Moderate; first dependency for all other work. |
| **R02** | Freeze claims, estimands, endpoints, information boundaries, and decision rules before the new test set. | R1-W3/W6; R2-W4; R3-W3/W4; EIC-W1/W5; DA-M5 | A protocol table separates known-asset trajectory shift, asset-disjoint ranking, topology/PMU-layout shift, and new-graph transfer. It declares primary endpoints, equal-target/event aggregation, noninferiority or operational margins where trade-offs are claimed, topology snapshot available at inference, threshold selection, and an untouched final holdout. | Moderate; follows R01 and precedes new experiments. |
| **R03** | Run a fixed-architecture physics/topology ablation. | EIC-W2; R1-W1; R2-W1/W2; R3-W1/W2; DA-C1/m3 | The same candidate ranker, candidate rows, paired negative samples, routing, tuning opportunity, features unrelated to topology, and compute/tree budget are used. Compare topology off, true topology on, and permuted/placebo topology; fully define `Y_bus`, reference, units, transformer/shunt treatment, topology time, line-candidate mapping, normalization, and distance range. Report event-family results and paired, design-aligned intervals in both known-asset and asset-disjoint regimes. | High; depends on R02 and feeds R12 title/claim choice. |
| **R04** | Test asset-disjoint localization and rule out template lookup. | EIC-W1/W2; R1-W2; R2-W4; R3-W3; DA-C2 and Alternative 1 | Leave-bus-out and leave-line-out folds exclude all positive trajectories of held-out assets from fitting and tuning. Only candidate-conditioned methods capable of scoring every admissible asset are compared; class heads are labeled not applicable or zero-coverage controls. The paper discloses whether held-out candidates appeared as negatives and includes nearest-signature/retrieval and candidate-ID permutation/placebo controls. Report exact, Top-k/rank, distance/zone, and coverage results with target-level uncertainty. | High; depends on R02 and a shared scorer from R03. |
| **R05** | Isolate hierarchy, candidate representation, and temporal/graph alternatives with matched baselines. | EIC-W2; R1-W7; R2-W1/W2; R3-W2; DA Alternatives 2 and 4 | At minimum, compare: global versus routed event stage under the same localizer; global versus typed location heads with the same event predictions; class head versus candidate-conditioned no-topology ranker; deterministic availability gate; simple distance or nearest-PMU/model-residual baseline; and a small causal temporal control. Add a lightweight graph control if a graph/topology advantage is claimed. All use identical causal inputs, candidate universes, split logic, and documented resource budgets. | High; depends on R02; can run in parallel with R03 after common infrastructure is fixed. |
| **R06** | Broaden and deconfound the event and PMU measurement design. | EIC-W1/W4; R1-W2/W5; R2-W3/W5; R3-W4; DA-M1/M4/M6 | Publish a channel/event table specifying source variables, PMU/PDC approximation, rate/filtering, angle reference, noise/bias, missingness, timing/alignment/latency, event mechanisms, severity, duration, and clearing/ramp rules. Randomize or hold out onset/duration and broaden operating, severity, noise, benign missingness, and communications regimes. Break deterministic physical–integrity target pairing and score every constituent target in mixed events or justify a pre-declared privileged target. | Very high; depends on R02; precedes final robustness and operational tests. |
| **R07** | Establish sparse-placement identifiability and systems breadth. | EIC-W1; R1-W2; R2-W4 and Domain Experiments 4–5; R3-W3/W4; DA Unexamined Premise | Evaluate several PMU placements/densities, topology/parameter errors, and PMU removal/replacement. Quantify candidate-pair separability or ambiguity zones and performance versus nearest-PMU/signature similarity. Repeat the frozen protocol on a second, materially different transmission network. Claims explicitly distinguish fixed-graph asset holdout from topology shift and new-graph transfer. | Very high; depends on R02/R06 and shared model interfaces. |
| **R08** | Rebuild uncertainty and aggregation around the experimental hierarchy. | R1-W3/W4/W6 and Statistical Reporting; EIC-W1; R2-W4; R3-W3; DA-M2/M5 | Refit models and reselect thresholds inside repeated outer grouped splits. Use trajectory-within-target and target-level paired inference appropriate to each estimand; report 95% interval type, independent-unit counts at every level, absolute metrics and paired differences, equal-target/macro-event/duration-weighted summaries, and seed variation as descriptive unless adequately replicated. Scenario localization starts from predicted online alarm episodes, not oracle active rows; delay includes misses through a censored or miss-penalized analysis. | High; analysis design follows R02 and execution follows R03–R07. |
| **R09** | Demonstrate an operator-relevant detection and integrity service. | EIC-W4; R1-W6/W7; R2-W5; R3-W5; DA-M1/M2/M5 and Alternatives 3/5 | Pre-specify an alarm budget and select thresholds/persistence/hysteresis on validation data. Test long contiguous normal streams plus benign packet loss, time skew, frozen/duplicated/out-of-order data, and full-PMU dropout. Report alarms per hour, misses, episode duration, time to stable correct shortlist, rank stability, abstention/coverage–risk, and event-family service levels. Evaluate the deterministic quality/presence gate on physical-only, integrity-only, and composite events; do not claim generic localization for load or missing-only routes unless their declared criteria are met. | High after R06; depends on online aggregation defined in R08. |
| **R10** | Replace the model-only speed claim with controlled end-to-end resource evidence. | EIC-W4; R1-W8; R2-W3; R3-W5; DA-M3/m1 | Measure ingestion/alignment, causal feature update, routing, candidate expansion, scoring, alarm aggregation, and output. Use warm-up and repeated controlled runs; report median/p95/p99 wall latency, throughput, CPU/thread setting, peak memory, model initialization, candidate rows/evaluations, and scaling with candidates/PMUs. Retain the current values only as model-only vectorized batch throughput and compare resource–accuracy frontiers against a stated deployment constraint. | Moderate to high; requires the frozen pipeline from R03–R09. |
| **R11** | Release an immutable, externally reproducible evidence package. | EIC-S1/W3; R1-W9; R2-W3; R3-W4; DA-C3/M4 | Archive code, environment lock/container, simulator and solver versions, manifests, inputs or regeneration scripts, hashes, all seeds, split/threshold/negative-sampling rules, commands, predictions, expected checksums, licenses, and the frozen analysis plan. Clearly mark which historical artifacts are unreproducible and which successor results can be regenerated independently. | Moderate; begins after R02 and closes after all experiments. |
| **R12** | Rewrite positioning, literature, terminology, title, abstract, and conclusion after evidence is frozen. | EIC-W5/Minor Issues; R2 Title/Related Work/Terminology; R3 Title/Related Work/Minor Issues; R1 reporting comments; DA-C1/C3/m2/m3 | The title uses “physics-informed” only if R03 identifies the claimed contribution; otherwise use physics-guided/network-aware or audit/benchmark language. The abstract defines phasor measurement unit, states simulation/network/known-target boundaries, and distinguishes model-only from end-to-end timing. A claim-to-mechanism table and related-work comparison establish novelty. Terms distinguish candidate coverage, observability, identifiability, alarm episodes/minute, line outage versus fault, and model/feature/communication/event latency. The conclusion names viable event families and limitations. Convert to current Transactions format. | Moderate; final dependency after R03–R11. |

## Required-revision publication gates

- **Gate A — Claim validity:** R01–R04 pass. The paper has one research object, pre-specified claims, an isolated physical intervention, and asset-disjoint evidence.
- **Gate B — Systems evidence:** R05–R08 pass. Baselines are matched, simulations and measurements are credible, system/placement/topology breadth is demonstrated, and uncertainty matches the sampling hierarchy.
- **Gate C — Operational evidence:** R09–R10 pass. Alarm burden, integrity behavior, online aggregation, and end-to-end resource use meet declared criteria.
- **Gate D — Reproducibility and presentation:** R11–R12 pass. Evidence is independently reproducible and the manuscript claims only what the frozen results support.

Failure of Gate A should trigger retitling/repositioning rather than post-hoc rescue of the physics-informed claim. Failure of Gate B or C should narrow the application claim and may make TPWRS unsuitable even if the algorithmic analysis is publishable elsewhere.

---

# Part VI — Suggested Revisions

These items are valuable but should not displace the required gates.

| ID | Suggested revision | Source | Expected improvement |
|---|---|---|---|
| **S01** | Report oracle-route and end-to-end routed localization side by side. | R1 Strongly Recommended 1; R2-W5; R3 assumption audit | Separates event-routing failure from localizer failure. |
| **S02** | Add region/corridor or ambiguity-set localization, reciprocal rank, and rank stability. | R2-W4/Results; R3 Secondary 2 and Minor Issues; DA Alternative 5/Unexamined Premise | Aligns the output with sparse-sensor identifiability and operator shortlist use. |
| **S03** | Add learning curves and accuracy–latency–memory curves across training and tree/compute budgets. | R1 Strongly Recommended 4 and W7; R3-W5 | Shows whether conclusions persist beyond the arbitrary 360-tree point. |
| **S04** | Compare pointwise with pairwise/listwise candidate-ranking objectives. | R3-W2 and Secondary 1 | Tests whether ranking supervision, rather than candidate representation alone, improves retrieval. |
| **S05** | Add a compatible field-derived/public detection subtask or cross-simulator check if exact location labels are unavailable. | EIC-W1; R2 Domain Experiment 5; R1-W2 | Strengthens measurement realism without pretending incompatible labels validate exact localization. |
| **S06** | Integrate the specific domain and adjacent-field references listed by R2 and R3 analytically rather than as a citation list. | R2 Missing Key References; R3 Cross-Disciplinary Reading Recommendations; EIC Minor Issues | Clarifies novelty and connects each experiment to the relevant research boundary. |

---

# Part VII — Phased Execution Checklist

The phases below encode dependencies rather than exact dates. R06 and R07 may dominate the schedule because simulator, PMU-chain, second-network, or compatible data infrastructure may not yet exist.

## Phase 0 — Editorial and protocol freeze (moderate effort)

- [ ] **R01:** Select the causal successor as the primary research object; demote the archive to an audit/motivation role.
- [ ] **R02:** Write and freeze the claim/estimand/endpoint/information-boundary table.
- [ ] Reserve an untouched final evaluation set and version the protocol before inspecting its outcomes.
- [ ] Define decision rules for retaining “physics-informed,” preferring typed heads, and claiming operational usefulness.

**Dependency exit:** No new final experiments begin until the protocol and decision rules are frozen.

## Phase 1 — Shared modeling and ablation infrastructure (high effort)

- [ ] **R03:** Implement one shared candidate scorer with topology off/on/permuted under paired candidate rows and negatives.
- [ ] **R04:** Implement asset-disjoint BUS and LINE folds plus nearest-signature and ID-placebo controls.
- [ ] **R05:** Implement the matched hierarchy, ranking, temporal, simple physical, and—if claimed—graph controls.
- [ ] Verify that every baseline uses the same causal inputs, candidate universe, splits, and tuning opportunity.

**Dependency exit:** The physics, hierarchy, and candidate-ranking effects can each be attributed to one changed factor.

## Phase 2 — Data-generating and physical-validity expansion (very high effort)

- [ ] **R06:** Document and implement randomized event timing, broadened operating/event regimes, realistic PMU/PDC and benign data-quality mechanisms, and deconfounded composite targets.
- [ ] **R07:** Add PMU-placement/density and topology/parameter stress, candidate-separability analysis, and a second network.
- [ ] Freeze the final data manifests and input hashes before final model fitting.

**Dependency exit:** The test population matches the intended TPWRS systems claim and no schedule/target pairing can serve as an unintended shortcut.

## Phase 3 — Repeated evaluation and uncertainty (high effort)

- [ ] **R08:** Run nested grouped fits with threshold reselection and target/trajectory-aware intervals.
- [ ] Report absolute and paired results at row, episode, trajectory, target, event-family, placement, topology, and network levels as pre-specified.
- [ ] Use prediction-defined online episode aggregation and include misses in delay analysis.
- [ ] Apply the frozen decision rules without choosing a favorable endpoint after seeing the test results.

**Dependency exit:** Every central claim has a stated population, independent unit, effect estimate, and uncertainty interval.

## Phase 4 — Operational and resource evaluation (high effort)

- [ ] **R09:** Tune and freeze alarm persistence/abstention under a validation alarm budget; test long normal and communications-artifact streams.
- [ ] Verify missing-only, load, and composite-event service levels or explicitly remove unsupported routes from the generic claim.
- [ ] **R10:** Benchmark the full streaming path with controlled repetitions, tail latency, memory, candidate scaling, and topology-feature refresh.

**Dependency exit:** The operator-facing claim is supported by alarm burden and end-to-end behavior, not model-only throughput.

## Phase 5 — Release and manuscript reconstruction (moderate effort after experiments)

- [ ] **R11:** Publish the immutable reproduction package and regenerate all paper tables/figures from it.
- [ ] **R12:** Rewrite the title, abstract, contribution statement, related work, methods, results, limitations, and conclusion from the frozen evidence.
- [ ] Integrate S01–S06 where they materially clarify the evidence.
- [ ] Convert to the current Transactions template and check terminology, units, candidate definitions, equation references, and table denominators.
- [ ] Prepare an internal point-by-point traceability response mapping every R01–R12 and adopted/declined S01–S06 item to the new manuscript and artifacts.

**Dependency exit:** All four publication gates pass and every claim can be traced to a frozen experiment and reproducible artifact.

---

# Part VIII — Reviewer Summaries

## Editor-in-Chief

- **Recommendation:** Reject in present form; encourage a substantially redeveloped new submission.
- **Confidence:** 4/5.
- **Core assessment:** Strong topical fit, transparency, ontology work, and practical trade-off reporting, but the present evidence does not identify the physics contribution or supply the systems-level breadth expected by TPWRS. The archive and controlled benchmark are not yet one journal contribution.

## Reviewer 1 — Methodology and statistical evaluation

- **Recommendation:** Major Revision; current version not journal-ready.
- **Confidence:** 5/5.
- **Core assessment:** Whole-scenario splitting, non-anticipation testing, validation-only thresholds, and scenario-blocked pairing are sound foundations. The topology treatment is compound, the split is target-covered, the inferential design conditions on one split and three seeds, thresholds do not match alarm objectives, and timing is not end to end.

## Reviewer 2 — Synchrophasor/WAMS domain

- **Recommendation:** Major Revision, near reject/resubmit boundary.
- **Confidence:** 5/5.
- **Core assessment:** The physical/integrity ontology and causal contract are useful, but “physics-informed,” the impedance-derived proximity, PMU measurement realism, event phenomenology, sparse-placement identifiability, and composite-event ground truth are under-specified or unsupported. A second network and broader measurement/event validation are needed.

## Reviewer 3 — Graph/physics-informed ML and candidate ranking

- **Recommendation:** Major Revision.
- **Confidence:** 4/5.
- **Core assessment:** The most credible contribution is the prediction contract, typed candidate ontology, and compact deployment path. A new version must distinguish feature-level physics guidance from constraints, use matched temporal/graph/ranking controls, reformulate asset holdout around shared candidate scoring, and separate asset-disjoint, topology-shift, and new-graph claims.

## Devil's Advocate — Utility-facing model risk

- **Recommendation/confidence/score:** Not assigned by the DA protocol.
- **Critical issues:** (C1) the physics-informed thesis is not identified; (C2) asset-signature lookup is a more parsimonious explanation; (C3) the benchmark cannot validate the archived submission.
- **Additional challenge:** The present alarm burden, oracle-active-row scenario metric, missing-only and load failures, fixed event schedule, and model-only resource claim do not demonstrate situational-awareness utility.
- **Editorial disposition:** All three CRITICAL issues are corroborated and valid. Each is mapped to mandatory acceptance criteria in R01/R03/R04/R05/R11/R12.

---

# Part IX — Resubmission Instructions

This decision does not invite a conventional revision of the current six-page manuscript. If the authors pursue TPWRS, they should prepare a **new journal manuscript** after completing R01–R12 and should treat this roadmap as an internal design and traceability document.

Before submission:

1. Verify all four publication gates.
2. Prepare a point-by-point internal response with evidence links for each required item and a reason for any declined suggested item.
3. Ensure the cover letter describes the resulting contribution directly and does not imply that the new benchmark validates the unreproducible historical model.
4. If the physics ablation or systems-breadth gates do not pass, retitle and reposition the work and select the alternative venue whose contribution expectations match the completed evidence.

The recommended editorial outcome is therefore: **do not submit the current manuscript to TPWRS; build and submit a new TPWRS version only after the required experimental and reproducibility gates are met.**
