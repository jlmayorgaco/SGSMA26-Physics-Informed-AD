# Peer Review Report — Cross-Disciplinary / Practical Perspective

## Process disclosure

This is an independent, read-only Peer Reviewer 3 report. I did not inspect the other reviewer reports and did not modify the manuscript. The installed reviewer skill refers to a frozen v3.6.2 sprint contract, but no such contract was supplied for this review. Per the review assignment, I therefore use the skill's legacy independent-review format and its 0–100 quality rubrics. The numerical scores are uncalibrated and should be interpreted ordinally, not as venue-acceptance probabilities.

## Manuscript information

- **Title:** *Physics-Informed Hierarchical Tree Ensembles for Sparse-PMU Event Detection and Localization*
- **Target venue:** *IEEE Transactions on Power Systems*
- **Review date:** 2026-08-16
- **Review round:** Journal-readiness review

## Reviewer information

### Reviewer role

Peer Reviewer 3 (Perspective)

### Reviewer identity

A researcher in graph signal processing and physics-informed machine learning for networked dynamical systems, with experience in candidate ranking and inductive generalization to unseen nodes and edges.

### Review focus

I assess what the term “physics-informed” denotes in this work, whether the empirical design distinguishes physical priors from architecture and representation changes, and what temporal, graph, and ranking alternatives would provide meaningful conceptual controls. I also examine asset-disjoint versus graph-inductive generalization and the gap between a competition prototype and an operator-facing monitoring pipeline. I do not attempt a statistical audit or a comprehensive review of the PMU literature.

## Overall assessment

### Recommendation

- [ ] **Accept**
- [ ] **Minor Revision**
- [x] **Major Revision**
- [ ] **Reject**

### Confidence score

**4/5.** The machine-learning, graph-signal-processing, candidate-ranking, and inductive-generalization questions are directly within my expertise. Some utility operating practices and synchrophasor-specific conventions are outside my primary specialization.

### Summary assessment

The manuscript reconstructs a sparse-PMU competition system and compares flat, typed, and typed-plus-topology ExtraTrees variants under a past-only, whole-scenario protocol. Its strongest contribution is not a demonstrated topology gain; it is the disciplined reconciliation of a non-anticipative prediction contract, a typed candidate ontology, and a compact tree-based deployment path. The paper is commendably explicit that every target is represented in training, that the topology comparison is not controlled, and that the topology-aware ranker does not improve aggregate localization. Those disclosures make the work more credible.

For *IEEE Transactions on Power Systems*, however, the title-level “physics-informed” claim is not yet empirically identified. The manuscript combines physically interpretable signal features, a domain-informed hierarchy, and one network-derived distance feature, but it neither enforces governing physics nor isolates the contribution of topology. It also lacks matched temporal, graph, and ranking comparators. Most importantly, the proposed leave-target-out direction requires a candidate-conditioned scoring formulation: conventional class heads cannot predict an unseen class. The paper should distinguish known-asset trajectory generalization, asset-disjoint ranking on a known graph, topology-shift robustness, and transfer to a new graph. A journal-ready revision needs a claim taxonomy, matched ablations, a fair candidate-ranking protocol, and end-to-end operational testing. I therefore recommend major revision rather than rejection; the conceptual foundation is promising and the necessary next experiments are well defined.

## Dimension scores

| Dimension | Score (0–100) | Descriptor | Perspective-specific rationale |
|---|---:|---|---|
| Originality (20%) | 72 | Adequate | The combination of non-anticipation, typed candidate spaces, archive reconstruction, and deployment trade-offs is useful, but the learning components are established and the topology contribution is not demonstrated. |
| Methodological Rigor (25%) | 52 | Weak | The protocol is careful for known-asset trajectories, yet the central physics/topology question is not isolated and the alternative model classes are not compared under a common candidate-scoring contract. |
| Evidence Sufficiency (25%) | 48 | Weak | One grid, one PMU placement, fixed event timing, target-covered splits, and no feature-inclusive deployment evaluation do not support a Transactions-level generalization claim. |
| Argument Coherence (15%) | 82 | Strong | The paper clearly separates prototype behavior from the controlled evaluation and appropriately limits several conclusions. |
| Writing Quality (15%) | 84 | Strong | The prose is precise, compact, and unusually candid about negative and non-identifiable results. |
| Literature Integration (optional) | 63 | Adequate | Domain references are present, but the manuscript lacks the adjacent-field concepts needed to define physics-informed learning, graph inductivity, causal temporal modeling, and learning to rank. |
| Significance & Impact (optional) | 76 | Strong | Sparse-sensor event triage is practically important, and a compact typed model could be useful, provided alarm burden, topology dependence, and transfer limits are resolved. |
| **Weighted average** | **64.3** | **Major Revision** | Computed from the five required dimensions using the template weights. |

## Strengths

### S1: A precise causal prediction contract

Section III-C, Eq. (4), defines predictions as functions of samples available no later than time (t), and Section IV-B reports a direct non-anticipation unit test. This is stronger than merely asserting that the method is “online.” It creates a reusable contract under which a tree model, a temporal model, and a graph-temporal model can be compared fairly.

### S2: Candidate ontology is treated as a first-class design object

Sections III-B and III-C distinguish BUS, LINE, and PMU candidate spaces and audit fitted labels against the admissible network ontology. This is an important structured-prediction insight. In practical systems, an accurate score attached to an invalid asset identifier is not a useful localization result. The manuscript's separation of physical and integrity targets also avoids forcing semantically incompatible candidates into one flat class space.

### S3: The paper reports engineering trade-offs rather than accuracy alone

Table III reports model size, node count, and model-only timing in addition to accuracy. The typed model's reduction from 98.3 MB to 28.1 MB and from 0.051 to 0.025 s per input minute is operationally relevant. Small tree ensembles may be easier to inspect, package, and run on control-center infrastructure than a deep graph model; the paper should retain this pragmatic positioning even if graph and temporal baselines are added.

### S4: Negative results and limits are stated honestly

The Introduction, Results, Discussion, and Conclusion consistently acknowledge that the topology comparison changes the candidate representation and training-row construction, that its intervals include no clear gain, and that all targets are observed during fitting. This intellectual restraint is a major strength. It turns an unsuccessful topology result into a clear experimental agenda rather than an inflated claim.

## Weaknesses

### W1: “Physics-informed” is not operationally defined or isolated

**Problem:** The title and keywords foreground physics-informed machine learning, but the manuscript currently combines several different kinds of prior knowledge: robust signal summaries and residuals (Section III-A, Eqs. (2)–(3)); an asset-type hierarchy and candidate ontology (Sections III-B–C); and effective electrical distance (Eq. (1)) used explicitly only by the typed-plus-topology ranker. No governing equation, conservation law, feasibility constraint, or physics-consistency loss is imposed on the learner or its outputs. Moreover, the flat model—the strongest detector and row-level localizer—does not rely on the explicit distance input used to justify the network-informed claim.

**Why it matters:** In physics-informed ML, feature engineering, architectural bias, constraints, residual coupling, and hybrid mechanistic/data-driven modeling are materially different interventions. Without naming the intervention and isolating it, readers cannot tell whether the reported benefit comes from physics, from typed decomposition, from a different candidate representation, or simply from model capacity.

**Suggestion:** Add a short definition and a claim-to-mechanism table. At present, “physics-guided feature and ontology engineering” is the most defensible description. Then run matched ablations that change exactly one component: (i) generic causal signal summaries, (ii) physically motivated residual/proxy features, (iii) typed routing, and (iv) true versus removed/permuted distance on an otherwise identical candidate ranker. If the authors retain the present evidence, narrow the title; if they retain “physics-informed,” demonstrate a measurable benefit attributable to a stated physical prior.

**Severity:** Critical for acceptance under the current title and positioning.

### W2: The comparison set cannot identify the value of temporal, graph, or ranking structure

**Problem:** Section IV-B compares three ExtraTrees systems, while typed-plus-topology changes the localizer, candidate rows, and topology variables together. The Discussion correctly asks for a temporal baseline but does not specify what scientific question each missing comparator would answer. The paper cites graph-based work but provides neither a graph-signal-processing control nor a shallow message-passing comparator.

**Why it matters:** The proposed system claims advantages from transient evolution, network propagation, and candidate ranking. A tree-only family comparison tests deployment choices inside one model class; it does not establish that handcrafted rolling features capture temporal evidence sufficiently, that effective distance is the right graph operator, or that the pointwise candidate construction is preferable to a ranking objective.

**Suggestion:** Add the minimum matched suite below. This is not a request to replace trees with a preferred deep model; a linear graph filter and a small causal temporal convolution are adequate controls.

| Scientific question | Minimal comparator | Fairness constraint |
|---|---|---|
| Do handcrafted causal summaries retain the useful dynamics? | Small causal TCN or GRU on the same 3-s raw PMU window | Same train/validation/test scenarios, past-only receptive field, threshold rule, candidate universe, and a reported parameter/latency budget |
| Does graph structure help beyond jointly observing all PMUs? | Polynomial graph filter plus linear/tree head, or a shallow message-passing model | Same dynamic PMU channels and same candidate scorer; only the graph operator is added |
| Does electrical distance help? | The exact typed-plus-topology ranker with distance features removed, permuted, and restored | Identical training rows, candidate sampling, hyperparameters, and tree budget |
| Does ranking supervision help localization? | Pointwise scorer versus pairwise/listwise loss within each event window | Same candidate descriptors and negatives; report Top-(k), reciprocal rank, and distance error |

**Severity:** Major.

### W3: Leave-target-out is necessary but is not, by itself, graph-inductive generalization

**Problem:** The manuscript appropriately states that the present splits test new trajectories at known assets and proposes withholding targets. However, the flat and typed class localizers in Eq. (3) cannot emit a label absent from their fitted classes. Only a candidate-conditioned scorer can rank a withheld target using attributes shared across candidates. In addition, withholding positive examples for BUS 17 while BUS 17 and its topology remain available during training as part of the known graph is asset-disjoint or zero-shot target localization on a fixed graph; it is not equivalent to generalization to an unseen node, an unseen edge, a changed topology, or a new network.

**Why it matters:** An unfair leave-target-out test would make class heads fail by construction and make the ranker appear uniquely capable for representational—not physical—reasons. Conversely, calling a fixed-graph asset holdout “inductive graph generalization” would overstate external validity.

**Suggestion:** Define a four-level evaluation ladder and align claims to it:

1. **Known-asset trajectory shift (current):** same targets and topology; new severities, scales, and seeds.
2. **Asset-disjoint ranking on a known graph:** hold out all positive events for selected buses/lines, but allow their non-ID physical descriptors and topology at inference. Compare only candidate-conditioned scorers capable of scoring every admissible asset.
3. **Topology and sensor-layout shift:** modify switch/line status, impedances, or PMU placement while preserving the task and candidate schema.
4. **Graph-inductive transfer:** train on one or more networks and test on a separate network without asset-specific embeddings or retraining.

For line localization, use an endpoint-symmetric edge scorer or a line-graph representation so predictions do not depend on arbitrary endpoint ordering. Report whether held-out candidates were visible as negatives during training; that detail separates zero-positive-shot ranking from genuinely unseen-candidate inference.

**Severity:** Major.

### W4: Topology and PMU placement are treated as static metadata, although deployment makes them uncertain inputs

**Problem:** Equation (1) derives effective distance from \(Y_{\mathrm{bus}}\) and \(Z_{\mathrm{bus}}\), but the paper does not state whether inference uses nominal, verified pre-event, or post-event topology. That choice is consequential for line outages: a post-event network model may be unavailable at detection time or may encode the very outage being localized, while a pre-event model may be stale. The 136-feature representation is also tied to exactly eight PMUs and their ordering.

**Why it matters:** A topology-aware model can be brittle when breakers are misreported, line parameters are approximate, or PMUs are temporarily unavailable. A fixed vector over eight sites also does not naturally transfer to another PMU placement. These are not merely data-quality details; they define whether the claimed inductive bias is available without leakage and whether the learned function is permutation- and size-compatible.

**Suggestion:** Specify the topology timestamp and information boundary. Cache only pre-event/nominal graph features for the principal experiment, then stress them with line-status errors, impedance perturbations, and PMU removal/replacement. Include feature recomputation in latency. If layout transfer is a stated goal, use set or graph aggregation with explicit masks rather than a position-specific concatenation, or explicitly limit the claim to one installed PMU configuration.

**Severity:** Major.

### W5: Model-only speed and row-level false-positive fraction do not yet establish operational usefulness

**Problem:** Section IV-C excludes feature computation from timing, and Table III reports 3.22–4.01 alarm episodes per normal-labeled minute. The topology ranker takes 0.586 s per input minute versus 0.025 s for typed—roughly 23 times the model-only cost—even though feature extraction and topology refresh are excluded. The manuscript does not define an operator-facing alarm state machine, abstention policy, or confidence/rank display.

**Why it matters:** Several alarms per minute would create alarm fatigue even when row-level false-positive fraction is small. Operators typically need a stable shortlist, evidence trace, and a “none/uncertain” option, not a rapidly changing forced Top-1 candidate. End-to-end latency and throughput will also scale with the number of candidates and PMUs, which the 39-bus timing does not show.

**Suggestion:** Evaluate a complete streaming path: ingest/alignment, causal features, candidate scoring, temporal debounce/aggregation, and output serialization. Report median and tail latency, memory, candidates scored per second, and scaling with candidate count. Select an alarm persistence/hysteresis rule on validation data and report alarms per hour on long normal streams. Add calibrated confidence or selective prediction, measure coverage–risk, and show rank stability and Top-(k) shortlist quality. Position the first deployment as operator decision support, not autonomous protective action.

**Severity:** Major.

## Detailed comments

### Title and abstract

- The abstract is admirably specific about the known-target condition, negative topology result, and interval crossing zero. Retain this candor.
- The title currently implies that the hierarchical tree ensemble itself is physics-informed and validated as such. Unless a matched intervention establishes that claim, a title such as “Causal Hierarchical Tree Ensembles with Physics-Guided Features for Sparse-PMU Event Detection and Localization” would more accurately describe the evidence.
- Add one phrase distinguishing target-covered evaluation from asset-disjoint/generalized localization. “Unseen targets” should not be used without specifying whether the graph and candidate descriptors were visible during training.

### Introduction

- The Introduction clearly states the two experimental questions, but the second is presently unanswerable because architecture and representation change with distance. Recast it as a hypothesis to be tested by a fixed-ranker distance ablation.
- The paper's most defensible novelty may be the *prediction contract plus candidate ontology plus deployment trade-off*, not a new physics-informed learning algorithm. Bringing that contribution forward would make the paper stronger even if topology remains neutral.

### Related work and conceptual framing

- I do not recommend a broad additional PMU citation sweep from this perspective. Instead, add a compact adjacent-field paragraph that distinguishes scientific knowledge injected through data/features, model architecture, objective/constraints, and hybrid coupling. State exactly which levels the proposed system occupies.
- Frame localization as structured retrieval: each event window is a query, admissible assets are candidates, and their relevance depends on dynamic PMU evidence plus static candidate descriptors. This framing naturally motivates fair unseen-target evaluation and ranking metrics.
- Graph signal processing offers a useful intermediate baseline between topology-free trees and a full GNN. The eight PMU measurements are samples of a partially observed graph signal; localized graph filters or candidate-centered kernels can encode propagation without a large deep network.

### Physics-information audit

| Manuscript mechanism | Knowledge injected | What is currently demonstrated | Claim boundary |
|---|---|---|---|
| Robust local departure, rolling summaries, RLS/Kalman residuals (Section III-A) | Signal-processing and dynamic prior | Interpretable causal descriptors | Physics-motivated features; not enforcement of grid equations |
| BUS/LINE/PMU routing (Section III-B) | Candidate ontology and task semantics | Smaller/faster typed model, with mixed accuracy effects | Domain-informed structured prediction |
| Effective electrical distance (Eq. (1)) | Static network-model descriptor | No aggregate localization gain in a confounded comparison | Topology-informed input, not yet an identified contribution |
| Cross-PMU contrasts and timing | Propagation intuition | Present in the reconstructed representation; explicit distance weighting was not used by the submitted call | Physical interpretation is plausible but requires ablation |
| Governing-equation residuals, feasibility constraints, conservation laws | First-principles physics | Not present | Should not be implied unless added and evaluated |

The absence of a physics constraint is not inherently a flaw; lightweight feature-level integration may be preferable in this application. The problem is terminological and empirical: the manuscript should define the level of integration it claims and show what that level contributes.

### Assumption audit

#### Explicit assumptions

- **Sparse PMU transients contain enough spatial information to localize unobserved origins.** This is plausible, but the current experiment establishes it only for repeated targets on one fixed graph and PMU placement.
- **Typed routing is useful because event families have different location spaces.** The ontology argument is strong. Its accuracy value is mixed, whereas its storage and model-only timing value is supported.
- **Effective electrical distance is a useful candidate descriptor.** This is a testable premise, but the present comparison does not isolate it and the observed aggregate result is negative.
- **A 3-s trailing window is an appropriate causal representation.** The paper treats it as fixed; temporal baselines and window-sensitivity tests are needed before treating it as a generally sufficient horizon.

#### Implicit assumptions

- **The network model available at inference is correct and timely.** In practice, topology and parameters can be stale, and the event being localized may itself change topology.
- **The candidate universe is closed and enumerated.** This is reasonable for a known grid but should lead to an explicit “unknown/none” path for out-of-ontology or compound events.
- **Asset-specific signatures are stable across operation.** The target-covered split allows this assumption; an asset-disjoint ranker is required to test whether shared descriptors, rather than asset identity, drive localization.
- **Routing errors are an acceptable bottleneck.** Missing-only events show that a localizer cannot recover if the event stage never routes to it. A structured multi-task or joint ranking model may retain alternative hypotheses better than hard routing.
- **Fixed PMU ordering and availability are part of the task, not a nuisance variable.** This limits transfer to changed sensor layouts unless the representation is made set/graph compatible.

#### Paradigmatic assumptions

The manuscript adopts a closed-set classification-first paradigm: detect, assign an event type, then choose a location from a type-specific label set. From a GSP/inverse-problem perspective, localization is instead inference over a graph-conditioned candidate field with partial observations. From an information-retrieval perspective, it is a query-dependent ranking problem. These views do not make the hierarchy wrong; they reveal a more appropriate experimental unit for unseen-target claims and suggest preserving ranked alternative hypotheses rather than committing early to one route and one class.

### Cross-disciplinary connections and borrowing opportunities

#### Graph signal processing

Represent the PMU disturbance score at time (t) as a partially observed graph signal. A simple baseline can apply one or more polynomial filters (h(L)x_t) using a Laplacian or admittance-derived shift and then score each candidate with a shared function. Candidate-centered heat/wavelet kernels would make “propagation from candidate to sensors” explicit and testable. Crucially, the coefficients are shared across locations rather than attached to an asset ID.

#### Candidate ranking and retrieval

Treat one event window as a query and every admissible BUS or LINE as an item. Train a shared scorer (s(x_t,c)) from dynamic evidence (x_t) and candidate descriptors (c). Compare pointwise binary supervision with pairwise or listwise ranking. This formulation can score a target with no positive training events, provided its descriptors are available, and avoids the structural impossibility faced by an unseen class label.

#### Temporal modeling

A causal TCN is a clean comparator because its receptive field can be restricted to exactly the same past-only 3-s window. It tests whether hand-designed slopes, residual summaries, and extrema preserve the discriminative timing information. A small model is sufficient; the goal is a diagnostic control, not a deep-learning leaderboard.

#### Physics-guided learning

If the authors want a stronger physics-informed claim without abandoning trees, consider adding physically defined residual channels before classification: for example, deviations from a reduced linear dynamic model or from nodal balance estimates that can be computed under sparse sensing. Another viable route is output regularization: penalize rankings whose candidate-to-PMU propagation pattern conflicts with the network model. Either intervention must be compared with the same learner without the residual/penalty.

### Results and discussion

- The Results correctly state that typed is worse by row and only has a higher scenario-level point estimate with an interval including zero. Do not promote the latter as an accuracy benefit.
- The per-event heterogeneity is valuable. In a revision, connect each gain/loss to candidate geometry or event duration rather than only listing numbers. In particular, investigate why topology helps generation and mixed cases yet substantially harms line outages. This may reveal descriptor mismatch rather than a universally poor topology prior.
- Load Top-1 below 0.05 for every method is a practical boundary condition, not a minor per-class issue. Discuss whether sparse PMU observability makes some load candidates indistinguishable and whether the appropriate output is a region/cluster rather than an exact bus.
- The missing-only failure usefully demonstrates that deterministic integrity rules can complement learned models. Present the proposed presence gate as a hybrid observation-model component and evaluate its false-alarm consequences; do not characterize it as grid physics.

### Practical impact

#### Real-world application

The most credible initial use is operator triage: detect an event, assign a coarse type, and present a stable ranked shortlist of candidate assets with supporting PMUs/features. The typed tree model's compactness is attractive for an advisory service or edge deployment. Exact forced Top-1 localization is less defensible than Top-(k) decision support under the current results.

#### Implementation feasibility

The principal barriers are alarm burden, topology freshness, PMU-layout dependence, end-to-end rather than model-only latency, and model maintenance after network changes. A utility also needs versioned network data, explicit handling of out-of-ontology events, monitoring for input/score drift, and a rollback path. Scaling tests should vary PMU and candidate counts rather than extrapolating from the 39-bus case.

#### Stakeholders

- **Control-room operators** need stable, prioritized alarms and an abstention option.
- **Protection and dynamics engineers** need physical evidence traces and known failure modes before trusting a shortlist.
- **Network-model maintainers** become part of the inference chain when \(Y_{\mathrm{bus}}\)-derived features are used.
- **Cybersecurity and data-quality teams** need to know whether topology or PMU corruption can systematically redirect the ranker.
- **Reliability management** needs performance expressed at an operational alarm budget, not only row-level classification metrics.

### Broader implications

#### Ethical and model-risk dimensions

The immediate ethical issue is not demographic fairness but automation risk: a confident wrong location could redirect operator attention during a disturbance. Until field evidence and selective-prediction behavior are established, the system should be framed as advisory. Predictions should be logged with model version, topology version, data-quality state, and ranked alternatives.

#### Social and institutional impact

Lightweight models may be valuable to utilities with limited accelerator infrastructure, including resource-constrained systems. That benefit will materialize only if the method does not require fragile, continuously perfect topology inputs or frequent retraining. Reporting the operational dependencies plainly would broaden the practical relevance of the work.

#### Future directions

The most valuable progression is: fixed-ranker distance ablation; asset-disjoint candidate ranking; topology/PMU-layout stress; then multi-network transfer. A complementary direction is hierarchical location at multiple resolutions—region, corridor, then exact asset—when sparse sensing makes exact buses observationally indistinguishable.

## Cross-disciplinary reading recommendations

The following references were verified against primary author/publisher/proceedings pages. They are recommendations for conceptual framing and experimental design, not claims that these methods must replace the proposed trees.

1. **J. Willard, X. Jia, S. Xu, M. Steinbach, and V. Kumar, “Integrating Scientific Knowledge with Machine Learning for Engineering and Environmental Systems,” 2022.** The paper provides a taxonomy of where scientific knowledge enters an ML system. It would help the authors distinguish feature-level guidance, architectural priors, objective constraints, and hybrid physical/data-driven coupling. [Primary author manuscript](https://arxiv.org/abs/2003.04919), [publisher DOI](https://doi.org/10.1145/3514228).
2. **D. I. Shuman, S. K. Narang, P. Frossard, A. Ortega, and P. Vandergheynst, “The Emerging Field of Signal Processing on Graphs,” *IEEE Signal Processing Magazine*, 2013.** This supplies the graph-filter, localized-kernel, and partial graph-signal language needed for a lightweight topology baseline. [Primary author manuscript](https://arxiv.org/abs/1211.0053), [IEEE DOI](https://doi.org/10.1109/MSP.2012.2235192).
3. **W. Hamilton, Z. Ying, and J. Leskovec, “Inductive Representation Learning on Large Graphs,” NeurIPS 2017.** GraphSAGE makes the key distinction between node-specific embeddings and a shared function of attributes/neighborhoods. It is directly relevant to defining what an unseen-node claim would require. [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html).
4. **S. Bai, J. Z. Kolter, and V. Koltun, “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling,” 2018.** This is a practical source for a small causal TCN baseline with a controlled receptive field, useful for testing whether the handcrafted temporal summaries lose relevant dynamics. [Primary paper](https://arxiv.org/abs/1803.01271).
5. **C. J. C. Burges et al., “Learning to Rank Using Gradient Descent,” ICML 2005.** The query/item and pairwise-ranking formulation maps naturally to event-window/candidate localization and motivates an objective aligned with candidate order rather than closed-set asset classification. [Microsoft Research publication page](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/), [primary paper PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf).

## Questions for the authors

1. Which precise property is necessary for your use of “physics-informed”: physically interpretable features, network-derived inputs, a physics-constrained learner, or measurable out-of-distribution benefit from a physical prior? Would you retain the term if effective distance were removed and performance were unchanged?
2. In the proposed leave-target-out study, how will flat and typed class heads assign nonzero probability to an asset label absent during fitting? Will all compared methods be reformulated as shared candidate scorers, and will held-out candidates be visible as negatives?
3. At inference, is effective distance computed from nominal, verified pre-event, or post-event topology? How will you prevent event-label leakage while testing sensitivity to stale breaker status, parameter error, and changed PMU placement?
4. What operator action is the system intended to support, and what alarm budget is acceptable? How would the reported 3.22–4.01 episodes per normal minute change after a validation-selected debounce/abstention policy, and what is the feature-inclusive tail latency?

## Minor issues

### Terminology and reporting

- The source uses `\documentclass[conference]{IEEEtran}`. A Transactions submission should be converted to the journal format and reassessed for journal-scale content; this is more than a cosmetic page-count change.
- Use **model-only inference time** consistently wherever Table III timing is discussed. “Prediction time” without the qualifier can be read as end-to-end latency.
- Define whether effective distance is normalized across candidate types and clarify the reference/slack treatment used in the \(Y_{\mathrm{bus}}^{+}\) construction. This matters when distances enter \(\exp(-d)\) without a stated unit or scale.
- Distinguish **asset-disjoint**, **topology-shift**, and **new-graph** evaluation throughout; avoid using “unseen target” as an umbrella term.
- For the candidate ranker, report the number of candidate rows scored per timestamp and the negative-candidate sampling or weighting policy. These determine both runtime and ranking behavior.
- Top-3 is useful, but reciprocal rank and rank stability across adjacent timestamps would better represent an operator-facing shortlist.

## Priority revision package

### Required for journal readiness

1. Define the physics-information level and either narrow the title/claims or isolate a physical prior with a fixed-architecture ablation.
2. Reformulate unseen-target evaluation around a shared candidate scorer and report asset-disjoint bus and line holdouts without asset-ID leakage.
3. Add at least one causal temporal control and one lightweight graph-structured control under the same information and latency contract.
4. Specify the topology information boundary and test topology/PMU-layout perturbations.
5. Report an end-to-end streaming pipeline with alarm aggregation, abstention or uncertainty, long normal exposure, and feature-inclusive latency.

### Valuable but secondary

1. Compare pointwise and pairwise/listwise ranking objectives.
2. Add hierarchical region/corridor localization for observationally ambiguous targets.
3. Test transfer to a second network only if the paper makes a true graph-inductive claim; otherwise state that multi-network transfer is outside scope.
