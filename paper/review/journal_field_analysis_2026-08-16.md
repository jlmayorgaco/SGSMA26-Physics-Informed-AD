# Journal field analysis and reviewer configuration

Review date: 2026-08-16  
Manuscript: *Physics-Informed Hierarchical Tree Ensembles for Sparse-PMU Event Detection and Localization*

## Paper profile

| Dimension | Assessment |
|---|---|
| Primary discipline | Power-system dynamic monitoring and synchrophasor analytics |
| Secondary disciplines | Statistical machine learning; signal processing; cyber-physical measurement integrity |
| Research paradigm | Quantitative computational experiment |
| Methodology | Physics-informed feature engineering, hierarchical tree ensembles, simulation benchmark, paired resampling |
| Target tier | Q1 field journal ambition; currently a strong conference paper but not a journal-complete study |
| Maturity | Conference pre-submission; journal revision requires new experiments rather than prose expansion |

Current length: approximately 3,300 manuscript words, 156-word abstract, 18 references, and six IEEE conference pages.

## Journal fit

1. **IEEE Transactions on Power Systems (primary target).** The paper fits power-system dynamic measurements, computing applications, intelligent systems, and transmission-system operation. A first submission is limited to ten pages.
2. **IEEE Open Access Journal of Power and Energy (strong alternative).** Its broad scope explicitly includes synchrophasor technology, situational awareness, and artificial intelligence in power-system analysis. It is fully open access.
3. **International Journal of Electrical Power & Energy Systems (alternative).** Its scope explicitly covers wide-area monitoring and data analytics/AI for power systems.

**Do not target IEEE Transactions on Smart Grid in the present form.** Its current scope lists transmission-system PMU applications such as WAMS and WACS as out of scope.

## Reviewer configuration cards

### 1. Editor-in-Chief / journal-fit reviewer

**Identity:** A senior editor for *IEEE Transactions on Power Systems* working in power-system dynamic performance and control-center analytics.

**Focus:**

1. Whether the paper delivers a systems-level contribution of lasting value beyond the SGSMA task.
2. Whether the physics-informed claim is technically precise and central to the experiments.
3. Whether the evidence is broad enough for a Transactions article.

**Will particularly care about:** A clear contribution statement, a journal-scale experimental center, and relevance to transmission-system monitoring rather than competition-specific engineering.

**Possible blind spot:** May give less attention to statistical leakage and uncertainty details than the methodology reviewer.

### 2. Methodology reviewer

**Identity:** A researcher specializing in evaluation of machine learning for power-system dynamics, including grouped validation, distribution shift, uncertainty intervals, and reproducibility.

**Focus:**

1. Target leakage and the distinction between trajectory holdout and asset holdout.
2. Matched ablations that vary one factor at a time.
3. Repeated splits, calibration, baselines, timing protocol, and uncertainty reporting.

**Will particularly care about:** Whether the topology claim can be identified causally from the reported comparison and whether three seeds are adequate.

**Possible blind spot:** May underweight the physical meaning of PMU signatures and network models.

### 3. Synchrophasor-domain reviewer

**Identity:** A senior WAMS/PMU researcher working on dynamic-event detection, localization, observability, and measurement-quality monitoring in transmission systems.

**Focus:**

1. PMU placement, partial observability, event phenomenology, and candidate ontology.
2. Physical interpretation of effective distance, cross-PMU timing, residuals, and power proxies.
3. External validity across operating points, event schedules, networks, and measurement conditions.

**Will particularly care about:** Whether simulated signatures are physically credible and whether the method remains useful under realistic PMU noise, missingness, latency, and topology changes.

**Possible blind spot:** May accept conventional classifiers without demanding strong machine-learning baselines.

### 4. Graph/physics-informed learning reviewer

**Identity:** A machine-learning researcher specializing in graph signal processing, physics-guided learning, candidate ranking, and inductive generalization to unseen nodes or edges.

**Focus:**

1. Whether “physics-informed” means more than physically interpretable feature engineering.
2. Comparison with temporal and graph baselines under identical causal inputs and candidate universes.
3. Leave-bus-out and leave-line-out generalization, including topology perturbations.

**Will particularly care about:** Whether the proposed hierarchy offers an identifiable advantage over a graph or sequence model and whether topology helps unseen-target ranking.

**Possible blind spot:** May underestimate deployment simplicity and the value of small tree ensembles.

### 5. Devil's Advocate

**Identity:** A utility-facing reviewer responsible for wide-area situational awareness and model-risk assessment.

**Focus:**

1. The strongest counterclaim: the target-covered classifier memorizes asset-specific signatures and does not demonstrate physics-informed localization.
2. Whether the reconstructed benchmark and development choices introduce confirmation bias.
3. Whether false-alarm exposure, end-to-end latency, and missing-only failure make the method operationally unusable.

**Will particularly care about:** A prospective evaluation in which targets, operating conditions, and event timing are genuinely unseen.

**Possible blind spot:** May set an industrial-validation bar higher than is usual for a methodological simulation paper.

## Review strategy

The five reviewers cover non-overlapping questions: journal contribution, statistical identification, power-system validity, modern learning baselines, and the strongest operational counterargument. The central arbitration question is whether a fixed-architecture, leave-target-out experiment can turn the current physics-informed narrative into an empirically isolated contribution.
