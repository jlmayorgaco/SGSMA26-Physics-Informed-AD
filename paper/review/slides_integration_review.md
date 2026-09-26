# Slide-to-paper integration review

Review date: 2026-08-12

## Material inspected

- `slides/slide.pdf`: 40-slide conference presentation.
- `slides/slide.tex`: source for the 40-slide presentation.
- `presentation/sgsma_sgms_tutorial_beamer_en_full117.pdf`: 117-slide technical tutorial and backup material.
- The corresponding LaTeX source, frozen model inventory, feature schema, representation sweep, and causal benchmark evidence.

## Material retained in the manuscript

The revised paper now reconstructs the method that was actually presented:

- eight sparse PMU streams grouped into 30-second segments;
- pre-event median/MAD normalization and temporal subsegments;
- base statistical, rolling, residual/innovation, and cross-PMU feature families;
- the 45,162-variable frozen feature schema and its block counts;
- hierarchical ExtraTrees event logic with separate integrity detectors;
- event-conditioned BUS, LINE, and PMU localization;
- the exact rolling windows, RLS forgetting factors, Kalman noise settings, and forest inventory supported by code or serialized artifacts;
- compact equations for the robust local reference, adaptive residuals, innovation, cross-PMU severity, and typed localization rule;
- a rebuilt method figure that connects PMU placement, network-derived distance, the four evidence views, hierarchical event logic, and typed origin spaces.

The title, abstract, introduction, method section, first figure, and feature table were rewritten around this system. Prototype-to-benchmark differences remain in Table II, where they can be checked without controlling the paper's narrative.

## Slide claims that were not copied verbatim

1. **Topology weighting.** The slides state that graph-temporal variables use effective electrical-distance weights. The deployed call does not pass `topology_distances`; missing pair distances therefore default to 1.0. The paper calls the frozen block spatial-temporal. Electrical distance appears only in the separately evaluated candidate-ranker variant.

2. **Ablation.** The 16-representation sweep is not a controlled feature-family ablation. The selected 45,162-variable representation scores 0.8524 localization Top-1 on the simulation split, while a smaller rolling-plus-spatial representation scores 0.8558. RAW0001 was used to choose the final representation. The manuscript reports this as model selection, not proof that every feature block improves performance.

3. **RAW0001 transfer.** The 1.000 detection/classification values and 8/12 localization result come from an inspected set also used during representation selection. They remain development evidence, not an independent transfer test.

4. **Timestamp-level inference.** The slide narrative says the method produces timestamp-level output. The executable predicts once from each complete 30-second segment and copies that decision to every row, including rows earlier than samples used by the prediction. The causal reassessment replaces this with trailing features and one prediction per row.

5. **Line ontology.** The frozen line localizer contains 24 valid transmission lines, omits 10 admissible lines, and fits 10 transformer labels. Aggregate development accuracy is conditional on that incompatible vocabulary.

6. **Physics-informed terminology.** The artifact contains physically motivated transforms, but no governing-equation residual, conservation constraint, or state update. The revised title follows the physics-guided design into its evaluation. The body states that the archived block uses uniform spatial weights and that electrical distance enters only through the separately tested ranker, whose architecture also changes.

## Result

The slides supply the paper's technical backbone without weakening the evidence standard. The principal quantitative claims come from the scenario-disjoint non-anticipative benchmark; the hackathon metrics are preserved as development provenance with their selection and label limitations stated explicitly.

## Submission-facing self-review

- **Contribution:** clear and bounded. The manuscript reconstructs the submitted system and supplies a non-anticipative benchmark rather than claiming a new generic detector.
- **Method:** internally consistent. Feature counts, formulas, routing, fitted forests, candidate universes, and the distinction between the frozen and reassessed pipelines agree with code or serialized evidence.
- **Evaluation:** appropriate to the claim. All methods share scenario splits, past-only inputs, seeds, tree budget, and validation selection. Missing-data failure, limited normal exposure, known assets, and the absence of a clean distance ablation remain explicit.
- **Presentation:** the six-page IEEE layout now carries one method figure, one evaluation-protocol figure, two diagnostic figures, and the core equations without shrinking body text or captions below a readable size. The protocol figure uses five spaced stages and a second band for target coverage, split isolation, non-anticipation, matched budgets, and paired inference. Outer margins remain open, and every arrow terminates in whitespace before the next node.
- **Claim discipline:** the final prose scan found no unsupported novelty language, stock AI transitions, topology overclaim, causal ambiguity, or conversion of serialized size into memory use.

## Manuscript architecture after slide integration

1. **Introduction:** partial spatial observability, the hierarchy used to address it, and the controlled comparison.
2. **Related work:** four ways network information enters PMU localization, followed by the paper's exact choice.
3. **Method:** sparse-PMU evidence, the $Y_{\mathrm{bus}}/Z_{\mathrm{bus}}$ distance path, event routing, candidate ontology, and prototype realization.
4. **Evaluation:** target-covered scenario design, past-only variables, matched baselines, metrics, and reproducibility controls.
5. **Results and discussion:** operating-point comparison, event-level failure analysis, scope, and the fixed-architecture leave-target-out experiment still required.

The paragraph roles follow a claim--mechanism--evidence pattern rather than repeating the section heading. Short result statements are mixed with longer explanatory paragraphs to avoid uniform, template-like rhythm.

## Claim--evidence map

| Manuscript claim | Supporting evidence | Boundary kept in prose |
|---|---|---|
| The presented method is hierarchical and typed | slide routing diagrams, serialized forests, fitted class inventories | no novelty claim |
| The representation uses local, multiscale, adaptive, and cross-PMU evidence | slide derivation, extractor source, 45,162-column schema | effective-distance weights defaulted to one in the prototype |
| Event conditioning is smaller and faster | matched three-seed benchmark, serialized sizes, model-only timing | no end-to-end latency claim |
| Flat heads lead row-level accuracy | paired scenario-block intervals and test predictions | scenario-level advantage is not claimed |
| The current topology result is inconclusive | typed versus typed+topology comparison | architecture and sampling change together |
| Spatial generalization remains open | target-complete split manifest | no leave-target-out result is reported |

## Prose audit

The final rewrite removed reader-steering phrases such as “the study establishes,” “clear architectural trade-off,” and “decisive next experiment.” It also reduced defensive labels: “Executable reconstruction” became “Prototype realization,” and Table II now describes the transition from the presented design to the controlled evaluation. Technical uses of “robust” were retained only for median/MAD statistics, where the term has a precise meaning.
