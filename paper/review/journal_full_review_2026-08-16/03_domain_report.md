# Peer Review Report — Synchrophasor Domain Review

## Manuscript information

- **Title:** *Physics-Informed Hierarchical Tree Ensembles for Sparse-PMU Event Detection and Localization*
- **Target venue:** *IEEE Transactions on Power Systems*
- **Review date:** 2026-08-16
- **Review round:** Pre-submission journal-readiness review
- **Reviewer role:** Peer Reviewer 2 (Domain)

## Protocol and independence disclosure

This is an independent, read-only review from the perspective of a senior researcher in synchrophasor/WAMS dynamic-event detection, localization, PMU observability and placement, and measurement-quality monitoring in transmission systems. I reviewed `paper/main.tex` and the supplied field-analysis/reviewer-configuration document. I did not inspect any other reviewer report and did not modify the manuscript. I intentionally do not repeat the methodology reviewer's statistical audit; comments on experiments below concern power-system meaning, physical fidelity, and external validity.

The installed reviewer skill describes a frozen two-phase sprint-contract process, but no frozen contract was supplied for this review. As requested, I therefore use the legacy independent-review format. The numerical scores are uncalibrated and should be interpreted ordinally rather than as acceptance probabilities.

## Reviewer identity and focus

**Identity:** Senior synchrophasor/WAMS researcher specializing in dynamic-event detection and localization, sparse-PMU observability and placement, topology-aware inference, and measurement-quality monitoring in transmission systems.

**Review focus:** Whether the claimed physical interpretation is sound; whether “physics-informed” is an accurate description of the contribution; whether the simulated event signatures and PMU measurement assumptions support transmission-system conclusions; whether the impedance-derived distance is well defined and suitable for each event type; and whether novelty and validation meet *IEEE Transactions on Power Systems* expectations.

## Overall assessment

### Recommendation

- [ ] Accept
- [ ] Minor Revision
- [x] **Major Revision — not ready for journal submission in the present form**
- [ ] Reject

For *IEEE Transactions on Power Systems*, this is near the boundary between a major revision and reject/resubmit because the required changes include new experiments and a sharper contribution, not merely additional exposition. I select **Major Revision** because the causal benchmark, candidate audit, and negative topology result provide a credible foundation for a substantially expanded paper.

### Confidence score

**5/5.** The paper lies directly within my domain expertise. My confidence is high on the synchrophasor, transmission-dynamics, observability, localization, and measurement-quality issues. I leave inferential-statistics details to the methodology reviewer.

### Summary assessment

The manuscript reconstructs a competition system for event detection, event classification, and typed localization from eight PMUs on the IEEE 39-bus system, then compares flat, typed, and typed-plus-topology ExtraTrees variants under a past-only prediction contract. It reports strong detection, modest physical localization, a clear size/runtime advantage for typed heads, no demonstrated gain from the topology ranker, and failure on missing-only events. From a domain perspective, the paper is commendably transparent about non-anticipation, candidate-set defects, target-covered evaluation, and the confounded topology comparison.

The principal problem is that the title and framing promise a physics-informed localization contribution that the evidence does not yet establish. Most inputs are physically motivated summaries; explicit network information appears only in an under-specified impedance-derived distance, and that addition is neither isolated nor beneficial. The simulated trajectories are also not shown to pass through a credible PMU/PDC measurement chain, and all events share a fixed start time on one legacy test system and one PMU placement. The manuscript therefore establishes a careful competition-system audit and a useful negative result, but not yet a general transmission-WAMS method. A journal-ready revision needs a precise definition of “physics-informed,” a physically defensible and fully specified topology formulation, realistic measurement/event perturbations, and tests across operating points, placements, and topology conditions.

## Strengths

### S1: Exceptional candor about the evaluation boundary

The Abstract, the “Submitted system and benchmark alignment” subsection, and the Discussion explicitly state that every target occurs in training, that results concern new trajectories at known assets, that the original 30-s executable is temporally anticipative, and that the topology comparison changes multiple factors. This is unusually good scientific hygiene in PMU analytics, where random row splitting and hidden future context can produce misleadingly high results. The candidate-universe audit in Table II is also valuable: identifying missing admissible lines and transformer labels prevents a nominal localization score from being mistaken for physical coverage.

### S2: Operationally meaningful separation of physical and measurement-integrity evidence

The hierarchy distinguishes physical disturbances from missing/corrupted PMU records and routes BUS, LINE, and PMU targets to different candidate spaces. That separation is appropriate in a WAMS: a sensor/data-path problem is not a grid event, even when both generate abrupt changes. The manuscript also avoids forcing incompatible asset types into one flat location vocabulary. This is a sound systems-design principle even though the current routing logic still needs refinement.

### S3: A clear non-anticipative prediction contract

Equation (5) defines inference from samples available up to time `t`, and the paper distinguishes complete-segment diagnosis from streaming labeling. This matters physically and operationally because an alarm or locator cannot use post-event samples that have not reached the PDC. The direct non-anticipation code test described in the evaluation section is a meaningful safeguard, and the manuscript correctly separates model-only inference time from feature construction.

### S4: Spatial metrics go beyond exact label accuracy

Candidate coverage, Top-`k` recovery, scenario aggregation, and electrical-distance error are more informative than a single exact-location score under sparse placement. Exact bus identification can be ill posed when several unobserved candidates induce nearly indistinguishable responses. Reporting both row- and scenario-level behavior also exposes prediction instability that an aggregate score could conceal.

### S5: The negative result is reported rather than hidden

The typed-plus-topology ranker performs worse in aggregate than typed localization, and the paper says directly that the comparison cannot attribute the result to electrical distance. The per-event breakdown further shows that generation and mixed cases improve while line-outage localization degrades. That heterogeneity is physically plausible and should become a central diagnostic result rather than being compressed into a general claim that topology “does not help.”

## Weaknesses and required revisions

### W1: “Physics-informed” currently overstates both the mechanism and the novelty

**Problem:** The manuscript calls the full system physics-informed, but most evaluated inputs are local departures, slopes, residuals, missing fractions, and cross-PMU summaries. These are physically interpretable or physics-guided features, not enforcement of power-flow, network-dynamic, conservation, or measurement equations. Explicit topology is confined to the typed-plus-topology ranker, which changes architecture and training-row construction and does not improve aggregate localization. Equation (3)'s impedance weight was not active in the submitted extractor. Consequently, neither the title nor the claimed novelty is supported by an isolated benefit from physics or topology.

**Why it matters:** In current power-system ML literature, “physics-informed” can legitimately include more than PINNs, but the paper must identify exactly where knowledge enters and show what it contributes. Otherwise, the work reads as a well-engineered hierarchical ExtraTrees pipeline whose domain features are conventional. The typed routing and fixed tree budget are useful engineering contributions, but not by themselves a Transactions-level methodological novelty relative to prior PMU event-identification/localization pipelines.

**Suggestion:** Choose one of two defensible positions. (1) Reframe the work as **physics-guided/network-aware feature engineering and causal evaluation**, revise the title and contribution claims, and foreground the forensic reconstruction, candidate audit, and negative ablation. Or (2) retain “physics-informed” only after introducing a fixed-architecture experiment in which the sole difference is the physically derived input or constraint, along with a mechanistic hypothesis for each event family. If a stronger method is intended, consider candidate-conditioned ranking tied to a linearized network sensitivity, event-specific transfer functions, or a consistency penalty between predicted origin and measured voltage-angle/current response. Do not imply that adding one static distance scalar makes the entire hierarchy physics-informed.

**Severity:** Critical for positioning and novelty.

### W2: The “effective electrical distance” is under-specified and is not yet justified for these dynamic events

**Problem:** Equation (1), `d_ab = |Z_aa + Z_bb - Z_ab - Z_ba|` with `Z = Y^+`, resembles a two-point driving/transfer-impedance quantity, but the manuscript does not define the exact `Y_bus` construction, reference treatment, per-unit base, shunt treatment, transformer taps/phase shifts, or whether pre- or post-contingency topology is used. It also does not establish that the complex-valued magnitude is a metric or explain the scale used by `1/(d + epsilon)` and `exp(-d)`. Most importantly, the mapping from a **line candidate** to distances from eight bus PMUs is absent (minimum endpoint distance, mean, both endpoints, or another construction).

**Why it matters:** Static impedance closeness is not universally the right proximity notion for all classes. Fault signatures depend on fault type, sequence networks, fault impedance, and protection clearing. Line-outage signatures depend strongly on pre-contingency flow and redistribution sensitivities such as PTDF/LODF, not merely endpoint impedance. Generator/load events excite inertial, governor, load, and modal responses; cross-PMU timing and amplitude can reflect dynamic coherency rather than static electrical closeness. A single distance may therefore help one event family and harm another, exactly as the per-event results suggest.

**Suggestion:** Fully specify and test the construction. State the matrix convention and units; define line-to-PMU distance; state which topology snapshot is available at inference; normalize any distance before exponentiation; and distinguish “impedance-derived proximity” from a proven effective-resistance metric. Compare at least three physically motivated alternatives under the same candidate-ranker architecture: impedance-based proximity, linearized voltage/angle sensitivity (or PTDF/LODF for line events), and a data-derived transient-response similarity. Report event-family-specific results. A topology-error/outage test is also required because a WAMS model usually sees a network model that may be stale or uncertain.

**Severity:** Critical for physical correctness of the topology claim.

### W3: The benchmark generates dynamic bus trajectories, but PMU measurement realism is not demonstrated

**Problem:** The scenario section states that scale, noise, and sampling rate are estimated from RAW0001, but it does not define the stochastic noise model, PMU estimator/filter, reporting rate, channel latency, time alignment, or whether ANDES' PMU model is used. The controlled features use voltage magnitude, unwrapped angle, and frequency, despite the Introduction's broader description of current and ROCOF. The ANDES documentation describes its simple PMU model as low-pass-filtered bus-voltage magnitude and angle; the manuscript must say whether that model, raw algebraic bus states, or another measurement emulator generated the inputs. Model-derived positive-sequence bus trajectories should not automatically be called PMU measurements.

Event phenomenology is also too uniform: every event begins at 2 s; faults last 0.2 s; missing/corruption lasts 0.7 s; and operating changes persist. If the standard ANDES fault device was used, it represents a three-phase-to-ground fault, so claims should not imply coverage of unsymmetrical faults. The paper does not specify fault impedance, clearing/tripping sequence, generator/load change mechanism and ramp, load composition/power factor, ambient fluctuations, or whether the corruption is a spike, bias, scaling error, frozen value, time error, duplicated frame, or packet loss.

**Why it matters:** PMU algorithms often fail because measurement artifacts interact with genuine electromechanical dynamics. TVE/FE/RFE, estimator transient response, PDC alignment, timing skew, bursts of loss, repeated/out-of-order frames, and channel-specific latency can change both detection delay and apparent cross-PMU timing. A fixed event schedule can also become a signature of the data generator rather than the disturbance physics. The reported 19–29 ms detection delays should therefore be interpreted only as delays on aligned simulator-derived samples, not end-to-end WAMS response.

**Suggestion:** Add a reproducible measurement-and-event table. For each channel, state the source variable, PMU/PDC emulation, reporting rate, filtering, noise/bias distribution, missingness process, and synchronization/latency assumptions. Randomize event onset and duration. Vary fault resistance and clearing behavior; generator/load step size, ramp, and operating point; and multiple data-quality mechanisms. Include tests with time skew, burst loss, PMU dropout, bad quality flags, and delayed/out-of-order packets. If a standard-conforming PMU model is out of scope, call the signals **PMU-like sampled dynamic trajectories** and limit operational claims.

**Severity:** Critical for external validity and the interpretation of cross-PMU timing.

### W4: Sparse placement is treated as a count, not as an identifiability/observability problem

**Problem:** “Sparse-PMU” is operationalized only as eight instrumented buses out of 39, using one fixed competition placement. The manuscript does not state whether this placement is optimized, merely prescribed, or what observability/localization resolution it provides. It reports exact labels but does not identify pairs or zones of candidates that are physically indistinguishable from the eight measurement locations. Only one network, one placement, and target-covered fitting are evaluated.

**Why it matters:** State observability and event-source identifiability are different concepts. A grid may be unobservable for state estimation yet still permit coarse event localization, while exact source discrimination may be impossible for electrically similar candidates. Under a fixed utility asset universe, target-covered closed-set training is not inherently invalid; in fact it can be the intended application. However, performance then depends on operating-point and topology transfer. Conversely, leave-target-out is meaningful only for a candidate-conditioned model capable of scoring a class not seen as a label; it cannot be imposed naively on a standard multiclass localizer.

**Suggestion:** Characterize the placement and localization resolution. Compute candidate-pair separability or event-response similarity from the eight PMUs, cluster ambiguous buses/lines into electrical zones, and show errors versus distance to the nearest PMU and versus pairwise signature similarity. Repeat across several PMU placements/densities, including at least one observability-driven or localization-driven placement. Evaluate operating-point and topology transfer as the primary deployment test. For the topology ranker, add leave-bus-out/leave-line-out evaluation; for closed-set heads, retain target-covered testing but label it explicitly as asset-specific calibration. A second network larger and structurally different from IEEE 39-bus is needed for Transactions-level generality.

**Severity:** Major.

### W5: The diagnostic ontology and measurement-integrity path are not yet operationally coherent

**Problem:** The paper presents a single event label and then introduces separate physical and integrity locations, but event 6 follows a physical component and event 8 combines generator and load changes with a corrupted PMU. It is unclear which physical origin is authoritative when more than one physical asset changes, how a single BUS answer represents simultaneous generator and load changes, and how physical and integrity outputs map back to the competition's one location field. “Corrupted measurement” is also too broad a category for a physically meaningful detector. Finally, missing-only events are essentially never detected even though missing fraction is an input and the absence pattern is described as deterministic.

**Why it matters:** In an operating WAMS, physical alarms and telemetry-quality alarms have different owners, actions, and ground-truth semantics. A missing-only recall near zero is not a minor class imbalance; it means the advertised integrated diagnosis cannot identify one of its most directly observable conditions. Multi-component events require multi-label or structured output if more than one origin is operationally relevant.

**Suggestion:** Provide a candidate-ontology table for every event ID: physical mechanism, number and type of true origins, admissible output fields, routing rule, and scoring rule. Separate a deterministic availability/quality gate from the learned physical-event stage, then test whether that gate increases false alarms or masks physical events. Subdivide bad data into at least missing packets/frames, frozen/repeated values, spikes, bias/scaling, and time-synchronization errors, or narrow the claim to the exact corruption used. For composite events, use a structured pair/set target or explain why the task defines only one privileged physical location.

**Severity:** Major.

## Detailed comments

### Title and abstract

- The title is stronger than the evidence. “Physics-Informed” implies a central, validated physical mechanism, whereas the only explicit network input is confined to a non-improving and confounded variant. “Physics-guided” or “network-aware” would be more accurate unless a controlled physical ablation is added.
- “Sparse-PMU” should be tied to “eight prescribed PMU buses on the IEEE 39-bus case,” not interpreted as a general sparse-observability result.
- The abstract is commendably honest about known targets and the null topology result. It should additionally state that validation uses one simulated network and PMU-like signals rather than field PMU records.
- “Halves model-only prediction time” is precise; retain “model-only.” It is not end-to-end latency.

### Introduction

- The sentence that voltage, angle, current, frequency, and ROCOF do not identify an unobserved origin on their own is broadly sensible, but the evaluated feature set later uses only voltage magnitude, angle, and frequency. Align the claimed measurement channels with those actually used.
- “Propagation across PMUs” is potentially misleading. Electromagnetic fault effects are effectively system-wide at ordinary PMU reporting intervals; observed lags are more likely to reflect estimator filtering, modal/electromechanical response, control action, or threshold-crossing differences. Use “cross-location dynamic response” unless a propagation model is explicitly derived.
- The contribution statement should separate (i) reconstruction/audit of the submitted system, (ii) causal benchmark creation, (iii) typed-head compression trade-off, and (iv) topology ablation. At present these are blended into a new-method narrative.

### Related work and novelty positioning

- The section is too short for a Transactions paper and mostly enumerates methods. It needs an explicit comparison table covering measurement source (field/simulated), network scale, PMU density/placement, event families, online causality, location representation, topology use, measurement-quality treatment, and cross-system/topology validation.
- Pandey et al. (2020), already cited, is the nearest transmission-PMU detection/classification/localization comparator. Explain exactly what is new relative to its real-time pipeline.
- Prior work on sparse-PMU fault-location limits and placement is needed to support the identifiability discussion; this is not equivalent to general PMU observability literature.
- Recent work on transfer across grids/label spaces and on real-world online event detection is important because the current benchmark is target-covered and single-system.
- The literature on physically realistic synthetic PMU generation is a major omission given that the main evidence is simulated.

### Sparse-PMU evidence representation

- Baseline normalization against the first 3 s is physically interpretable only if the reference interval is known to be event-free. In the controlled benchmark the event starts at 2 s, so clarify how a “first 3 s” baseline from the submitted 30-s representation relates to the 3-s causal benchmark, whose event occurs inside the first 3 s. The current notation risks suggesting contamination of the baseline by the event.
- Angle unwrapping and differencing require a clearly stated common angular/time reference. Clarify how slack/reference-bus choice, islanding, and PMU time error affect angle features.
- “Power proxies” must be defined. Without current phasors, active/reactive injection or line-flow quantities are not directly observed.
- Cross-PMU lag and peak-time features need resolution/sensitivity analysis at the reporting rate; otherwise one- or two-frame differences may be dominated by filtering or alignment.

### Hierarchical event and typed origin inference

- Typed BUS/LINE/PMU heads are reasonable, but the physical rationale for separate heads should be distinguished from evidence of superior accuracy. The present evidence supports smaller storage and faster inference, not better localization.
- The route for composite physical-plus-integrity events should return both structured outputs if both are evaluated. Equation (4) currently displays only one final location even though Equation (5) introduces two target types.
- Event-specific priors could be physically useful, but they can also encode competition ontology. Explain which routing rules would remain valid outside SGSMA.

### Prediction contract and candidate ontology

- The candidate audit is a strong contribution and should be moved earlier or highlighted in a dedicated reproducibility box.
- Do not equate a model's fitted-class coverage with physical observability. It is **label coverage**. Observability/identifiability depends on the measurement model and signatures.
- For line labels, specify whether direction is ignored and how parallel lines, transformer branches, and circuit identifiers are represented.

### Scenario design and external validity

- The event population is too deterministic to support a broad claim. Random severity and operating “scale” are not substitutes for distinct unit commitment/dispatch, topology, inertia, governor/exciter settings, dynamic load composition, and renewable/IBR penetration.
- The IEEE 39-bus case is a useful controlled system but an old, small benchmark. Add a larger synthetic transmission network with a different PMU layout and dynamic model mix.
- Five-second windows may capture immediate transient signatures but not slower governor, voltage-recovery, oscillatory, or delayed protection responses. Use event-dependent horizons or justify why the task only concerns onset localization.
- RAW0001-based calibration is not field validation. If RAW0001 is itself simulated, say so directly. If it is measured, describe provenance and which statistics were transferred.

### Results and physical interpretation

- The class-specific topology result is more informative than the aggregate: topology helps generation and composite classes but materially harms line outage. Investigate whether the distance definition matches generator/load disturbance spread but mismatches outage flow redistribution.
- Load localization below 0.05 for every method indicates near-total failure for that event family. This deserves equal prominence with the missing-only failure and limits any general localization claim.
- Scenario voting raises Top-1 because repeated rows smooth unstable predictions, but it is an offline episode decision unless the vote is defined causally over elapsed event time. Distinguish post-event scenario localization from an online rolling consensus.
- Report error geography: confusion between adjacent/electrically close candidates is much less serious than confusion across the network. The stated electrical-distance error should be broken out by event type and compared with a nearest-PMU or electrical-zone baseline.

### Discussion and conclusion

- The current Discussion is appropriately conservative but should go further: the paper does not establish that topology is useless; it establishes that this specific impedance-distance candidate representation, bundled with architectural changes, provides no aggregate gain in this target-covered setting.
- The “next test” should prioritize operating-point/topology shift alongside unseen targets. In a fixed transmission network, unseen targets are a useful inductive-learning stress test, but distribution shift at known assets is at least as operationally important.
- The conclusion should not imply a generally successful event-localization system while load localization is below 0.05 and missing-only detection is essentially zero. State which event families are presently viable.

## Required domain experiments for a journal revision

1. **Fixed-architecture physical ablation:** Use one candidate-conditioned ranker and change only the physical variables. Compare no topology, impedance proximity, event-specific sensitivity features, and (if feasible) transient-response similarity. Include known-target and leave-target-out regimes.
2. **Measurement-chain stress suite:** PMU estimator/filter or a documented approximation; 30/60 fps; realistic TVE/FE/RFE perturbations; time skew; PDC alignment/latency; packet-loss bursts; duplicated/frozen/out-of-order data; full-PMU dropout. Report changes in event-family performance and delay.
3. **Event-physics suite:** Random onset and duration; multiple fault impedances and, where the simulator supports them, fault types; explicit clearing/tripping sequences; generator/load step and ramp mechanisms; multiple operating points and inertia/control settings.
4. **Sparse-placement/identifiability study:** Several PMU placements and densities; candidate-pair or zone separability; performance versus nearest-PMU and electrical proximity; one localization-oriented placement baseline.
5. **External validation:** At least one larger transmission network and either field event data, a public field-derived benchmark such as pmuBAGE for a compatible subtask, or cross-simulator validation. The paper should separate event detection generalization from exact asset localization when labels are not compatible.
6. **Structured integrity baseline:** A deterministic quality/presence gate plus learned physical classifier, evaluated on composite events and realistic data-quality mechanisms.

## Missing key references and how to use them

The current 18-reference bibliography is adequate for a conference paper but too narrow for a Transactions contribution. The following additions were verified against publisher, institutional, or official records. They should be integrated analytically, not appended as a list.

1. **M. Jamei, R. Ramakrishna, T. Tesfay, R. Gentz, C. Roberts, A. Scaglione, and S. Peisert, “Phasor Measurement Units Optimal Placement and Performance Limits for Fault Localization,” *IEEE Journal on Selected Areas in Communications*, 38(1), 180–192, 2020.** [DOI](https://doi.org/10.1109/JSAC.2019.2951971). Use this to frame source identifiability, ambiguity clusters, and PMU placement when the grid is not fully observable.
2. **T. L. Baldwin, L. Mili, M. B. Boisen, Jr., and R. Adapa, “Power System Observability With Minimal Phasor Measurement Placement,” *IEEE Transactions on Power Systems*, 8(2), 707–715, 1993.** [DOI](https://doi.org/10.1109/59.260810). Use as the foundational reference when distinguishing PMU count/placement and state observability from event-source identifiability.
3. **H. Li, Z. Ma, and Y. Weng, “A Transfer Learning Framework for Power System Event Identification,” *IEEE Transactions on Power Systems*, 37(6), 4424–4435, 2022.** [DOI](https://doi.org/10.1109/TPWRS.2022.3153445). Directly relevant to event-type and event-zone transfer across systems with different measurement and location-label spaces.
4. **Y. Cheng, N. Yu, B. Foggo, and K. Yamashita, “Online Power System Event Detection via Bidirectional Generative Adversarial Networks,” *IEEE Transactions on Power Systems*, 37(6), 4807–4818, 2022.** [DOI](https://doi.org/10.1109/TPWRS.2022.3153591). Use to position streaming detection and real-world PMU evaluation, not as a required architecture choice.
5. **X. Mao, W. Zhu, L. Wu, and B. Zhou, “Comparative Study on Methods for Computing Electrical Distance,” *International Journal of Electrical Power & Energy Systems*, 130, 106923, 2021.** [DOI](https://doi.org/10.1016/j.ijepes.2021.106923). Use to distinguish impedance-, sensitivity-, and transient-based distance and justify the distance appropriate to each event family.
6. **I. Idehen, W. Jang, and T. J. Overbye, “Large-Scale Generation and Validation of Synthetic PMU Data,” *IEEE Transactions on Smart Grid*, 11(5), 4290–4298, 2020.** [DOI](https://doi.org/10.1109/TSG.2020.2977349). Essential for validating variability, noise, ambient behavior, bad data, and other realism properties of synthetic PMU datasets.
7. **B. Foggo, K. Yamashita, and N. Yu, “pmuBAGE: The Benchmarking Assortment of Generated PMU Data for Power System Events,” *IEEE Transactions on Power Systems*, 39, 3485–3496, 2024.** [DOI](https://doi.org/10.1109/TPWRS.2023.3280430); [arXiv](https://arxiv.org/abs/2210.14204). Use as a field-derived synthetic benchmark and as context for validation of generated event signatures.
8. **J. Zhao, J. Tan, L. Wu, L. Zhan, W. Yao, and Y. Liu, “Impact of the Measurement Errors on Synchrophasor-Based WAMS Applications,” *IEEE Access*, 7, 143960–143972, 2019.** [DOI](https://doi.org/10.1109/ACCESS.2019.2945786). Directly supports testing how PMU errors affect disturbance-location output rather than adding generic Gaussian noise.
9. **K. D. Jones, A. Pal, and J. S. Thorp, “Methodology for Performing Synchrophasor Data Conditioning and Validation,” *IEEE Transactions on Power Systems*, 30(3), 1121–1130, 2015.** [DOI](https://doi.org/10.1109/TPWRS.2014.2347047). Use to position the integrity path relative to field-tested PMU conditioning and repair.
10. **D. Dwivedi, P. K. Yemula, and M. Pal, “DynamoPMU: A Physics Informed Anomaly Detection, Clustering, and Prediction Method Using Nonlinear Dynamics on μPMU Measurements,” *IEEE Transactions on Instrumentation and Measurement*, 72, 1–9, 2023.** [DOI](https://doi.org/10.1109/TIM.2023.3327481); [arXiv](https://arxiv.org/abs/2304.00092). Use to clarify how the manuscript's meaning of “physics-informed” differs from a dynamics-based use of the term.

Two official technical sources should also inform the measurement description: the [ANDES PMU model documentation](https://docs.andes.app/en/v1.10.0_a/reference/models/PhasorMeasurement.html), which documents its voltage-magnitude/angle low-pass model, and [IEEE C37.118.2-2024](https://standards.ieee.org/ieee/C37.118.2/7077/), which defines synchrophasor data transfer between PMUs, PDCs, and applications. These sources do not substitute for experimental validation.

## Terminology and notation corrections

- Replace **“effective electrical distance”** with **“impedance-derived two-point proximity/dissimilarity”** unless metric properties and the precise network construction are established.
- Replace **“propagation across PMUs”** with **“cross-PMU dynamic response”** or **“cross-location response timing”** unless physical propagation speed is modeled.
- Use **“PMU-like sampled dynamic trajectories”** rather than “PMU data” when no PMU estimator, timing, and data-transfer chain is simulated.
- Reserve **“observability”** for a defined measurement-model property. Use **“candidate coverage”** for labels present in a fitted model and **“localization identifiability/resolution”** for distinguishable origins.
- Distinguish **fault**, **faulted bus**, **line outage/trip**, and **faulted line**. A line trip without a simulated fault/relay sequence is not a faulted-line event.
- Define **bad data/corruption** by mechanism. Missing frames, frozen values, spikes, biases, and time errors have different signatures and mitigations.
- Clarify whether **FAR** means alarm episodes per normal-labeled minute; this is different from the row-level false-positive fraction.
- Use **model inference time**, **feature-computation time**, **communication/PDC latency**, and **event-to-alarm delay** as distinct quantities.
- State the angular reference and units for all voltage-angle features, and the per-unit base for the `Y_bus`/`Z_bus` calculation.

## Questions for the authors

1. What exact signals are exported from ANDES: raw bus algebraic variables, the built-in PMU model, or a separate PMU estimator? How are frequency and any ROCOF/current/power channels obtained?
2. What is the complete event model for each class: disturbance device, parameter changed, magnitude distribution, fault type and impedance, clearing action, load model, generator control response, and corruption mechanism?
3. How is the first-3-s reference in Equation (2) kept event-free when controlled-benchmark events begin at 2 s? Is the equation only for the 30-s archived representation, and if so, what is the exact causal-baseline rule in the 3-s evaluation?
4. How is `Y_bus` formed and referenced before taking the pseudoinverse? Are shunts, transformer taps, and phase shifts included, and is the topology pre-contingency, post-contingency, or event-dependent?
5. How is a line candidate represented by distances to PMU buses? Are both endpoints retained, aggregated, or encoded through another rule?
6. Is the distance vector normalized before `exp(-d)`? Could its main effect be to provide a unique candidate fingerprint that the target-covered ranker memorizes?
7. For event 8, which physical location is the ground truth if a generator and a load both change? Can the system return both physical origins plus the corrupted PMU, or is one component privileged by the competition ontology?
8. Why do missing-only events bypass the integrity localizer despite explicit missing-fraction inputs? Can a deterministic availability gate resolve them without suppressing simultaneous physical events?
9. Were the eight PMU buses selected for state observability, event localization, or only prescribed by the task? Which candidate pairs are least distinguishable under this placement?
10. What operating-point variations are represented by “operating scale”? Do they change dispatch, commitment, power factor, inertia, governor/exciter behavior, or merely multiply injections?
11. Are scenario-level modal predictions available online at a specified elapsed time, or are they computed retrospectively over all active rows?
12. Which part of the contribution is intended to be lasting beyond the competition: the hierarchy, the reconstructed audit, the causal benchmark protocol, or the topology-aware ranker? The journal revision should choose and validate one primary claim.

## Minor issues

- Equation numbering should be cross-checked in prose after any journal-format conversion; several concepts are introduced in one equation but used differently in the controlled variant.
- Define the sample/reporting rate before giving rolling-window lengths in samples.
- Specify whether Top-3 is meaningful for PMU integrity localization when only eight sites exist and whether empty predictions are included consistently.
- The term “electrical-distance error” needs a formula and units, especially for line candidates.
- Report whether buses 30–39 to high-side generator buses is a competition-specific mapping or a physically intended transformer-side reporting convention.
- The Introduction invokes current and ROCOF, but the 136-feature controlled representation does not list them. Make the channel inventory consistent throughout.
- Do not use a line drawing as the contrast for electrical distance; the relevant alternatives are hop distance, impedance/sensitivity distance, and dynamic-response similarity.

## Dimension scores

These scores apply the supplied 0–100 rubric at *IEEE Transactions on Power Systems* standards. The Methodological Rigor score reflects physical modeling and domain-validity only, not a statistical audit.

| Dimension | Score | Descriptor | Domain rationale |
|---|---:|---|---|
| Originality (20%) | 56 | Weak | Useful integration and audit, but typed ExtraTrees and physically motivated summaries are incremental; the topology contribution is not isolated or beneficial. |
| Methodological Rigor (25%) | 58 | Weak | Causal prediction and candidate checks are strong, but event physics, PMU measurement chain, and topology construction are under-specified. |
| Evidence Sufficiency (25%) | 54 | Weak | One network, one PMU placement, fixed event schedule, target-covered assets, and no field/cross-simulator validation do not support broad WAMS claims. |
| Argument Coherence (15%) | 82 | Strong | The limitation-aware narrative is clear and the conclusions generally follow the reported domain evidence. |
| Writing Quality (15%) | 84 | Strong | Concise, technically readable prose with mostly precise claims; several terms need tighter power-system definitions. |
| Literature Integration (R2 focus) | 60 | Adequate | Relevant recent event/localization papers are cited, but sparse-PMU limits, electrical-distance alternatives, synthetic-PMU realism, and transfer literature are missing. |
| Significance & Impact | 68 | Adequate | Transparent causal evaluation and model compression are useful, but operational impact is limited by weak load/missing-event performance and external validity. |
| **Weighted average** | **64.1/100** | **Major Revision** | Below the 65-point minor-revision threshold and not presently journal-ready. |

## Bottom line

The paper has a credible journal seed, chiefly because it exposes anticipation, candidate-universe defects, target coverage, and a negative topology result with unusual clarity. Its current domain contribution, however, is better described as a careful causal audit of a competition pipeline than as a validated physics-informed WAMS locator. A successful *IEEE Transactions on Power Systems* revision should make that choice explicit: either narrow and retitle the paper around the audit/benchmark contribution, or add a physically justified candidate-conditioned localizer and the measurement-, placement-, topology-, and operating-condition experiments needed to support the stronger title.
