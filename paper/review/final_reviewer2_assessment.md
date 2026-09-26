# Final adversarial assessment

Audit date: 2026-08-11  
Decision: **weak accept**  
Confidence: **0.98 / very high**  
Submission-ready: **yes**  
Remaining blocking defects: **none**

## Re-review outcome

The first final review returned weak reject. It identified an unfair detector
comparison: the typed detector used a validation-selected threshold while flat
used a hard multiclass decision. The benchmark was changed so both event models
select an abnormal threshold on the same 33-point validation grid. All nine
model-seed combinations were rerun on the unchanged 690-scenario corpus.

The matched comparison reverses the earlier claim. Flat reaches detection F1
0.977 versus 0.965 for typed, and 3.22 versus 4.01 false-alarm episodes per
normal-labeled minute. The typed pipeline's serialized artifact is 71% smaller,
and its recorded model-prediction time is half that of flat on the benchmark
host. The manuscript now presents this as a pipeline-level efficiency result
rather than an accuracy or isolated routing gain.

| Prior issue | Final status |
|---|---|
| Unmatched flat/typed operating points | Both use validation abnormal-F1 on the same grid; thresholds are 0.75 and 0.45. |
| Label-free calibration claim | Corrected: calibration uses RAW0001 rows labeled `Event == 0`. |
| Uncertainty based only on forest seeds | Added a 5,000-replicate paired event-stratified scenario-block and model-seed bootstrap. |
| Operational alarm-burden language | Removed; FAR is benchmark-specific and normalized by normal-labeled exposure. |
| Candidate-ranker feature attribution | Removed; the full composite variant is evaluated without an isolated distance-feature claim. |
| Runtime scope | Defined as one prediction pass per fitted seed after feature computation; CPU, cores, RAM, OS, `n_jobs=-1`, and absence of a separate warm-up are disclosed. |
| Serialized size versus memory | Corrected throughout; `serialized_bytes` is reported as serialized model size, not RAM use. |
| Frozen-training provenance | The paper now states that the original corpus and generated feature tables are absent, so exact retraining is unavailable. |
| Integrity-label persistence | Disclosed as a simulation-label limitation. |
| Learning-curve applicability | Corrected to state that no training-size curve was run. |

## Independent checks

- All nine model-seed combinations and 138 paired test scenarios are present,
  with no duplicate prediction keys.
- Precision, recall, F1, FAR, and all four paired bootstrap intervals reproduce
  from the saved row-level predictions.
- The focused benchmark and ontology tests pass.
- The compiled manuscript is synchronized with source, exactly six US-letter
  pages, visually clean, and contains no overfull boxes, unresolved references,
  or compilation errors.
- Every PDF font is embedded. The sole layout diagnostic is a non-blocking
  underfull-page warning; there is no clipping or unreadable figure text.
- The citation audit verifies all 18 sources and leaves no unresolved context or
  metadata defect.
- A final prose scan found no stock AI vocabulary, canned transitions, inflated
  novelty claim, em dash, or hidden causal overstatement.

## Residual limitations

The fixed onset, one network and PMU placement, known assets across splits,
limited normal exposure, missing-event failure, absence of an isolated distance
ablation, and lack of a contemporary SOTA comparison constrain the claim. The
paper states each limitation directly. None is a blocking correctness defect for
the six-page system-reconstruction and non-anticipative-reassessment contribution.

The Windows Anaconda environment emits a late `gmpy2`/ANDES teardown exception
after pytest has passed and returned exit code 0. This does not change any
result, but a clean reproducibility container should pin a compatible binary
stack before external release.

## Slide-narrative revision, 2026-08-11

The final slide-integration pass adds the exact feature-view narrative, compact
method equations, event-to-localizer routing, and a denser method/reconstruction diagram.
It does not reintroduce the presentation's unsupported topology-weighting,
timestamp-causality, independent-ablation, or transfer claims. The scientific
verdict is unchanged.

## External-review adjustment, 2026-08-12

The manuscript now follows the intended physics-guided design into a
non-anticipative evaluation. It also reports one modal location per
scenario from the saved test predictions. Typed has the higher physical
scenario Top-1 point estimate (0.614 versus 0.590), but the paired 95% interval
[-0.0551, 0.1019] includes zero. This addition answers the duration-weighting
concern without changing fitted models or selecting a new representation.

## Voice and title revision, 2026-08-12

The title is now `Physics-Informed Sparse-PMU Event Diagnosis: From Hierarchical
Design to Non-Anticipative Evaluation`. The abstract, introduction, runtime
reconstruction, discussion, conclusion, and first-figure wording were rebuilt
to remove defensive disclaimers and repeated audit language. Scientific scope
and all numerical claims remain unchanged.
