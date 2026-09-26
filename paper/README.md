# IEEE conference paper

This directory contains the six-page IEEE conference manuscript, the non-anticipative
reassessment code, and the evidence used in every quantitative claim.

## Build

Run from `paper/`:

```powershell
python figures/build_figures.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The manuscript uses the standard `IEEEtran` conference class, US Letter paper,
10-pt text, and the `IEEEtran` BibTeX style.

## Reproduce the evidence

The full simulation benchmark is deterministic but takes several minutes:

```powershell
python experiments/causal_grouped_benchmark.py --output evidence/revised/causal_benchmark/target_complete
python experiments/summarize_per_event.py
python experiments/summarize_scenario_localization.py
python experiments/bootstrap_differences.py
python experiments/audit_candidate_contract.py
```

Run the focused checks from the repository root:

```powershell
pytest -q paper/experiments/test_causal_grouped_benchmark.py paper/experiments/test_scenario_localization.py
```

## Evidence map

- `evidence/revised/candidate_audit.json`: fitted, missing, and invalid classes
  from the frozen submission, with the archive hash and topology counts.
- `evidence/revised/causal_benchmark/target_complete/target_complete/`: scenario
  manifest, aggregate and row-level metrics, predictions, model metadata, input
  hashes, and self-checks for the 690-scenario reassessment.
- `evidence/revised/per_event_metrics_summary.csv`: per-event and joint-exact
  statistics used in the Results section and Fig. 4.
- `evidence/revised/scenario_localization_summary.csv`: modal physical and
  integrity Top-1 results after consolidating each test scenario once.
- `evidence/revised/paired_scenario_bootstrap.csv`: paired 95% intervals from
  event-stratified resampling of complete test scenarios and model seeds.
- `.aris/citation-audit/contexts.txt`: sentence-level citation context extracted
  for independent reference verification.
- `figures/build_figures.py`: deterministic source for all four vector figures.
- `review/slides_integration_review.md`: claim-by-claim record of what the
  conference slides contributed and which slide claims were narrowed after
  checking the executable artifact.

The older files under `evidence/` preserve the hackathon development results and
hidden-stream execution records. They are retained as provenance, not used as an
independent labeled test set.

## Claim boundary

The reassessment evaluates independent trajectories at known assets on one
simulated IEEE 39-bus configuration and one PMU placement. Unseen-target
transfer, field performance, and operational deployment require separate
experiments. The title follows the intended physics-guided design into a
non-anticipative evaluation. The archived model uses uniform spatial-temporal
weights; electrical distance reaches inference through the candidate-ranker
variant, whose architecture also changes.

Before submission, confirm the author order, affiliations, funding statement,
conflicts of interest, and the target venue's disclosure policy.
