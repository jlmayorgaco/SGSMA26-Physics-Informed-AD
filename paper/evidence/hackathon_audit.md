# Hackathon evidence audit

Audit date: 2026-08-07.

## What is retained

- `Competition_Testing Data Set 1.zip`: SHA-256 `99e14dc3062070c02fbcde5a4137788e9e968ac05859212e9b0f66c2971972f2`; eight aligned PMU CSV files; 43,472 rows per PMU; 24.1506 min.
- `Competition_Testing Data Set 2.zip`: SHA-256 `660103453d015cb95a07644baa97446402f359052589331e9cc033933f6dfd22`; eight aligned PMU CSV files; 42,750 rows per PMU; 23.7495 min.
- `data/RAW0001`: eight labeled PMU files; 161,379 rows per PMU; 89.65 min; 21 promoted windows, of which 12 have localizable events.
- IEEE 39-bus topology: 39 buses and 46 physical branches.
- Frozen release `sgsma_2026_final_submission.zip`: SHA-256 `e0bfc17e4c5ee9b595fa51fd94d736145c2f48755696c8d2eadcff00cb23a243`; its `models/` directory occupies 205,827,807 bytes (196.29 MiB) before compression.

## Competition-test limitation

Neither test archive contains an event label, organizer timeline, blank label column, or `DATA_PRESENT` field. No organizer-returned result workbook, hidden-test labels, leaderboard entry, official score, or rank is present in the repository. The current artifact can therefore demonstrate end-to-end execution on those streams, but it cannot supply hidden-test accuracy, F1, localization accuracy, false-alarm rate, or rank.

The frozen artifact emits 49 window decisions for Test 1 and 48 for Test 2. The output profiles are recorded in `hackathon_execution.csv`. They are predictions, not measured event prevalence.

## Reproducibility boundary

The ordinary repository retains inference code, fitted models, topology, aggregate metrics, RAW0001, and the two hidden measurement archives. The 5,000-window simulated training/development corpus, generated feature matrices, and split indices are absent, so exact retraining and independent reconstruction of the aggregate simulation scores are not possible from the current tree.

## Excluded claims

Stale exploratory reports and alternative model directories contain incompatible scores or unsupported claims about placement invariance, topology weighting, model size, causal improvements, and near-miss localization. They are not used as evidence in the manuscript. The paper reports only the selected frozen artifact, the complete 16-representation selection table, the separate placement stress test under its own protocol, and outputs reconstructed from the two unlabeled competition archives.
