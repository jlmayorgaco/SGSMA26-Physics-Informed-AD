# M9 Simulator Validation

Overall working: `True`

## Downstream Readiness
- estimator_training_ready: `True`
- identifier_training_ready: `True`
- detector_training_ready: `True`
- cyber_detector_training_ready: `True`
- physical_detector_training_ready: `True`
- classifier_training_ready: `True`
- localizer_training_ready: `True`
- estimator_assisted_localizer_ready: `True`

## Explicit Answers
- pmu_csvs_match_expected_schema: `True`
- non_pmu_buses_exported: `True`
- event_labels_coherent: `True`
- data_present_semantics_correct: `True`
- physical_and_cyber_events_represented: `True`
- synthetic_pmu_statistically_useful_vs_raw0001: `True`

## Templates Tested
- EVENT0_NORMAL: `True` (data\scenarios\SIM0002)
- EVENT1_FAULT: `True` (data\scenarios\SIM0003)
- EVENT2_LINE_OUTAGE: `True` (data\scenarios\SIM0004)
- EVENT3_GENERATION_CHANGE: `True` (data\scenarios\SIM0005)
- EVENT4_LOAD_CHANGE: `True` (data\scenarios\SIM0006)
- EVENT5_MISSING_ONLY: `True` (data\scenarios\SIM0007)
- EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL: `True` (data\scenarios\SIM0008)
- EVENT7_BAD_DATA: `True` (data\scenarios\SIM0009)
- EVENT8_UNKNOWN_COMPOSITE: `True` (data\scenarios\SIM0010)
- OFFICIAL_STYLE_MULTI_EVENT: `True` (data\scenarios\SIM0001)

## Strengths
- Exports official-like 8-PMU CSV files with exact aligned timestamps.
- Exports full 39-bus truth and full-state targets for estimator/localizer supervision.
- Separates physical truth from cyber/data-quality corruption.
- Exercises Event labels 0 through 8 plus the official-style multi-event timeline.
