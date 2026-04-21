# Detector Training Report

## Model
- Hybrid detector: cyber heuristics + tabular ML, physical temporal model + heuristics, rule-gated fusion, state-machine postprocessing.

## NaN / DATA_PRESENT
- strategy: `numeric coercion + ffill/bfill + mask channels + z-score normalization`
- data_present_feature_used: `True`
- nan_count_before_preprocessing_train: `54800`
- nan_count_before_preprocessing_val: `29784`
- nan_count_before_preprocessing_test: `30211`
- nan_count_after_preprocessing_train: `0`
- nan_count_after_preprocessing_val: `0`
- nan_count_after_preprocessing_test: `0`

## Threshold Policy
- threshold/start/stop: `0.7000` / `0.7000` / `0.6500`
- margin: `0.0500`
- min_on_frames: `1`
- min_off_frames: `1`

## Metrics
- validation_f1_abnormal: `0.8030`
- validation_recall_abnormal: `0.7162`
- test_f1_abnormal: `0.8163`
- test_precision_abnormal: `0.9302`
- test_recall_abnormal: `0.7273`
- test_false_positives_per_minute: `1.8947`
- test_detection_delay_s: `2.133`
- test_roc_auc: `0.9854545454545455`
- test_pr_auc: `0.9714628423067125`
- test_brier_score: `0.0892`
- test_calibration_ece: `0.18948506329525575`

## Cyber vs Physical
- cyber_heavy_recall: `None`
- physical_heavy_recall: `0.7272727272727273`
- concurrent_heavy_recall: `None`

## Readiness
- binary_detector_ready_for_classifier_localizer: `False`
- verdict: `not_ready`
