
============================================================
PROJECT SCAN METADATA
============================================================
ROOT: /c/Users/walla/Documents/Github/SGSMA26-Physics-Informed-AD
GENERATED_AT_UTC: 2026-04-18T23:27:28Z
HOST_OS: Windows via Git Bash / MINGW / MSYS
IGNORED: .git/

============================================================
FULL DIRECTORY AND FILE TREE
============================================================
./.gitignore
./.pytest_cache
./.pytest_cache/.gitignore
./.pytest_cache/CACHEDIR.TAG
./.pytest_cache/README.md
./.pytest_cache/v
./.pytest_cache/v/cache
./.pytest_cache/v/cache/nodeids
./README.md
./configs
./configs/data.yaml
./configs/simulation.yaml
./configs/training_task1.yaml
./configs/training_task2.yaml
./configs/training_task3.yaml
./data
./data/RAW0001
./data/RAW0001/Bus10_Competition_Data_nanmask.csv
./data/RAW0001/Bus19_Competition_Data_nanmask.csv
./data/RAW0001/Bus22_Competition_Data_nanmask.csv
./data/RAW0001/Bus29_Competition_Data_nanmask.csv
./data/RAW0001/Bus2_Competition_Data_nanmask.csv
./data/RAW0001/Bus39_Competition_Data_nanmask.csv
./data/RAW0001/Bus5_Competition_Data_nanmask.csv
./data/RAW0001/Bus6_Competition_Data_nanmask.csv
./data/metadata
./data/metadata/Event Timeline & Location.xlsx
./data/metadata/IEEE_39_Bus_Power_System.raw
./data/metadata/PMUbus_ Location.txt
./data/synth
./docs
./docs/architecture.md
./docs/migration_plan_phase1.md
./docs/parity_contracts.md
./docs/testing_strategy.md
./project_context_report.txt
./pyproject.toml
./pytest.ini
./scan_project.sh
./src
./src/__init__.py
./src/__pycache__
./src/__pycache__/__init__.cpython-313.pyc
./src/application
./src/application/__init__.py
./src/application/__pycache__
./src/application/__pycache__/__init__.cpython-313.pyc
./src/application/dto
./src/application/dto/__init__.py
./src/application/use_cases
./src/application/use_cases/__init__.py
./src/application/use_cases/__pycache__
./src/application/use_cases/__pycache__/__init__.cpython-313.pyc
./src/application/use_cases/__pycache__/chunk_raw_data.cpython-313.pyc
./src/application/use_cases/build_report_assets.py
./src/application/use_cases/calibrate_event0.py
./src/application/use_cases/chunk_raw_data.py
./src/application/use_cases/estimate_unobserved_buses.py
./src/application/use_cases/evaluate_models.py
./src/application/use_cases/export_submission.py
./src/application/use_cases/normalize_data.py
./src/application/use_cases/prepare_windows.py
./src/application/use_cases/profile_noise.py
./src/application/use_cases/simulate_cyber_event.py
./src/application/use_cases/simulate_physical_event.py
./src/application/use_cases/train_task1.py
./src/application/use_cases/train_task2.py
./src/application/use_cases/train_task3.py
./src/calibration
./src/calibration/__init__.py
./src/calibration/current_mapping.py
./src/calibration/event0_calibrator.py
./src/calibration/measurement_model.py
./src/calibration/noise_profiler.py
./src/cli
./src/cli/__init__.py
./src/cli/__pycache__
./src/cli/__pycache__/__init__.cpython-313.pyc
./src/cli/__pycache__/generate_data.cpython-313.pyc
./src/cli/__pycache__/prepare_dataset.cpython-313.pyc
./src/cli/build_report.py
./src/cli/evaluate.py
./src/cli/export_submission.py
./src/cli/generate_data.py
./src/cli/prepare_dataset.py
./src/cli/train.py
./src/data_engineering
./src/data_engineering/__init__.py
./src/data_engineering/__pycache__
./src/data_engineering/__pycache__/__init__.cpython-313.pyc
./src/data_engineering/__pycache__/chunking.cpython-313.pyc
./src/data_engineering/chunking.py
./src/data_engineering/feature_factory.py
./src/data_engineering/imputation.py
./src/data_engineering/scaling.py
./src/data_engineering/splitting.py
./src/data_engineering/windowing.py
./src/domain
./src/domain/__init__.py
./src/domain/constants.py
./src/domain/events.py
./src/domain/models.py
./src/domain/topology.py
./src/estimation
./src/estimation/__init__.py
./src/estimation/__pycache__
./src/estimation/__pycache__/__init__.cpython-313.pyc
./src/estimation/__pycache__/metrics.cpython-313.pyc
./src/estimation/metrics.py
./src/estimation/temporal_regularized_estimator.py
./src/estimation/ybus_estimator.py
./src/evaluation
./src/evaluation/__init__.py
./src/evaluation/classification_metrics.py
./src/evaluation/detection_metrics.py
./src/evaluation/efficiency_metrics.py
./src/evaluation/localization_metrics.py
./src/evaluation/report_tables.py
./src/evaluation/runtime_profiler.py
./src/infrastructure
./src/infrastructure/__init__.py
./src/infrastructure/__pycache__
./src/infrastructure/__pycache__/__init__.cpython-313.pyc
./src/infrastructure/andes
./src/infrastructure/andes/__init__.py
./src/infrastructure/andes/engine.py
./src/infrastructure/andes/extractors.py
./src/infrastructure/andes/scenarios.py
./src/infrastructure/io
./src/infrastructure/io/__init__.py
./src/infrastructure/io/artifact_repository.py
./src/infrastructure/io/csv_repository.py
./src/infrastructure/io/json_repository.py
./src/infrastructure/legacy
./src/infrastructure/legacy/__init__.py
./src/infrastructure/legacy/__pycache__
./src/infrastructure/legacy/__pycache__/__init__.cpython-313.pyc
./src/infrastructure/legacy/__pycache__/m0_adapter.cpython-313.pyc
./src/infrastructure/legacy/__pycache__/m1_adapter.cpython-313.pyc
./src/infrastructure/legacy/__pycache__/m2_adapter.cpython-313.pyc
./src/infrastructure/legacy/__pycache__/m3_adapter.cpython-313.pyc
./src/infrastructure/legacy/__pycache__/m4_adapter.cpython-313.pyc
./src/infrastructure/legacy/m0_adapter.py
./src/infrastructure/legacy/m1_adapter.py
./src/infrastructure/legacy/m2_adapter.py
./src/infrastructure/legacy/m3_adapter.py
./src/infrastructure/legacy/m4_adapter.py
./src/ml
./src/ml/__init__.py
./src/ml/datasets.py
./src/ml/model_registry.py
./src/ml/models
./src/ml/models/__init__.py
./src/ml/models/baselines.py
./src/ml/models/graph.py
./src/ml/models/sequence.py
./src/ml/tasks
./src/ml/tasks/__init__.py
./src/ml/tasks/classifier.py
./src/ml/tasks/detector.py
./src/ml/tasks/localizer.py
./src/ml/trainer.py
./src/physics
./src/physics/__init__.py
./src/physics/__pycache__
./src/physics/__pycache__/__init__.cpython-313.pyc
./src/physics/__pycache__/positive_sequence.cpython-313.pyc
./src/physics/electrical_distance.py
./src/physics/positive_sequence.py
./src/physics/state_estimation.py
./src/physics/ybus.py
./src/physics/zbus.py
./src/signals
./src/signals/__init__.py
./src/signals/__pycache__
./src/signals/__pycache__/__init__.cpython-313.pyc
./src/signals/__pycache__/rocof.cpython-313.pyc
./src/signals/__pycache__/transforms.cpython-313.pyc
./src/signals/normalization_stats.py
./src/signals/phasors.py
./src/signals/rocof.py
./src/signals/transforms.py
./src/simulation
./src/simulation/__init__.py
./src/simulation/cyber.py
./src/simulation/orchestrator.py
./src/simulation/physical.py
./src/submission
./src/submission/__init__.py
./src/submission/formats.py
./src/submission/validator.py
./src/submission/writer.py
./tests
./tests/__pycache__
./tests/__pycache__/conftest.cpython-313-pytest-8.4.2.pyc
./tests/conftest.py
./tests/fixtures
./tests/fixtures/estimated_small
./tests/fixtures/json
./tests/fixtures/json/snapshot_chunk_metadata.json
./tests/fixtures/json/snapshot_estimation_summary.json
./tests/fixtures/json/snapshot_normalization_baselines.csv
./tests/fixtures/raw_small
./tests/fixtures/raw_small/Bus10_Competition_Data_nanmask.csv
./tests/fixtures/raw_small/Bus19_Competition_Data_nanmask.csv
./tests/fixtures/raw_small/Bus22_Competition_Data_nanmask.csv
./tests/fixtures/simulated_small
./tests/integration
./tests/integration/__pycache__
./tests/integration/__pycache__/test_legacy_m0_pipeline.cpython-313-pytest-8.4.2.pyc
./tests/integration/__pycache__/test_legacy_m1_pipeline.cpython-313-pytest-8.4.2.pyc
./tests/integration/__pycache__/test_legacy_m2_pipeline.cpython-313-pytest-8.4.2.pyc
./tests/integration/__pycache__/test_legacy_m3_pipeline.cpython-313-pytest-8.4.2.pyc
./tests/integration/__pycache__/test_legacy_m4_metrics_pipeline.cpython-313-pytest-8.4.2.pyc
./tests/integration/__pycache__/test_m0_chunk_use_case.cpython-313-pytest-8.4.2.pyc
./tests/integration/__pycache__/test_submission_alignment.cpython-313-pytest-8.4.2.pyc
./tests/integration/test_legacy_m0_pipeline.py
./tests/integration/test_legacy_m1_pipeline.py
./tests/integration/test_legacy_m2_pipeline.py
./tests/integration/test_legacy_m3_pipeline.py
./tests/integration/test_legacy_m4_metrics_pipeline.py
./tests/integration/test_m0_chunk_use_case.py
./tests/integration/test_submission_alignment.py
./tests/regression
./tests/regression/__pycache__
./tests/regression/__pycache__/test_snapshot_chunk_metadata.cpython-313-pytest-8.4.2.pyc
./tests/regression/__pycache__/test_snapshot_estimation_summary.cpython-313-pytest-8.4.2.pyc
./tests/regression/__pycache__/test_snapshot_m0_parity_vs_legacy.cpython-313-pytest-8.4.2.pyc
./tests/regression/__pycache__/test_snapshot_normalization_baselines.cpython-313-pytest-8.4.2.pyc
./tests/regression/test_snapshot_chunk_metadata.py
./tests/regression/test_snapshot_estimation_summary.py
./tests/regression/test_snapshot_m0_parity_vs_legacy.py
./tests/regression/test_snapshot_normalization_baselines.py
./tests/smoke
./tests/smoke/__pycache__
./tests/smoke/__pycache__/test_cli_generate_data.cpython-313-pytest-8.4.2.pyc
./tests/smoke/__pycache__/test_cli_prepare_dataset.cpython-313-pytest-8.4.2.pyc
./tests/smoke/test_cli_generate_data.py
./tests/smoke/test_cli_prepare_dataset.py
./tests/unit
./tests/unit/__pycache__
./tests/unit/__pycache__/test_angle_math.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_chunk_boundaries.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_chunk_metadata.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_m0_chunk_subroutines.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_metrics_math.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_normalization_baselines.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_positive_sequence.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_rocof.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_sanity_rows.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_schema_contracts.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_ybus_helpers.cpython-313-pytest-8.4.2.pyc
./tests/unit/chunks
./tests/unit/chunks/test_chunk_boundaries.py
./tests/unit/chunks/test_chunk_logic.py
./tests/unit/chunks/test_chunk_mask.py
./tests/unit/chunks/test_chunk_metadata.py
./tests/unit/test_angle_math.py
./tests/unit/test_chunk_boundaries.py
./tests/unit/test_chunk_metadata.py
./tests/unit/test_m0_chunk_subroutines.py
./tests/unit/test_metrics_math.py
./tests/unit/test_normalization_baselines.py
./tests/unit/test_positive_sequence.py
./tests/unit/test_rocof.py
./tests/unit/test_sanity_rows.py
./tests/unit/test_schema_contracts.py
./tests/unit/test_ybus_helpers.py

============================================================
DIRECTORIES ONLY
============================================================
.
./.pytest_cache
./.pytest_cache/v
./.pytest_cache/v/cache
./configs
./data
./data/RAW0001
./data/metadata
./data/synth
./docs
./src
./src/__pycache__
./src/application
./src/application/__pycache__
./src/application/dto
./src/application/use_cases
./src/application/use_cases/__pycache__
./src/calibration
./src/cli
./src/cli/__pycache__
./src/data_engineering
./src/data_engineering/__pycache__
./src/domain
./src/estimation
./src/estimation/__pycache__
./src/evaluation
./src/infrastructure
./src/infrastructure/__pycache__
./src/infrastructure/andes
./src/infrastructure/io
./src/infrastructure/legacy
./src/infrastructure/legacy/__pycache__
./src/ml
./src/ml/models
./src/ml/tasks
./src/physics
./src/physics/__pycache__
./src/signals
./src/signals/__pycache__
./src/simulation
./src/submission
./tests
./tests/__pycache__
./tests/fixtures
./tests/fixtures/estimated_small
./tests/fixtures/json
./tests/fixtures/raw_small
./tests/fixtures/simulated_small
./tests/integration
./tests/integration/__pycache__
./tests/regression
./tests/regression/__pycache__
./tests/smoke
./tests/smoke/__pycache__
./tests/unit
./tests/unit/__pycache__
./tests/unit/chunks

============================================================
FILES ONLY
============================================================
./.gitignore
./.pytest_cache/.gitignore
./.pytest_cache/CACHEDIR.TAG
./.pytest_cache/README.md
./.pytest_cache/v/cache/nodeids
./README.md
./configs/data.yaml
./configs/simulation.yaml
./configs/training_task1.yaml
./configs/training_task2.yaml
./configs/training_task3.yaml
./data/RAW0001/Bus10_Competition_Data_nanmask.csv
./data/RAW0001/Bus19_Competition_Data_nanmask.csv
./data/RAW0001/Bus22_Competition_Data_nanmask.csv
./data/RAW0001/Bus29_Competition_Data_nanmask.csv
./data/RAW0001/Bus2_Competition_Data_nanmask.csv
./data/RAW0001/Bus39_Competition_Data_nanmask.csv
./data/RAW0001/Bus5_Competition_Data_nanmask.csv
./data/RAW0001/Bus6_Competition_Data_nanmask.csv
./data/metadata/Event Timeline & Location.xlsx
./data/metadata/IEEE_39_Bus_Power_System.raw
./data/metadata/PMUbus_ Location.txt
./docs/architecture.md
./docs/migration_plan_phase1.md
./docs/parity_contracts.md
./docs/testing_strategy.md
./project_context_report.txt
./pyproject.toml
./pytest.ini
./scan_project.sh
./src/__init__.py
./src/__pycache__/__init__.cpython-313.pyc
./src/application/__init__.py
./src/application/__pycache__/__init__.cpython-313.pyc
./src/application/dto/__init__.py
./src/application/use_cases/__init__.py
./src/application/use_cases/__pycache__/__init__.cpython-313.pyc
./src/application/use_cases/__pycache__/chunk_raw_data.cpython-313.pyc
./src/application/use_cases/build_report_assets.py
./src/application/use_cases/calibrate_event0.py
./src/application/use_cases/chunk_raw_data.py
./src/application/use_cases/estimate_unobserved_buses.py
./src/application/use_cases/evaluate_models.py
./src/application/use_cases/export_submission.py
./src/application/use_cases/normalize_data.py
./src/application/use_cases/prepare_windows.py
./src/application/use_cases/profile_noise.py
./src/application/use_cases/simulate_cyber_event.py
./src/application/use_cases/simulate_physical_event.py
./src/application/use_cases/train_task1.py
./src/application/use_cases/train_task2.py
./src/application/use_cases/train_task3.py
./src/calibration/__init__.py
./src/calibration/current_mapping.py
./src/calibration/event0_calibrator.py
./src/calibration/measurement_model.py
./src/calibration/noise_profiler.py
./src/cli/__init__.py
./src/cli/__pycache__/__init__.cpython-313.pyc
./src/cli/__pycache__/generate_data.cpython-313.pyc
./src/cli/__pycache__/prepare_dataset.cpython-313.pyc
./src/cli/build_report.py
./src/cli/evaluate.py
./src/cli/export_submission.py
./src/cli/generate_data.py
./src/cli/prepare_dataset.py
./src/cli/train.py
./src/data_engineering/__init__.py
./src/data_engineering/__pycache__/__init__.cpython-313.pyc
./src/data_engineering/__pycache__/chunking.cpython-313.pyc
./src/data_engineering/chunking.py
./src/data_engineering/feature_factory.py
./src/data_engineering/imputation.py
./src/data_engineering/scaling.py
./src/data_engineering/splitting.py
./src/data_engineering/windowing.py
./src/domain/__init__.py
./src/domain/constants.py
./src/domain/events.py
./src/domain/models.py
./src/domain/topology.py
./src/estimation/__init__.py
./src/estimation/__pycache__/__init__.cpython-313.pyc
./src/estimation/__pycache__/metrics.cpython-313.pyc
./src/estimation/metrics.py
./src/estimation/temporal_regularized_estimator.py
./src/estimation/ybus_estimator.py
./src/evaluation/__init__.py
./src/evaluation/classification_metrics.py
./src/evaluation/detection_metrics.py
./src/evaluation/efficiency_metrics.py
./src/evaluation/localization_metrics.py
./src/evaluation/report_tables.py
./src/evaluation/runtime_profiler.py
./src/infrastructure/__init__.py
./src/infrastructure/__pycache__/__init__.cpython-313.pyc
./src/infrastructure/andes/__init__.py
./src/infrastructure/andes/engine.py
./src/infrastructure/andes/extractors.py
./src/infrastructure/andes/scenarios.py
./src/infrastructure/io/__init__.py
./src/infrastructure/io/artifact_repository.py
./src/infrastructure/io/csv_repository.py
./src/infrastructure/io/json_repository.py
./src/infrastructure/legacy/__init__.py
./src/infrastructure/legacy/__pycache__/__init__.cpython-313.pyc
./src/infrastructure/legacy/__pycache__/m0_adapter.cpython-313.pyc
./src/infrastructure/legacy/__pycache__/m1_adapter.cpython-313.pyc
./src/infrastructure/legacy/__pycache__/m2_adapter.cpython-313.pyc
./src/infrastructure/legacy/__pycache__/m3_adapter.cpython-313.pyc
./src/infrastructure/legacy/__pycache__/m4_adapter.cpython-313.pyc
./src/infrastructure/legacy/m0_adapter.py
./src/infrastructure/legacy/m1_adapter.py
./src/infrastructure/legacy/m2_adapter.py
./src/infrastructure/legacy/m3_adapter.py
./src/infrastructure/legacy/m4_adapter.py
./src/ml/__init__.py
./src/ml/datasets.py
./src/ml/model_registry.py
./src/ml/models/__init__.py
./src/ml/models/baselines.py
./src/ml/models/graph.py
./src/ml/models/sequence.py
./src/ml/tasks/__init__.py
./src/ml/tasks/classifier.py
./src/ml/tasks/detector.py
./src/ml/tasks/localizer.py
./src/ml/trainer.py
./src/physics/__init__.py
./src/physics/__pycache__/__init__.cpython-313.pyc
./src/physics/__pycache__/positive_sequence.cpython-313.pyc
./src/physics/electrical_distance.py
./src/physics/positive_sequence.py
./src/physics/state_estimation.py
./src/physics/ybus.py
./src/physics/zbus.py
./src/signals/__init__.py
./src/signals/__pycache__/__init__.cpython-313.pyc
./src/signals/__pycache__/rocof.cpython-313.pyc
./src/signals/__pycache__/transforms.cpython-313.pyc
./src/signals/normalization_stats.py
./src/signals/phasors.py
./src/signals/rocof.py
./src/signals/transforms.py
./src/simulation/__init__.py
./src/simulation/cyber.py
./src/simulation/orchestrator.py
./src/simulation/physical.py
./src/submission/__init__.py
./src/submission/formats.py
./src/submission/validator.py
./src/submission/writer.py
./tests/__pycache__/conftest.cpython-313-pytest-8.4.2.pyc
./tests/conftest.py
./tests/fixtures/json/snapshot_chunk_metadata.json
./tests/fixtures/json/snapshot_estimation_summary.json
./tests/fixtures/json/snapshot_normalization_baselines.csv
./tests/fixtures/raw_small/Bus10_Competition_Data_nanmask.csv
./tests/fixtures/raw_small/Bus19_Competition_Data_nanmask.csv
./tests/fixtures/raw_small/Bus22_Competition_Data_nanmask.csv
./tests/integration/__pycache__/test_legacy_m0_pipeline.cpython-313-pytest-8.4.2.pyc
./tests/integration/__pycache__/test_legacy_m1_pipeline.cpython-313-pytest-8.4.2.pyc
./tests/integration/__pycache__/test_legacy_m2_pipeline.cpython-313-pytest-8.4.2.pyc
./tests/integration/__pycache__/test_legacy_m3_pipeline.cpython-313-pytest-8.4.2.pyc
./tests/integration/__pycache__/test_legacy_m4_metrics_pipeline.cpython-313-pytest-8.4.2.pyc
./tests/integration/__pycache__/test_m0_chunk_use_case.cpython-313-pytest-8.4.2.pyc
./tests/integration/__pycache__/test_submission_alignment.cpython-313-pytest-8.4.2.pyc
./tests/integration/test_legacy_m0_pipeline.py
./tests/integration/test_legacy_m1_pipeline.py
./tests/integration/test_legacy_m2_pipeline.py
./tests/integration/test_legacy_m3_pipeline.py
./tests/integration/test_legacy_m4_metrics_pipeline.py
./tests/integration/test_m0_chunk_use_case.py
./tests/integration/test_submission_alignment.py
./tests/regression/__pycache__/test_snapshot_chunk_metadata.cpython-313-pytest-8.4.2.pyc
./tests/regression/__pycache__/test_snapshot_estimation_summary.cpython-313-pytest-8.4.2.pyc
./tests/regression/__pycache__/test_snapshot_m0_parity_vs_legacy.cpython-313-pytest-8.4.2.pyc
./tests/regression/__pycache__/test_snapshot_normalization_baselines.cpython-313-pytest-8.4.2.pyc
./tests/regression/test_snapshot_chunk_metadata.py
./tests/regression/test_snapshot_estimation_summary.py
./tests/regression/test_snapshot_m0_parity_vs_legacy.py
./tests/regression/test_snapshot_normalization_baselines.py
./tests/smoke/__pycache__/test_cli_generate_data.cpython-313-pytest-8.4.2.pyc
./tests/smoke/__pycache__/test_cli_prepare_dataset.cpython-313-pytest-8.4.2.pyc
./tests/smoke/test_cli_generate_data.py
./tests/smoke/test_cli_prepare_dataset.py
./tests/unit/__pycache__/test_angle_math.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_chunk_boundaries.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_chunk_metadata.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_m0_chunk_subroutines.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_metrics_math.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_normalization_baselines.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_positive_sequence.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_rocof.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_sanity_rows.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_schema_contracts.cpython-313-pytest-8.4.2.pyc
./tests/unit/__pycache__/test_ybus_helpers.cpython-313-pytest-8.4.2.pyc
./tests/unit/chunks/test_chunk_boundaries.py
./tests/unit/chunks/test_chunk_logic.py
./tests/unit/chunks/test_chunk_mask.py
./tests/unit/chunks/test_chunk_metadata.py
./tests/unit/test_angle_math.py
./tests/unit/test_chunk_boundaries.py
./tests/unit/test_chunk_metadata.py
./tests/unit/test_m0_chunk_subroutines.py
./tests/unit/test_metrics_math.py
./tests/unit/test_normalization_baselines.py
./tests/unit/test_positive_sequence.py
./tests/unit/test_rocof.py
./tests/unit/test_sanity_rows.py
./tests/unit/test_schema_contracts.py
./tests/unit/test_ybus_helpers.py

============================================================
FILES WITH SIZE IN BYTES
============================================================
         127  .gitignore
          39  .pytest_cache/.gitignore
         191  .pytest_cache/CACHEDIR.TAG
         310  .pytest_cache/README.md
        2643  .pytest_cache/v/cache/nodeids
         287  configs/data.yaml
         133  configs/simulation.yaml
         106  configs/training_task1.yaml
         116  configs/training_task2.yaml
         112  configs/training_task3.yaml
       10700  data/metadata/Event Timeline & Location.xlsx
       16185  data/metadata/IEEE_39_Bus_Power_System.raw
        2124  data/metadata/PMUbus_ Location.txt
    38077730  data/RAW0001/Bus10_Competition_Data_nanmask.csv
    38070799  data/RAW0001/Bus19_Competition_Data_nanmask.csv
    37606138  data/RAW0001/Bus2_Competition_Data_nanmask.csv
    38339889  data/RAW0001/Bus22_Competition_Data_nanmask.csv
    34834438  data/RAW0001/Bus29_Competition_Data_nanmask.csv
    36600134  data/RAW0001/Bus39_Competition_Data_nanmask.csv
    38077650  data/RAW0001/Bus5_Competition_Data_nanmask.csv
    38076823  data/RAW0001/Bus6_Competition_Data_nanmask.csv
        1111  docs/architecture.md
        3380  docs/migration_plan_phase1.md
        1101  docs/parity_contracts.md
         677  docs/testing_strategy.md
       23014  project_context_report.txt
         618  pyproject.toml
         304  pytest.ini
         689  README.md
        3016  scan_project.sh
          34  src/__init__.py
         226  src/__pycache__/__init__.cpython-313.pyc
          34  src/application/__init__.py
         238  src/application/__pycache__/__init__.cpython-313.pyc
          34  src/application/dto/__init__.py
          34  src/application/use_cases/__init__.py
         248  src/application/use_cases/__pycache__/__init__.cpython-313.pyc
         918  src/application/use_cases/__pycache__/chunk_raw_data.cpython-313.pyc
         128  src/application/use_cases/build_report_assets.py
         125  src/application/use_cases/calibrate_event0.py
         613  src/application/use_cases/chunk_raw_data.py
         134  src/application/use_cases/estimate_unobserved_buses.py
         124  src/application/use_cases/evaluate_models.py
         126  src/application/use_cases/export_submission.py
         123  src/application/use_cases/normalize_data.py
         124  src/application/use_cases/prepare_windows.py
         122  src/application/use_cases/profile_noise.py
         129  src/application/use_cases/simulate_cyber_event.py
         132  src/application/use_cases/simulate_physical_event.py
         120  src/application/use_cases/train_task1.py
         120  src/application/use_cases/train_task2.py
         120  src/application/use_cases/train_task3.py
          34  src/calibration/__init__.py
         124  src/calibration/current_mapping.py
         126  src/calibration/event0_calibrator.py
         126  src/calibration/measurement_model.py
         123  src/calibration/noise_profiler.py
          48  src/cli/__init__.py
         244  src/cli/__pycache__/__init__.cpython-313.pyc
        1344  src/cli/__pycache__/generate_data.cpython-313.pyc
        1011  src/cli/__pycache__/prepare_dataset.cpython-313.pyc
         232  src/cli/build_report.py
         228  src/cli/evaluate.py
         242  src/cli/export_submission.py
         796  src/cli/generate_data.py
         503  src/cli/prepare_dataset.py
         224  src/cli/train.py
          34  src/data_engineering/__init__.py
         243  src/data_engineering/__pycache__/__init__.cpython-313.pyc
       21185  src/data_engineering/__pycache__/chunking.cpython-313.pyc
       14621  src/data_engineering/chunking.py
         124  src/data_engineering/feature_factory.py
         119  src/data_engineering/imputation.py
         116  src/data_engineering/scaling.py
         118  src/data_engineering/splitting.py
         118  src/data_engineering/windowing.py
          34  src/domain/__init__.py
         118  src/domain/constants.py
         115  src/domain/events.py
         115  src/domain/models.py
         117  src/domain/topology.py
          34  src/estimation/__init__.py
         237  src/estimation/__pycache__/__init__.cpython-313.pyc
        2445  src/estimation/__pycache__/metrics.cpython-313.pyc
        1188  src/estimation/metrics.py
         139  src/estimation/temporal_regularized_estimator.py
         123  src/estimation/ybus_estimator.py
          34  src/evaluation/__init__.py
         131  src/evaluation/classification_metrics.py
         126  src/evaluation/detection_metrics.py
         659  src/evaluation/efficiency_metrics.py
         129  src/evaluation/localization_metrics.py
         122  src/evaluation/report_tables.py
         125  src/evaluation/runtime_profiler.py
          34  src/infrastructure/__init__.py
         241  src/infrastructure/__pycache__/__init__.cpython-313.pyc
          34  src/infrastructure/andes/__init__.py
         115  src/infrastructure/andes/engine.py
         119  src/infrastructure/andes/extractors.py
         118  src/infrastructure/andes/scenarios.py
          34  src/infrastructure/io/__init__.py
         128  src/infrastructure/io/artifact_repository.py
         123  src/infrastructure/io/csv_repository.py
         124  src/infrastructure/io/json_repository.py
          58  src/infrastructure/legacy/__init__.py
         272  src/infrastructure/legacy/__pycache__/__init__.cpython-313.pyc
        1170  src/infrastructure/legacy/__pycache__/m0_adapter.cpython-313.pyc
        3363  src/infrastructure/legacy/__pycache__/m1_adapter.cpython-313.pyc
        1527  src/infrastructure/legacy/__pycache__/m2_adapter.cpython-313.pyc
        1244  src/infrastructure/legacy/__pycache__/m3_adapter.cpython-313.pyc
        2769  src/infrastructure/legacy/__pycache__/m4_adapter.cpython-313.pyc
         927  src/infrastructure/legacy/m0_adapter.py
        2112  src/infrastructure/legacy/m1_adapter.py
         863  src/infrastructure/legacy/m2_adapter.py
         555  src/infrastructure/legacy/m3_adapter.py
        1545  src/infrastructure/legacy/m4_adapter.py
          34  src/ml/__init__.py
         117  src/ml/datasets.py
         123  src/ml/model_registry.py
          34  src/ml/models/__init__.py
         118  src/ml/models/baselines.py
         114  src/ml/models/graph.py
         117  src/ml/models/sequence.py
          34  src/ml/tasks/__init__.py
         119  src/ml/tasks/classifier.py
         117  src/ml/tasks/detector.py
         118  src/ml/tasks/localizer.py
         116  src/ml/trainer.py
          34  src/physics/__init__.py
         234  src/physics/__pycache__/__init__.cpython-313.pyc
         796  src/physics/__pycache__/positive_sequence.cpython-313.pyc
         128  src/physics/electrical_distance.py
         340  src/physics/positive_sequence.py
         125  src/physics/state_estimation.py
         113  src/physics/ybus.py
         113  src/physics/zbus.py
          34  src/signals/__init__.py
         234  src/signals/__pycache__/__init__.cpython-313.pyc
        1669  src/signals/__pycache__/rocof.cpython-313.pyc
        1081  src/signals/__pycache__/transforms.cpython-313.pyc
         128  src/signals/normalization_stats.py
         116  src/signals/phasors.py
         855  src/signals/rocof.py
         543  src/signals/transforms.py
          34  src/simulation/__init__.py
         114  src/simulation/cyber.py
         121  src/simulation/orchestrator.py
         117  src/simulation/physical.py
          34  src/submission/__init__.py
         116  src/submission/formats.py
         118  src/submission/validator.py
         115  src/submission/writer.py
        2691  tests/__pycache__/conftest.cpython-313-pytest-8.4.2.pyc
        1316  tests/conftest.py
         598  tests/fixtures/json/snapshot_chunk_metadata.json
         121  tests/fixtures/json/snapshot_estimation_summary.json
         154  tests/fixtures/json/snapshot_normalization_baselines.csv
        1290  tests/fixtures/raw_small/Bus10_Competition_Data_nanmask.csv
        1298  tests/fixtures/raw_small/Bus19_Competition_Data_nanmask.csv
        1285  tests/fixtures/raw_small/Bus22_Competition_Data_nanmask.csv
        3986  tests/integration/__pycache__/test_legacy_m0_pipeline.cpython-313-pytest-8.4.2.pyc
        4006  tests/integration/__pycache__/test_legacy_m1_pipeline.cpython-313-pytest-8.4.2.pyc
        2901  tests/integration/__pycache__/test_legacy_m2_pipeline.cpython-313-pytest-8.4.2.pyc
        1608  tests/integration/__pycache__/test_legacy_m3_pipeline.cpython-313-pytest-8.4.2.pyc
        4271  tests/integration/__pycache__/test_legacy_m4_metrics_pipeline.cpython-313-pytest-8.4.2.pyc
        5721  tests/integration/__pycache__/test_m0_chunk_use_case.cpython-313-pytest-8.4.2.pyc
         905  tests/integration/__pycache__/test_submission_alignment.cpython-313-pytest-8.4.2.pyc
         723  tests/integration/test_legacy_m0_pipeline.py
         755  tests/integration/test_legacy_m1_pipeline.py
         749  tests/integration/test_legacy_m2_pipeline.py
         429  tests/integration/test_legacy_m3_pipeline.py
        1435  tests/integration/test_legacy_m4_metrics_pipeline.py
        1016  tests/integration/test_m0_chunk_use_case.py
         228  tests/integration/test_submission_alignment.py
        2063  tests/regression/__pycache__/test_snapshot_chunk_metadata.cpython-313-pytest-8.4.2.pyc
        3898  tests/regression/__pycache__/test_snapshot_estimation_summary.cpython-313-pytest-8.4.2.pyc
       37548  tests/regression/__pycache__/test_snapshot_m0_parity_vs_legacy.cpython-313-pytest-8.4.2.pyc
        2856  tests/regression/__pycache__/test_snapshot_normalization_baselines.cpython-313-pytest-8.4.2.pyc
         570  tests/regression/test_snapshot_chunk_metadata.py
         781  tests/regression/test_snapshot_estimation_summary.py
        9488  tests/regression/test_snapshot_m0_parity_vs_legacy.py
        1091  tests/regression/test_snapshot_normalization_baselines.py
        1603  tests/smoke/__pycache__/test_cli_generate_data.cpython-313-pytest-8.4.2.pyc
        1609  tests/smoke/__pycache__/test_cli_prepare_dataset.cpython-313-pytest-8.4.2.pyc
         203  tests/smoke/test_cli_generate_data.py
         209  tests/smoke/test_cli_prepare_dataset.py
        5021  tests/unit/__pycache__/test_angle_math.cpython-313-pytest-8.4.2.pyc
        3306  tests/unit/__pycache__/test_chunk_boundaries.cpython-313-pytest-8.4.2.pyc
        3815  tests/unit/__pycache__/test_chunk_metadata.cpython-313-pytest-8.4.2.pyc
        5115  tests/unit/__pycache__/test_m0_chunk_subroutines.cpython-313-pytest-8.4.2.pyc
        6730  tests/unit/__pycache__/test_metrics_math.cpython-313-pytest-8.4.2.pyc
        5398  tests/unit/__pycache__/test_normalization_baselines.cpython-313-pytest-8.4.2.pyc
        3312  tests/unit/__pycache__/test_positive_sequence.cpython-313-pytest-8.4.2.pyc
        4350  tests/unit/__pycache__/test_rocof.cpython-313-pytest-8.4.2.pyc
         772  tests/unit/__pycache__/test_sanity_rows.cpython-313-pytest-8.4.2.pyc
        4204  tests/unit/__pycache__/test_schema_contracts.cpython-313-pytest-8.4.2.pyc
         769  tests/unit/__pycache__/test_ybus_helpers.cpython-313-pytest-8.4.2.pyc
        1090  tests/unit/chunks/test_chunk_boundaries.py
        5477  tests/unit/chunks/test_chunk_logic.py
           0  tests/unit/chunks/test_chunk_mask.py
         868  tests/unit/chunks/test_chunk_metadata.py
         592  tests/unit/test_angle_math.py
         650  tests/unit/test_chunk_boundaries.py
         588  tests/unit/test_chunk_metadata.py
        1297  tests/unit/test_m0_chunk_subroutines.py
         783  tests/unit/test_metrics_math.py
        1054  tests/unit/test_normalization_baselines.py
         433  tests/unit/test_positive_sequence.py
         515  tests/unit/test_rocof.py
         167  tests/unit/test_sanity_rows.py
         543  tests/unit/test_schema_contracts.py
         151  tests/unit/test_ybus_helpers.py

============================================================
TEXT / CODE / CONFIG FILE CONTENT PREVIEW
============================================================

------------------------------------------------------------
FILE: .gitignore
SIZE_BYTES: 127
