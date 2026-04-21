# Migration Plan (Phase Ledger)

## Function migration ledger
Status values: `pending`, `migrated`, `parity-verified`.

### Slice a) estimation/metrics + signals/transforms/rocof + physics/positive_sequence
| Legacy source | New target | Status |
|---|---|---|
| `m4_andes_faults_type1.wrap_deg` | `src/signals/transforms.wrap_deg` | parity-verified |
| `m4_andes_faults_type1.angle_diff_deg` | `src/signals/transforms.angle_diff_deg` | parity-verified |
| `m4_andes_faults_type1.robust_rocof` | `src/signals/rocof.robust_rocof` | parity-verified |
| `m4_andes_faults_type1.positive_sequence_from_abc` | `src/physics/positive_sequence.positive_sequence_from_abc` | parity-verified |
| `m4_andes_faults_type1.rmse` | `src/estimation/metrics.rmse` | parity-verified |
| `m4_andes_faults_type1.mae` | `src/estimation/metrics.mae` | parity-verified |
| `m4_andes_faults_type1.rel_rmse` | `src/estimation/metrics.rel_rmse` | parity-verified |
| `m4_andes_faults_type1.safe_corr` | `src/estimation/metrics.safe_corr` | parity-verified |

### Slice b) data_engineering/chunking from m0
| Legacy source | New target | Status |
|---|---|---|
| `m0_chunks.load_and_synchronize_data` | `src/data_engineering/chunking.load_and_synchronize_data` | parity-verified |
| `m0_chunks.find_chunk_boundaries` | `src/data_engineering/chunking.find_chunk_boundaries` | parity-verified |
| `m0_chunks.get_chunk_metadata` | `src/data_engineering/chunking.get_chunk_metadata` | parity-verified |
| `m0_chunks._chunk_mask` | `src/data_engineering/chunking.chunk_mask` | parity-verified |
| `m0_chunks._append_sanity_rows` | `src/data_engineering/chunking.append_sanity_rows` | parity-verified |
| `m0_chunks.process_pipeline` | `src/data_engineering/chunking.run_chunk_pipeline` | parity-verified |
| m0 orchestration entrypoint | `src/application/use_cases/chunk_raw_data.run_chunk_raw_data_use_case` | migrated |
| legacy adapter call-path | `src/infrastructure/legacy/m0_adapter.run_chunk_pipeline` | migrated |

### Slice c) normalization/scaling from m1
| Legacy source | New target | Status |
|---|---|---|
| `m1_preprocessing.normalize_bus_data` and pipeline | `src/data_engineering/scaling.py` + use case wiring | pending |

### Slice d) calibration/noise_profiling from m2
| Legacy source | New target | Status |
|---|---|---|
| `m2_noise_profiling_raw` analyzers and profile generation | `src/calibration/noise_profiler.py` | pending |

### Slice e) event0 calibration/current mapping from m3
| Legacy source | New target | Status |
|---|---|---|
| `m3_andes_calibration_raw` internals | `src/calibration/event0_calibrator.py` / `measurement_model.py` / `current_mapping.py` | pending |

### Slice f) ybus estimation + temporal regularization + orchestration from m4
| Legacy source | New target | Status |
|---|---|---|
| `m4` YBUS and temporal estimation flow | `src/estimation/ybus_estimator.py` + `temporal_regularized_estimator.py` + `src/simulation/orchestrator.py` | pending |

## Migration order
1. metrics helpers
2. angle/signal transforms
3. chunking
4. normalization
5. noise profiling
6. calibration
7. YBUS estimation
8. orchestration and CLI

## Guardrails
- Keep legacy scripts runnable until final cutover.
- Move one slice at a time behind parity tests.
- Keep output paths and artifact naming conventions unchanged.
- Require regression snapshots for every migrated slice.
