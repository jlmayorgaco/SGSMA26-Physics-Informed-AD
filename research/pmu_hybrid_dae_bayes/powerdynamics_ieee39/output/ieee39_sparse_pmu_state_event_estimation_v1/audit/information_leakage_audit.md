# Information leakage audit

All estimator inputs contain only canonical time, 32 PMU channels, and a deterministic contract hash. Inference receives no truth object, support, severity, native state, or non-PMU bus. BMA reconstruction consumes only the frozen state manifold and normalized posterior. Ground truth is first loaded by `score_reconstruction`, after inference CSV artifacts exist and are hashed. The oracle reconstruction is isolated and labeled evaluation-only.

| artifact                   | check                             | status   | detail                                       |
|:---------------------------|:----------------------------------|:---------|:---------------------------------------------|
| estimator_input_case_a.npz | allowed_keys_only                 | PASS     | contract_sha256|pmu_32|time_s                |
| estimator_input_case_a.npz | 32_channels_only                  | PASS     | (120, 32)                                    |
| estimator_input_case_b.npz | allowed_keys_only                 | PASS     | contract_sha256|pmu_32|time_s                |
| estimator_input_case_b.npz | 32_channels_only                  | PASS     | (120, 32)                                    |
| estimator_input_case_c.npz | allowed_keys_only                 | PASS     | contract_sha256|pmu_32|time_s                |
| estimator_input_case_c.npz | 32_channels_only                  | PASS     | (120, 32)                                    |
| infer_observation          | truth_object_not_in_signature     | PASS     | path,D,Q,Qcross,var,y0 only                  |
| bma_reconstruct            | truth_not_in_signature            | PASS     | state manifold and normalized posterior only |
| score_reconstruction       | truth_loaded_after_inference_hash | PASS     | separate scoring stage                       |
