# G1 PMU measurement operator audit

G1-SYNTHETIC = PASS

G1-EXTERNAL = BLOCKED_UNRESOLVED

Static audit summary: `{"max_current_error_pu": 1.432144669219779e-14, "max_power_error_mva": 1.575355130137589e-12, "max_voltage_error_pu": 2.70921440061811e-07, "worst_current_pmu": "PMU4", "worst_current_scale": 0.98, "worst_voltage_pmu": "PMU7"}`

Frequency audit summary: `{"andes_busrocof_configured_count": 0, "andes_busrocof_model_present": true, "causal_future_independent": true, "max_ideal_frequency_error_hz": 7.105427357601002e-15, "native_busfreq_buses": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39], "native_busfreq_pmu_overlap": [39]}`

Frequency semantics: `IDEAL_STATE_DERIVATIVE` is the explicit offline reference; the causal baseline is a trailing phase-slope plus finite-difference ROCOF. ANDES `BusFreq` is configured on buses 30-39, so only PMU bus 39 has a direct native device. The ANDES `BusROCOF` model class is present but has zero configured devices in this workbook; no PMU filter is claimed for the other seven buses.

Tiny native ANDES TDS smoke: `{"error": "", "frame_count": 6, "pflow_ok": true, "tds_ok": true}`. It exercises only the simulator API; the package has no PMU dynamic wrapper, so it is not promoted to a dynamic PMU parity claim.

Scope excludes estimator, localization, Bayesian inference, ML, and large Monte Carlo metrics. The static exact operator and explicit timing modes are the bounded G1 evidence.
