# SGSMA 2026 V1

Production V1 pipeline for physics-informed PMU anomaly detection on the IEEE
39-bus system. This folder is self-contained: run commands from `V1/`.

## Reproduce

```bash
make setup
make test
make augment-smoke
make augment
make benchmark-models
make infer
make simulations
make report
```

Generated synthetic data, model files, report build products, and simulation
artifacts are reproducible outputs and are ignored by git.
