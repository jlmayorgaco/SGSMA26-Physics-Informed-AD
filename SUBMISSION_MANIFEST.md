# Submission Manifest

## Reviewer Entrypoint

- `main.py`

## Required Runtime Assets

- `src/`
- `models_bus_agnostic/`
- `data/topology/`
- `data/metadata/`
- `requirements.txt`

## Documentation

- `README.md`
- `Final.md`
- `report.md`
- `REPRODUCIBILITY.md`
- `SUBMISSION_MANIFEST.md`

## Output Contract

The generated prediction CSV contains exactly:

```text
TIMESTAMP, Bus, Predicted_Event, Predicted_Location
```

## Notes

- PMU buses are discovered from `Bus*.csv` filenames at runtime.
- The final path does not hardcode the RAW0001 PMU placement.
- The compact archive intentionally excludes local training workbench outputs and raw validation data.
