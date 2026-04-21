# V6 Cyber (Simple)

Implementacion minima para deteccion cyber en chunks RAW:

1. `event 5` (missing data): basado en `NaN` y `DATA_PRESENT`.
2. `event 7` (bad data): score simple por bus con 4 senales:
   - outlier rate
   - jump rate
   - stuck rate
   - replay-like rate
3. Fusion:
   - si hay `event5` -> etiqueta `5`
   - si no hay `event5` y hay corrupcion localizada -> etiqueta `7`
   - en otro caso -> `0`

## Ejecutar sobre M0

```bash
python -m src.detectors.v6.raw_eval --m0-root output/M0_RAW0001_NEWARCH --output-root output/detector_m10/v6_cyber_raw_eval
```

Artefactos generados:

- `output/detector_m10/v6_cyber_raw_eval/v6_cyber_raw_predictions.csv`
- `output/detector_m10/v6_cyber_raw_eval/v6_cyber_raw_eval.json`
- `output/detector_m10/v6_cyber_raw_eval/v6_cyber_raw_eval.md`

