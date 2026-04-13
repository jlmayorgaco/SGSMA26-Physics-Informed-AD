# SGSMA 2026 - Resumen final de entrega

Generado el 2026-04-11 en Windows 11 / Python 3.13.12. Este resumen reemplaza
las notas historicas anteriores; los numeros de abajo salen de la version final
del codigo, predicciones, reporte y artefactos de ingenieria.

## Veredicto

La entrega esta lista para submit desde el punto de vista de auditoria visible y
reproducibilidad local. El resultado de deteccion 1.000 se reporta como
auditoria sobre los datos visibles, no como garantia de hidden test. Nota: el
PDF actual es una version tecnica extendida de 22 paginas creada para explicar
todo el proyecto sin limite de espacio; si el jurado exige estrictamente 3--6
paginas, conviene generar una version corta aparte.

La fuente potencial de optimismo que se encontro fue corregida: la calibracion
del detector ya no filtra por `Event == 0`. Ahora usa candidatos robustos de los
primeros 60 s filtrados por `DATA_PRESENT` y consistencia trifasica. `Event` se
usa para scoring visible y para etiquetas supervisadas, no para disparar alarmas
ni calibrar el umbral del detector.

## Metricas actuales

- Deteccion visible final: precision `1.000`, recall `1.000`, F1 `1.000`, FP/min
  `0.000`, delay medio `0.05 s`, `12 TP / 0 FP / 0 FN`.
- Detector base chi2 + DATA_PRESENT sin bad-data guard: precision `1.000`,
  recall `0.750`, F1 `0.857`, `9 TP / 0 FP / 3 FN`.
- Clasificacion en alarmas emparejadas: macro-F1 `1.000`, weighted-F1 `1.000`,
  etiquetas visibles `1..7`.
- Localizacion visible por alarma: Top-1 `11/12`, Top-3 `11/12`.
- Sample-level por los 8 archivos PMU contra sus propios `Event`: macro-F1
  `0.8607491845` sobre etiquetas `0..8`, macro-F1 visible `0.9683428326` sobre
  `0..7`, weighted-F1 `0.9988057149`.

## Semantica por bus

Las predicciones finales ya no copian una etiqueta global a los 8 PMUs.

- Eventos fisicos `1..4`: se escriben en todos los archivos PMU.
- Missing-data `5`: se escribe solo en el PMU con `DATA_PRESENT == 0`.
- Cyber + physical `6`: se escribe en el PMU caido; los otros PMUs conservan el
  evento fisico subyacente.
- Bad data `7`: se escribe solo en el PMU localizado como corrupto.
- Si un evento fisico ocurre mientras un PMU esta caido, ese PMU se marca como
  `6` y los demas como el evento fisico.

Conteos agregados en `predictions/submission.csv`:

```text
0: 1131464
1: 1200
2: 1200
3: 75878
4: 72000
5: 8646
6: 554
7: 90
```

## Artefactos de ingenieria

La carpeta `report/engineering_review/` esta incluida en el ZIP y contiene:

- `detection_and_scenario_audit.json`: umbrales, fuente de cada alarma, labels,
  ubicaciones, top residual buses y comentario de auditoria del 1.000.
- `per_bus_sample_metrics.json`: metricas por cada PMU contra su propio
  `Event`.
- `units.json`: unidades de PMU, estimados, residual y campos de salida.
- `ieee39_topology.json`: nodos, ramas, PMUs, generadores y coordenadas del
  diagrama IEEE 39.
- `scenario_table.csv`: tabla plana para revision en hoja de calculo.
- `figures/`: 13 PNG, incluyendo diagrama IEEE 39 y un plot por alarma con PMU
  real, bus estimado, eta y score de bad data.

## Verificacion ejecutada

- `python -m pytest tests -q` -> `205 passed, 1 xfailed, 1 warning`; exit code
  `0`. En Windows aparece ruido post-exit conocido de `gmpy2/sympy/ANDES`, sin
  fallo de pytest.
- `python -m src.pipeline.run_inference --data data/raw --out predictions --raw
  "data/metadata/IEEE 39 Bus Power System.raw" --synthetic data/synthetic
  --seed 42 --log-level WARNING` -> metricas de alarma finales indicadas arriba.
- `python -m src.pipeline.validate_submission --pred predictions --data
  data/raw --required-labels 1,2,3,4,5,6,7 --json-out
  report/submission_validation.json` -> `ok: true`.
- `python -m src.pipeline.generate_engineering_review --data data/raw --raw
  "data/metadata/IEEE 39 Bus Power System.raw" --synthetic data/synthetic --out
  report/engineering_review --seed 42` -> `12` alarmas, `13` figuras.
- `make report` -> `report/sgsma2026_report.pdf`, 22 paginas, version tecnica
  extendida con formulas, figuras, ejemplos, auditoria de leakage y explicacion
  de por que algunas metricas son `1.000`.
- `python -m src.pipeline.build_submission_zip --zip submission_sgsma2026.zip`
  -> `ok: true`, `106` archivos, `0` archivos prohibidos.
- `python -m src.pipeline.build_submission_zip --zip submission_sgsma2026.zip
  --audit-only` -> `ok: true`.

## Hashes de artefactos

```text
predictions/submission.csv
SHA256 108ABAB907761952084B33E7FADDCD73963564C4C6F160E2CDE0D65AC7543958

report/sgsma2026_report.pdf
SHA256 3F5BD8B2BA26131FA22F21699348BE241AF272512C8A46B67718B5F4E4BF6925

report/engineering_review/detection_and_scenario_audit.json
SHA256 4CCE50F49A7B9EA05C1913E51D177B869FA5C39D0E351DFCB0E185ACE7C080CF

report/engineering_review/per_bus_sample_metrics.json
SHA256 53B1342B20C8E561A4C53D9D015970F4318A6902DA9F4B1AB604E3DA0E5AEB72
```

## Riesgos residuales

- Label `8` no tiene soporte visible; se mantiene como clase abierta sin claims
  fuertes.
- El 1.000 de deteccion depende de los eventos visibles y de un guard
  deterministico para bad data; el reporte lo presenta como auditoria publica,
  no como garantia de generalizacion.
- La duracion de eventos fisicos persistentes en hidden data se infiere con
  reglas transparentes; una duracion distinta podria afectar sample-level F1.
