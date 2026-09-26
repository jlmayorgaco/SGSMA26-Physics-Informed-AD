# Tutorial — Run the model on a new RAW folder (macOS)

How to take a **fresh `git clone`** on a Mac and corroborate the pipeline against a
new dataset such as `RAW0002`. Commands are for **zsh/bash** (the default macOS shell).

> ⚠️ **The most important step is #2.** `models/` is **gitignored**, so a fresh clone
> has **no model bundle** and `main.py` silently falls back to the degraded physics
> route. You must restore `models/` from the tracked submission zip first.

---

## 0. Requirements

- **Python 3.10+** (`python3 --version`)
- The clone already contains the tracked bundle `sgsma_2026_final_submission.zip`
  and the input data under `data/`.

```bash
cd /path/to/SGSMA26-Physics-Informed-AD
python3 --version          # must be 3.10 or newer
```

---

## 1. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Installs: `numpy`, `pandas`, `scipy`, `scikit-learn==1.7.2`, `joblib`, `matplotlib`,
`openpyxl`.

> ⚠️ **scikit-learn must be exactly `1.7.2`** — the model bundle was pickled with that
> version. `requirements.txt` pins it. If you ever see
> `'SimpleImputer' object has no attribute '_fit_dtype'` (or a similar
> `InconsistentVersionWarning`), you are on the wrong version — fix it with:
> ```bash
> pip install "scikit-learn==1.7.2"
> ```

---

## 2. Restore the model bundle (REQUIRED on a fresh clone)

A fresh clone has **no `models/`** folder. Extract it from the tracked zip:

```bash
unzip -o sgsma_2026_final_submission.zip -d _bundle
cp -R _bundle/models .
cp -R _bundle/models_bus_agnostic .
```

Confirm the file that switches on the validated ML route exists:

```bash
test -f models/final_model_config.json && echo "OK: ML bundle present" || echo "MISSING"
```

You must see `OK: ML bundle present`. (Without it the run still works but uses the
weaker physics fallback.)

---

## 3. Drop in the new RAW0002 data

Put the 8 PMU CSVs in `data/RAW0002/`. The model is **bus-agnostic**: it discovers the
buses from the **file names** (`Bus<N>_*.csv`, e.g. `Bus2_*.csv`, `Bus29_*.csv`), so it
works even if `RAW0002` places PMUs on different buses than `RAW0001`.

```bash
ls data/RAW0002          # should list Bus2_*.csv, Bus5_*.csv, ...
```

---

## 4. Run inference (single entrypoint)

```bash
python main.py --input-dir data/RAW0002 --output data/RAW0002/predictions.csv
```

CPU-only; expect roughly **~14 minutes** for a RAW0001-sized folder. When it finishes,
`main.py` prints the diagnostics JSON to the terminal.

---

## 5. Corroborate the result

Two files are written into `data/RAW0002/`:

### a) `predictions.csv` — the deliverable
Exact columns (graders parse these):

```
TIMESTAMP, Bus, Predicted_Event, Predicted_Location
```

```bash
head data/RAW0002/predictions.csv
```

### b) `prediction_diagnostics.json` — confirm the right model ran
Check the route and model name:

```bash
python3 -c "import json; d=json.load(open('data/RAW0002/prediction_diagnostics.json')); print('route =', d.get('route')); print('model =', d.get('model'))"
```

Expected:

```
route = ml_windowed
model = sgms_extra_trees_windowed_v2
```

If instead you see `route = physics_fallback_missing_ml_bundle`, the `models/` bundle
was not restored — go back to **step 2**.

The diagnostics file also includes per-window predictions, timing, and the **Top-3
location candidates** (only Top-1 goes into the official CSV).

---

## 6. (Optional) Quick smoke test without waiting 14 min

Point the same command at a small SIM folder to confirm everything runs end to end:

```bash
python main.py --input-dir <small_SIM_folder> --output /tmp/pred.csv
```

## 6b. (Optional) Measure accuracy
If `RAW0002` ships ground-truth `Event` labels, compare them to `Predicted_Event`.
Note the labels are **not consistent across the 8 CSVs** — derive a global per-row
label as the **max non-zero** across the per-bus `Event` columns before scoring.

---

## 7. Apple Silicon (M1/M2/M3) and troubleshooting

- **Python:** macOS may ship an old `python3`. Use 3.10+ (Homebrew: `brew install python@3.12`).
  Check with `python3 --version`.
- **arm64 wheels:** `numpy`, `scipy`, `pandas`, `scikit-learn` all ship native Apple-Silicon
  wheels — `pip install -r requirements.txt` just works. If pip tries to build from source,
  run `pip install --upgrade pip` first.
- **scikit-learn version on `joblib.load`:** the bundle needs **exactly `scikit-learn==1.7.2`**.
  A different version throws `'SimpleImputer' object has no attribute '_fit_dtype'` (or an
  `InconsistentVersionWarning`). Fix: `pip install "scikit-learn==1.7.2"`.
- **Run from the repo root** (so `src/`, `models/`, `data/` resolve).
- **No internet at the venue?** Pre-install everything beforehand; the venv must be self-contained.

## 8. Pre-flight checklist (do this BEFORE the on-site)

Run the full pipeline once on the known-good `RAW0001` and confirm it matches the contract:

```bash
source .venv/bin/activate
python main.py --input-dir data/RAW0001 --output data/RAW0001/predictions.csv
python3 -c "import json; d=json.load(open('data/RAW0001/prediction_diagnostics.json')); assert d.get('route')=='ml_windowed', d.get('route'); print('route OK:', d['route'], d['model'])"
```

You should see `route OK: ml_windowed sgms_extra_trees_windowed_v2`. If so, the Mac is ready:
on the day, you only swap `--input-dir` to the new RAW folder. Keep the `.venv` and the restored
`models/` in place so nothing needs reinstalling under time pressure (the on-site round is 90 min).

---

## Event label convention (0–8)

```
0 = normal              5 = missing data
1 = fault               6 = missing data + physical event
2 = line outage         7 = bad data
3 = generation change   8 = unknown / mixed proxy
4 = load change/drop
```

## Reference

- `MODEL.md` — authoritative runtime contract and reported metrics.
- `CLAUDE.md` — repo conventions and data-handling gotchas.
