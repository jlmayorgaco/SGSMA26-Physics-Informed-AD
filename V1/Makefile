.PHONY: setup test train infer validate engineering-review package report submission clean augment augment-smoke benchmark-models simulations

PYTHON        := python
DATA_RAW      := data/raw
DATA_META     := data/metadata
DATA_SYN      := data/synthetic
PREDICTIONS   := predictions
SUBMISSION_ZIP := submission_sgsma2026.zip
SEED          := 42
N_SYN_SCENARIOS := 5000
SYN_FEATURE_CACHE := $(DATA_SYN)/features_v1_5000.npz

# ── Environment setup ─────────────────────────────────────────────────────────
setup:
	pip install -e ".[dev]"
	$(PYTHON) -c "import lightgbm, sklearn, scipy, numpy, pandas; print('All dependencies OK')"

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# ── Augmentation (generate synthetic training data) ───────────────────────────
augment-smoke:
	$(PYTHON) -m src.augmentation.andes_sim \
	    --n-scenarios 16 \
	    --out $(DATA_SYN)_smoke \
	    --manifest $(DATA_SYN)_smoke/manifest.json \
	    --feature-cache $(DATA_SYN)_smoke/features_smoke.npz \
	    --seed $(SEED)

augment:
	$(PYTHON) -m src.augmentation.andes_sim \
	    --n-scenarios $(N_SYN_SCENARIOS) \
	    --out $(DATA_SYN) \
	    --manifest $(DATA_SYN)/manifest.json \
	    --feature-cache $(SYN_FEATURE_CACHE) \
	    --seed $(SEED)

# ── Training (saves model to models/lgbm_model.pkl) ──────────────────────────
train:
	$(PYTHON) -c "\
import logging, pickle; from pathlib import Path; \
logging.basicConfig(level=logging.INFO); \
from src.io.load_csv import load_all; \
from src.eval.splits import make_splits; \
from src.classifier.train_lgbm import train, _build_detector; \
from src.grid.load_case import load_case; \
from src.grid.jacobians import compute_jacobians, bus_sensitivity_columns; \
df = load_all('$(DATA_RAW)'); \
s  = make_splits(df); \
df_train = df.iloc[s['train']].reset_index(drop=True); \
df_val   = df.iloc[s['val']].reset_index(drop=True); \
grid = load_case('$(DATA_META)/IEEE 39 Bus Power System.raw'); \
J    = compute_jacobians(grid); \
J_cols = bus_sensitivity_columns(J, grid); \
Path('models').mkdir(exist_ok=True); \
clf = train(df_train, df_val, J_cols=J_cols, \
            lgbm_params={'n_estimators': 200, 'verbose': -1}, \
            model_path=Path('models/lgbm_model.pkl'), \
            synthetic_dir='$(DATA_SYN)', grid=grid); \
print('Model saved to models/lgbm_model.pkl')"

benchmark-models:
	$(PYTHON) -m src.pipeline.benchmark_models \
	    --data $(DATA_RAW) \
	    --raw "$(DATA_META)/IEEE 39 Bus Power System.raw" \
	    --synthetic $(DATA_SYN) \
	    --feature-cache $(SYN_FEATURE_CACHE) \
	    --out-json report/model_benchmark.json \
	    --out-csv report/model_benchmark.csv \
	    --model-dir models \
	    --seed $(SEED)

# ── Inference (produces prediction CSVs) ──────────────────────────────────────
infer:
	$(PYTHON) -m src.pipeline.run_inference \
	    --data $(DATA_RAW) \
	    --out  $(PREDICTIONS) \
	    --raw  "$(DATA_META)/IEEE 39 Bus Power System.raw" \
	    --model models/best_event_classifier.pkl \
	    --synthetic $(DATA_SYN) \
	    --seed $(SEED) \
	    --log-level INFO

validate:
	$(PYTHON) -m src.pipeline.validate_submission \
	    --pred $(PREDICTIONS) \
	    --data $(DATA_RAW) \
	    --required-labels 1,2,3,4,5,6,7

engineering-review:
	$(PYTHON) -m src.pipeline.generate_engineering_review \
	    --data $(DATA_RAW) \
	    --raw "$(DATA_META)/IEEE 39 Bus Power System.raw" \
	    --synthetic $(DATA_SYN) \
	    --out report/engineering_review \
	    --seed $(SEED)

simulations:
	$(PYTHON) -m src.pipeline.generate_event_simulations \
	    --data $(DATA_RAW) \
	    --raw "$(DATA_META)/IEEE 39 Bus Power System.raw" \
	    --out experiments/event_simulations \
	    --labels 1,2,3,4,5,6,7 \
	    --seed $(SEED)

# ── Report (requires latexmk) ─────────────────────────────────────────────────
report: simulations
	cd report && pdflatex -interaction=nonstopmode sgsma2026_report.tex && pdflatex -interaction=nonstopmode sgsma2026_report.tex

# ── Submission package ────────────────────────────────────────────────────────
package:
	$(PYTHON) -m src.pipeline.build_submission_zip --zip $(SUBMISSION_ZIP)

submission: infer validate engineering-review simulations report package
	@echo "Packaging submission..."
	@echo "Created $(SUBMISSION_ZIP)"

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f report/*.aux report/*.log report/*.fdb_latexmk report/*.fls report/*.synctex.gz
