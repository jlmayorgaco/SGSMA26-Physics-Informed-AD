.PHONY: setup test train infer report submission clean

PYTHON := python
DATA_RAW := data/raw
SUBMISSION_ZIP := submission_sgsma2026.zip

setup:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

train:
	$(PYTHON) -m src.classifier.train_lgbm

infer:
	$(PYTHON) -m src.pipeline.run_inference --data $(DATA_RAW)

report:
	cd report && latexmk -pdf sgsma2026_report.tex

submission: infer report
	zip -r $(SUBMISSION_ZIP) src/ data/raw/ predictions/ report/sgsma2026_report.pdf README.md requirements.txt

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
