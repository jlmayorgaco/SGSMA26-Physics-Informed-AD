"""Post-contest comparative POC for SGSMA 2026.

The production competition code lives under ``src/``.  This package is an
isolated research bench for journal-paper ablations and must remain swappable:
estimators emit a common innovation object, classifiers consume common feature
views, and shared detector/localizer logic stays estimator-agnostic.
"""

