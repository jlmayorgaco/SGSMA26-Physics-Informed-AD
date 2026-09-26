# Non-anticipative grouped-split benchmark

This directory contains the replacement experiment requested by the paper
review.  It is deliberately separate from the competition submission model:
the benchmark measures past-only, per-timestamp diagnosis and never reuses the
30 s label-replication protocol.

## Primary protocol

`causal_grouped_benchmark.py` generates calibrated IEEE-39 trajectories with
`src.simulation.transient_scenarios`, extracts features from a trailing window
(1--5 s), and keeps every timestamp from a scenario in one split.  Five
independent replicas are generated for every target key; three are assigned to
training, one to validation, and one to test.  Consequently, every eligible
target occurs under independent operating conditions in all three splits.

The target universes are declared before simulation:

* faults: all 39 buses;
* outages: the 34 transmission lines in `branches_physical.csv`; the 12
  transformers are excluded using tap and phase-shift fields;
* generation changes: the ten high-side network buses through which the
  dynamic generators are connected (the simulator perturbs the corresponding
  low-side device terminals and labels the reportable high-side bus);
* load changes: the eighteen buses with a nonzero PQ device (the zero-power
  record at BUS31 is excluded);
* integrity events: the eight observed PMUs.

Events 6 and 8 have two output heads: a primary physical target and an
integrity target.  Event 8 also perturbs a randomly selected load in the
existing simulator.  That additional load is retained in the scenario
metadata but is not treated as a single-label localization target.

Three ExtraTrees baselines receive the same non-anticipative measurement features and a
matched tree budget per task:

1. `flat`: one event classifier and global physical/integrity classifiers;
2. `typed`: detector then abnormal-event classifier, followed by
   event-specific location classifiers;
3. `typed_topology`: the typed event predictor plus a candidate-conditioned
   location ranker.  Its ranker features contain the candidate's electrical
   distances to every PMU and interactions with the contemporaneous PMU
   disturbance scores.  Topology is therefore attached to candidates rather
   than appended as a global feature to a class classifier.

The report includes abnormal precision/recall/F1, false-alarm episodes per
normal minute, detection delay, event macro/weighted/per-class F1, confusion
matrices, physical and integrity Top-1/Top-3 accuracy, Zbus electrical-distance
error, training/prediction runtime, tree/node/depth counts, and serialized
model size.

`summarize_scenario_localization.py` consolidates each saved test scenario to
one modal Top-1 location. Empty output votes as `NONE`; a tie goes to the
candidate that first reached Top-1. `bootstrap_differences.py` reports paired
scenario-level intervals alongside the row-level metrics.

## Commands

Run deterministic checks without ANDES simulation:

```powershell
python paper/experiments/causal_grouped_benchmark.py --self-check
pytest -q paper/experiments/test_causal_grouped_benchmark.py
```

Run an end-to-end engineering smoke test (one target key per event; these
numbers are **not paper evidence**):

```powershell
python paper/experiments/causal_grouped_benchmark.py --smoke --model-seeds 11
```

Run the primary target-complete protocol with three model seeds:

```powershell
python paper/experiments/causal_grouped_benchmark.py \
  --model-seeds 11,29,47 --window-s 3 --force
```

This target-complete run contains 690 ANDES scenarios and is intentionally not
called a quick experiment.  Cached trajectories are reused unless `--force`
is supplied.

The optional `--stress-leave-target-out` run is a separate inductive
localization stress test.  Class-based flat and typed localizers are not
reported there because they cannot emit labels absent from training; only the
candidate-conditioned topology ranker is eligible.  Its results must not be
mixed with the matched primary comparison.

Outputs are written under `paper/evidence/revised/causal_benchmark/`.  A smoke
run writes to a `smoke/` subdirectory and marks every table as non-evidentiary.
