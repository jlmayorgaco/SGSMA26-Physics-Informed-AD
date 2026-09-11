# G1 source evidence and semantics resolution

## Authoritative evidence

- `guidelines.pdf`, pp. 2, 6-7 and 10: the official guide defines PMU1..PMU8 placement as buses 39, 29, 10, 22, 19, 2, 5, 6, and identifies the supplied CSV channels.
- `guidelines.pdf`, p. 11 (section 5.1): the external schema is three-phase voltage/current magnitudes and angles plus frequency, ROCOF, DATA_PRESENT and Event. DATA_PRESENT=0 means all measurement columns are NaN.
- `data/metadata/PMUbus_ Location.txt`: supporting placement metadata; it contains no branch, CT, terminal, polarity, or current orientation assignment.

## Operator evidence in this repository

- `src/pmu_hybrid/physics/network.py:branch_terminal_currents` is the exact pi-model primitive, including charging and complex tap.
- `src/pmu_hybrid/physics/measurement.py` freezes a deterministic current-leaving-PMU convention without reading competition currents.
- `src/simulation/raw_static_case.py` contains a calibrated/inferred mapping helper. It is supporting evidence only, not organizer-authoritative metadata, and is not promoted into the map.

## Resolution

No authoritative SGSMA branch-terminal or CT-polarity semantics were found in the official guide or available metadata. `configs/pmu_map.yaml` therefore remains explicitly `controlled_synthetic` with `AMBIGUOUS` confidence. The positive-sequence internal operator is not a claim that the external CSV is positive-sequence; it is an estimator-internal representation with a separate balanced compatibility export.
