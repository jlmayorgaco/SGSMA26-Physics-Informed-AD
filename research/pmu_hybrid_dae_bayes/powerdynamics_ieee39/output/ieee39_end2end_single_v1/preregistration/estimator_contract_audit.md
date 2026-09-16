# IEEE39-END2END-SINGLE-V1 estimator contract audit

This contract was frozen before the first new TDS solve at START_HEAD
`c9d4a1404950e06acc55d2da39c532eb0f89d348`. The expected scientific
ancestor `218fdfda69ea94c15664fa1f2e79096997432ee2` is present.

## Canonical paths

- Production model: PowerDynamics 5.0.0 IEEE39 example compiled as a
  NetworkDynamics 1.3.0 descriptor network.
- Production callback: `PresetTimeComponentCallback` mutating the actual
  `ZIPLoad₊Pset` and `ZIPLoad₊Qset` parameters without reinitializing stored
  states.
- PMU extraction: `scripts/e06h_corrected_m6_static.py::load_branch_rows` and
  `measurement`.
- Candidate sources, priors and cardinality contract:
  `scripts/load_multi_pilot_v1.py`.
- Operational 137-support evidence:
  `scripts/exact_weak_regime_resolution_v2.py::make_prefix_model` and
  `gh31_prefix`, preserving the local adaptive GH31 convention.
- OP-conditioned corrected PMU dictionary:
  `first_flow_hessian_closure_v1/results/corrected_dictionary_op_m085.npz`.
- Production first-flow derivatives and homogeneous corrections:
  `scripts/first_flow_hessian_closure_v1.py` and its frozen state-map bank.
- Native descriptor ordering:
  `second_order_op_robustness_v1/analytic_op_m085_state/metadata/state_order.csv`.

All required quantities resolve without ambiguity. The PMU ordering is the
repository ordering `(2,5,6,10,19,22,29,39)`. The first 16 channels are
interleaved voltage Re/Im by bus; the final 16 are interleaved terminal-current
Re/Im for frozen branch rows 11, 37, 18, 35, 31, 1, 8 and 5.

## Fixed pilot choices

- Operating point: `m=0.85`, implemented by multiplying the Bus-3 ZIP P/Q by
  `1 + 0.03*m`.
- Event: hidden candidate Bus 7, callback at 2.0 s.
- Severity: `a=+0.0033` (0.33%), the pre-existing singleton `MODERATE` value.
- Noise seeds: frozen in `random_seeds.json`.
- No threshold, feature, prior, dictionary, covariance or model component may
  be changed after results are observed.

The reconstruction is an event-conditioned local-manifold prediction around a
known pre-event operating point. It is not a general unknown-initial-state DAE
state estimator.
