from __future__ import annotations

import numpy as np

from src.estimation.state_estimation.priors import build_loadflow_prior


def test_loadflow_prior_shape_and_values() -> None:
    meta = {
        "buses": [
            {"bus_label_canonical": "BUS1", "v_pu": 1.0, "theta_deg": 0.0},
            {"bus_label_canonical": "BUS2", "v_pu": 0.98, "theta_deg": -5.0},
        ]
    }
    prior = build_loadflow_prior(["BUS1", "BUS2"], meta)
    assert prior.shape == (2,)
    assert np.isclose(np.abs(prior[0]), 1.0)
    assert np.isclose(np.abs(prior[1]), 0.98)

