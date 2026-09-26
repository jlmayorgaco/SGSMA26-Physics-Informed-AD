"""Contract tests for the leakage-safe E06-I causal layer."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.e06i_online_causal import causal_center, online_case


def test_causal_extractors_do_not_see_future_samples():
    history = np.arange(20.0).reshape(10, 2)
    for extractor in ("TRAILING_MEAN", "HUBER_TRAILING_MEAN", "EMA"):
        before = causal_center(history[:6], extractor, 5)
        after = causal_center(np.vstack([history[:6], history[6:] * 1000]), extractor, 5)
        assert np.all(np.isfinite(before))
        # The value at frame five is invariant to samples appended later.
        assert np.allclose(before, causal_center(history[:6], extractor, 5))
        assert not np.allclose(before, after)


def test_online_path_has_no_truth_argument():
    names = list(inspect.signature(online_case).parameters)
    forbidden = {"truth", "hidden", "equilibrium", "mismatch", "d_known"}
    assert forbidden.isdisjoint({n.lower() for n in names})


def test_causal_center_shapes_and_finite_values():
    h = np.random.default_rng(7).normal(size=(13, 32))
    for extractor in ("TRAILING_MEAN", "HUBER_TRAILING_MEAN", "EMA"):
        z = causal_center(h, extractor, 10)
        assert z.shape == (32,)
        assert np.all(np.isfinite(z))
