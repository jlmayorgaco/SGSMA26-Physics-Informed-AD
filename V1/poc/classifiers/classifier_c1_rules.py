"""C1: zero-parameter physics rule classifier."""

from __future__ import annotations

import numpy as np

from poc.classifiers.base import Classifier, hard_label_proba
from poc.features import FEATURE_INDEX, ensure_tabular


class RuleBasedClassifier(Classifier):
    """Decision tree of physical thresholds on innovation-window features.

    Method: hand-crafted thresholds over residual energy, missing-data fraction,
    voltage/current excursions, frequency, ROCOF, and spike ratio.  Reference:
    physics-first residual classifier from the SGSMA Tier-1 plan.  Parameter
    count derivation: thresholds are fixed rules, not learned from labels, so
    ``count_parameters() = 0``.
    """

    def fit(self, features, labels) -> None:
        x = ensure_tabular(features)
        self.scale_ = np.nanmedian(np.abs(x), axis=0) + 1e-9
        self._fitted = True

    def predict(self, features) -> np.ndarray:
        x = ensure_tabular(features)
        idx = FEATURE_INDEX
        out = np.zeros(len(x), dtype=int)
        nan_frac = x[:, idx["nan_fraction"]]
        pmu_missing = x[:, idx["pmu_missing_fraction"]]
        vmag = x[:, idx["v_mag_max_abs"]]
        imag = x[:, idx["i_mag_max_abs"]]
        freq = x[:, idx["freq_max_abs"]]
        rocof = x[:, idx["rocof_max_abs"]]
        spike = x[:, idx["spike_ratio"]]
        vstep = x[:, idx["v_mag_step"]]
        istep = x[:, idx["i_mag_step"]]

        missing = (nan_frac > 0.05) | (pmu_missing > 0.05)
        out[missing] = 5
        out[missing & ((rocof > 5.0) | (np.abs(vstep) > 1_000.0))] = 6
        out[((vmag > 50_000.0) | (imag > 1_000.0) | (rocof > 50.0) | ((vmag > 20_000.0) & (imag > 100.0))) & (out == 0)] = 1
        out[(spike > 40.0) & (~missing) & (out == 0)] = 7
        out[((imag > 500.0) | ((rocof > 5.0) & (np.abs(istep) > 20.0))) & (out == 0)] = 2
        out[((rocof > 0.08) | (freq > 0.08)) & (out == 0)] = 3
        out[((np.abs(vstep) > 120.0) | (vmag > 1_000.0) | (imag > 20.0)) & (out == 0)] = 4
        return out

    def predict_proba(self, features) -> np.ndarray:
        return hard_label_proba(self.predict(features), confidence=0.82)

    def count_parameters(self) -> int:
        return 0
