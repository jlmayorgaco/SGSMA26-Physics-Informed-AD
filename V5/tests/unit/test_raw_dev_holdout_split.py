from __future__ import annotations

import numpy as np

from src.detectors.pipelines.non0_detector_v2 import _build_raw_dev_holdout_mask


def test_raw_dev_holdout_split_has_guard_band_separation() -> None:
    y = np.asarray([0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=int)
    dev_mask, holdout_mask = _build_raw_dev_holdout_mask(y, guard_band=1, holdout_mod=2)
    assert len(dev_mask) == len(y)
    assert len(holdout_mask) == len(y)
    assert not bool(np.any(dev_mask & holdout_mask))
    # second abnormal interval (idx 6-7) plus guard belongs to holdout
    assert bool(holdout_mask[6])
    assert bool(holdout_mask[5])

