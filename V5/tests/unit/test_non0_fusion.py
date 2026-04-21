from __future__ import annotations

import numpy as np

from src.detectors.fusion.non0_fusion import Non0Fusion


def test_non0_fusion_allows_event5_override() -> None:
    fusion = Non0Fusion(decision_threshold=0.6, event5_override=0.75)
    out = fusion.fuse(
        p_event5=np.asarray([0.8, 0.1], dtype=float),
        p_event7=np.asarray([0.2, 0.2], dtype=float),
        p_physical=np.asarray([0.2, 0.2], dtype=float),
    )
    assert int(out.y_frame[0]) == 1
    assert int(out.y_frame[1]) == 0
    assert out.attribution[0] == "event5"

