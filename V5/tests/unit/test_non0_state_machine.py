from __future__ import annotations

import numpy as np

from src.detectors.postprocessing.non0_state_machine import Non0StateMachine


def test_non0_state_machine_requires_persistence() -> None:
    sm = Non0StateMachine(start_threshold=0.6, stop_threshold=0.4, min_on_frames=2, min_off_frames=2, warmup_frames=0)
    probs = np.asarray([0.2, 0.62, 0.65, 0.58, 0.39, 0.35], dtype=float)
    y, states = sm.run(probs)
    assert len(y) == len(probs)
    assert int(y[1]) == 0
    assert int(y[2]) == 1
    assert states[-1] == "NORMAL"

