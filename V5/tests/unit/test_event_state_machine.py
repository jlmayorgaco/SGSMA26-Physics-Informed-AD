from __future__ import annotations

import numpy as np

from src.detectors.postprocessing.event_state_machine import EventStateMachine


def test_event_state_machine_hysteresis_transitions() -> None:
    sm = EventStateMachine(start_threshold=0.6, stop_threshold=0.4, min_on_frames=2, min_off_frames=2)
    p = np.array([0.2, 0.65, 0.7, 0.55, 0.3, 0.2, 0.7, 0.75], dtype=float)
    y, states = sm.run(p)
    assert y.shape == (8,)
    assert len(states) == 8
    assert int(y[2]) == 1
    assert int(y[5]) == 1
    assert states[5] in {"ABNORMAL_ACTIVE", "RECOVERY"}


def test_event_state_machine_quiet_state_veto_clears_latched_tail() -> None:
    sm = EventStateMachine(
        start_threshold=0.6,
        stop_threshold=0.45,
        min_on_frames=1,
        min_off_frames=2,
        quiet_abnormal_max=0.6,
        quiet_cyber_max=0.46,
        quiet_physical_max=0.28,
        quiet_frames_to_reset=1,
    )
    p = np.array([0.2, 0.82, 0.78, 0.57, 0.48], dtype=float)
    p_c = np.array([0.1, 0.9, 0.86, 0.40, 0.39], dtype=float)
    p_p = np.array([0.1, 0.2, 0.2, 0.18, 0.16], dtype=float)
    y, _ = sm.run(p, cyber_probabilities=p_c, physical_probabilities=p_p)
    assert y.tolist() == [0, 1, 1, 0, 0]


def test_event_state_machine_frame_normal_veto_clears_non_quiet_tail() -> None:
    sm = EventStateMachine(
        start_threshold=0.6,
        stop_threshold=0.4,
        min_on_frames=1,
        min_off_frames=2,
        frame_normal_abnormal_max=0.68,
        frame_normal_cyber_max=0.45,
        frame_normal_frames_to_reset=1,
    )
    p = np.array([0.1, 0.92, 0.88, 0.65, 0.42], dtype=float)
    p_c = np.array([0.2, 0.89, 0.89, 0.40, 0.40], dtype=float)
    p_p = np.array([0.1, 0.60, 0.55, 0.36, 0.14], dtype=float)
    y_frame = np.array([0, 1, 1, 0, 0], dtype=int)
    y, _ = sm.run(p, cyber_probabilities=p_c, physical_probabilities=p_p, frame_predictions=y_frame)
    assert y.tolist() == [0, 1, 1, 0, 0]
