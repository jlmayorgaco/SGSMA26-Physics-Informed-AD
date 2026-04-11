"""K-frame debouncing logic for binary event signals.

State machine:
  normal  → k_on  consecutive frames above threshold → alarm
  alarm   → k_off consecutive frames below threshold → normal

The alarm turns ON at frame k_on (zero-indexed: frame 0 is the first high frame,
alarm becomes True at frame k_on-1). During the k_on accumulation frames, the alarm
is still False — detection delay is exactly k_on frames.
"""
from __future__ import annotations

import numpy as np


def debounce(
    signal: "np.ndarray[bool]",
    k_on: int,
    k_off: int | None = None,
) -> "np.ndarray[bool]":
    """Apply k-frame rising/falling debounce to a binary signal.

    Args:
        signal: (T,) bool or 0/1 array — raw per-frame anomaly indicator.
        k_on:   number of consecutive high frames required to enter alarm state.
        k_off:  number of consecutive low frames required to exit alarm state.
                Defaults to k_on.

    Returns:
        alarm: (T,) bool array — alarm state (True = anomalous) per frame.
    """
    if k_off is None:
        k_off = k_on

    T = len(signal)
    alarm = np.zeros(T, dtype=bool)

    state = 0     # 0 = normal, 1 = alarm
    run_hi = 0    # consecutive high frames since last low (state=0)
    run_lo = 0    # consecutive low frames since last high (state=1)

    for t in range(T):
        s = bool(signal[t])
        if state == 0:
            if s:
                run_hi += 1
                if run_hi >= k_on:
                    state = 1
                    run_lo = 0
            else:
                run_hi = 0
        else:   # state == 1
            if not s:
                run_lo += 1
                if run_lo >= k_off:
                    state = 0
                    run_hi = 0
            else:
                run_lo = 0
        alarm[t] = (state == 1)

    return alarm


def alarm_onsets(alarm: "np.ndarray[bool]") -> "np.ndarray[int]":
    """Return frame indices where alarm transitions from False to True."""
    diff = np.diff(alarm.astype(np.int8), prepend=0)
    return np.where(diff == 1)[0]


def alarm_offsets(alarm: "np.ndarray[bool]") -> "np.ndarray[int]":
    """Return frame indices where alarm transitions from True to False."""
    diff = np.diff(alarm.astype(np.int8), prepend=0)
    return np.where(diff == -1)[0]
