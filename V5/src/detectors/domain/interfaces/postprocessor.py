from __future__ import annotations

from typing import Protocol

import numpy as np


class Postprocessor(Protocol):
    """Stabilize frame-level decisions and produce event chunks."""

    def smooth(self, probabilities: np.ndarray) -> np.ndarray: ...

    def state_machine(self, binary: np.ndarray) -> np.ndarray: ...

    def chunk(self, binary: np.ndarray, timestamps: np.ndarray, probabilities: np.ndarray) -> list[dict[str, object]]: ...

