"""Kron reduction of Ybus to the generator internal nodes.

Y_red = Y_gg - Y_gl @ inv(Y_ll) @ Y_lg
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def kron_reduce(Ybus: np.ndarray, retain_indices: list[int]) -> np.ndarray:
    """Kron-reduce Ybus keeping only the specified node indices.

    Args:
        Ybus: (N, N) complex admittance matrix
        retain_indices: indices of nodes to retain (generators)

    Returns:
        Y_red: (m, m) reduced admittance matrix where m = len(retain_indices)
    """
    N = Ybus.shape[0]
    retain_idx = sorted(set(retain_indices))
    elim_idx = sorted(set(range(N)) - set(retain_idx))

    Y_gg = Ybus[np.ix_(retain_idx, retain_idx)]
    Y_gl = Ybus[np.ix_(retain_idx, elim_idx)]
    Y_ll = Ybus[np.ix_(elim_idx, elim_idx)]
    Y_lg = Ybus[np.ix_(elim_idx, retain_idx)]

    try:
        Y_ll_inv = np.linalg.inv(Y_ll)
    except np.linalg.LinAlgError:
        log.warning("Y_ll singular during Kron reduction — using pseudo-inverse")
        Y_ll_inv = np.linalg.pinv(Y_ll)

    Y_red = Y_gg - Y_gl @ Y_ll_inv @ Y_lg
    log.info(
        "Kron reduction: %d×%d → %d×%d (retained %d generator nodes)",
        N, N, len(retain_idx), len(retain_idx), len(retain_idx),
    )
    return Y_red
