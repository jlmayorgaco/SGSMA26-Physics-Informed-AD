"""Deterministic seed derivation without global RNG side effects."""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(root_seed: int, namespace: str) -> int:
    """Return a stable unsigned 64-bit child seed for a named simulation unit."""
    payload = f"{int(root_seed)}::{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def rng_for(root_seed: int, namespace: str) -> np.random.Generator:
    """Create an independent reproducible generator for one declared purpose."""
    return np.random.default_rng(derive_seed(root_seed, namespace))
