from __future__ import annotations

from typing import Protocol

from src.detectors.domain.interfaces.branch_detector import BranchDetector


class PhysicalDetector(BranchDetector, Protocol):
    """Phase-1 physical branch contract stub."""

