from __future__ import annotations

from enum import Enum


class DetectorMode(str, Enum):
    TRAIN = "train"
    EVAL = "eval"
    INFER = "infer"

