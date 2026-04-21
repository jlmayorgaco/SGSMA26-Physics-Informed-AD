from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.configs import FusionConfig, PostprocessingConfig
from src.detectors.fusion.strategy import WeightedFusionStrategy
from src.detectors.postprocessing.chunker import chunk_events
from src.detectors.postprocessing.hysteresis import apply_hysteresis
from src.detectors.postprocessing.state_machine import enforce_event_state_machine


def test_fusion_and_postprocessing_pipeline() -> None:
    fusion = WeightedFusionStrategy(FusionConfig(cyber_weight=0.7, physical_weight=0.3))
    cyber = np.array([0.1, 0.2, 0.8, 0.9, 0.2, 0.1], dtype=float)
    physical = np.array([0.2, 0.2, 0.6, 0.7, 0.2, 0.2], dtype=float)
    meta = pd.DataFrame({"estimator_difficulty_score": [0.5] * len(cyber)})
    scores = fusion.fuse(cyber, physical, meta)
    cfg = PostprocessingConfig(start_threshold=0.6, stop_threshold=0.4, min_on_frames=2, min_off_frames=2, min_chunk_frames=2)
    h = apply_hysteresis(scores, cfg)
    s = enforce_event_state_machine(h, cfg)
    chunks = chunk_events(s, np.arange(len(scores), dtype=float), scores, cfg)
    assert len(chunks) == 1
    assert chunks[0].frame_count >= 2

