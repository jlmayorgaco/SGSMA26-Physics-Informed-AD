from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.domain.enums.event_family import EventFamily
from src.detectors.domain.models.branch_score import BranchScore
from src.detectors.domain.models.detection_input import DetectionInput
from src.detectors.domain.models.detection_output import DetectionOutput
from src.detectors.domain.models.detector_config import DetectorConfigV2
from src.detectors.domain.models.detector_metrics import DetectorMetricsV2
from src.detectors.domain.models.window_sample import WindowSample


def test_detection_models_instantiate_and_serialize() -> None:
    sample = WindowSample(
        scenario_id="SIMA",
        split="train",
        window_index=0,
        start_idx=0,
        end_idx=3,
        center_timestamp=0.05,
        x=np.ones((4, 2)),
        y_binary=1,
        event_coarse=5,
        event_family=EventFamily.CYBER_HEAVY,
        is_cyber_event=True,
        is_physical_event=False,
        is_concurrent_event=False,
        data_present_ratio=0.8,
    )
    assert sample.x.shape == (4, 2)

    inp = DetectionInput(
        scenario_id="SIMA",
        split="train",
        x_windows=np.ones((3, 4, 2)),
        timestamps=np.array([0.1, 0.2, 0.3]),
        feature_names=["A", "B"],
        metadata=pd.DataFrame({"window_index": [0, 1, 2]}),
    )
    assert inp.to_summary_dict()["n_windows"] == 3

    out = DetectionOutput(
        p_abnormal=np.array([0.2, 0.8]),
        p_cyber=np.array([0.3, 0.9]),
        p_physical=np.array([0.1, 0.7]),
        y_pred_frame=np.array([0, 1]),
        y_pred_stable=np.array([0, 1]),
    )
    assert len(out.p_abnormal) == 2

    branch = BranchScore(probability=np.array([0.2, 0.3]), backend="stub")
    assert branch.backend == "stub"

    cfg = DetectorConfigV2()
    assert "window" in cfg.to_dict()

    metrics = DetectorMetricsV2(
        f1_abnormal=0.8,
        precision_abnormal=0.75,
        recall_abnormal=0.85,
        balanced_accuracy=0.81,
        false_positives_per_minute=1.2,
        detection_delay_s=0.5,
        roc_auc=0.9,
        pr_auc=0.88,
        brier_score=0.1,
        ece=0.05,
        support_abnormal=20,
        support_normal=30,
    )
    assert metrics.to_dict()["support_abnormal"] == 20

