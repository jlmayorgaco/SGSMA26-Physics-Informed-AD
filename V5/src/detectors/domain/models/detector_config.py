from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class WindowingConfig:
    size: int = 64
    stride: int = 16
    positive_ratio_threshold: float = 0.10


@dataclass(slots=True)
class PreprocessingConfigV2:
    fill_method: str = "ffill_bfill"
    include_angle_features: bool = True
    include_positive_sequence: bool = False
    include_estimator_features: bool = True
    scaling_clip_quantile: float = 0.995


@dataclass(slots=True)
class BranchConfig:
    enabled: bool = True
    backend: str = "stub"
    random_seed: int = 12345


@dataclass(slots=True)
class FusionConfigV2:
    strategy: str = "rule_gated_fusion"
    decision_threshold: float = 0.50


@dataclass(slots=True)
class PostprocessingConfigV2:
    start_threshold: float = 0.62
    stop_threshold: float = 0.42
    min_on_frames: int = 2
    min_off_frames: int = 2
    warmup_frames: int = 2
    min_chunk_frames: int = 2
    smoothing_alpha: float = 0.35
    strong_alpha: float = 0.82
    strong_cyber_threshold: float = 0.82
    strong_physical_threshold: float = 0.86
    quiet_abnormal_max: float = 0.58
    quiet_cyber_max: float = 0.46
    quiet_physical_max: float = 0.28
    quiet_frames_to_reset: int = 1
    frame_normal_abnormal_max: float = 0.68
    frame_normal_cyber_max: float = 0.45
    frame_normal_frames_to_reset: int = 1


@dataclass(slots=True)
class DetectorConfigV2:
    window: WindowingConfig = field(default_factory=WindowingConfig)
    preprocessing: PreprocessingConfigV2 = field(default_factory=PreprocessingConfigV2)
    cyber: BranchConfig = field(default_factory=BranchConfig)
    physical: BranchConfig = field(default_factory=BranchConfig)
    fusion: FusionConfigV2 = field(default_factory=FusionConfigV2)
    postprocessing: PostprocessingConfigV2 = field(default_factory=PostprocessingConfigV2)
    artifact_root: Path = Path("output/detector_m10")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifact_root"] = str(self.artifact_root)
        return payload
