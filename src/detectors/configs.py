from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class WindowConfig:
    size: int = 64
    stride: int = 16
    positive_ratio_threshold: float = 0.10


@dataclass(slots=True)
class PreprocessingConfig:
    fill_method: str = "ffill_bfill"
    clip_quantile: float = 0.995
    include_estimator_assist: bool = True


@dataclass(slots=True)
class CyberRulesConfig:
    missing_ratio_weight: float = 0.45
    stuck_weight: float = 0.20
    spike_weight: float = 0.20
    timestamp_weight: float = 0.15
    spike_z_threshold: float = 4.5
    stuck_std_threshold: float = 1e-6
    timestamp_jitter_threshold: float = 0.020


@dataclass(slots=True)
class CyberModelConfig:
    prefer_lightgbm: bool = True
    learning_rate: float = 0.05
    l2: float = 1e-4
    epochs: int = 240
    random_seed: int = 12345


@dataclass(slots=True)
class PhysicalModelConfig:
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    learning_rate: float = 0.05
    l2: float = 1e-4
    epochs: int = 260
    use_baseline_fallback: bool = True
    random_seed: int = 12345


@dataclass(slots=True)
class FusionConfig:
    cyber_weight: float = 0.50
    physical_weight: float = 0.50
    estimator_assist_weight: float = 0.10
    decision_threshold: float = 0.50


@dataclass(slots=True)
class PostprocessingConfig:
    start_threshold: float = 0.62
    stop_threshold: float = 0.42
    min_on_frames: int = 2
    min_off_frames: int = 2
    min_chunk_frames: int = 2


@dataclass(slots=True)
class DetectorConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    cyber_rules: CyberRulesConfig = field(default_factory=CyberRulesConfig)
    cyber_model: CyberModelConfig = field(default_factory=CyberModelConfig)
    physical_model: PhysicalModelConfig = field(default_factory=PhysicalModelConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    pmu_pattern: str = "Bus*_Competition_Data_*.csv"
    model_path: Path = Path("output/detector_m10/detector_model.pkl")
    report_dir: Path = Path("output/detector_m10")

