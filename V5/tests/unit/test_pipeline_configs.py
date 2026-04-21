from __future__ import annotations

from src.detectors.domain.models.detector_config import DetectorConfigV2


def test_pipeline_configs_load_cleanly() -> None:
    cfg = DetectorConfigV2()
    payload = cfg.to_dict()
    assert "window" in payload
    assert "fusion" in payload
    assert payload["window"]["size"] > 0

