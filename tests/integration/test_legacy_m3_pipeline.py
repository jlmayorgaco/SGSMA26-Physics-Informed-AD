from __future__ import annotations

import pytest

from src.infrastructure.legacy.m3_adapter import load_signal_specs, run_raw_event0_calibration


@pytest.mark.integration
@pytest.mark.andes
def test_legacy_m3_pipeline_callable() -> None:
    # This is intentionally light in phase-1; full runtime is expensive.
    # We only verify the adapter function is importable/callable under ANDES environments.
    assert callable(run_raw_event0_calibration)
    assert isinstance(load_signal_specs(), list)
