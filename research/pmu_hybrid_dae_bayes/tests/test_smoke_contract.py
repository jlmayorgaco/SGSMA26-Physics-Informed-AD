from __future__ import annotations

from pathlib import Path

from pmu_hybrid.experiments import smoke


def test_smoke_runner_is_a_module_entrypoint() -> None:
    assert callable(smoke.run)
    assert Path(smoke.__file__).name == "smoke.py"
