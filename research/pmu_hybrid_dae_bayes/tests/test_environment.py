from __future__ import annotations

from pmu_hybrid.experiments.e00_environment import package_versions
from pmu_hybrid.utils.seeds import derive_seed, rng_for


def test_versions_always_include_required_audit_keys() -> None:
    versions = package_versions()
    assert versions["python"]
    assert "andes" in versions
    assert "pandapower" in versions


def test_named_seed_is_deterministic_and_partitioned() -> None:
    assert derive_seed(7, "scenario/a") == derive_seed(7, "scenario/a")
    assert derive_seed(7, "scenario/a") != derive_seed(7, "scenario/b")
    assert rng_for(7, "scenario/a").normal() == rng_for(7, "scenario/a").normal()
