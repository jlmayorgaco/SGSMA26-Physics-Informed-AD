from __future__ import annotations

from src.simulation.m9.hardening import aggregate_readiness_with_reasons


def test_validator_consistency_aggregation() -> None:
    flags = {"a": True, "b": False, "c": True}
    agg = {"overall": ["a", "b", "c"]}
    out = aggregate_readiness_with_reasons(flags, agg)
    assert out["aggregates"]["overall"] is False
    assert "child:b=false" in out["reasons"]["overall"]

