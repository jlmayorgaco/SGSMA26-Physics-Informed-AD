from __future__ import annotations

from src.domain.topology import bus_token, canonical_bus_name


def test_canonical_bus_name_variants() -> None:
    assert canonical_bus_name(" bus39 ") == "BUS39"
    assert canonical_bus_name("'BUS30x1'") == "BUS30X1"
    assert canonical_bus_name(2) == "BUS2"
    assert bus_token("BUS39") == "39"

