from __future__ import annotations

from pathlib import Path

from src.infrastructure.io.raw_network_loader import load_network_model_from_raw


def test_raw_parser_builds_network_model() -> None:
    model = load_network_model_from_raw(Path("data/metadata/IEEE_39_Bus_Power_System.raw"))
    assert model.ybus.shape[0] == model.ybus.shape[1]
    assert model.ybus.shape[0] == len(model.bus_order)
    assert len(model.bus_order) > 0
    assert model.base_mva > 0

