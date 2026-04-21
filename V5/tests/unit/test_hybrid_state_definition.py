from __future__ import annotations

from src.estimation.dynamic_state_estimation.hybrid_state_definition import build_hybrid_state_layout


def test_hybrid_state_layout_deterministic_indices() -> None:
    layout = build_hybrid_state_layout(["BUS1", "BUS2", "BUS3"], ["BUS3", "BUS2"])
    assert layout.total_dim == 2 * 3 + 2 * 2
    assert layout.bus_to_index["BUS2"] == 1
    assert layout.gen_to_index["BUS3"] == 0

