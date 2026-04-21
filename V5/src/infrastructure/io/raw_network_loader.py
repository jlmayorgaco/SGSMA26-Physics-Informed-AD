"""Infrastructure loader for RAW network + Ybus model."""

from __future__ import annotations

from pathlib import Path

from src.estimation.state_estimation.models import NetworkModel
from src.metadata.raw_parser import parse_raw_file
from src.metadata.ybus_builder import build_ybus


def load_network_model_from_raw(
    raw_path: str | Path,
    rated_frequency_hz: float = 60.0,
) -> NetworkModel:
    """Load RAW file and build canonical network model."""
    raw = parse_raw_file(Path(raw_path))
    ybus, bus_order, _ = build_ybus(raw)
    bus_kv = {str(b["bus_label_canonical"]): float(b.get("kv_ll", 345.0)) for b in raw.get("buses", [])}
    slack_bus = next((str(b["bus_label_canonical"]) for b in raw.get("buses", []) if int(b.get("bus_type_raw", 1)) == 3), None)
    return NetworkModel(
        ybus=ybus,
        bus_order=bus_order,
        bus_kv_map=bus_kv,
        base_mva=float(raw.get("base_mva", 100.0)),
        rated_frequency_hz=float(rated_frequency_hz),
        slack_bus=slack_bus,
    )

