from __future__ import annotations

from pathlib import Path

import pytest

from src.simulation.type1_config import Type1Config, validate_fault_bus


def test_fault_bus_validation_accepts_range() -> None:
    assert validate_fault_bus("1") == "1"
    assert validate_fault_bus("39") == "39"


def test_fault_bus_validation_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        validate_fault_bus("0")


def test_output_dir_naming_helpers() -> None:
    cfg = Type1Config(fault_bus="22", base_output_dir="output")
    run_name = cfg.run_folder_name()
    assert run_name == "SIM0001_BUS22_Event1"
    root = Path(cfg.base_output_dir) / run_name
    assert root.name.endswith("Event1")
