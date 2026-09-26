from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pandas as pd
import pytest


TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"


def andes_available() -> bool:
    # Default to skip unless explicitly opted in by environment.
    if os.getenv("RUN_ANDES_TESTS", "0") != "1":
        return False
    return importlib.util.find_spec("andes") is not None


@pytest.fixture(scope="session")
def raw_small_dir() -> Path:
    return FIXTURES_DIR / "raw_small"


@pytest.fixture(scope="session")
def snapshot_dir() -> Path:
    return FIXTURES_DIR / "json"


@pytest.fixture
def tiny_event_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Bus10": [0, 0, 1, 1, 0],
            "Bus19": [0, 0, 0, 0, 0],
        },
        index=[0.0, 0.1, 0.2, 0.3, 0.4],
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "andes: mark test as requiring andes")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if andes_available():
        return

    skip_andes = pytest.mark.skip(reason="ANDES not installed in this environment")
    for item in items:
        if "andes" in item.keywords:
            item.add_marker(skip_andes)
