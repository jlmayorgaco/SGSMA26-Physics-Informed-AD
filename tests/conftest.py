"""Shared fixtures for the test suite."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# Repository root
REPO_ROOT = Path(__file__).parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_META = REPO_ROOT / "data" / "metadata"
RAW_PATH = DATA_META / "IEEE 39 Bus Power System.raw"


@pytest.fixture(scope="session")
def data_raw() -> Path:
    return DATA_RAW


@pytest.fixture(scope="session")
def data_meta() -> Path:
    return DATA_META


@pytest.fixture(scope="session")
def raw_path() -> Path:
    return RAW_PATH


@pytest.fixture(scope="session")
def merged_df():
    """Load the merged DataFrame once per test session."""
    from src.io.load_csv import load_all
    return load_all(DATA_RAW)


@pytest.fixture(scope="session")
def grid_case():
    """Load and solve the grid case once per test session."""
    from src.grid.load_case import load_case
    return load_case(RAW_PATH)
