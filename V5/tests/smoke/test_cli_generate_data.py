from __future__ import annotations

import pytest


@pytest.mark.smoke
def test_cli_generate_data_importable() -> None:
    from src.cli import generate_data

    assert callable(generate_data.main)
