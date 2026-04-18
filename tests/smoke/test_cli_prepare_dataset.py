from __future__ import annotations

import pytest


@pytest.mark.smoke
def test_cli_prepare_dataset_importable() -> None:
    from src.cli import prepare_dataset

    assert callable(prepare_dataset.main)
