from __future__ import annotations

from pathlib import Path

import yaml

from poc.evaluation.ablation import _contiguous_split, run_full_ablation


def test_contiguous_split_order():
    split = _contiguous_split(20)
    assert split["train"].tolist() == list(range(14))
    assert split["test"][0] > split["val"][-1]


def test_smoke_ablation_writes_16_rows(tmp_path):
    cfg = yaml.safe_load(Path("poc/config.yaml").read_text(encoding="utf-8"))
    cfg["ablation"]["max_existing_synthetic"] = 36
    cfg["paths"]["results"] = str(tmp_path)
    table = run_full_ablation(cfg=cfg, repo_root=Path.cwd(), smoke=True)
    assert len(table) == 16
    assert (tmp_path / "ablation_table.csv").exists()
