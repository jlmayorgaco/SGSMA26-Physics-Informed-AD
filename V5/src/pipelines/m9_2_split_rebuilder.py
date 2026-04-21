from __future__ import annotations

import argparse
from pathlib import Path

from src.simulation.m9.final_polish import build_splits_v2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M9.2 split rebuilder")
    p.add_argument("--output-root", type=Path, default=Path("."))
    p.add_argument("--split-strategy", type=str, default="leakage_safe_balanced")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--rebuild-splits", action="store_true", default=True)
    p.add_argument("--save-json", action="store_true", default=True)
    p.add_argument("--save-csv", action="store_true", default=True)
    p.add_argument("--save-plots", action="store_true", default=True)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    if not a.rebuild_splits:
        return 0
    build_splits_v2(a.output_root / "data" / "scenarios", a.output_root, split_strategy=a.split_strategy, seed=a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
