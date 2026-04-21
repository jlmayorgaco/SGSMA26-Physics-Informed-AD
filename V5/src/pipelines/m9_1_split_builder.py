from __future__ import annotations

import argparse
from pathlib import Path

from src.simulation.m9.hardening import build_splits_no_leakage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M9.1 split builder")
    p.add_argument("--output-root", type=Path, default=Path("."))
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--build-splits", action="store_true", default=True)
    p.add_argument("--save-json", action="store_true", default=True)
    p.add_argument("--save-csv", action="store_true", default=True)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    if not a.build_splits:
        return 0
    build_splits_no_leakage(a.output_root / "data" / "scenarios", a.output_root, seed=a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

