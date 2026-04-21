from __future__ import annotations

import argparse
from pathlib import Path

from src.simulation.m9.hardening import run_estimator_scoring


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M9.1 estimator-in-the-loop scoring")
    p.add_argument("--output-root", type=Path, default=Path("."))
    p.add_argument("--save-json", action="store_true", default=True)
    p.add_argument("--save-csv", action="store_true", default=True)
    p.add_argument("--save-plots", action="store_true", default=True)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    scenarios_root = a.output_root / "data" / "scenarios"
    scenario_dirs = sorted([p for p in scenarios_root.iterdir() if p.is_dir() and p.name.upper().startswith("SIM")]) if scenarios_root.exists() else []
    run_estimator_scoring(scenario_dirs, a.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

