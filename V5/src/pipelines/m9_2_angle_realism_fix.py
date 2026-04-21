from __future__ import annotations

import argparse
from pathlib import Path

from src.simulation.m9.final_polish import run_angular_realism_v2, run_freq_rocof_coherence_v2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M9.2 angular realism / coherence recompute")
    p.add_argument("--output-root", type=Path, default=Path("."))
    p.add_argument("--reference-pmu-dir", type=Path, default=Path("data/RAW0001"))
    p.add_argument("--recompute-angular-realism", action="store_true", default=True)
    p.add_argument("--recompute-freq-rocof", action="store_true", default=True)
    p.add_argument("--save-json", action="store_true", default=True)
    p.add_argument("--save-csv", action="store_true", default=True)
    p.add_argument("--save-plots", action="store_true", default=True)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    scenarios_root = a.output_root / "data" / "scenarios"
    scenario_dirs = sorted([p for p in scenarios_root.iterdir() if p.is_dir() and p.name.upper().startswith("SIM")]) if scenarios_root.exists() else []
    if a.recompute_angular_realism:
        run_angular_realism_v2(scenario_dirs, a.reference_pmu_dir, a.output_root)
    if a.recompute_freq_rocof:
        run_freq_rocof_coherence_v2(scenario_dirs, a.reference_pmu_dir, a.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
