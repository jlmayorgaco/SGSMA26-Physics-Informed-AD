"""Phase 2 entrypoint: gate every later phase on AC static parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pmu_hybrid.cases.parity import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps({"status": result["status"], "manifest": "output/manifests/static_parity.json"}))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
