"""Report helpers for M8 dynamic benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_m8_reports(report_dir: Path, payload: dict[str, Any]) -> None:
    """Write canonical JSON and markdown report."""
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "m8_benchmark_results.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    best = payload.get("best_estimator", {}).get("name", "N/A")
    overall = payload.get("overall_conclusion", {})
    md_lines = [
        "# M8 Hybrid Dynamic State Estimation Benchmark",
        "",
        f"- Best estimator: **{best}**",
        f"- M8 working: **{overall.get('is_m8_working')}**",
        f"- Improves over M7: **{overall.get('does_m8_improve_over_m7')}**",
        "",
        "## Ranking (overall score)",
    ]
    for name, score in payload.get("ranking", {}).get("by_overall_score", []):
        md_lines.append(f"- {name}: {score:.6f}")
    md_lines.extend(
        [
            "",
            "## Strengths",
            *[f"- {x}" for x in overall.get("main_strengths", [])],
            "",
            "## Failures",
            *[f"- {x}" for x in overall.get("main_failures", [])],
            "",
            "## Next Actions",
            *[f"- {x}" for x in overall.get("next_actions", [])],
        ]
    )
    (report_dir / "m8_benchmark_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

