from __future__ import annotations

from pathlib import Path
import json


def write_phase2_report(report_payload: dict, report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "phase_2_branches_and_fusion.json"
    md_path = report_dir / "phase_2_branches_and_fusion.md"
    json_path.write_text(json.dumps(report_payload, indent=2, default=str), encoding="utf-8")
    lines = [
        "# Phase 2 Branches And Fusion",
        f"- phase_2_complete: `{report_payload.get('overall_verdict', {}).get('phase_2_complete')}`",
        f"- ready_for_phase_3: `{report_payload.get('overall_verdict', {}).get('ready_for_phase_3')}`",
        "",
        "## Initial Metrics",
    ]
    for key, value in report_payload.get("initial_metrics", {}).items():
        lines.append(f"- {key}: `{value}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path

