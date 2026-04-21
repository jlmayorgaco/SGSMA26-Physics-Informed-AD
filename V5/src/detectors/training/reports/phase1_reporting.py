from __future__ import annotations

from pathlib import Path
import json


def write_phase1_foundations_report(report_payload: dict, report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "phase_1_foundations.json"
    md_path = report_dir / "phase_1_foundations.md"
    json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    summary = [
        "# Phase 1 Foundations Report",
        "",
        f"- phase_1_complete: `{report_payload.get('overall_verdict', {}).get('phase_1_complete')}`",
        f"- ready_for_phase_2: `{report_payload.get('overall_verdict', {}).get('ready_for_phase_2')}`",
        "",
        "## Known Gaps For Phase 2",
    ]
    for gap in report_payload.get("known_gaps_for_phase_2", []):
        summary.append(f"- {gap}")
    md_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return json_path, md_path

