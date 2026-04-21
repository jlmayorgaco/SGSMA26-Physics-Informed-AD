"""Report writer for M6 topology-aware state estimation outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_state_estimation_report(
    output_dir: str | Path,
    summary: dict[str, Any],
    global_metrics: dict[str, Any],
    plot_paths: dict[str, str],
) -> dict[str, str]:
    """Write markdown and JSON summary report artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_json = out / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "summary": summary,
                "global_metrics": global_metrics,
                "plots": plot_paths,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    md = out / "report.md"
    md.write_text(
        "\n".join(
            [
                "# M6 Topology-Aware PMU State Estimation",
                "",
                "## Run Summary",
                f"- timestamps: {summary.get('timestamp_count')}",
                f"- bus_count: {summary.get('bus_count')}",
                f"- pmu_bus_count: {summary.get('pmu_bus_count')}",
                f"- lambda_reg: {summary.get('lambda_reg')}",
                f"- mu_reg: {summary.get('mu_reg')}",
                "",
                "## Global Metrics",
                f"- RMSE |V| (p.u.): {global_metrics.get('rmse_mag_pu_global')}",
                f"- MAE |V| (p.u.): {global_metrics.get('mae_mag_pu_global')}",
                f"- RMSE angle (deg): {global_metrics.get('rmse_ang_deg_global')}",
                f"- MAE angle (deg): {global_metrics.get('mae_ang_deg_global')}",
                "",
                "## Notes",
                "- This estimator is regularized and does not claim full observability from 8 PMUs alone.",
                "- Reconstruction quality will vary across buses depending on topology and PMU placement.",
            ]
        ),
        encoding="utf-8",
    )
    return {"report_md": str(md), "summary_json": str(summary_json)}

