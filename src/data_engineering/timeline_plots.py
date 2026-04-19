"""Timeline plotting for normalized chunks."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src.data_engineering.chunking import CATEGORY_BACKGROUNDS, EVENT_MAP


def plot_event_timeline(
    chunk_meta_list: list[dict],
    max_time_s: float,
    output_path: str,
    bus_name: str | None = None,
) -> None:
    """Plot event timeline in legacy m1 style."""
    fig, ax1 = plt.subplots(figsize=(18, 3), dpi=200)
    ax1.set_xlim(0, max_time_s)
    ax1.set_ylim(-0.5, len(EVENT_MAP) - 0.5)

    ax1.set_yticks(list(EVENT_MAP.keys()))
    ax1.set_yticklabels(list(EVENT_MAP.values()), fontsize=6)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Event label")

    ax2 = ax1.twiny()
    ax2.set_xlim(0, max_time_s / 60.0)
    ax2.set_xlabel("Time (min)")
    ax1.axhline(y=0, color="black", linewidth=0.5)

    legend_patches = [
        mpatches.Patch(color=props["color"], label=cat, alpha=props["alpha"])
        for cat, props in CATEGORY_BACKGROUNDS.items()
    ]

    for chunk in chunk_meta_list:
        start = chunk["start_time_s"]
        duration = chunk["duration_s"]
        category = chunk.get("category")

        if category:
            props = CATEGORY_BACKGROUNDS[category]
            ax1.axvspan(
                start,
                start + duration,
                facecolor=props["color"],
                alpha=props["alpha"],
                edgecolor="black",
                linestyle="--",
                linewidth=0.5,
            )

        events_to_plot = (
            [chunk["per_bus_labels"].get(bus_name, 0)]
            if bus_name
            else [event_id for event_id in chunk["labels_present"] if event_id != 0]
        )

        for event_id in events_to_plot:
            if event_id == 0:
                continue
            color = (
                "#8da0cb"
                if event_id in [5, 7]
                else "#66c2a5"
                if event_id == 3
                else "#fc8d62"
                if event_id in [1, 2]
                else "gray"
            )
            ax1.broken_barh(
                [(start, duration)],
                (event_id - 0.4, 0.8),
                facecolors=color,
                edgecolors="black",
                linewidth=0.5,
            )

    title = f"{bus_name} Normalized - Event timeline" if bus_name else "All PMU buses Normalized - Event timeline"
    plt.title(title, pad=20)
    ax1.legend(handles=legend_patches, loc="upper right", fontsize=6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
