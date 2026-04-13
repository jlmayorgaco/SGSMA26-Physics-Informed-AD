"""Professional IEEE 39-bus one-line diagram plotter.

This module is intentionally dedicated to topology diagrams. The time-series
plotters live in ``synth_review.py``; keeping the single-line diagram separate
makes it easier to polish the electrical drawing without touching CSV or PMU
signal plotting logic.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


GEN_BUSES = {30: 1, 31: 2, 32: 3, 33: 4, 34: 5, 35: 6, 36: 7, 37: 8, 38: 9, 39: 10}
LOAD_BUSES = {3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29}
XFO_BRANCHES = {
    tuple(sorted(edge))
    for edge in [
        (30, 2),
        (31, 6),
        (32, 10),
        (33, 19),
        (34, 20),
        (35, 22),
        (36, 23),
        (37, 25),
        (38, 29),
        (39, 1),
        (12, 11),
        (12, 13),
        (6, 11),
        (19, 20),
    ]
}

# Fixed coordinates approximating the classic IEEE 39 New England one-line
# diagram. These are deliberately hand-placed; spectral layouts look unstable
# across scenarios and are harder for engineers to compare visually.
IEEE39_POSITIONS = {
    1: (4.5, 2.95),
    2: (3.15, 3.35),
    3: (4.65, 4.85),
    4: (7.25, 4.15),
    5: (8.55, 4.15),
    6: (10.15, 4.95),
    7: (10.15, 3.05),
    8: (11.35, 3.05),
    9: (12.55, 3.05),
    10: (12.25, 6.75),
    11: (10.55, 5.75),
    12: (9.65, 6.55),
    13: (8.65, 5.75),
    14: (7.55, 5.00),
    15: (6.35, 5.80),
    16: (5.65, 7.00),
    17: (4.05, 6.55),
    18: (4.05, 5.55),
    19: (8.35, 7.90),
    20: (8.95, 9.05),
    21: (5.65, 8.15),
    22: (5.65, 9.25),
    23: (7.60, 9.55),
    24: (4.25, 8.25),
    25: (1.50, 4.85),
    26: (1.50, 7.30),
    27: (3.00, 7.15),
    28: (1.50, 8.45),
    29: (1.50, 9.55),
    30: (3.15, 1.65),
    31: (11.45, 4.95),
    32: (13.75, 6.75),
    33: (10.10, 7.90),
    34: (10.10, 9.05),
    35: (4.95, 10.20),
    36: (9.05, 9.55),
    37: (0.45, 3.80),
    38: (0.85, 10.70),
    39: (6.35, 2.05),
}

LOAD_SIDE = {bus: "down" for bus in LOAD_BUSES}
GEN_SIDE = {bus: "up" for bus in GEN_BUSES}

BUS_LABEL_OFFSETS = {
    bus: (0.55, 0.22 if bus not in {11, 12, 13, 19, 20, 22, 23, 24} else 0.30)
    for bus in IEEE39_POSITIONS
}
BUS_LABEL_OFFSETS.update(
    {
        6: (0.55, -0.28),
        7: (0.55, -0.28),
        8: (0.55, -0.28),
        9: (0.55, -0.28),
        30: (0.55, -0.28),
        37: (0.55, -0.28),
        38: (0.55, -0.28),
        39: (0.55, -0.28),
    }
)

EVENT_STYLES = {
    1: {"name": "Fault", "color": "#7c3aed"},
    2: {"name": "Line outage", "color": "#f97316"},
    3: {"name": "Generation change", "color": "#0ea5e9"},
    4: {"name": "Load change", "color": "#16a34a"},
    5: {"name": "PMU dropout", "color": "#64748b"},
    6: {"name": "Cyber + physical", "color": "#db2777"},
    7: {"name": "Bad data", "color": "#eab308"},
    8: {"name": "Ambiguous overlap", "color": "#0f766e"},
}

EVENT_LABEL_OFFSETS = {
    2: (-0.95, 0.72),
    7: (1.08, 0.66),
    23: (0.0, 0.82),
    24: (0.0, -0.82),
    29: (-1.00, 0.55),
    39: (0.0, -0.85),
}

NORMAL_LINE = "#4b5563"
NORMAL_BUS = "#111827"
PMU_TAG = "#9f1239"
EVENT_PURPLE = "#7c3aed"
BUS_BAR_HALF_WIDTH = 0.34

# Absolute dogleg routes for the crowded portions of the New England diagram.
# Any branch not listed falls back to a small "rise, run, drop" orthogonal path.
BRANCH_ROUTES = {
    (1, 2): [(3.15, 2.95)],
    (1, 39): [(4.50, 2.05)],
    (2, 3): [(3.15, 4.85)],
    (2, 25): [(2.55, 3.35), (2.55, 4.85)],
    (2, 30): [],
    (3, 4): [(5.70, 4.85), (5.70, 4.15)],
    (3, 18): [(4.65, 5.55)],
    (4, 5): [],
    (4, 14): [(7.25, 5.00)],
    (5, 6): [(9.35, 4.15), (9.35, 4.95)],
    (5, 8): [(10.55, 4.15), (10.55, 3.05)],
    (6, 7): [],
    (6, 11): [(10.15, 5.75)],
    (6, 31): [],
    (7, 8): [],
    (8, 9): [],
    (9, 39): [(12.55, 2.05)],
    (10, 11): [(11.70, 6.75), (11.70, 5.75)],
    (10, 13): [(11.15, 6.75), (11.15, 5.75)],
    (10, 32): [],
    (11, 12): [(10.55, 6.55)],
    (12, 13): [(8.65, 6.55)],
    (13, 14): [],
    (14, 15): [(6.95, 5.00), (6.95, 5.80)],
    (15, 16): [(6.35, 7.00)],
    (16, 17): [(4.05, 7.00)],
    (16, 19): [(6.65, 7.00), (6.65, 7.90)],
    (16, 21): [],
    (16, 24): [(5.65, 8.25)],
    (17, 18): [],
    (17, 27): [(4.05, 7.15)],
    (19, 20): [],
    (19, 33): [],
    (20, 34): [],
    (21, 22): [],
    (22, 23): [(6.55, 9.25), (6.55, 9.55)],
    (22, 35): [(5.65, 10.20)],
    (23, 24): [(7.60, 8.25)],
    (23, 36): [],
    (25, 26): [],
    (25, 37): [(1.50, 3.80)],
    (26, 27): [],
    (26, 28): [],
    (26, 29): [(0.95, 7.30), (0.95, 9.55)],
    (27, 17): [(3.00, 6.55)],
    (28, 29): [],
    (29, 38): [(0.85, 9.55)],
}


def _offset(side: str, distance: float = 0.55) -> tuple[float, float]:
    return {
        "up": (0.0, distance),
        "down": (0.0, -distance),
        "left": (-distance, 0.0),
        "right": (distance, 0.0),
    }[side]


def _event_style(label: int) -> dict[str, str]:
    return EVENT_STYLES.get(label, {"name": "Event", "color": EVENT_PURPLE})


def _event_location_text(event: dict[str, Any]) -> str:
    if event.get("line"):
        return f"line {event['line'][0]}-{event['line'][1]}"
    if event.get("pmu_bus"):
        return f"bus {event['pmu_bus']}"
    nodes = event.get("nodes") or []
    if nodes:
        return "bus " + ",".join(str(node) for node in nodes)
    return "system"


def _event_buses(event: dict[str, Any]) -> set[int]:
    buses = {int(node) for node in event.get("nodes", [])}
    if event.get("line"):
        buses.update(int(node) for node in event["line"])
    if event.get("pmu_bus"):
        buses.add(int(event["pmu_bus"]))
    return buses


def _branch_is_event(edge: tuple[int, int], events: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    edge_set = set(edge)
    for event in events:
        line = event.get("line")
        if line and set(int(node) for node in line) == edge_set:
            return event
    return None


def _route_points(left: int, right: int) -> list[tuple[float, float]]:
    """Return a stable one-line route with mostly 90-degree bends."""

    key = tuple(sorted((left, right)))
    if key in BRANCH_ROUTES:
        p1 = IEEE39_POSITIONS[key[0]]
        p2 = IEEE39_POSITIONS[key[1]]
        bends = BRANCH_ROUTES[key]
        if bends or abs(p1[0] - p2[0]) < 0.03 or abs(p1[1] - p2[1]) < 0.03:
            points = [p1, *bends, p2]
            if left != key[0]:
                points = list(reversed(points))
            return _dedupe_points(points)

    p1 = IEEE39_POSITIONS[left]
    p2 = IEEE39_POSITIONS[right]
    if abs(p1[0] - p2[0]) < 0.03 or abs(p1[1] - p2[1]) < 0.03:
        return [p1, p2]

    track_y = max(p1[1], p2[1]) + 0.35
    return _dedupe_points([p1, (p1[0], track_y), (p2[0], track_y), p2])


def _dedupe_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    for point in points:
        if kept and abs(kept[-1][0] - point[0]) < 1e-6 and abs(kept[-1][1] - point[1]) < 1e-6:
            continue
        kept.append(point)
    return kept


def _polyline_midpoint(points: list[tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    segments: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    total = 0.0
    for start, end in zip(points[:-1], points[1:]):
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if length <= 1e-9:
            continue
        segments.append((start, end, length))
        total += length
    if not segments:
        return points[0], (1.0, 0.0)

    target = total * 0.5
    walked = 0.0
    for start, end, length in segments:
        if walked + length >= target:
            ratio = (target - walked) / length
            point = (start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio)
            tangent = ((end[0] - start[0]) / length, (end[1] - start[1]) / length)
            return point, tangent
        walked += length
    start, end, length = segments[-1]
    return end, ((end[0] - start[0]) / length, (end[1] - start[1]) / length)


def _draw_generator(ax: plt.Axes, bus_pos: tuple[float, float], gen_label: int, side: str) -> None:
    dx, dy = _offset(side, 0.74)
    center = (bus_pos[0] + dx, bus_pos[1] + dy)
    ax.plot([bus_pos[0], center[0]], [bus_pos[1], center[1]], color=NORMAL_BUS, lw=1.05, zorder=4)
    ax.add_patch(
        patches.Circle(center, 0.31, facecolor="white", edgecolor=NORMAL_BUS, lw=1.25, zorder=7)
    )
    ax.text(
        center[0],
        center[1],
        f"G{gen_label}",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=NORMAL_BUS,
        zorder=8,
    )


def _draw_load(ax: plt.Axes, bus_pos: tuple[float, float], side: str) -> None:
    dx, dy = _offset("down", 0.56)
    mid = (bus_pos[0] + dx * 0.55, bus_pos[1] + dy * 0.55)
    end = (bus_pos[0] + dx, bus_pos[1] + dy)
    ax.plot([bus_pos[0], mid[0]], [bus_pos[1], mid[1]], color=NORMAL_BUS, lw=1.0, zorder=4)
    ax.annotate(
        "",
        xy=end,
        xytext=mid,
        arrowprops={"arrowstyle": "-|>", "color": NORMAL_BUS, "lw": 1.2, "mutation_scale": 12},
        zorder=5,
    )


def _draw_transformer(ax: plt.Axes, points: list[tuple[float, float]]) -> None:
    midpoint, tangent = _polyline_midpoint(points)
    ux, uy = tangent
    radius = 0.115
    centers = [
        (midpoint[0] - ux * radius, midpoint[1] - uy * radius),
        (midpoint[0] + ux * radius, midpoint[1] + uy * radius),
    ]
    for center in centers:
        ax.add_patch(patches.Circle(center, radius, fill=False, edgecolor=NORMAL_BUS, lw=1.0, zorder=6))


def _draw_branch(
    ax: plt.Axes,
    left: int,
    right: int,
    events: list[dict[str, Any]],
) -> None:
    points = _route_points(left, right)
    event = _branch_is_event((left, right), events)
    if event:
        style = _event_style(int(event["label"]))
        color = style["color"]
        lw = 3.4
        alpha = 0.98
        zorder = 3
    else:
        color = NORMAL_LINE
        lw = 1.15
        alpha = 0.62
        zorder = 1

    ax.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        color=color,
        lw=lw,
        alpha=alpha,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=zorder,
    )
    if tuple(sorted((left, right))) in XFO_BRANCHES:
        _draw_transformer(ax, points)
    if event:
        (mx, my), _ = _polyline_midpoint(points)
        ax.text(
            mx,
            my + 0.20,
            f"L{event['label']} {_event_location_text(event)}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": color, "lw": 1.0},
            zorder=14,
        )


def _draw_event_marker(
    ax: plt.Axes,
    bus: int,
    events: list[dict[str, Any]],
) -> None:
    pos = IEEE39_POSITIONS[bus]
    labels = sorted({int(event["label"]) for event in events})
    primary = labels[0] if labels else 1
    style = _event_style(primary)
    color = style["color"]

    ax.add_patch(
        patches.Circle(
        pos,
        0.58,
        facecolor=color,
        edgecolor=EVENT_PURPLE,
        lw=2.4,
        alpha=0.18,
        zorder=24,
        )
    )
    ax.scatter(
        [pos[0]],
        [pos[1] + 0.02],
        marker="*",
        s=260,
        color=EVENT_PURPLE,
        edgecolor="white",
        linewidth=0.8,
        zorder=34,
    )
    label_text = "/".join(f"L{label}" for label in labels)
    is_line_endpoint_only = all(
        event.get("line") and bus not in {int(node) for node in event.get("nodes", [])} and event.get("pmu_bus") != bus
        for event in events
    )
    suffix = "LINE END" if is_line_endpoint_only else "ORIGIN"
    label_dx, label_dy = EVENT_LABEL_OFFSETS.get(bus, (0.0, 0.72))
    ax.text(
        pos[0] + label_dx,
        pos[1] + label_dy,
        f"{label_text} {suffix}",
        ha="center",
        va="center",
        fontsize=7.3,
        fontweight="bold",
        color=EVENT_PURPLE,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": EVENT_PURPLE, "lw": 1.0},
        zorder=35,
    )


def _draw_bus(
    ax: plt.Axes,
    bus: int,
    pmu_set: set[int],
    events_by_bus: dict[int, list[dict[str, Any]]],
) -> None:
    pos = IEEE39_POSITIONS[bus]
    is_event_bus = bus in events_by_bus
    bus_color = EVENT_PURPLE if is_event_bus else NORMAL_BUS
    bus_width = 6.2 if is_event_bus else 4.5

    ax.plot(
        [pos[0] - BUS_BAR_HALF_WIDTH, pos[0] + BUS_BAR_HALF_WIDTH],
        [pos[1], pos[1]],
        color=bus_color,
        lw=bus_width,
        solid_capstyle="butt",
        zorder=10,
    )

    label_dx, label_dy = BUS_LABEL_OFFSETS[bus]
    ax.text(
        pos[0] + label_dx,
        pos[1] + label_dy,
        str(bus),
        ha="left",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color="#111827",
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "#64748b", "lw": 0.65},
        zorder=18,
    )

    if bus in GEN_BUSES:
        _draw_generator(ax, pos, GEN_BUSES[bus], GEN_SIDE.get(bus, "up"))
    if bus in LOAD_BUSES:
        _draw_load(ax, pos, LOAD_SIDE.get(bus, "down"))
    if bus in pmu_set:
        tag_x = pos[0] - 1.05
        tag_y = pos[1] + 0.08
        ax.add_patch(
            patches.FancyBboxPatch(
                (tag_x, tag_y),
                0.70,
                0.30,
                boxstyle="round,pad=0.025",
                facecolor=PMU_TAG,
                edgecolor="white",
                lw=0.9,
                zorder=19,
            )
        )
        ax.text(
            tag_x + 0.35,
            tag_y + 0.15,
            "PMU",
            color="white",
            fontsize=6.8,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=20,
        )

    if is_event_bus:
        _draw_event_marker(ax, bus, events_by_bus[bus])


def _event_summary_lines(events: list[dict[str, Any]]) -> list[str]:
    lines = []
    for event in events:
        label = int(event["label"])
        style = _event_style(label)
        start = float(event["start_sec"])
        end = float(event["end_sec"])
        kind = str(event.get("kind", style["name"])).replace("_", " ")
        lines.append(f"L{label} {kind} @ {_event_location_text(event)}: {start:.3f}-{end:.3f} s")
    return lines


def _draw_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D([0], [0], color=NORMAL_LINE, lw=1.6, label="Transmission/transformer branch"),
        Line2D([0], [0], color=PMU_TAG, lw=5.0, label="PMU-equipped bus tag"),
        Line2D([0], [0], marker="*", color="white", markerfacecolor=EVENT_PURPLE, markersize=12, label="Event origin bus"),
        Line2D([0], [0], color="#f97316", lw=4.0, label="Highlighted outage/event line"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.94, fontsize=8.0)


def plot_ieee39_diagram(
    branches: Iterable[tuple[int, int] | list[int]],
    pmu_buses: Iterable[int],
    events: Iterable[dict[str, Any]],
    output_dir: Path,
    scenario_id: str | None = None,
    filename: str = "ieee39_event_diagram.png",
) -> Path:
    """Plot an engineer-readable IEEE 39 single-line event diagram.

    Parameters
    ----------
    branches:
        Iterable of bus pairs using IEEE bus numbers.
    pmu_buses:
        The eight PMU buses from the competition.
    events:
        Scenario event dictionaries from ``SIM_####.json``.
    output_dir:
        Directory where the PNG should be written.
    scenario_id:
        Optional scenario identifier printed in the title.
    filename:
        Output PNG name. Defaults to the canonical dataset diagram name.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    pmu_set = {int(bus) for bus in pmu_buses}
    event_list = list(events)
    events_by_bus: dict[int, list[dict[str, Any]]] = {}
    for event in event_list:
        for bus in _event_buses(event):
            if bus in IEEE39_POSITIONS:
                events_by_bus.setdefault(bus, []).append(event)

    fig, ax = plt.subplots(figsize=(15.6, 10.8), facecolor="white")

    for branch in branches:
        if len(branch) < 2:
            continue
        left = int(branch[0])
        right = int(branch[1])
        if left not in IEEE39_POSITIONS or right not in IEEE39_POSITIONS:
            continue
        _draw_branch(ax, left, right, event_list)

    for bus in sorted(IEEE39_POSITIONS):
        _draw_bus(ax, bus, pmu_set, events_by_bus)

    summary_lines = _event_summary_lines(event_list)
    if summary_lines:
        ax.text(
            0.012,
            0.018,
            "\n".join(summary_lines[:12]),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.0,
            color="#111827",
            bbox={"boxstyle": "round,pad=0.42", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.96},
            zorder=30,
        )

    _draw_legend(ax)
    title = "IEEE 39-Bus New England System - Event Single-Line Diagram"
    if scenario_id:
        title = f"{scenario_id} - {title}"
    ax.set_title(title, fontsize=15.5, fontweight="bold", pad=18)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(-0.85, 14.75)
    ax.set_ylim(0.40, 11.85)

    path = output_dir / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
