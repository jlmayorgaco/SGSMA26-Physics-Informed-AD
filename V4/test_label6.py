#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, '.')

from src.analysis.loader import load_all_buses
from src.analysis.events import build_global_event_spans
from src.config.config import DEFAULT_EVENT_LABELS

input_dir = Path("data/RAW0001")
buses = load_all_buses(input_dir, "Bus*_Competition_Data*.csv")
label_map = {int(k): v for k, v in DEFAULT_EVENT_LABELS.items()}
spans = build_global_event_spans(buses, label_map)

label6 = [sp for sp in spans if sp['event_id'] == 6]
print(f"Label 6 spans: {len(label6)}")
for sp in label6:
    print(f"  {sp['start_time']:.1f}-{sp['end_time']:.1f}s, duration {sp['duration_s']:.2f}s, buses involved: {sp.get('involved_buses', [])}")
    print(f"    per_bus_labels: {sp.get('per_bus_labels', {})}")

# Also label 5 spans count
label5 = [sp for sp in spans if sp['event_id'] == 5]
print(f"\nLabel 5 spans: {len(label5)} (first 10)")
for sp in label5[:10]:
    print(f"  {sp['start_time']:.1f}-{sp['end_time']:.1f}s, duration {sp['duration_s']:.2f}s")

# Label 7 spans
label7 = [sp for sp in spans if sp['event_id'] == 7]
print(f"\nLabel 7 spans: {len(label7)}")
for sp in label7:
    print(f"  {sp['start_time']:.1f}-{sp['end_time']:.1f}s, duration {sp['duration_s']:.2f}s, buses involved: {sp.get('involved_buses', [])}")