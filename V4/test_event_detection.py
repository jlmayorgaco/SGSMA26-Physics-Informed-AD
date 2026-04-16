#!/usr/bin/env python3
"""
Test event detection with global spans.
"""
import sys
from pathlib import Path
sys.path.insert(0, '.')

from src.analysis.loader import load_all_buses
from src.analysis.events import build_global_event_spans
from src.config.config import DEFAULT_EVENT_LABELS

def main():
    input_dir = Path("data/RAW0001")
    buses = load_all_buses(input_dir, "Bus*_Competition_Data*.csv")
    print(f"Loaded {len(buses)} buses: {[b.bus_id for b in buses]}")
    
    label_map = {int(k): v for k, v in DEFAULT_EVENT_LABELS.items()}
    spans = build_global_event_spans(buses, label_map)
    
    print(f"\nTotal events detected: {len(spans)}")
    for sp in spans:
        print(f"  Event {sp['event_id']} ({sp['label']}): "
              f"{sp['start_time']:.1f}-{sp['end_time']:.1f}s, "
              f"duration {sp['duration_s']:.2f}s, "
              f"buses involved: {sp.get('involved_buses', [])}")
    
    # Count by event label
    from collections import Counter
    counts = Counter(sp['event_id'] for sp in spans)
    print("\nCounts per event label:")
    for event_id in sorted(counts):
        print(f"  {event_id}: {counts[event_id]} occurrences")
    
    # Check if we have all expected labels (0-7)
    expected = set(range(1, 8))  # 1..7 (0 is normal, 8 absent)
    detected = set(counts.keys())
    missing = expected - detected
    if missing:
        print(f"\nWARNING: Missing event labels: {missing}")
    else:
        print("\nSUCCESS: All expected event labels detected.")

if __name__ == "__main__":
    main()