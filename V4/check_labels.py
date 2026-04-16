#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

input_dir = Path("data/RAW0001")
for csv in input_dir.glob("Bus*.csv"):
    df = pd.read_csv(csv, nrows=0)  # just header
    cols = df.columns.tolist()
    if 'Event' not in cols:
        print(f"{csv.name}: No Event column")
        continue
    # read first few rows to see labels
    df = pd.read_csv(csv, usecols=['Event'])
    unique = df['Event'].dropna().unique()
    print(f"{csv.name}: {len(df)} rows, unique Events: {sorted(unique)}")
    # count each label
    counts = df['Event'].value_counts().sort_index()
    for label, count in counts.items():
        print(f"  {label}: {count}")