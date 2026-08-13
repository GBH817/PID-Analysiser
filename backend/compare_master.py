"""Compare our Python decoder against the official blackbox_decode CSV output.

Loads samples/master/...master.csv (official reference) and our parsed
main_frames, then compares every numeric field value row by row.
"""
import re
import sys

import numpy as np
import pandas as pd

from app.core.blackbox import parse_bbl

BBL = r"..\samples\BTFL_KWONGKAN_10inch_0326_00_Filter.BBL"
CSV = r"..\samples\master\BTFL_KWONGKAN_10inch_0326_00_Filter.01.master.csv"

with open(BBL, "rb") as f:
    data = f.read()
logs = parse_bbl(data)
log = logs[0]
print(f"our main_frames: {log.main_frames.shape}, fields={len(log.field_names)}")

# ---- load official CSV ----
df = pd.read_csv(CSV, nrows=None)
print(f"master CSV: {df.shape}, columns={len(df.columns)}")
print("master columns:", list(df.columns))

# ---- build column name -> cleaned name mapping (strip units) ----
clean_map = {}
for col in df.columns:
    clean = re.sub(r"\s*\([^)]*\)$", "", str(col)).strip()
    clean_map[clean] = col

# column names may carry a leading space, store a stripped-accessor helper
def col_series(name):
    return df[name].values

our_names = log.field_names
arr = log.main_frames

print("\n==== field-by-field comparison ====")
all_ok = True
n = min(len(df), arr.shape[0])
print(f"comparing {n} rows (master {len(df)} vs ours {arr.shape[0]})")

for i, name in enumerate(our_names):
    if name not in clean_map:
        print(f"  {name:24s} NOT IN MASTER CSV")
        continue
    col = clean_map[name]
    master_vals = col_series(col)[:n]
    # try to convert to numeric (flags fields may be strings)
    try:
        master_num = pd.to_numeric(master_vals, errors="raise").astype(np.float64)
    except (ValueError, TypeError):
        print(f"  {name:24s} (col '{col}') non-numeric, skipped")
        continue
    ours = arr[:n, i].astype(np.float64)

    # time: master is us (rollover applied) -> should match raw
    # vbatLatest (V) / amperageLatest (A) / energy: converted, try common scales
    if name == "vbatLatest":
        candidates = {"raw/100": ours / 100.0, "raw/10": ours / 10.0, "raw": ours}
    elif name == "amperageLatest":
        candidates = {"raw/100": ours / 100.0, "raw/1000": ours / 1000.0, "raw": ours}
    else:
        candidates = {"raw": ours}

    best = None
    for label, conv in candidates.items():
        diff = np.abs(conv - master_num)
        mismatch = int((diff > 0.01).sum())
        maxdiff = float(diff.max()) if len(diff) else 0.0
        score = mismatch
        if best is None or score < best[0]:
            best = (score, label, maxdiff)
    score, label, maxdiff = best
    ok = score == 0
    all_ok &= ok
    print(f"  {name:24s} match={str(ok):5s} conv={label:10s} mismatches={score:6d} max|diff|={maxdiff:.6g}")

print("\n==== RESULT ====")
print("ALL FIELDS MATCH" if all_ok else "SOME FIELDS DIFFER")
