"""Investigate mismatches in rcCommand[1], rcCommand[2], setpoint[3]."""
import re
import numpy as np
import pandas as pd

from app.core.blackbox import parse_bbl

BBL = r"..\samples\BTFL_KWONGKAN_10inch_0326_00_Filter.BBL"
CSV = r"..\samples\master\BTFL_KWONGKAN_10inch_0326_00_Filter.01.master.csv"

with open(BBL, "rb") as f:
    data = f.read()
log = parse_bbl(data)[0]
arr = log.main_frames
df = pd.read_csv(CSV)

def col(name):
    return df[name].values

idx = {name: i for i, name in enumerate(log.field_names)}
names = log.field_names

for target in ["rcCommand[1]", "rcCommand[2]", "setpoint[3]"]:
    i = idx[target]
    master = pd.to_numeric(col(" " + target), errors="raise").astype(np.float64)
    ours = arr[:len(master), i].astype(np.float64)
    diff = np.abs(master - ours)
    bad = np.where(diff > 0.01)[0]
    print(f"\n### {target}: {len(bad)} mismatches of {len(master)}")
    print("first 12 mismatch rows (idx, master, ours, time, loopIteration):")
    for r in bad[:12]:
        ti = idx["time"]
        print(f"  row={r} master={master[r]:.0f} ours={ours[r]:.0f} "
              f"time={arr[r, ti]} loop={arr[r, idx['loopIteration']]}")
    # where do mismatches cluster? check timespan of first/last
    if len(bad):
        ti = idx["time"]
        print(f"  mismatch rows span time {arr[bad[0], ti]} .. {arr[bad[-1], ti]}")
        # check if mismatches coincide with i-frame boundaries
        # print master/ours distributions
        print(f"  master diff values sample: {master[bad][:20]}")
        print(f"  ours   diff values sample: {ours[bad][:20]}")
