"""Inspect raw rows in the mismatch region."""
import numpy as np
import pandas as pd

from app.core.blackbox import parse_bbl

with open(r"..\samples\BTFL_KWONGKAN_10inch_0326_00_Filter.BBL", "rb") as f:
    data = f.read()
log = parse_bbl(data)[0]
arr = log.main_frames
df = pd.read_csv(r"..\samples\master\BTFL_KWONGKAN_10inch_0326_00_Filter.01.master.csv")

idx = {n: i for i, n in enumerate(log.field_names)}

for r in [9489, 9489 + 1, 26982, 35463]:
    print(f"=== row {r} ===")
    for n in ("rcCommand[0]", "rcCommand[1]", "rcCommand[2]", "rcCommand[3]",
              "setpoint[0]", "setpoint[1]", "setpoint[2]", "setpoint[3]",
              "gyroADC[0]", "motor[0]"):
        mcol = " " + n
        m = df[mcol].iloc[r] if mcol in df.columns else None
        print(f"  {n:12s} master={m}  ours={arr[r, idx[n]]}")
