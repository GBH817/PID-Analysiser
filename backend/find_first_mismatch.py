"""Find first mismatch row for each field and identify frame type (I/P)."""
import numpy as np
import pandas as pd

from app.core.blackbox import parse_bbl

with open(r"..\samples\BTFL_KWONGKAN_10inch_0326_00_Filter.BBL", "rb") as f:
    data = f.read()
log = parse_bbl(data)[0]
arr = log.main_frames
df = pd.read_csv(r"..\samples\master\BTFL_KWONGKAN_10inch_0326_00_Filter.01.master.csv")

idx = {n: i for i, n in enumerate(log.field_names)}
ii = log.frame_interval_i

for target in ["rcCommand[1]", "rcCommand[2]", "setpoint[3]"]:
    i = idx[target]
    mcol = " " + target
    master = pd.to_numeric(df[mcol], errors="raise").astype(np.float64).values
    ours = arr[:len(master), i].astype(np.float64)
    diff = np.abs(master - ours)
    bad = np.where(diff > 0.01)[0]
    first = bad[0]
    # frame index -> which iteration; check if first mismatch is near an I frame
    print(f"### {target}: first mismatch row={first} (row%128={first % 128})")
    # examine surrounding rows
    for r in range(max(0, first - 3), min(len(master), first + 4)):
        m = master[r]
        o = ours[r]
        is_i = "I" if (arr[r, 0] % ii == 0) else "P"
        print(f"    row={r} iter={arr[r, 0]:8d} ({is_i}) master={m:8.0f} ours={o:8.0f} diff={o - m:8.0f}")
