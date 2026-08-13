"""Inspect predictor/encoding/signed for rcCommand/setpoint fields."""
from app.core.blackbox import parse_bbl

with open(r"..\samples\BTFL_KWONGKAN_10inch_0326_00_Filter.BBL", "rb") as f:
    data = f.read()
log = parse_bbl(data)[0]
fd = log.frame_defs[ord('I')]
for i, name in enumerate(fd.field_names):
    if name in ("rcCommand[0]", "rcCommand[1]", "rcCommand[2]", "rcCommand[3]",
                "setpoint[0]", "setpoint[1]", "setpoint[2]", "setpoint[3]",
                "axisD[0]", "axisD[1]", "axisD[2]"):
        print(f"{name:14s} signed={fd.field_signed[i]} predictor={fd.predictor[i]} encoding={fd.encoding[i]}")

# raw first rows for these fields
idx = {n: i for i, n in enumerate(fd.field_names)}
arr = log.main_frames
for n in ("rcCommand[0]", "rcCommand[1]", "rcCommand[2]", "rcCommand[3]", "setpoint[3]"):
    print(n, arr[:10, idx[n]].tolist())
