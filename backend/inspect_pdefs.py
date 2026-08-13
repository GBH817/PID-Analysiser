"""Inspect P-frame field defs vs I-frame."""
from app.core.blackbox import parse_bbl

with open(r"..\samples\BTFL_KWONGKAN_10inch_0326_00_Filter.BBL", "rb") as f:
    data = f.read()
log = parse_bbl(data)[0]
fi = log.frame_defs[ord('I')]
fp = log.frame_defs[ord('P')]
print("I vs P defs for rcCommand/setpoint:")
for i, name in enumerate(fi.field_names):
    if name.startswith(("rcCommand", "setpoint")):
        print(f"  {name:14s} I: sig={fi.field_signed[i]} pred={fi.predictor[i]} enc={fi.encoding[i]}  |  P: sig={fp.field_signed[i]} pred={fp.predictor[i]} enc={fp.encoding[i]}")
