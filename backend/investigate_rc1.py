"""Investigate the remaining 44 rcCommand[1] mismatches after the v2 fix."""
import numpy as np
import pandas as pd

import app.core.blackbox.parser as parser_mod
from app.core.blackbox.encodings import read_tag8_4s16_v1, read_tag8_4s16_v2 as orig_v2
from app.core.blackbox.constants import FRAME_MAIN, FRAME_INTER
from app.core.blackbox import parse_bbl

BBL = r"..\samples\BTFL_KWONGKAN_10inch_0326_00_Filter.BBL"
CSV = r"..\samples\master\BTFL_KWONGKAN_10inch_0326_00_Filter.01.master.csv"

frame_records = []
_current = None


def wrap_v2(stream):
    off = stream.pos
    v = orig_v2(stream)
    if _current is not None:
        _current.append((off, bytes(stream.data[off:stream.pos]), list(v)))
    return v


orig_parse_frame = parser_mod.FlightLogParser._parse_frame


def patched_parse_frame(self, frame_type, frame, previous, previous2, skipped_frames):
    global _current
    if frame_type not in (FRAME_MAIN, FRAME_INTER):
        return orig_parse_frame(self, frame_type, frame, previous, previous2, skipped_frames)
    is_inter = frame_type == FRAME_INTER
    _current = []
    r = orig_parse_frame(self, frame_type, frame, previous, previous2, skipped_frames)
    frame_records.append((is_inter, int(frame[0]), _current))
    _current = None
    return r


parser_mod.read_tag8_4s16_v2 = wrap_v2
parser_mod.FlightLogParser._parse_frame = patched_parse_frame

with open(BBL, "rb") as f:
    data = f.read()
log = parse_bbl(data)[0]
arr = log.main_frames

df = pd.read_csv(CSV)
master_rc1 = pd.to_numeric(df[" rcCommand[1]"], errors="raise").astype(np.float64)
idx = log.field_names.index("rcCommand[1]")
n = min(len(master_rc1), arr.shape[0])
diff = np.abs(master_rc1[:n] - arr[:n, idx])
bad = np.where(diff > 0.01)[0]
print(f"rcCommand[1] mismatches: {len(bad)}")
for r in bad:
    loop = int(arr[r, 0])
    recs = [rec for rec in frame_records if rec[1] == loop]
    g = recs[0][2][0] if recs else None
    print(f"row={r} mod128={r % 128} loop={loop} master={master_rc1[r]:.0f} ours={arr[r, idx]:.0f} "
          f"group_bytes={g[1].hex(' ') if g else '?'} group_vals={g[2] if g else '?'}")
