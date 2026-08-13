"""Trace TAG8_4S16 groups, aligned to master rows via loopIteration."""
import numpy as np
import pandas as pd

import app.core.blackbox.parser as parser_mod
from app.core.blackbox.encodings import read_tag8_4s16_v1, read_tag8_4s16_v2 as orig_v2
from app.core.blackbox.constants import FRAME_MAIN, FRAME_INTER
from app.core.blackbox import parse_bbl

BBL = r"..\samples\BTFL_KWONGKAN_10inch_0326_00_Filter.BBL"
CSV = r"..\samples\master\BTFL_KWONGKAN_10inch_0326_00_Filter.01.master.csv"

frame_records = []   # per MAIN/INTER frame: (is_inter, loop_iter, [ (byte_off, raw, values) ... ])
_current = None


def wrap_v2(stream):
    off = stream.pos
    v = orig_v2(stream)
    if _current is not None:
        _current.append((off, bytes(stream.data[off:stream.pos]), list(v)))
    return v


def wrap_v1(stream):
    off = stream.pos
    v = read_tag8_4s16_v1(stream)
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
parser_mod.read_tag8_4s16_v1 = wrap_v1
parser_mod.FlightLogParser._parse_frame = patched_parse_frame

with open(BBL, "rb") as f:
    data = f.read()
log = parse_bbl(data)[0]
arr = log.main_frames
print(f"frames={arr.shape}, records={len(frame_records)}")

df = pd.read_csv(CSV)
master_rc1 = pd.to_numeric(df[" rcCommand[1]"], errors="raise").astype(np.float64)
idx_rc1 = log.field_names.index("rcCommand[1]")
n = min(len(master_rc1), arr.shape[0])
diff = np.abs(master_rc1[:n] - arr[:n, idx_rc1])
bad = np.where(diff > 0.01)[0]
r0 = bad[0]
print(f"first mismatch row={r0} (mod128={r0 % 128}) master={master_rc1[r0]:.0f} ours={arr[r0, idx_rc1]:.0f}")

# find the matching trace record by loopIteration
loop = int(arr[r0, 0])
recs = [rec for rec in frame_records if rec[1] == loop]
print(f"loopIteration={loop}, matching records={len(recs)}")
if recs:
    is_inter, loopv, groups = recs[0]
    print(f"inter={is_inter}, groups={len(groups)}")
    for g in groups:
        print(f"  off={g[0]} bytes={g[1].hex(' ')} values={g[2]}")

    # check the trace of the PREVIOUS matching record
    prev_loop = int(arr[r0 - 1, 0])
    prev_recs = [rec for rec in frame_records if rec[1] == prev_loop]
    if prev_recs:
        _, _, pg = prev_recs[0]
        print(f"\nprev frame (loop={prev_loop}) groups:")
        for g in pg:
            print(f"  off={g[0]} bytes={g[1].hex(' ')} values={g[2]}")

    # all-field comparison at this frame
    print("\nall fields at row", r0, "(ours vs master):")
    for i, name in enumerate(log.field_names):
        col = " " + name
        if col in df.columns:
            mv = pd.to_numeric(df[col].iloc[r0], errors="coerce")
            if not pd.isna(mv):
                flag = " <-- DIFF" if abs(float(mv) - arr[r0, i]) > 0.01 else ""
                print(f"  {name:16s} master={float(mv):12.3f} ours={arr[r0, i]:12d}{flag}")
