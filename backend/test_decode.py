"""Smoke test for the blackbox decoder against a real .bbl sample."""
import json
import sys
import time

from app.core.blackbox import parse_bbl, build_summary

path = r"..\samples\BTFL_KWONGKAN_10inch_0326_00_Filter.BBL"
if len(sys.argv) > 1:
    path = sys.argv[1]

t0 = time.time()
with open(path, "rb") as f:
    data = f.read()
print(f"file size: {len(data)} bytes")

logs = parse_bbl(data)
print(f"logs found: {len(logs)}")

for i, log in enumerate(logs):
    s = build_summary(log)
    print(f"--- log {i} ---")
    print(f"  fc_version      : {s['fc_version']}")
    print(f"  firmware_type   : {s['firmware_type']}")
    print(f"  data_version    : {s['data_version']}")
    print(f"  date_time       : {s['date_time']}")
    print(f"  frame_intervals : {s['frame_intervals']}")
    print(f"  field_count     : {s['field_count']}")
    print(f"  n_main_frames   : {s['n_main_frames']}")
    print(f"  n_events        : {s['n_events']}")
    print(f"  n_gps_frames    : {s['n_gps_frames']}")
    print(f"  n_slow_frames   : {s['n_slow_frames']}")
    print(f"  corrupt_frames  : {s['total_corrupt_frames']}")
    print(f"  duration_us     : {s.get('duration_us')}")
    print(f"  frame_counts    : {s['frame_counts']}")
    print(f"  field_names     : {s['field_names'][:20]} ...")
    print(f"  field_indexes   : {json.dumps(s['field_indexes'], default=str)}")
    if log.main_frames is not None:
        names = log.field_names
        idx = log.main_field_indexes
        arr = log.main_frames
        print(f"  main_frames shape: {arr.shape}")
        if idx.gyro_adc[0] >= 0 and len(arr) > 5:
            print(f"  gyro[0] first 5: {arr[:5, idx.gyro_adc[0]].tolist()}")
            print(f"  gyro[0] min/max: {arr[:, idx.gyro_adc[0]].min()} / {arr[:, idx.gyro_adc[0]].max()}")
        if idx.motor[0] >= 0 and len(arr) > 5:
            print(f"  motor[0] first 5: {arr[:5, idx.motor[0]].tolist()}")
        if idx.time >= 0:
            t = arr[:, idx.time]
            print(f"  time first/last: {t[0]} / {t[-1]}  (rollover?)")
            print(f"  dt min/max(us): {int((t[1:] - t[:-1]).min())} / {int((t[1:] - t[:-1]).max())}")
    print()

print(f"total time: {time.time() - t0:.2f}s")
