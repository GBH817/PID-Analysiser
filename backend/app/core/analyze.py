"""Analysis helpers: unit conversion, downsampling for the web frontend."""

import math
from typing import Optional, Tuple

import numpy as np

from .blackbox.parser import FlightLogParser


# ---------------------------------------------------------------------------
# Field metadata / units
# ---------------------------------------------------------------------------

def field_unit(name: str) -> str:
    """Human-readable unit for a main-frame field (display only)."""
    if name == "time":
        return "s"
    if name == "loopIteration":
        return ""
    if name.startswith("gyroADC[") or name.startswith("gyroUnfilt[") or name.startswith("setpoint["):
        return "deg/s"
    if name.startswith("axisP[") or name.startswith("axisD[") or name.startswith("axisF["):
        return "deg/s"
    if name.startswith("axisI["):
        return "raw"
    if name.startswith("rcCommand[") or name.startswith("motor["):
        return "us"
    if name == "vbatLatest":
        return "V"
    if name == "amperageLatest":
        return "A"
    if name.startswith("accSmooth["):
        return "g"
    if name.startswith("eRPM["):
        return "rpm"
    return "raw"


def _convert_column(parser: FlightLogParser, name: str, raw: np.ndarray) -> np.ndarray:
    """Convert raw blackbox values to display units."""
    if name == "time":
        return raw.astype(np.float64) * 1e-6  # us -> s
    if name.startswith("gyroADC[") or name.startswith("gyroUnfilt["):
        # raw gyro is stored in (rad/s * 1e6) scale; one scale factor per log
        factor = parser.sys_config.gyro_scale * 1e6 * (180.0 / math.pi)
        return raw.astype(np.float64) * factor
    if name == "vbatLatest":
        divisor = 100.0 if parser._fc_version_gte("4.3.0") else 10.0
        return raw.astype(np.float64) / divisor
    if name == "amperageLatest":
        return raw.astype(np.float64) / 100.0
    if name.startswith("accSmooth["):
        acc_1g = parser.sys_config.acc_1g or 2048
        return raw.astype(np.float64) / acc_1g
    return raw.astype(np.float64)


def channel_series(parser: FlightLogParser, name: str,
                   max_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return (t_sec, values, unit) for a main-frame channel.

    Values are converted to display units. If max_points is given, the series
    is decimated with a min/max decimator (preserves spikes).
    """
    if name == "time":
        # time in seconds, offset from log start
        defs0 = parser.frame_defs[ord("I")]
        arr0 = parser.main_frames
        if arr0 is None or arr0.shape[0] == 0:
            return np.array([]), np.array([]), "s"
        ti = defs0.field_names.index("time")
        t = (arr0[:, ti].astype(np.float64) - arr0[0, ti]) * 1e-6
        if max_points is not None and max_points >= 2 and t.shape[0] > max_points:
            t, _ = _downsample_minmax(t, t, max_points)
        return t, None, "s"

    defs = parser.frame_defs[ord("I")]
    if name not in defs.field_names:
        raise KeyError(f"field '{name}' not in log")
    idx = defs.field_names.index(name)

    arr = parser.main_frames
    if arr is None or arr.shape[0] == 0:
        return np.array([]), np.array([]), field_unit(name)

    time_idx = defs.field_names.index("time")
    t = (arr[:, time_idx].astype(np.float64) - arr[0, time_idx]) * 1e-6
    v = _convert_column(parser, name, arr[:, idx])
    unit = field_unit(name)

    if max_points is not None and max_points >= 2 and v.shape[0] > max_points:
        t, v = _downsample_minmax(t, v, max_points)
    return t, v, unit


def _downsample_minmax(t: np.ndarray, v: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Min/max decimation to <= max_points points (spike-preserving)."""
    n = v.shape[0]
    if n <= max_points:
        return t, v
    bins = max_points // 2
    edges = np.linspace(0, n, bins + 1).astype(np.int64)
    starts = edges[:-1]
    counts = np.diff(edges)
    ends = starts + counts

    tmin = np.take(t, starts)
    tmax = np.take(t, ends - 1)
    vmin = np.minimum.reduceat(v, starts)
    vmax = np.maximum.reduceat(v, starts)

    tt = np.empty(2 * bins, dtype=np.float64)
    vv = np.empty(2 * bins, dtype=np.float64)
    tt[0::2] = tmin
    tt[1::2] = tmax
    vv[0::2] = vmin
    vv[1::2] = vmax
    return tt, vv
