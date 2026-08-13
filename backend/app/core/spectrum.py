"""Spectral analysis (FFT/PSD, STFT spectrogram) and PID error analysis.

Follows the PIDtoolbox methodology: setpoint-vs-gyro PID error spectrum,
propwash (30-90 Hz) energy, broadband gyro noise, and error vs throttle.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.ndimage import binary_closing

from .blackbox.parser import FlightLogParser
from .analyze import _convert_column


# Frequency bands of interest (Hz) for PID tuning.
PROPWASH_BAND = (30.0, 90.0)    # propwash / motor-induced oscillation
NOISE_BAND = (10.0, 150.0)      # broadband gyro noise
AXIS_LABELS = ["roll", "pitch", "yaw"]


# ---------------------------------------------------------------------------
# Raw series access (full resolution, display units)
# ---------------------------------------------------------------------------

def _time_axis(parser: FlightLogParser) -> np.ndarray:
    defs = parser.frame_defs[ord("I")]
    arr = parser.main_frames
    if arr is None or arr.shape[0] == 0:
        return np.array([])
    ti = defs.field_names.index("time")
    return (arr[:, ti].astype(np.float64) - arr[0, ti]) * 1e-6


def _field_values(parser: FlightLogParser, name: str) -> np.ndarray:
    defs = parser.frame_defs[ord("I")]
    if name not in defs.field_names:
        raise KeyError(f"field '{name}' not in log")
    arr = parser.main_frames
    idx = defs.field_names.index(name)
    return _convert_column(parser, name, arr[:, idx])


def estimate_sample_rate(t: np.ndarray) -> float:
    """Robust sample-rate estimate (Hz), tolerant of 32-bit time wraparound."""
    if t.shape[0] < 2:
        return 0.0
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.shape[0] == 0:
        return 0.0
    return 1.0 / np.median(dt)


def _welch(v: np.ndarray, fs: float, max_freq: Optional[float],
           nperseg: int) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v, dtype=np.float64)
    nperseg = int(min(nperseg, v.shape[0]))
    if nperseg < 8:
        return np.array([]), np.array([])
    freqs, psd = signal.welch(v, fs=fs, nperseg=nperseg, window="hann",
                              detrend="constant", scaling="density")
    if max_freq is not None:
        m = freqs <= max_freq
        freqs, psd = freqs[m], psd[m]
    return freqs, psd


def _band_energy(freqs: np.ndarray, psd: np.ndarray, f0: float, f1: float) -> float:
    if freqs.size == 0:
        return 0.0
    m = (freqs >= f0) & (freqs <= f1)
    if not m.any():
        return 0.0
    return float(np.trapezoid(psd[m], freqs[m]))


# ---------------------------------------------------------------------------
# Public analysis functions
# ---------------------------------------------------------------------------

def power_spectrum(parser: FlightLogParser, name: str,
                   max_freq: float = 200.0, nperseg: int = 1024):
    """Welch PSD of a single field. Returns (freq, psd, sample_rate)."""
    t = _time_axis(parser)
    v = _field_values(parser, name)
    fs = estimate_sample_rate(t)
    if v.shape[0] < 32 or fs <= 0:
        return np.array([]), np.array([]), fs
    v = v[np.isfinite(v)]
    freqs, psd = _welch(v, fs, max_freq, nperseg)
    return freqs, psd, fs


def motor_frequency(parser: FlightLogParser, pole_pairs: int = 7,
                    harmonics: int = 3) -> Optional[Dict]:
    """Estimate the motor mechanical frequency (Hz) from the eRPM fields.

    Blackbox stores eRPM as (electrical RPM / 100). For an outrunner motor the
    mechanical RPM equals the electrical RPM divided by the number of pole
    pairs (7 for the common 14-pole / N14P motors), and the mechanical
    frequency is that divided by 60. Returns the base frequency and its first
    few harmonics, or None when no RPM data is present.
    """
    names = parser.frame_defs[ord("I")].field_names
    erpm_names = [n for n in names if n.startswith("eRPM[")]
    if not erpm_names:
        return None

    medians = []
    for name in erpm_names:
        v = _field_values(parser, name)
        v = v[np.isfinite(v)]
        v = v[v > 0]
        if v.size:
            medians.append(float(np.median(v)))
    if not medians:
        return None

    erpm_field = float(np.median(medians))          # electrical RPM / 100
    base_hz = erpm_field * 100.0 / pole_pairs / 60.0
    if base_hz <= 0:
        return None
    return {
        "base_hz": base_hz,
        "harmonics_hz": [base_hz * (i + 1) for i in range(harmonics)],
    }


def spectrogram(parser: FlightLogParser, name: str,
                max_freq: float = 200.0, nperseg: int = 1024,
                max_time_bins: int = 256) -> Dict:
    """STFT spectrogram (dB) of a field, for a heatmap waterfall plot."""
    t = _time_axis(parser)
    v = _field_values(parser, name)
    fs = estimate_sample_rate(t)
    if v.shape[0] < 32 or fs <= 0:
        return {"times": [], "freqs": [], "power": [], "sample_rate": fs}

    v = np.asarray(v[np.isfinite(v)], dtype=np.float64)
    nperseg = int(min(nperseg, v.shape[0]))
    noverlap = nperseg // 2
    freqs, times, Sxx = signal.spectrogram(
        v, fs=fs, nperseg=nperseg, noverlap=noverlap,
        window="hann", mode="psd", detrend="constant")
    Sxx = np.clip(10.0 * np.log10(Sxx + 1e-30), -80.0, None)

    if max_freq is not None:
        m = freqs <= max_freq
        freqs = freqs[m]
        Sxx = Sxx[m, :]

    # time-bin decimation for the frontend
    if Sxx.shape[1] > max_time_bins:
        idx = np.linspace(0, Sxx.shape[1] - 1, max_time_bins).astype(int)
        times = times[idx]
        Sxx = Sxx[:, idx]

    return {
        "times": times.tolist(),
        "freqs": freqs.tolist(),
        "power": Sxx.T.tolist(),  # (n_times, n_freqs)
        "sample_rate": fs,
    }


def pid_error_analysis(parser: FlightLogParser,
                       max_freq: float = 200.0, nperseg: int = 1024) -> Dict:
    """PID error (setpoint - gyro) spectra, propwash metrics and error vs throttle."""
    t = _time_axis(parser)
    fs = estimate_sample_rate(t)
    names = parser.frame_defs[ord("I")].field_names

    throttle = None
    if "rcCommand[3]" in names:
        throttle = _field_values(parser, "rcCommand[3]")

    axes = []
    for axis in range(3):
        sp_name = f"setpoint[{axis}]"
        gy_name = f"gyroADC[{axis}]"
        if sp_name not in names or gy_name not in names:
            continue

        setpoint = _field_values(parser, sp_name)
        gyro = _field_values(parser, gy_name)
        error = setpoint - gyro

        err_finite = error[np.isfinite(error)]
        rms = float(np.sqrt(np.mean(err_finite ** 2))) if err_finite.size else 0.0

        freqs, psd = _welch(err_finite, fs, max_freq, nperseg)
        pw = _band_energy(freqs, psd, *PROPWASH_BAND)
        noise = _band_energy(freqs, psd, *NOISE_BAND)
        ratio = (pw / noise) if noise > 0 else 0.0

        axes.append({
            "axis": axis,
            "label": AXIS_LABELS[axis],
            "rms_error_deg_s": rms,
            "error_psd": {"freq": freqs.tolist(), "psd": psd.tolist()},
            "propwash_energy": pw,
            "noise_energy": noise,
            "propwash_ratio": ratio,
            "error_vs_throttle": _error_vs_throttle(error, throttle),
        })

    return {"sample_rate": fs, "axes": axes}


def _error_vs_throttle(error: np.ndarray, throttle: Optional[np.ndarray],
                       n_bins: int = 20) -> Dict:
    """Mean |PID error| binned by normalized throttle (rcCommand[3], 1000-2000 us)."""
    if throttle is None or error.size == 0 or throttle.size != error.size:
        return {"throttle": [], "mean_abs_error": []}

    thr = np.asarray(throttle, dtype=np.float64)
    norm = (thr - 1000.0) / 1000.0  # 0..1
    valid = (norm >= 0.0) & (norm <= 1.0) & np.isfinite(error)
    if not valid.any():
        return {"throttle": [], "mean_abs_error": []}

    norm = norm[valid]
    ae = np.abs(error[valid])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = ((edges[:-1] + edges[1:]) / 2.0).tolist()
    bin_idx = np.digitize(norm, edges[1:-1])  # 0 .. n_bins-1

    means = []
    for b in range(n_bins):
        m = bin_idx == b
        means.append(float(ae[m].mean()) if m.any() else None)

    return {"throttle": centers, "mean_abs_error": means}


# ---------------------------------------------------------------------------
# Step response analysis (PIDtoolbox style)
# ---------------------------------------------------------------------------

def step_response_analysis(parser: FlightLogParser, threshold: float = 100.0,
                           pre: float = 0.2, post: float = 0.8,
                           max_steps: int = 10) -> Dict:
    """Detect setpoint step changes and extract the gyro response.

    For each step, returns the response window plus overshoot / rise-time.
    """
    t = _time_axis(parser)
    fs = estimate_sample_rate(t)
    names = parser.frame_defs[ord("I")].field_names

    axes_out = []
    for axis in range(3):
        sp_name = f"setpoint[{axis}]"
        gy_name = f"gyroADC[{axis}]"
        if sp_name not in names or gy_name not in names:
            continue
        sp = _field_values(parser, sp_name)
        gy = _field_values(parser, gy_name)
        # yaw is under-actuated on most quads, so its commanded rates are much
        # lower than roll/pitch; use a proportionally lower step threshold so
        # low-amplitude yaw steps are still detected.
        axis_threshold = threshold * 0.5 if axis == 2 else threshold
        steps = _detect_steps(t, sp, gy, fs, axis_threshold, pre, post, max_steps)
        axes_out.append({"axis": axis, "label": AXIS_LABELS[axis], "steps": steps})

    return {"sample_rate": fs, "axes": axes_out}


def _detect_steps(t: np.ndarray, sp: np.ndarray, gy: np.ndarray, fs: float,
                  threshold: float, pre: float, post: float, max_steps: int) -> List[Dict]:
    if sp.size < 32 or fs <= 0:
        return []
    sp = np.asarray(sp, dtype=np.float64)
    gy = np.asarray(gy, dtype=np.float64)

    # Betaflight's setpoint is smoothed by RC smoothing, so a "step" is a ramp
    # rather than a single-sample jump. The setpoint is also stored as int16,
    # which leaves occasional zero-diff samples inside the ramp; close those
    # small gaps so a full stick deflection is seen as one contiguous run.
    d = np.diff(sp)
    moving = np.abs(d) >= 0.5
    gap = max(int(0.02 * fs), 10)
    moving = binary_closing(moving, structure=np.ones(gap))

    pre_n = int(pre * fs)
    post_n = int(post * fs)
    settle_n = max(int(0.15 * fs), 4)

    steps = []
    used_until = -1
    i = 0
    while i < sp.size - 1:
        if not moving[i]:
            i += 1
            continue

        # contiguous motion run: setpoint moved from sp[seg_start] to sp[seg_end]
        seg_start = i
        while i < sp.size - 1 and moving[i]:
            i += 1
        seg_end = i

        # the step edge is where the setpoint changed fastest
        e = seg_start + int(np.argmax(np.abs(d[seg_start:seg_end])))
        if e < used_until:
            continue

        # steady-state levels just before / after the motion run
        b0 = max(0, seg_start - settle_n)
        baseline = float(np.median(sp[b0:seg_start])) if seg_start > b0 else float(sp[seg_start])
        t1 = min(sp.size, seg_end + settle_n)
        target = float(np.median(sp[seg_end:t1])) if t1 > seg_end else float(sp[seg_end])
        step_mag = target - baseline
        if abs(step_mag) < threshold:
            continue

        win_start = max(0, e - pre_n)
        win_end = min(sp.size, e + 1 + post_n)
        if win_end - win_start < 8:
            continue

        seg_gy = gy[e + 1:win_end]
        if seg_gy.size == 0:
            continue
        if step_mag > 0:
            overshoot = float(np.max(seg_gy) - target)
        else:
            overshoot = float(target - np.min(seg_gy))
        overshoot_pct = overshoot / abs(step_mag) * 100.0

        rise_time = _rise_time(gy[win_start:win_end], baseline, target, fs)

        seg_t, seg_sp, seg_gy2 = _decimate(
            t[win_start:win_end] - t[e], sp[win_start:win_end], gy[win_start:win_end])

        steps.append({
            "t0": float(t[e]),
            "step_mag": float(step_mag),
            "overshoot_pct": float(overshoot_pct),
            "rise_time_s": float(rise_time),
            "data": {"t": seg_t, "setpoint": seg_sp, "gyro": seg_gy2},
        })
        used_until = win_end
        if len(steps) >= max_steps:
            break
    return steps


def _rise_time(gy: np.ndarray, baseline: float, target: float, fs: float) -> float:
    """Time (s) for gyro to go from 10% to 90% of the step."""
    step = target - baseline
    if abs(step) < 1e-6 or fs <= 0:
        return 0.0
    lo = baseline + 0.1 * step
    hi = baseline + 0.9 * step
    t10 = t90 = None
    for i, v in enumerate(gy):
        if step > 0:
            if t10 is None and v >= lo:
                t10 = i
            if t90 is None and v >= hi:
                t90 = i
        else:
            if t10 is None and v <= lo:
                t10 = i
            if t90 is None and v <= hi:
                t90 = i
        if t10 is not None and t90 is not None:
            break
    if t10 is None or t90 is None:
        return 0.0
    return (t90 - t10) / fs


def _decimate(t: np.ndarray, sp: np.ndarray, gy: np.ndarray,
              max_pts: int = 400):
    """Uniform stride decimation of a step window for the frontend."""
    n = t.shape[0]
    if n <= max_pts:
        return t.tolist(), sp.tolist(), gy.tolist()
    idx = np.linspace(0, n - 1, max_pts).astype(int)
    return t[idx].tolist(), sp[idx].tolist(), gy[idx].tolist()
