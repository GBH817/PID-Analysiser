"""PID tuning diagnosis engine (Bardwell / PIDtoolbox style) + CLI advice.

Reads the current PID/filter values from the blackbox header (Betaflight 4.x
logs them) and produces concrete target values, e.g. ``set d_pitch = 26``.
"""

from typing import Dict, List, Optional

import numpy as np

from .blackbox.parser import FlightLogParser
from .spectrum import (
    AXIS_LABELS,
    NOISE_BAND,
    PROPWASH_BAND,
    _band_energy,
    _field_values,
    pid_error_analysis,
    power_spectrum,
)


_SEV_RANK = {"high": 0, "medium": 1, "low": 2, "info": 3}


def _finding(axis, category, severity, title, detail, action, cli):
    return {
        "axis": axis,
        "category": category,
        "severity": severity,
        "title": title,
        "detail": detail,
        "action": action,
        "cli": cli,
    }


def _adjust(cur: int, pct: float) -> Optional[int]:
    """Apply a relative percent change; None when the current value is zero/absent."""
    if cur <= 0:
        return None
    return int(round(cur * (1.0 + pct / 100.0)))


def _pid_map(sc) -> Dict[str, Dict[str, int]]:
    return {
        "roll":  {"p": sc.pid_roll[0],  "i": sc.pid_roll[1],  "d": sc.pid_roll[2]},
        "pitch": {"p": sc.pid_pitch[0], "i": sc.pid_pitch[1], "d": sc.pid_pitch[2]},
        "yaw":   {"p": sc.pid_yaw[0],   "i": sc.pid_yaw[1],   "d": sc.pid_yaw[2]},
    }


# ---------------------------------------------------------------------------
# Per-axis rules (driven by pid_error_analysis)
# ---------------------------------------------------------------------------

def _rule_propwash(ax: Dict, pid: Dict, findings: List):
    ratio = ax["propwash_ratio"]
    # Skip near-inactive axes: a high ratio on a tiny signal is meaningless.
    if ratio < 0.20 or ax["rms_error_deg_s"] < 10.0:
        return
    severity = "high" if ratio > 0.35 else "medium"
    axis = ax["label"]

    freq = ax["error_psd"]["freq"]
    psd = ax["error_psd"]["psd"]
    band = [(f, v) for f, v in zip(freq, psd) if PROPWASH_BAND[0] <= f <= PROPWASH_BAND[1]]
    peak = max(band, key=lambda x: x[1])[0] if band else 0.0

    cur_d = pid[axis]["d"]
    cur_p = pid[axis]["p"]
    cli = []
    new_d = _adjust(cur_d, 12)
    new_p = _adjust(cur_p, -8)
    if new_d is not None:
        cli.append(f"set d_{axis} = {new_d}")
    if new_p is not None:
        cli.append(f"set p_{axis} = {new_p}")

    detail = (f"PID error 在 30–90 Hz 有显著能量（占比 {ratio * 100:.0f}%，峰值约 {peak:.0f} Hz），"
              "是螺旋桨洗流/桨尖涡导致的典型振荡。")
    if new_d is not None:
        detail += f" 当前 d_{axis}={cur_d}，建议提高到 {new_d}。"
    findings.append(_finding(
        axis, "propwash", severity,
        f"{axis} 轴 propwash 振荡", detail,
        "提高 D 增益约 12%（或略微降 P）以抑制洗流振荡。",
        cli,
    ))


def _rule_noise_vs_throttle(ax: Dict, pid: Dict, sc, findings: List):
    evt = ax["error_vs_throttle"]
    thr = evt["throttle"]
    err = evt["mean_abs_error"]
    pts = [(t, e) for t, e in zip(thr, err) if e is not None]
    if len(pts) < 4:
        return
    t = np.array([p[0] for p in pts])
    e = np.array([p[1] for p in pts])
    slope = float(np.polyfit(t, e, 1)[0])
    if slope > 25.0:
        axis = ax["label"]
        cur_d = pid[axis]["d"]
        cli = []
        new_d = _adjust(cur_d, -10)
        new_lpf = _adjust(sc.gyro_lpf2_static_hz, -20) or _adjust(sc.gyro_lpf1_static_hz, -20)
        if new_d is not None:
            cli.append(f"set d_{axis} = {new_d}")
        if new_lpf is not None:
            if sc.gyro_lpf2_static_hz > 0:
                cli.append(f"set gyro_lpf2_static_hz = {new_lpf}")
            else:
                cli.append(f"set gyro_lpf1_static_hz = {new_lpf}")
        findings.append(_finding(
            axis, "noise_throttle", "medium",
            f"{axis} 轴噪声随油门升高",
            f"mean |error| 随油门明显上升（斜率 {slope:.0f} deg/s per 满油门），"
            "指向电机/桨叶不平衡导致的振动噪声。",
            "检查桨叶平衡与电机轴承；可增强 gyro 低通滤波或适度降低 D。",
            cli,
        ))


# ---------------------------------------------------------------------------
# Global rules
# ---------------------------------------------------------------------------

def _rule_gyro_hf_noise(parser: FlightLogParser, sc, findings: List):
    names = parser.frame_defs[ord("I")].field_names
    for axis in range(3):
        name = f"gyroADC[{axis}]"
        if name not in names:
            continue
        freq, psd, _ = power_spectrum(parser, name, max_freq=200.0)
        if freq.size == 0:
            continue
        high = _band_energy(freq, psd, 100.0, 150.0)
        noise = _band_energy(freq, psd, NOISE_BAND[0], NOISE_BAND[1])
        if noise <= 0:
            continue
        ratio = high / noise
        if ratio > 0.25:
            cli = []
            if sc.gyro_lpf2_static_hz > 0:
                new = _adjust(sc.gyro_lpf2_static_hz, -20)
                if new is not None:
                    cli.append(f"set gyro_lpf2_static_hz = {new}")
            elif sc.gyro_lpf1_static_hz > 0:
                new = _adjust(sc.gyro_lpf1_static_hz, -20)
                if new is not None:
                    cli.append(f"set gyro_lpf1_static_hz = {new}")
            findings.append(_finding(
                AXIS_LABELS[axis], "gyro_hf_noise", "medium",
                f"{AXIS_LABELS[axis]} 轴 gyro 高频噪声",
                f"gyro 频谱 >100 Hz 能量占比 {ratio * 100:.0f}%，存在明显高频噪声（电机或机架振动）。",
                "增强 gyro 低通滤波（降低截止频率约 20%），或降低 D。",
                cli,
            ))


def _rule_dterm_hf(parser: FlightLogParser, sc, pid: Dict, findings: List):
    names = parser.frame_defs[ord("I")].field_names
    for axis in range(2):
        name = f"axisD[{axis}]"
        if name not in names:
            continue
        freq, psd, _ = power_spectrum(parser, name, max_freq=200.0)
        if freq.size == 0:
            continue
        high = _band_energy(freq, psd, 100.0, 200.0)
        total = _band_energy(freq, psd, 10.0, 200.0)
        if total <= 0:
            continue
        ratio = high / total
        if ratio > 0.30:
            label = AXIS_LABELS[axis]
            cur_d = pid[label]["d"]
            cli = []
            new_d = _adjust(cur_d, -15)
            new_lpf = _adjust(sc.dterm_lpf2_static_hz, -20) or _adjust(sc.dterm_lpf1_static_hz, -20)
            if new_d is not None:
                cli.append(f"set d_{label} = {new_d}")
            if new_lpf is not None:
                if sc.dterm_lpf2_static_hz > 0:
                    cli.append(f"set dterm_lpf2_static_hz = {new_lpf}")
                else:
                    cli.append(f"set dterm_lpf1_static_hz = {new_lpf}")
            findings.append(_finding(
                label, "dterm_hf", "medium",
                f"{label} 轴 D-term 高频噪声",
                f"D-term 输出 >100 Hz 能量占比 {ratio * 100:.0f}%，D 正在放大高频噪声。当前 d_{label}={cur_d}。",
                "降低 D 增益约 15%，或增强 D-term 低通滤波。",
                cli,
            ))


def _rule_motor_saturation(parser: FlightLogParser, sc, pid: Dict, findings: List):
    names = parser.frame_defs[ord("I")].field_names
    high = sc.motor_output_high
    if high <= 0:
        return
    motor_max = 0.0
    for axis in range(4):
        name = f"motor[{axis}]"
        if name in names:
            v = _field_values(parser, name)
            if v.size:
                motor_max = max(motor_max, float(v.max()))
    if motor_max >= high * 0.98:
        cli = []
        for axis in ("roll", "pitch"):
            cur_p = pid[axis]["p"]
            new_p = _adjust(cur_p, -10)
            if new_p is not None:
                cli.append(f"set p_{axis} = {new_p}")
        findings.append(_finding(
            None, "motor_saturation", "high",
            "电机输出接近饱和",
            f"电机输出峰值 {motor_max:.0f} 接近上限 {high}，动力裕量不足，P 增益过高会加剧饱和/发热。"
            f"当前 p_roll={pid['roll']['p']}、p_pitch={pid['pitch']['p']}。",
            "降低 roll/pitch 的 P 增益约 10%；确认电机/桨叶匹配，必要时降低 maxthrottle。",
            cli,
        ))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def diagnose(parser: FlightLogParser) -> Dict:
    """Run all diagnostic rules and return severity-ranked findings with
    concrete CLI target values."""
    findings: List[Dict] = []
    sc = parser.sys_config
    pid = _pid_map(sc)

    pe = pid_error_analysis(parser)
    for ax in pe["axes"]:
        _rule_propwash(ax, pid, findings)
        _rule_noise_vs_throttle(ax, pid, sc, findings)

    _rule_gyro_hf_noise(parser, sc, findings)
    _rule_dterm_hf(parser, sc, pid, findings)
    _rule_motor_saturation(parser, sc, pid, findings)

    findings.sort(key=lambda f: _SEV_RANK.get(f["severity"], 9))
    return {"sample_rate": pe["sample_rate"], "findings": findings}
