"""REST API: upload / decode / query flight logs."""

import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Query

from .core.blackbox import parse_bbl, BlackboxError
from .core.blackbox.parser import FlightLogParser, build_summary
from .core.analyze import channel_series, field_unit
from .core import spectrum, diagnose

router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# In-memory log session store (small LRU; logs can be tens of MB)
# ---------------------------------------------------------------------------

MAX_SESSIONS = 4
_sessions: dict = {}  # id -> {"parser": FlightLogParser, "name": str, "ts": float}


def _store(parser: FlightLogParser, name: str) -> str:
    if len(_sessions) >= MAX_SESSIONS:
        oldest = min(_sessions, key=lambda k: _sessions[k]["ts"])
        del _sessions[oldest]
    sid = uuid.uuid4().hex[:12]
    _sessions[sid] = {"parser": parser, "name": name, "ts": time.time()}
    return sid


def _get(sid: str) -> FlightLogParser:
    entry = _sessions.get(sid)
    if entry is None:
        raise HTTPException(status_code=404, detail="log session not found (expired?)")
    entry["ts"] = time.time()
    return entry["parser"]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/decode")
async def decode(file: UploadFile = File(...)):
    """Upload a .bbl file, parse it, and return the log summary."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    try:
        logs = parse_bbl(data)
    except BlackboxError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    name = file.filename or "log.bbl"
    summaries = []
    for i, log in enumerate(logs):
        summary = build_summary(log)
        summary["name"] = name if len(logs) == 1 else f"{name} #{i + 1}"
        sid = _store(log, summary["name"])
        summary["id"] = sid
        summaries.append(summary)
    return {"logs": summaries}


@router.get("/decode/{sid}/summary")
def summary(sid: str):
    log = _get(sid)
    return build_summary(log)


@router.get("/decode/{sid}/channels")
def channels(
    sid: str,
    names: str = Query(..., description="comma-separated field names"),
    max_points: int = Query(4000, ge=2, le=20000),
):
    """Return time series for the requested channels (min/max downsampled).

    Response: {"channels": {"gyroADC[0]": {"unit": "deg/s", "data": [[t,v],...]}}}
    """
    log = _get(sid)
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    if not wanted:
        raise HTTPException(status_code=400, detail="no channel names given")

    available = set(log.frame_defs[ord("I")].field_names)
    missing = [w for w in wanted if w not in available]
    if missing:
        raise HTTPException(status_code=404, detail=f"fields not in log: {missing}")

    result = {}
    for name in wanted:
        t, v, unit = channel_series(log, name, max_points=max_points)
        if v is None:  # the time channel itself
            result[name] = {"unit": unit, "data": [[float(x), None] for x in t]}
        else:
            result[name] = {"unit": unit, "data": [[float(x), float(y)] for x, y in zip(t, v)]}
    return {"channels": result}


@router.get("/decode/{sid}/fields")
def fields(sid: str):
    """List all main-frame fields with their units."""
    log = _get(sid)
    names = log.frame_defs[ord("I")].field_names
    return {"fields": [{"name": n, "unit": field_unit(n)} for n in names]}


@router.get("/decode/{sid}/spectrum")
def spectrum_endpoint(
    sid: str,
    names: str = Query(..., description="comma-separated field names"),
    max_freq: float = Query(200.0, ge=10.0, le=1000.0),
    nperseg: int = Query(1024, ge=64, le=8192),
):
    """Welch PSD for the requested channels (for a spectrum plot)."""
    log = _get(sid)
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    if not wanted:
        raise HTTPException(status_code=400, detail="no channel names given")

    available = set(log.frame_defs[ord("I")].field_names)
    missing = [w for w in wanted if w not in available]
    if missing:
        raise HTTPException(status_code=404, detail=f"fields not in log: {missing}")

    result = {"sample_rate": 0.0, "series": {}}
    for name in wanted:
        freqs, psd, fs = spectrum.power_spectrum(log, name, max_freq=max_freq, nperseg=nperseg)
        result["sample_rate"] = fs
        result["series"][name] = {
            "unit": field_unit(name),
            "freq": freqs.tolist(),
            "psd": psd.tolist(),
        }
    result["motor"] = spectrum.motor_frequency(log)
    return result


@router.get("/decode/{sid}/spectrogram")
def spectrogram_endpoint(
    sid: str,
    name: str = Query(..., description="field name"),
    max_freq: float = Query(200.0, ge=10.0, le=1000.0),
    nperseg: int = Query(1024, ge=64, le=8192),
):
    """STFT spectrogram (waterfall) for a single field."""
    log = _get(sid)
    available = set(log.frame_defs[ord("I")].field_names)
    if name not in available:
        raise HTTPException(status_code=404, detail=f"field not in log: {name}")
    return spectrum.spectrogram(log, name, max_freq=max_freq, nperseg=nperseg)


@router.get("/decode/{sid}/pid-error")
def pid_error_endpoint(
    sid: str,
    max_freq: float = Query(200.0, ge=10.0, le=1000.0),
    nperseg: int = Query(1024, ge=64, le=8192),
):
    """PID error (setpoint - gyro) spectra and propwash metrics per axis."""
    log = _get(sid)
    return spectrum.pid_error_analysis(log, max_freq=max_freq, nperseg=nperseg)


@router.get("/decode/{sid}/diagnose")
def diagnose_endpoint(sid: str):
    """Run PID tuning diagnosis rules and return CLI suggestions."""
    log = _get(sid)
    return diagnose.diagnose(log)


@router.get("/decode/{sid}/step-response")
def step_response_endpoint(
    sid: str,
    threshold: float = Query(100.0, ge=10.0, le=1000.0),
    max_steps: int = Query(10, ge=1, le=50),
):
    """Detect setpoint step changes and return gyro response windows."""
    log = _get(sid)
    return spectrum.step_response_analysis(log, threshold=threshold, max_steps=max_steps)
