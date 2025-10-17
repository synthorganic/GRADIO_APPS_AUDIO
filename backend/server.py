"""
FastAPI backend for audio separation and analysis.

Endpoints
- POST /api/separate: Accepts an audio file and returns stems + measures + tempo/key.
- GET  /api/health: Health probe.

This is a lightweight, CPU-friendly approximation using librosa HPSS and simple
filtering to derive stem tracks. It writes per-stem WAVs under ./backend/media
and serves them at /media/{filename} for the frontend to stream.

Run:
  uvicorn backend.server:app --host 127.0.0.1 --port 8001 --reload

Requirements:
  pip install -r backend/requirements_server.txt
"""
from __future__ import annotations

import io
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import librosa

ROOT = Path(__file__).resolve().parent
MEDIA_DIR = ROOT / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def hp_filter(y: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """Simple first-order high-pass via FFT masking."""
    n = len(y)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    Y = np.fft.rfft(y)
    mask = (freqs >= cutoff_hz).astype(float)
    return np.fft.irfft(Y * mask, n=n)


def lp_filter(y: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    n = len(y)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    Y = np.fft.rfft(y)
    mask = (freqs <= cutoff_hz).astype(float)
    return np.fft.irfft(Y * mask, n=n)


def derive_measures(duration: float, bpm: float) -> List[Dict[str, Any]]:
    beat_sec = 60.0 / max(bpm, 1e-6)
    measure = beat_sec * 4
    measures: List[Dict[str, Any]] = []
    t = 0.0
    while t < duration - 1e-3:
        measures.append(
            {
                "id": str(uuid.uuid4()),
                "start": float(t),
                "end": float(min(duration, t + measure)),
                "beatCount": 4,
                "isDownbeat": True,
            }
        )
        t += measure
    return measures


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    # Normalize lightly to avoid clipping
    peak = float(np.max(np.abs(audio))) or 1.0
    sf.write(str(path), (audio / peak * 0.98), sr)


def separate_basic(y: np.ndarray, sr: int) -> Dict[str, Tuple[np.ndarray, str]]:
    """Return basic stems using HPSS and simple bands.

    Keys are stem ids; value is (audio, label).
    """
    # Harmonic/Percussive
    y_harm, y_perc = librosa.effects.hpss(y)
    # Bass (low harmonic), Leads (mid-high harmonic)
    bass = lp_filter(y_harm, sr, 180.0)
    leads = y_harm - bass
    # Kicks (low percussive), Percussion (high percussive)
    kicks = lp_filter(y_perc, sr, 160.0)
    perc = y_perc - kicks
    # Vocals proxy (bandpass 300..6k on harmonic)
    voc_hi = hp_filter(y_harm, sr, 300.0)
    vocals = voc_hi - hp_filter(voc_hi, sr, 6000.0)
    full = y

    return {
        "full": (full, "Full Mix"),
        "vocals": (vocals, "Vocal"),
        "leads": (leads, "Leads"),
        "percussion": (perc, "High Drums"),
        "kicks": (kicks, "Kicks"),
        "bass": (bass, "Bassline"),
    }


def camelot_major_c_default() -> str:
    # Keep consistent with frontend expectation; C Major ≈ 8B typically, but
    # we use a stable default as key detection is out-of-scope here.
    return "8B"


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/separate")
def separate(file: UploadFile = File(...)) -> JSONResponse:
    data = file.file.read()
    y, sr = librosa.load(io.BytesIO(data), sr=44100, mono=True)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(max(60.0, min(180.0, tempo)))
    duration = float(len(y) / sr)

    stems = separate_basic(y, sr)

    # Write stems to disk and build JSON response
    out = []
    for stem_type, (audio, label) in stems.items():
        stem_id = uuid.uuid4().hex
        fname = f"{stem_id}.wav"
        fpath = MEDIA_DIR / fname
        write_wav(fpath, audio, sr)
        out.append(
            {
                "id": stem_id,
                "name": label,
                "type": stem_type,
                "color": "#9fe0d6",
                "url": f"/media/{fname}",
                "startOffset": 0,
                "duration": duration,
                "extractionModel": "HPSS-basic",
            }
        )

    measures = derive_measures(duration, tempo)
    key = camelot_major_c_default()
    payload = {"stems": out, "measures": measures, "bpm": tempo, "key": key}
    return JSONResponse(payload)

