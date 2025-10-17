Backend Separation Service (FastAPI)

Endpoints
- POST /api/separate: multipart form with `file` (audio). Returns JSON with stems, measures, bpm, key. Saves stems as WAV under `backend/media/` and serves them at `/media/{filename}`.
- GET /api/health: status probe.

Run locally
1. Create venv and install requirements:
   python -m venv .venv
   .venv/Scripts/activate  # Windows
   pip install -r backend/requirements_server.txt
2. Start the server:
   uvicorn backend.server:app --host 127.0.0.1 --port 8001 --reload

Or, launch the desktop app via `pywebview_app.py`; it will attempt to auto-start the backend (if `uvicorn` is available) and the Vite dev server.

Notes
- This service performs fast, HPSS-based separation (harmonic/percussive + simple band splitting) as a baseline. It’s wired so the frontend can use actual stem URLs when available.
- For production-quality models (e.g., UVR/MDX/Demucs), swap the `separate_basic` implementation with proper model inference and write the resulting stems into `media/` before returning.
