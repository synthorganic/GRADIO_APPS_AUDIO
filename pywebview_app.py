"""Launch the DAW Mixer interface inside a PyWebview window.

The script ensures the Vite development server for the React frontend is
running and then opens it inside a desktop window powered by PyWebview.
"""
from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

try:
    import webview  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "PyWebview is not installed. Install it with `pip install pywebview`."
    ) from exc

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "daw-mixer"
HOST = "127.0.0.1"
PORT = 5173
APP_URL = f"http://{HOST}:{PORT}"

# Backend (optional) — used for separation if available
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8001
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
DEV_COMMAND = [
    "npm",
    "run",
    "dev",
    "--",
    "--host",
    HOST,
    "--port",
    str(PORT),
]


def wait_for_server(url: str, timeout: float = 45.0) -> bool:
    """Return True when the given URL responds before the timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    return False


def start_dev_server() -> subprocess.Popen[str]:
    """Start the Vite development server."""
    if not FRONTEND_DIR.exists():
        raise SystemExit("daw-mixer frontend directory not found.")
    return subprocess.Popen(
        DEV_COMMAND,
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def start_backend_server() -> subprocess.Popen[str]:
    """Start the FastAPI backend server if available."""
    # Use uvicorn if installed; otherwise raise
    return subprocess.Popen(
        [
            "uvicorn",
            "backend.server:app",
            "--host",
            BACKEND_HOST,
            "--port",
            str(BACKEND_PORT),
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def ensure_dev_server() -> Tuple[Optional[subprocess.Popen[str]], bool]:
    """Ensure the dev server is running.

    Returns the process (or None if already running) and a boolean flag
    indicating whether the caller started it.
    """
    if wait_for_server(APP_URL, timeout=2.0):
        return None, False

    process = start_dev_server()
    if not wait_for_server(APP_URL):
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        raise RuntimeError("Vite development server did not start in time.")

    return process, True


def ensure_backend_server() -> Tuple[Optional[subprocess.Popen[str]], bool]:
    """Ensure the backend is running; start if not available."""
    if wait_for_server(f"{BACKEND_URL}/api/health", timeout=2.0):
        return None, False
    try:
        process = start_backend_server()
    except FileNotFoundError:
        # uvicorn not installed — backend optional
        return None, False
    if not wait_for_server(f"{BACKEND_URL}/api/health", timeout=20.0):
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        # optional, so don't crash app
        return None, False
    return process, True


def shutdown_process(process: Optional[subprocess.Popen[str]]) -> None:
    """Gracefully terminate a subprocess."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> None:
    """Entry point for launching the PyWebview app."""
    try:
        process, started = ensure_dev_server()
    except Exception as exc:  # pragma: no cover - startup path
        raise SystemExit(str(exc)) from exc

    cleanup_target: Optional[subprocess.Popen[str]] = process if started else None
    backend_proc, backend_started = ensure_backend_server()
    if backend_started:
        cleanup_target = cleanup_target  # keep primary ref

    def on_exit() -> None:
        shutdown_process(cleanup_target)
        shutdown_process(backend_proc)

    try:
        webview.create_window("DAW Mixer", APP_URL, width=1200, height=800)
        webview.start(on_exit=on_exit)
    finally:
        shutdown_process(cleanup_target)
        shutdown_process(backend_proc)


if __name__ == "__main__":
    main()
