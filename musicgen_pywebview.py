"""Launch the MusicGen stems interface inside a PyWebview window."""
from __future__ import annotations

import multiprocessing
import time
import urllib.error
import urllib.request
from typing import Optional

try:
    import webview  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "PyWebview is not installed. Install it with `pip install pywebview`."
    ) from exc

HOST = "127.0.0.1"
DEFAULT_PORT = 7860
STARTUP_TIMEOUT = 120.0
POLL_INTERVAL = 0.5


def _launch_musicgen_server(host: str, port: int) -> None:
    """Run the MusicGen Gradio application on *host*:*port*."""

    import logging
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.INFO)

    from musicgen_stems_continue2 import TMP_DIR, ui_full

    ui_full(
        {
            "server_name": host,
            "server_port": port,
            "allowed_paths": [str(TMP_DIR)],
        }
    )


def _start_server(host: str, port: int) -> multiprocessing.Process:
    """Spawn the MusicGen server in a separate process."""

    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_launch_musicgen_server,
        args=(host, port),
        daemon=True,
    )
    process.start()
    return process


def _wait_for_server(url: str, timeout: float) -> bool:
    """Return True when *url* responds before *timeout* seconds elapse."""

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except (urllib.error.URLError, TimeoutError, ConnectionRefusedError, OSError):
            time.sleep(POLL_INTERVAL)
    return False


def _terminate_process(process: Optional[multiprocessing.Process]) -> None:
    """Gracefully terminate *process* if it is still running."""

    if process is None:
        return
    if not process.is_alive():
        return
    process.terminate()
    process.join(timeout=5)


def main() -> None:
    """Start the MusicGen Gradio server and display it inside PyWebview."""

    host = HOST
    port = DEFAULT_PORT
    server: Optional[multiprocessing.Process] = None

    try:
        server = _start_server(host, port)
    except Exception as exc:  # pragma: no cover - startup failure
        raise SystemExit(f"Failed to launch MusicGen server: {exc}") from exc

    app_url = f"http://{host}:{port}"
    if not _wait_for_server(app_url, STARTUP_TIMEOUT):
        _terminate_process(server)
        raise SystemExit("MusicGen server did not become ready in time.")

    window_title = "MusicGen Stems"
    webview.create_window(window_title, app_url, width=1280, height=900)

    try:
        webview.start()
    finally:
        _terminate_process(server)


if __name__ == "__main__":  # pragma: no cover - script entry
    multiprocessing.freeze_support()
    main()
