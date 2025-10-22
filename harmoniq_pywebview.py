"""Launch the Harmoniq interface inside a PyWebview window.

This script loads the pre-built Harmoniq frontend bundle directly from the
filesystem so it can be packaged into a standalone desktop executable. The
runtime also ensures a ``music`` workspace exists alongside the executable so
captured loops or bundled stems can be stored in a predictable location.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import webview  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "PyWebview is not installed. Install it with `pip install pywebview`."
    ) from exc

ROOT = Path(__file__).resolve().parent
STAGED_DIST_NAME = "harmoniq_dist"
MUSIC_DIR_NAME = "music"


def _candidate_paths() -> list[Path]:
    """Return possible locations for the Harmoniq dist directory."""
    candidates: list[Path] = []
    # When running from a PyInstaller bundle ``_MEIPASS`` points at the
    # temporary extraction directory where we copy the static assets.
    base = getattr(sys, "_MEIPASS", None)
    if base is not None:
        candidates.append(Path(base) / STAGED_DIST_NAME)

    candidates.extend(
        [
            ROOT / "harmoniq" / "dist",
            ROOT / "build" / "harmoniq_pywebview" / "staging" / STAGED_DIST_NAME,
        ]
    )
    return candidates


def _candidate_music_paths() -> list[Path]:
    """Return possible locations for the packaged music workspace."""

    candidates: list[Path] = []
    base = getattr(sys, "_MEIPASS", None)
    if base is not None:
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / MUSIC_DIR_NAME)
        candidates.append(Path(base) / MUSIC_DIR_NAME)

    candidates.extend(
        [
            ROOT / MUSIC_DIR_NAME,
            Path.cwd() / MUSIC_DIR_NAME,
        ]
    )

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        unique.append(candidate)
        seen.add(candidate)
    return unique


def ensure_music_directory() -> Path:
    """Ensure the runtime music directory exists and return its path."""

    candidates = _candidate_music_paths()
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    target = candidates[0]
    target.mkdir(parents=True, exist_ok=True)
    return target


def locate_dist_directory() -> Path:
    """Return the path to the Harmoniq production bundle."""
    for candidate in _candidate_paths():
        if candidate.exists():
            return candidate
    raise SystemExit(
        "Harmoniq dist directory not found. Run `npm run build` inside the "
        "harmoniq folder before launching the PyWebview app."
    )


def main() -> None:
    """Create and run the Harmoniq PyWebview window."""

    music_dir = ensure_music_directory()
    os.environ.setdefault("HARMONIQ_MUSIC_DIR", str(music_dir))

    dist_dir = locate_dist_directory()
    index_file = dist_dir / "index.html"
    if not index_file.exists():
        raise SystemExit(
            "Harmoniq build is incomplete. Could not locate index.html at "
            f"{index_file}."
        )

    # Use a file URI so the bundle can reference its asset manifest.
    window_url = index_file.resolve().as_uri()
    webview.create_window("Harmoniq", window_url, width=1280, height=900)
    webview.start()


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
