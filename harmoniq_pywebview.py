"""Launch the Harmoniq interface inside a PyWebview window.

This script loads the pre-built Harmoniq frontend bundle directly from the
filesystem so it can be packaged into a standalone desktop executable.
"""
from __future__ import annotations

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
