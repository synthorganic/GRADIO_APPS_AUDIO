"""Launch the Soniq DAW mixer interface inside a PyWebview window."""
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
STAGED_DIST_NAME = "soniq_dist"


def _candidate_paths() -> list[Path]:
    """Return possible locations for the Soniq build directory."""

    candidates: list[Path] = []
    base = getattr(sys, "_MEIPASS", None)
    if base is not None:
        candidates.append(Path(base) / STAGED_DIST_NAME)

    candidates.extend(
        [
            ROOT / "daw-mixer" / "dist",
            ROOT / "build" / "soniq_pywebview" / "staging" / STAGED_DIST_NAME,
        ]
    )
    return candidates


def locate_dist_directory() -> Path:
    """Return the path to the packaged Soniq frontend bundle."""

    for candidate in _candidate_paths():
        if candidate.exists():
            return candidate
    raise SystemExit(
        "Soniq dist directory not found. Run `npm run build` inside the "
        "daw-mixer folder before launching the PyWebview app."
    )


def main() -> None:
    """Create and run the Soniq PyWebview window."""

    dist_dir = locate_dist_directory()
    index_file = dist_dir / "index.html"
    if not index_file.exists():
        raise SystemExit(
            "Soniq build is incomplete. Could not locate index.html at "
            f"{index_file}."
        )

    window_url = index_file.resolve().as_uri()
    webview.create_window("Soniq", window_url, width=1440, height=900)
    webview.start()


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
