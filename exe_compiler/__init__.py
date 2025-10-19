"""Utilities for building a standalone executable using PyInstaller.

This module provides a small command line interface that ensures PyInstaller
is available, downloads third-party dependencies into a local directory, and
invokes PyInstaller to build an executable. The goal is to make it easy to
produce distributable binaries without relying on globally installed tools or
network access during the build step.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence


DEFAULT_BUILD_ROOT = Path("build")
DEPS_DIRNAME = "deps"
DIST_DIRNAME = "dist"
WORK_DIRNAME = "pyinstaller"
SPEC_DIRNAME = "spec"

CommandRunner = Callable[[Sequence[str]], None]
ModuleChecker = Callable[[str], bool]


def module_available(module_name: str) -> bool:
    """Return True when *module_name* can be imported."""

    return importlib.util.find_spec(module_name) is not None


def run_command(command: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    """Execute *command* and raise :class:`RuntimeError` on failure."""

    result = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        message = os.linesep.join(
            [
                "Command failed:",
                " ".join(command),
                "Output:",
                result.stdout.strip(),
            ]
        )
        raise RuntimeError(message)


def ensure_pyinstaller(
    *,
    installer: Optional[Sequence[str]] = None,
    command_runner: Optional[CommandRunner] = None,
    module_checker: ModuleChecker = module_available,
) -> None:
    """Ensure PyInstaller is installed before building."""

    if module_checker("PyInstaller"):
        return

    runner = command_runner or run_command
    install_command = list(installer) if installer is not None else [
        sys.executable,
        "-m",
        "pip",
        "install",
        "pyinstaller",
    ]
    runner(install_command)


def download_dependencies(
    requirements: Path,
    destination: Path,
    *,
    command_runner: Optional[CommandRunner] = None,
) -> None:
    """Download dependencies listed in *requirements* to *destination*."""

    if not requirements.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements}")

    destination.mkdir(parents=True, exist_ok=True)
    runner = command_runner or run_command
    runner(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "-r",
            str(requirements),
            "--dest",
            str(destination),
        ]
    )


def install_requirements(
    requirements: Path,
    *,
    downloads: Optional[Path] = None,
    command_runner: Optional[CommandRunner] = None,
) -> None:
    """Install dependencies from *requirements* using optional downloads."""

    if not requirements.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements}")

    runner = command_runner or run_command
    command: List[str] = [sys.executable, "-m", "pip", "install"]
    if downloads is not None:
        command.extend(["--no-index", f"--find-links={downloads}"])
    command.extend(["-r", str(requirements)])
    runner(command)


def build_with_pyinstaller(
    entry_script: Path,
    *,
    dist_dir: Path,
    work_dir: Path,
    spec_dir: Path,
    name: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    windowed: bool = False,
    command_runner: Optional[CommandRunner] = None,
) -> None:
    """Invoke PyInstaller with sensible defaults for this repository."""

    if not entry_script.exists():
        raise FileNotFoundError(f"Entry script not found: {entry_script}")

    dist_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    command: List[str] = ["pyinstaller", str(entry_script), "--onefile"]
    command.extend(["--distpath", str(dist_dir)])
    command.extend(["--workpath", str(work_dir)])
    command.extend(["--specpath", str(spec_dir)])
    if name:
        command.extend(["--name", name])
    if extra_args:
        command.extend(list(extra_args))
    if windowed:
        command.append("--noconsole")

    runner = command_runner or run_command
    runner(command)


def build_executable(
    entry_script: Path,
    *,
    requirements: Optional[Path] = None,
    build_root: Optional[Path] = None,
    name: Optional[str] = None,
    extra_pyinstaller_args: Optional[Sequence[str]] = None,
    windowed: bool = False,
    command_runner: Optional[CommandRunner] = None,
    module_checker: ModuleChecker = module_available,
) -> Path:
    """Build an executable for *entry_script* and return the output path."""

    runner = command_runner or run_command
    ensure_pyinstaller(command_runner=runner, module_checker=module_checker)

    root = build_root or DEFAULT_BUILD_ROOT
    deps_dir = root / DEPS_DIRNAME
    dist_dir = root / DIST_DIRNAME
    work_dir = root / WORK_DIRNAME
    spec_dir = root / SPEC_DIRNAME

    requirements_path = requirements
    if requirements_path is None:
        default_req = entry_script.parent / "requirements.txt"
        if default_req.exists():
            requirements_path = default_req
        else:
            repo_req = Path("requirements.txt")
            if repo_req.exists():
                requirements_path = repo_req

    if requirements_path is not None:
        download_dependencies(requirements_path, deps_dir, command_runner=runner)
        install_requirements(
            requirements_path, downloads=deps_dir, command_runner=runner
        )

    build_with_pyinstaller(
        entry_script,
        dist_dir=dist_dir,
        work_dir=work_dir,
        spec_dir=spec_dir,
        name=name,
        extra_args=extra_pyinstaller_args,
        windowed=windowed,
        command_runner=runner,
    )

    output_name = name or entry_script.stem
    if sys.platform.startswith("win"):
        output_name += ".exe"
    return dist_dir / output_name


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the exe compiler."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "entry",
        nargs="?",
        default="pywebview_app.py",
        help="The Python script to package.",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        help="Optional requirements file to download and install before building.",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=DEFAULT_BUILD_ROOT,
        help="Directory used for downloads and PyInstaller artefacts.",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Explicit name for the generated executable.",
    )
    parser.add_argument(
        "--pyinstaller-arg",
        dest="pyinstaller_args",
        action="append",
        default=None,
        help="Extra arguments forwarded to PyInstaller (can be repeated).",
    )
    parser.add_argument(
        "--windowed",
        dest="windowed",
        action="store_true",
        help="Build the executable without a console window (PyWebview-style).",
    )
    parser.add_argument(
        "--console",
        dest="windowed",
        action="store_false",
        help="Keep a console window attached to the executable.",
    )
    parser.set_defaults(windowed=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Path:
    """Entry point for the command line interface."""

    args = parse_args(argv)
    entry_path = Path(args.entry).resolve()
    requirements = args.requirements
    if requirements is not None:
        requirements = requirements.resolve()
    build_root = Path(args.build_root).resolve()
    default_windowed = (
        args.windowed
        if args.windowed is not None
        else entry_path.name.lower().startswith("pywebview")
    )

    output = build_executable(
        entry_path,
        requirements=requirements,
        build_root=build_root,
        name=args.name,
        extra_pyinstaller_args=args.pyinstaller_args,
        windowed=default_windowed,
    )
    print(f"Executable created at: {output}")
    return output


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
