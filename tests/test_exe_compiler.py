from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import exe_compiler


class FakeRunner:
    def __init__(self) -> None:
        self.commands: List[List[str]] = []

    def __call__(self, command):
        self.commands.append(list(command))


def test_ensure_pyinstaller_installs_when_missing(monkeypatch):
    runner = FakeRunner()

    monkeypatch.setattr(exe_compiler, "module_available", lambda name: False)

    exe_compiler.ensure_pyinstaller(command_runner=runner)

    assert runner.commands == [
        [sys.executable, "-m", "pip", "install", "pyinstaller"]
    ]


def test_build_executable_orchestrates_steps(tmp_path: Path):
    entry = tmp_path / "demo.py"
    entry.write_text("print('hello')\n")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("requests==2.31.0\n")

    runner = FakeRunner()
    output = exe_compiler.build_executable(
        entry,
        requirements=requirements,
        build_root=tmp_path / "build",
        name="demo-app",
        extra_pyinstaller_args=["--noconsole"],
        command_runner=runner,
        module_checker=lambda _: True,
    )

    expected_dist = tmp_path / "build" / exe_compiler.DIST_DIRNAME
    expected_output = expected_dist / (
        "demo-app.exe" if sys.platform.startswith("win") else "demo-app"
    )

    assert output == expected_output

    download_cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "-r",
        str(requirements),
        "--dest",
        str(tmp_path / "build" / exe_compiler.DEPS_DIRNAME),
    ]
    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        f"--find-links={tmp_path / 'build' / exe_compiler.DEPS_DIRNAME}",
        "-r",
        str(requirements),
    ]
    pyinstaller_cmd = [
        "pyinstaller",
        str(entry),
        "--onefile",
        "--distpath",
        str(expected_dist),
        "--workpath",
        str(tmp_path / "build" / exe_compiler.WORK_DIRNAME),
        "--specpath",
        str(tmp_path / "build" / exe_compiler.SPEC_DIRNAME),
        "--name",
        "demo-app",
        "--noconsole",
    ]

    assert runner.commands == [download_cmd, install_cmd, pyinstaller_cmd]


def test_build_executable_uses_default_requirements(tmp_path: Path):
    entry = tmp_path / "demo.py"
    entry.write_text("print('hello')\n")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("requests==2.31.0\n")

    runner = FakeRunner()
    exe_compiler.build_executable(
        entry,
        build_root=tmp_path / "build",
        command_runner=runner,
        module_checker=lambda _: True,
    )

    assert str(requirements) in runner.commands[0]
