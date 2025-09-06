import types
import sys
from pathlib import Path
import numpy as np
import subprocess as sp
import pytest

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - minimal torch stub
    class _DummyTensor:
        def __init__(self, arr=None):
            self.arr = arr

        def float(self):
            return self

        def t(self):
            return self
    nn_stub = types.ModuleType("torch.nn")
    nn_stub.functional = types.ModuleType("torch.nn.functional")
    torch = types.SimpleNamespace(
        Tensor=object,
        from_numpy=lambda x: _DummyTensor(x),
        nn=nn_stub,
        cuda=types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        device=lambda *a, **k: types.SimpleNamespace(type=(a[0] if a else "cpu")),
    )
    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.nn", nn_stub)
    sys.modules.setdefault("torch.nn.functional", nn_stub.functional)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub heavy optional dependencies so module imports without them.
gradio_stub = types.ModuleType("gradio")
gradio_stub.Error = type("Error", (Exception,), {})
sys.modules.setdefault("gradio", gradio_stub)

audiocraft_stub = types.ModuleType("audiocraft")
sys.modules.setdefault("audiocraft", audiocraft_stub)

data_stub = types.ModuleType("audiocraft.data")
sys.modules.setdefault("audiocraft.data", data_stub)

audio_utils_stub = types.ModuleType("audiocraft.data.audio_utils")
audio_utils_stub.convert_audio = lambda x, *a, **k: x
sys.modules.setdefault("audiocraft.data.audio_utils", audio_utils_stub)

audio_stub = types.ModuleType("audiocraft.data.audio")
audio_stub.audio_write = lambda *a, **k: None
sys.modules.setdefault("audiocraft.data.audio", audio_stub)

models_stub = types.ModuleType("audiocraft.models")
class _Dummy:
    @staticmethod
    def get_pretrained(*a, **k):
        return None
models_stub.MusicGen = _Dummy
models_stub.MultiBandDiffusion = _Dummy
models_stub.AudioGen = _Dummy
sys.modules.setdefault("audiocraft.models", models_stub)

import musicgen_stems_continue2 as msc2


def test_apply_stereo_space_no_sample_fmt(monkeypatch, tmp_path):
    cmds = []
    def fake_run(cmd, check):
        cmds.append(cmd)
        if len(cmds) == 1:
            raise sp.CalledProcessError(1, cmd)
    monkeypatch.setattr(msc2.sp, "run", fake_run)
    inp = tmp_path / "in.wav"
    inp.touch()
    outp = tmp_path / "out.wav"
    msc2._apply_stereo_space(inp, outp, sr=32000, width=1.5, pan=0.0)
    for cmd in cmds:
        assert "-sample_fmt" not in cmd
    assert len(cmds) == 2


def test_apply_bass_narrow_builds_filter(monkeypatch, tmp_path):
    cmds = []
    monkeypatch.setattr(msc2.sp, "run", lambda cmd, check: cmds.append(cmd))
    inp = tmp_path / "in.wav"
    inp.touch()
    outp = tmp_path / "out.wav"
    msc2._apply_bass_narrow(inp, outp, sr=32000, width=0.0)
    assert "lowpass" in cmds[0][cmds[0].index("-af") + 1]


def test_apply_frequency_cuts_uses_equalizer(monkeypatch, tmp_path):
    cmds = []
    monkeypatch.setattr(msc2.sp, "run", lambda cmd, check: cmds.append(cmd))
    inp = tmp_path / "in.wav"
    inp.touch()
    outp = tmp_path / "out.wav"
    gains = [0.0] * len(msc2.FREQ_BANDS)
    gains[0] = -3.0
    msc2._apply_frequency_cuts(inp, outp, sr=32000, gains=gains)
    assert "equalizer" in cmds[0][cmds[0].index("-af") + 1]


def test_master_simple_uses_default_reference(monkeypatch, tmp_path):
    ref_dir = Path("/references")
    ref_dir.mkdir(exist_ok=True)
    ref_file = ref_dir / "reference.wav"
    ref_file.touch()

    def dummy_match(target, reference, output):
        assert reference == str(ref_file)
        Path(output).touch()
    monkeypatch.setattr(msc2, "_matchering_match", dummy_match)
    monkeypatch.setattr(msc2, "_apply_bass_boost", lambda a, b, c, d: Path(b).touch())
    monkeypatch.setattr(msc2, "_apply_stereo_space", lambda a, b, c, width, pan: Path(b).touch())
    monkeypatch.setattr(msc2, "_apply_bass_narrow", lambda a, b, c, w: Path(b).touch())
    monkeypatch.setattr(msc2, "_apply_frequency_cuts", lambda a, b, c, g: Path(b).touch())
    monkeypatch.setattr(msc2, "MATCHERING_AVAILABLE", True)

    sr = 32000
    wav = np.zeros((sr,), dtype=np.float32)
    out = msc2._master_simple((sr, wav))
    assert Path(out).exists()
