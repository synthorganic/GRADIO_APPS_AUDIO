import types
import sys
from pathlib import Path
import numpy as np

# --- dependency stubs (copied from test_master_simple) ---
try:  # pragma: no cover - optional runtime dep
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

gradio_stub = types.ModuleType("gradio")
gradio_stub.Error = type("Error", (Exception,), {})
sys.modules.setdefault("gradio", gradio_stub)

import musicgen_stems_continue2 as msc2


def test_apply_stem_fx_invokes_effects(monkeypatch):
    calls = []

    def mark(name):
        def _inner(*a, **k):
            calls.append(name)
            return a[0]

        return _inner

    monkeypatch.setattr(msc2, "_looper", mark("looper"))
    monkeypatch.setattr(msc2, "_glitch_audio", mark("glitch"))
    monkeypatch.setattr(msc2, "_rhythmic_gate", lambda a, sr, bpm, **kw: mark("rg")(a))
    monkeypatch.setattr(msc2, "_apply_pedalboard", mark("pedal"))
    monkeypatch.setattr(msc2, "_multiband_compress", mark("mb"))

    audio = np.zeros(100, dtype=np.float32)
    msc2._apply_stem_fx(
        audio,
        44100,
        reverb=0.1,
        dist=0.2,
        gate=0.3,
        glitch=0.4,
        rhythm_gate={"freq": "1/4", "dur": 0.5, "loc": "start", "pattern": "flat"},
        bpm=120,
        loop_measures=2.0,
        multiband_comp=True,
    )

    assert calls == ["looper", "glitch", "rg", "pedal", "mb"]
