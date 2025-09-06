import types
import sys

import torch
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies so that the module under test imports
# without requiring the real libraries.
# ---------------------------------------------------------------------------

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

from musicgen_stems_continue2 import _move_to_device

class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(1))
        # raw tensor not registered as buffer or parameter
        self.raw = torch.randn(1)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_move_to_device_moves_unregistered_tensors():
    m = Dummy().to("cpu")
    assert m.raw.device.type == "cpu"
    _move_to_device(m, torch.device("cuda:0"))
    assert m.raw.device.type == "cuda"
    # parameters should also be on cuda
    assert next(m.parameters()).device.type == "cuda"
