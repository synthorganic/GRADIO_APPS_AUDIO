import types
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import musicgen_stems_continue2 as msc2


def test_apply_gpus_moves_models(monkeypatch):
    calls = []

    def fake_move_musicgen(model, device):
        calls.append((model, device.type))
        return model

    def fake_move_to_device(obj, device, _seen=None):
        calls.append((obj, device.type))
        return obj

    monkeypatch.setattr(msc2, "_move_musicgen", fake_move_musicgen)
    monkeypatch.setattr(msc2, "_move_to_device", fake_move_to_device)
    monkeypatch.setattr(msc2.torch, "device", lambda d: types.SimpleNamespace(type=d))

    style = object()
    medium = object()
    large = object()
    audiogen = object()
    diffusion = types.SimpleNamespace(device="orig")

    msc2.STYLE_MODEL = style
    msc2.MEDIUM_MODEL = medium
    msc2.LARGE_MODEL = large
    msc2.AUDIOGEN_MODEL = audiogen
    msc2.STYLE_MBD = diffusion

    msc2._apply_gpus("cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:1")

    mapping = {id(obj): dev for obj, dev in calls}
    assert mapping[id(style)] == "cuda:0"
    assert mapping[id(medium)] == "cuda:1"
    assert mapping[id(large)] == "cuda:1"
    assert mapping[id(audiogen)] == "cuda:2"
    assert mapping[id(diffusion)] == "cuda:1"
    assert msc2.MEDIUM_DEVICE.type == "cuda:1"
    assert msc2.DIFFUSION_DEVICE.type == "cuda:1"
    assert msc2.STYLE_MBD.device.type == "cuda:1"
