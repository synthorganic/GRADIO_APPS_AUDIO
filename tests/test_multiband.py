import types
import sys
import numpy as np

# Stub optional heavy dependencies before importing module
gradio_stub = types.ModuleType('gradio')
gradio_stub.Error = Exception
sys.modules.setdefault('gradio', gradio_stub)

audiocraft_stub = types.ModuleType('audiocraft')
sys.modules.setdefault('audiocraft', audiocraft_stub)

data_stub = types.ModuleType('audiocraft.data')
sys.modules.setdefault('audiocraft.data', data_stub)

audio_utils_stub = types.ModuleType('audiocraft.data.audio_utils')
audio_utils_stub.convert_audio = lambda x, *a, **k: x
sys.modules.setdefault('audiocraft.data.audio_utils', audio_utils_stub)

audio_stub = types.ModuleType('audiocraft.data.audio')
audio_stub.audio_write = lambda *a, **k: None
sys.modules.setdefault('audiocraft.data.audio', audio_stub)

models_stub = types.ModuleType('audiocraft.models')
class _Dummy:
    @staticmethod
    def get_pretrained(*a, **k):
        return None
models_stub.MusicGen = _Dummy
models_stub.MultiBandDiffusion = _Dummy
models_stub.AudioGen = _Dummy
sys.modules.setdefault('audiocraft.models', models_stub)

import musicgen_stems_continue2 as msc2


def _sine_wave(sr=44100, dur=0.1):
    t = np.linspace(0, dur, int(sr*dur), False)
    return np.sin(2*np.pi*440*t).astype(np.float32)


def test_multiband_compressor_alters_signal():
    audio = _sine_wave()
    out = msc2._apply_stem_fx(audio.copy(), 44100, 0.0, 0.0, 0.0, 0.0, None, 120.0, 0.0, 1.0)
    assert out.shape == audio.shape
    # effect should modify waveform
    assert not np.allclose(out, audio)


def test_no_effect_returns_input():
    audio = _sine_wave()
    out = msc2._apply_stem_fx(audio.copy(), 44100, 0.0, 0.0, 0.0, 0.0, None, 120.0, 0.0, 0.0)
    assert np.allclose(out, audio)
