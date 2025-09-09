"""Tests for the batch analyze and rename utility."""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub heavy optional dependencies before importing module under test.
gradio_stub = types.ModuleType("gradio")
gradio_stub.Error = type("Error", (Exception,), {})
sys.modules.setdefault("gradio", gradio_stub)

audiocraft_stub = types.ModuleType("audiocraft")
sys.modules.setdefault("audiocraft", audiocraft_stub)
data_stub = types.ModuleType("audiocraft.data")
sys.modules.setdefault("audiocraft.data", data_stub)
audio_stub = types.ModuleType("audiocraft.data.audio")
audio_stub.audio_write = lambda *a, **k: None
sys.modules.setdefault("audiocraft.data.audio", audio_stub)
audio_utils_stub = types.ModuleType("audiocraft.data.audio_utils")
audio_utils_stub.convert_audio = lambda x, *a, **k: x
sys.modules.setdefault("audiocraft.data.audio_utils", audio_utils_stub)
models_stub = types.ModuleType("audiocraft.models")
class _Dummy:
    @staticmethod
    def get_pretrained(*a, **k):
        return None
models_stub.MusicGen = models_stub.MultiBandDiffusion = models_stub.AudioGen = _Dummy
sys.modules.setdefault("audiocraft.models", models_stub)

from musicgen_stems_continue2 import analyze_and_rename_batch


def test_analyze_and_rename_batch(tmp_path):
    """Multiple files should be analyzed and renamed."""

    file1 = tmp_path / "sample1.wav"
    file2 = tmp_path / "sample2.wav"
    file1.write_bytes(b"")
    file2.write_bytes(b"")

    original_paths = [str(file1), str(file2)]
    results = analyze_and_rename_batch(original_paths)

    assert len(results) == 2

    for orig, res in zip(original_paths, results):
        assert len(res) == 4
        desc, key, bpm, new_path = res
        assert isinstance(desc, str)
        assert isinstance(key, str)
        assert isinstance(bpm, float)
        assert Path(new_path).exists()
        assert not Path(orig).exists()
