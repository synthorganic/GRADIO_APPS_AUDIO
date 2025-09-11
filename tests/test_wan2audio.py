import logging
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wan2audio


def test_detect_key_bpm_parameter_error(tmp_path, caplog):
    audio = tmp_path / "bad.wav"
    audio.write_text("data")

    class DummyParameterError(Exception):
        pass

    dummy_librosa = SimpleNamespace(
        load=lambda path: (_ for _ in ()).throw(DummyParameterError("bad audio")),
        beat=SimpleNamespace(beat_track=lambda y, sr: (0.0, None)),
        feature=SimpleNamespace(chroma_cqt=lambda y, sr: [[0]]),
        util=SimpleNamespace(exceptions=SimpleNamespace(ParameterError=DummyParameterError)),
    )

    original_librosa = wan2audio.librosa
    original_param_error = wan2audio.ParameterError
    wan2audio.librosa = dummy_librosa
    wan2audio.ParameterError = DummyParameterError
    try:
        with caplog.at_level(logging.WARNING):
            key, bpm = wan2audio._detect_key_bpm(str(audio))
        assert (key, bpm) == ("C", 120.0)
        assert any("bad audio" in record.message for record in caplog.records)
    finally:
        wan2audio.librosa = original_librosa
        wan2audio.ParameterError = original_param_error


def test_detect_key_bpm_io_error(tmp_path, caplog):
    audio = tmp_path / "bad.wav"
    audio.write_text("data")

    class DummyParameterError(Exception):
        pass

    def raise_io_error(path):
        raise IOError("io problem")

    dummy_librosa = SimpleNamespace(
        load=raise_io_error,
        beat=SimpleNamespace(beat_track=lambda y, sr: (0.0, None)),
        feature=SimpleNamespace(chroma_cqt=lambda y, sr: [[0]]),
        util=SimpleNamespace(exceptions=SimpleNamespace(ParameterError=DummyParameterError)),
    )

    original_librosa = wan2audio.librosa
    original_param_error = wan2audio.ParameterError
    wan2audio.librosa = dummy_librosa
    wan2audio.ParameterError = DummyParameterError
    try:
        with caplog.at_level(logging.WARNING):
            key, bpm = wan2audio._detect_key_bpm(str(audio))
        assert (key, bpm) == ("C", 120.0)
        assert any("io problem" in record.message for record in caplog.records)
    finally:
        wan2audio.librosa = original_librosa
        wan2audio.ParameterError = original_param_error
