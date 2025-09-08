import os
import random

try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    librosa = None

WORDS = [
    "ambient", "bouncy", "calm", "digital", "echoing",
    "fuzzy", "groovy", "harsh", "intense", "jazzy",
    "light", "melodic", "noisy", "organic", "punchy",
    "quirky", "rhythmic", "smooth", "trembling", "warm",
]

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _five_words() -> str:
    """Return a random five word description."""
    return " ".join(random.sample(WORDS, 5))


def _detect_key_bpm(path: str) -> tuple[str, float]:
    """Best effort key and BPM detection using librosa."""
    if librosa is None or not os.path.exists(path):
        return "C", 120.0
    try:
        y, sr = librosa.load(path)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_idx = chroma.mean(axis=1).argmax()
        key = KEYS[int(key_idx) % 12]
        return key, float(tempo)
    except Exception:
        return "C", 120.0


def analyze(path: str) -> tuple[str, str, float]:
    """Return (description, key, bpm) for an audio file."""
    desc = _five_words()
    key, bpm = _detect_key_bpm(path)
    return desc, key, bpm

