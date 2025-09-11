#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===== Headers Legend =====
# [NEW]     brand new header I added
# [ALTERED] adapted since your last version
# [UNCHANGED] same intent as before

import argparse
import logging
import math
import os
import base64
import warnings
import uuid
from pathlib import Path
import subprocess as sp
import sys
import shutil
import json
import tempfile
import re
import numpy as np
import types
from typing import Iterable
from contextlib import contextmanager
from prompt_notebook import save_prompt, show_notebook, load_prompt

try:  # pragma: no cover - optional runtime dependency
    from scipy.signal import cheby1, lfilter
    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - allow import without scipy
    cheby1 = lfilter = None
    SCIPY_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    # Enable fast attention and TF32/FP16 matmuls when running on CUDA to
    # recover the throughput seen in the demo build.
    if torch.cuda.is_available():  # pragma: no cover - depends on runtime
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            if hasattr(torch.backends.cuda, "sdp_kernel"):
                torch.backends.cuda.sdp_kernel(
                    enable_flash_sdp=True,
                    enable_mem_efficient=True,
                    enable_math=False,
                )
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = True
        except Exception:
            pass
except Exception:  # pragma: no cover - allow import without torch
    torch = types.SimpleNamespace(
        nn=types.SimpleNamespace(Module=object, Parameter=lambda *a, **k: None),
        Tensor=object,
        device=lambda *a, **k: types.SimpleNamespace(type="cpu"),
        from_numpy=lambda a: types.SimpleNamespace(
            float=lambda: types.SimpleNamespace(t=lambda: a)
        ),
        cuda=types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
    )
    F = types.SimpleNamespace()
try:
    import gradio as gr
except Exception:  # pragma: no cover - allow import without gradio
    gr = types.SimpleNamespace(Error=Exception)

# ``audiocraft`` is a heavy optional dependency.  Importing this module should
# not fail if it is missing, so we provide minimal placeholders when the
# package cannot be imported.  Only a very small subset of the file is used by
# the tests and the rest of the application will gracefully error if the real
# implementation is required at runtime.
try:  # pragma: no cover - optional runtime dependency
    from audiocraft.data.audio_utils import convert_audio
    from audiocraft.data.audio import audio_write
    from audiocraft.models import MusicGen, MultiBandDiffusion, AudioGen
    AUDIOCRAFT_AVAILABLE = True
except Exception:  # pragma: no cover - allow import without audiocraft
    convert_audio = audio_write = MusicGen = MultiBandDiffusion = AudioGen = None
    AUDIOCRAFT_AVAILABLE = False

# ---------- Optional deps [UNCHANGED] ----------
try:
    import demucs.separate
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

try:
    import matchering as mg
    MATCHERING_AVAILABLE = True
except ImportError:
    mg = None  # ensure symbol exists for type checkers and wrappers
    MATCHERING_AVAILABLE = False

# numpy 1.24+ removed aliases like ``np.float`` that older libs still use
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

try:
    sys.path.append(str(Path(__file__).resolve().parent / "versatile_audio_super_resolution"))
    from audiosr import build_model as audiosr_build_model, super_resolution as audiosr_super_resolution
    AUDIOSR_AVAILABLE = True
except Exception:
    AUDIOSR_AVAILABLE = False

# Additional optional deps for stem combination / harmonization features
try:  # librosa for BPM detection + pitch operations
    import librosa  # type: ignore
    LIBROSA_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dep
    LIBROSA_AVAILABLE = False

try:  # pedalboard effects (reverb, distortion, gating)
    from pedalboard import Pedalboard, Reverb, Distortion, NoiseGate  # type: ignore
    PEDALBOARD_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dep
    PEDALBOARD_AVAILABLE = False

try:  # soundfile for harmonize output writing
    import soundfile as sf  # type: ignore
    SOUNDFILE_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dep
    SOUNDFILE_AVAILABLE = False

try:  # MIDI export for harmonization
    import mido  # type: ignore
    MIDO_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dep
    MIDO_AVAILABLE = False
 
try:  # matplotlib for waveform visualization
    import matplotlib.pyplot as plt  # type: ignore
    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dep
    MATPLOTLIB_AVAILABLE = False

try:  # Retrieval-based Voice Conversion (RVC) library
    from rvc import VoiceConverter  # type: ignore
    RVC_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dep
    VoiceConverter = None  # type: ignore
    RVC_AVAILABLE = False

try:  # WAN2Audio for sample analysis
    import wan2audio  # type: ignore
    WAN2AUDIO_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dep
    WAN2AUDIO_AVAILABLE = False

# ---------- Devices [ALTERED] ----------
# Allow overriding the default GPU placement via environment variables.  Each
# variable should be a string understood by ``torch.device`` such as
# ``"cuda:1"`` or ``"cpu"``.  When the variable is not provided we fall back
# to the original heuristic used by the script.

def _get_device(env_name: str, fallback: str) -> torch.device:
    return torch.device(os.environ.get(env_name, fallback))


# High‑VRAM GPUs 0 & 1 host the heavy generation models.  Smaller GPUs 2 & 3
# are reserved for diffusion/utility work so that section composer can always
# offload to MultiBandDiffusion without exhausting memory.  Users can override
# these placements with ``STYLE_DEVICE``, ``MEDIUM_DEVICE`` ... etc.
STYLE_DEVICE = _get_device("STYLE_DEVICE", "cuda:0")
MEDIUM_DEVICE = _get_device(
    "MEDIUM_DEVICE",
    "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0",
)
LARGE_DEVICE = STYLE_DEVICE
AUDIOGEN_DEVICE = _get_device("AUDIOGEN_DEVICE", str(MEDIUM_DEVICE))
DIFFUSION_DEVICE = _get_device(
    "DIFFUSION_DEVICE",
    (
        "cuda:3"
        if torch.cuda.is_available() and torch.cuda.device_count() > 3
        else (
            "cuda:1"
            if torch.cuda.is_available() and torch.cuda.device_count() > 1
            else "cpu"
        )
    ),
)
UTILITY_DEVICE = _get_device(
    "UTILITY_DEVICE",
    "cuda:3" if torch.cuda.is_available() and torch.cuda.device_count() > 3 else str(DIFFUSION_DEVICE),
)

# ``AudioSR`` is lightweight enough to duplicate across two GPUs.  The list of
# devices can also be overridden through ``AUDIOSR_DEVICES`` using a comma
# separated list of ``torch.device`` strings.
if "AUDIOSR_DEVICES" in os.environ:
    AUDIOSR_DEVICES = [torch.device(d.strip()) for d in os.environ["AUDIOSR_DEVICES"].split(",")]
else:
    AUDIOSR_DEVICES = (
        [torch.device("cuda:0"), torch.device("cuda:1")]
        if torch.cuda.is_available() and torch.cuda.device_count() > 1
        else [UTILITY_DEVICE]
    )
# First device retained for backward compatibility / logging.
AUDIOSR_DEVICE = AUDIOSR_DEVICES[0]

print(
    f"[Boot] STYLE: {STYLE_DEVICE} | MEDIUM: {MEDIUM_DEVICE} | "
    f"LARGE: {LARGE_DEVICE} | AUDIOGEN: {AUDIOGEN_DEVICE} | "
    f"DIFFUSION(MBD): {DIFFUSION_DEVICE} | UTILITY: {UTILITY_DEVICE} | "
    f"AUDIOSR: {AUDIOSR_DEVICES}"
)

# ---------- Constants & paths [ALTERED] ----------
TARGET_SR = 32000
TARGET_AC = 1
# Use a temporary directory within the repository so that generated files
# reside in a location automatically permitted by Gradio. This avoids
# InvalidPathError when returning files outside the working or system temp
# directories and makes the folder safe to expose via ``allowed_paths``.
TMP_DIR = (Path(__file__).resolve().parent / "outputs")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Persisted user settings (EQ, UI preferences, etc.) are stored as JSON within
# the temporary directory so that they survive application restarts.
SETTINGS_PATH = TMP_DIR / "settings.json"


# Local font embedding for offline usage. Warn if the file is missing so that
# missing or unreadable fonts surface during startup instead of failing
# silently when building the CSS.
FONT_PATH = Path(__file__).resolve().parent / "assets" / "nasalization.ttf"
FONT_BASE64 = ""
if FONT_PATH.exists():
    try:
        FONT_BASE64 = base64.b64encode(FONT_PATH.read_bytes()).decode("utf-8")
    except Exception as exc:  # pragma: no cover - log unexpected issues
        logging.warning("Could not read font file %s: %s", FONT_PATH, exc)
else:  # pragma: no cover - font file missing
    logging.warning("Font file not found: %s", FONT_PATH)

# Global font / theme overrides
CUSTOM_CSS = f"""
@font-face {{
    font-family: 'Nasalization';
    src: url(data:font/ttf;base64,{FONT_BASE64}) format('truetype');
}}
body, .gradio-container {{
    font-family: 'Nasalization', sans-serif;
    color: #E8E8E8;
    background-color: #475043;
}}
:root {{
    --primary-hue: 150;
    --color-primary: #434750;
    --color-secondary: #475043;
    --color-accent: #39FF14;
    --color-accent-soft: #39FF14;
    --color-background-primary: #475043;
    --color-background-secondary: #434750;
    --color-background-tertiary: #434750;
}}
.icon-btn {{
    width: 40px;
    height: 40px;
    background-color: var(--color-primary);
    background-size: 24px 24px;
    background-position: center;
    background-repeat: no-repeat;
    border: 1px solid var(--color-accent);
}}
.icon-btn:hover {{
    background-color: var(--color-accent);
}}
.icon-save {{
    background-image: url('file=assets/checkmark.png');
}}
.icon-view {{
    background-image: url('file=assets/light_looper.png');
}}
.icon-preview {{
    background-image: url('file=assets/light_render_preview_effects.png');
}}
.icon-harmonize {{
    background-image: url('file=assets/light_reverb.png');
}}
.icon-revert {{
    background-image: url('file=assets/light_revert_effects.png');
}}
button, .gr-button {{
    background-color: var(--color-accent) !important;
    border-color: var(--color-accent) !important;
    color: #000 !important;
}}
button:hover, .gr-button:hover {{
    background-color: var(--color-accent) !important;
    opacity: 0.9;
}}
::selection {{
    background: var(--color-accent);
    color: #000;
}}
mark, .highlight {{
    background: var(--color-accent);
    color: #000;
}}
#top-banner {{
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
    margin-left: 0;
    text-align: left;
}}
"""


def load_settings() -> dict:
    """Load saved UI settings if the settings file exists."""

    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


def save_settings(cfg: dict) -> None:
    """Persist ``cfg`` to :data:`SETTINGS_PATH`."""

    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)
    except Exception:
        pass


USER_SETTINGS = load_settings()

# Persisted model/sharding configuration pulled from settings
CUSTOM_SHARD_RAW = USER_SETTINGS.get("shard_devices", "")
CUSTOM_SHARD_DEVICES = [
    torch.device(d.strip()) for d in CUSTOM_SHARD_RAW.split(",") if d.strip()
]
MODEL_OPTIONS = USER_SETTINGS.get("model_options", ["Style", "AudioGen"])

# Note mapping and available musical scales for harmonization.  Values are
# sets of semitone numbers (C=0) that are considered in-key.
NOTE_TO_INT = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11,
}

def _notes_to_set(notes: str) -> set[int]:
    return {NOTE_TO_INT[n] for n in notes.split() if n in NOTE_TO_INT}

SCALE_NOTES = {
    "A Minor": _notes_to_set("A B C D E F G"),
    "C Minor": _notes_to_set("C D Eb F G Ab Bb"),
    "D# Minor": _notes_to_set("D# F F# G# A# B C#"),
    "E Minor": _notes_to_set("E F# G A B C D"),
    "F Minor": _notes_to_set("F G Ab Bb C Db Eb"),
    "G Minor": _notes_to_set("G A Bb C D Eb F"),
    "C Major": _notes_to_set("C D E F G A B"),
    "D Major": _notes_to_set("D E F# G A B C#"),
    "F# Minor": _notes_to_set("F# G# A B C# D E"),
    "G# Minor": _notes_to_set("G# A# B C# D# E F#"),
    "B Minor": _notes_to_set("B C# D E F# G A"),
    "D Dorian": _notes_to_set("D E F G A B C"),
    "E Phrygian": _notes_to_set("E F G A B C D"),
    "F Lydian": _notes_to_set("F G A B C D E"),
    "G Mixolydian": _notes_to_set("G A B C D E F"),
}

SCALE_NAMES = list(SCALE_NOTES.keys())

# ---------- Caches [UNCHANGED] ----------
STYLE_MODEL = None
STYLE_MBD = None
STYLE_USE_DIFFUSION = False
AUDIOGEN_MODEL = None
MEDIUM_MODEL = None
LARGE_MODEL = None
MELODY_MODEL = None
# Cache one ``AudioSR`` model per device and keep an index for load balancing
# across ``AUDIOSR_DEVICES``.
AUDIOSR_MODELS = {}
_AUDIOSR_NEXT_DEVICE = 0


def load_model_custom_shard(model_cls, devices):
    """Load ``model_cls`` on every device in ``devices``.

    This helper is a lightweight stand‑in for a real sharding implementation. It
    instantiates one model per device so that requests can be divided between
    them manually.  The function returns the list of models, which can then be
    passed to :func:`generate_sharded`.
    """

    models = []
    for dev in devices:
        model = model_cls() if callable(model_cls) else model_cls
        if hasattr(model, "to"):
            model = model.to(dev)
        models.append(model)
    return models


def generate_sharded(prompts, models):
    """Split ``prompts`` across ``models`` and combine the generated output.

    The same model architecture is expected to be loaded on every GPU.  Prompts
    are divided into equal chunks and dispatched to each model in turn.  Results
    are concatenated in the original order.  This keeps the implementation
    trivial while providing a way for advanced users to experiment with manual
    sharding strategies.
    """

    if not models:
        return []
    # Compute chunk size and dispatch work
    chunk = math.ceil(len(prompts) / len(models))
    outputs = []
    for i, model in enumerate(models):
        batch = prompts[i * chunk : (i + 1) * chunk]
        if not batch:
            continue
        if hasattr(model, "generate"):
            with _no_grad():
                gen = model.generate(batch)
        else:  # pragma: no cover - placeholder path
            gen = [None] * len(batch)
        outputs.extend(gen)
    return outputs

# ---------- FFmpeg noise control (for future waveform use) [UNCHANGED] ----------
_old_call = sp.call
def _call_nostderr(*args, **kwargs):
    kwargs["stderr"] = sp.DEVNULL
    kwargs["stdout"] = sp.DEVNULL
    return _old_call(*args, **kwargs)
sp.call = _call_nostderr

# ---------- Utils [NEW] ----------
def _ensure_2d(wav_t: torch.Tensor) -> torch.Tensor:
    return wav_t[None, :] if wav_t.dim() == 1 else wav_t

def _rms(x: torch.Tensor) -> float:
    x = x.float()
    return float(torch.sqrt(torch.clamp((x ** 2).mean(), min=1e-12)).item())


def _db_to_amp(db: float) -> float:
    """Convert decibel gain to linear amplitude."""
    return 10 ** (db / 20.0)


@contextmanager
def _no_grad():
    """Best effort context manager disabling gradient tracking."""
    if hasattr(torch, "inference_mode"):
        with torch.inference_mode():
            yield
    elif hasattr(torch, "no_grad"):
        with torch.no_grad():
            yield
    else:  # pragma: no cover - torch stub without no_grad
        yield


def _move_to_device(obj, device: torch.device, _seen: set[int] | None = None):
    """Recursively move ``obj`` and its tensors to ``device``.

    ``MultiBandDiffusion`` pulls in components that are not registered as
    ``nn.Module`` (e.g. raw ``Tensor`` codebooks inside the quantizer).  A
    plain ``.to(device)`` therefore leaves these tensors behind which later
    triggers *"Expected all tensors to be on the same device"* errors.  This
    helper walks through an arbitrary Python object and attempts to relocate
    every tensor it can find onto the requested device.

    ``_seen`` keeps track of already visited objects to avoid infinite
    recursion when objects reference themselves or share references.
    """

    if _seen is None:
        _seen = set()

    obj_id = id(obj)
    if obj_id in _seen:
        return obj
    _seen.add(obj_id)

    # Direct tensors: move and return early.
    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    # Modules: ``.to`` moves parameters/buffers but may leave raw tensors
    # hanging around.  We still traverse their attributes to catch any
    # unregistered tensors (e.g. quantizer codebooks in MultiBandDiffusion).
    if isinstance(obj, torch.nn.Module):
        obj.to(device)
        for name, val in vars(obj).items():
            if name in {"_parameters", "_buffers", "_modules"}:
                continue
            if isinstance(val, torch.Tensor):
                setattr(obj, name, val.to(device))
            else:
                _move_to_device(val, device, _seen)
        return obj

    # Containers: recurse over their items.
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            _move_to_device(item, device, _seen)
        return obj
    if isinstance(obj, dict):
        for key, val in obj.items():
            obj[key] = _move_to_device(val, device, _seen)
        return obj

    # Generic object: inspect attributes and move what we can. We only
    # consider attributes from ``__dict__`` to skip methods/properties that
    # might spawn new objects or recurse infinitely.
    if hasattr(obj, "__dict__"):
        for name, val in vars(obj).items():
            if isinstance(val, torch.Tensor):
                setattr(obj, name, val.to(device))
            else:
                _move_to_device(val, device, _seen)
    return obj


def _move_musicgen(model, device: torch.device):
    """Relocate a ``MusicGen`` model and all of its tensors."""
    if model is None:
        return
    _move_to_device(model, device)
    # ``MusicGen`` exposes ``set_device`` which takes care of moving
    # subordinate components such as the T5 text encoder.  Calling it here
    # prevents mismatched-device errors where parts of the conditioner remain
    # on an old GPU.
    try:
        model.set_device(device)
    except AttributeError:
        # Fallback for potential mocks or minimal stubs in tests.
        model.device = device

    # ``set_device`` on ``MusicGen`` does not always propagate the target
    # device to nested conditioner objects or to the lazily loaded HuggingFace
    # T5 model.  If the text conditioner is instantiated after ``set_device``
    # runs, it would default to CUDA:0, leading to ``Expected all tensors to be
    # on the same device`` errors during generation.  Explicitly update the
    # condition provider and any child conditioners here and move any existing
    # tensors they may already hold.
    try:  # ``model`` may not have an ``lm`` attribute in lightweight tests
        provider = model.lm.condition_provider
    except AttributeError:
        return

    # Update provider level device attribute
    if hasattr(provider, "device"):
        provider.device = device

    # Individual conditioners (e.g. text/T5) may have their own ``device``
    # attribute and internal tensors that need relocation.
    conds = getattr(provider, "conditioners", {})
    for cond in conds.values():
        if hasattr(cond, "device"):
            cond.device = device
        _move_to_device(cond, device)


def _offload_musicgen(model):
    """Push ``model`` back to CPU and clear the originating CUDA cache."""
    if model is None:
        return
    prev_dev = getattr(model, "device", None)
    _move_musicgen(model, torch.device("cpu"))
    if prev_dev is not None and prev_dev.type == "cuda":
        with torch.cuda.device(prev_dev):
            torch.cuda.empty_cache()


def _apply_gpus(style_d, mg_l, ag_d, asr_d, diff_d):
    """Reassign devices for already loaded models."""
    global STYLE_DEVICE, MEDIUM_DEVICE, LARGE_DEVICE, AUDIOGEN_DEVICE, AUDIOSR_DEVICES, AUDIOSR_DEVICE, DIFFUSION_DEVICE

    STYLE_DEVICE = torch.device(style_d)
    LARGE_DEVICE = torch.device(mg_l)
    MEDIUM_DEVICE = LARGE_DEVICE
    AUDIOGEN_DEVICE = torch.device(ag_d)
    AUDIOSR_DEVICES = [torch.device(asr_d)]
    AUDIOSR_DEVICE = AUDIOSR_DEVICES[0]
    DIFFUSION_DEVICE = torch.device(diff_d)

    _move_musicgen(STYLE_MODEL, STYLE_DEVICE)
    _move_musicgen(MEDIUM_MODEL, MEDIUM_DEVICE)
    _move_musicgen(LARGE_MODEL, LARGE_DEVICE)
    _move_musicgen(AUDIOGEN_MODEL, AUDIOGEN_DEVICE)
    if STYLE_MBD is not None:
        _move_to_device(STYLE_MBD, DIFFUSION_DEVICE)
        STYLE_MBD.device = DIFFUSION_DEVICE
    return "✅ GPU assignments updated"

def _prep_to_32k(audio_input, take_last_seconds: float | None = None, device: torch.device = UTILITY_DEVICE) -> torch.Tensor:
    """Return mono 32k tensor on device; optionally last N seconds only."""
    sr, wav_np = audio_input
    wav = torch.from_numpy(wav_np).float().t()             # (C, T)
    wav = _ensure_2d(wav)
    wav32 = convert_audio(wav, sr, TARGET_SR, TARGET_AC)   # (1, T32k)
    if take_last_seconds:
        tail = int(max(0.1, float(take_last_seconds)) * TARGET_SR)
        wav32 = wav32[..., -tail:]
    return wav32.to(device)

# ---------- Output post-processing + Robust WAV writer [NEW] ----------
def _postprocess_out(wav_ct: torch.Tensor, peak_ceiling_db: float = -1.0, trim_db: float = -3.0) -> torch.Tensor:
    """
    Enforce (C,T) float CPU; apply peak ceiling and user trim for headroom.
    """
    wav_ct = wav_ct.detach().cpu().float()
    if wav_ct.dim() == 1:
        wav_ct = wav_ct.unsqueeze(0)
    elif wav_ct.dim() != 2:
        raise gr.Error(f"Expected (C,T) waveform, got shape {tuple(wav_ct.shape)}")

    # Peak ceiling (true-peak proxy) to -1 dBTP by simple peak scale
    peak = wav_ct.abs().max().item()
    if peak > 0:
        ceiling = 10.0 ** (peak_ceiling_db / 20.0)  # ~0.891 for -1 dB
        scale = min(1.0, ceiling / peak)
        wav_ct = wav_ct * scale

    # Additional user trim (default -3 dB for safety)
    if trim_db != 0:
        wav_ct = wav_ct * (10.0 ** (trim_db / 20.0))

    return wav_ct

def _write_wav(wav_ct: torch.Tensor, sr: int, stem: str = "out", trim_db: float = -3.0, peak_db: float = -1.0) -> str:
    """
    Safely write a WAV:
      - post-process to predictable level (peak ceiling + trim)
      - write with strategy="peak" (no auto-boost) and add_suffix=False
      - atomic rename to final
      - verify existence and non-zero size
    """
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    wav_ct = _postprocess_out(wav_ct, peak_ceiling_db=peak_db, trim_db=trim_db)
    final_name = f"{stem}_{uuid.uuid4().hex}.wav"
    final_path = TMP_DIR / final_name
    tmp_path   = TMP_DIR / f".{final_name}.tmp"

    try:
        audio_write(
            str(tmp_path),
            wav_ct,
            sr,
            strategy="peak",      # <- predictable, no loudness boost
            add_suffix=False,     # <- prevent “.wav.wav”
        )
    except Exception as e:
        raise gr.Error(f"audio_write failed: {e}")

    # Verify tmp exists and is non-empty
    if not tmp_path.exists():
        raise gr.Error(f"Internal write error: temp file missing: {tmp_path}")
    sz = tmp_path.stat().st_size if tmp_path.exists() else 0
    if sz <= 0:
        raise gr.Error(f"Internal write error: temp file size is {sz} bytes: {tmp_path}")

    os.replace(tmp_path, final_path)

    # Verify final exists and non-empty
    if not final_path.exists():
        raise gr.Error(f"Internal write error: final file missing after rename: {final_path}")
    fsz = final_path.stat().st_size if final_path.exists() else 0
    if fsz <= 0:
        raise gr.Error(f"Internal write error: final file size is {fsz} bytes: {final_path}")

    print(f"[I/O] Wrote WAV: {final_path} ({fsz} bytes)")
    return str(final_path)

# ---------- AudioGen helpers (compat + crossfade) [ALTERED] ----------
def _extract_audio_batch(out):
    # Accepts: Tensor[B,C,T] OR (Tensor, ...) OR [Tensor, ...]
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for item in out:
            if isinstance(item, torch.Tensor):
                return item
    raise RuntimeError("Unexpected model output format – no Tensor found.")

def _crossfade_concat(src: torch.Tensor, gen: torch.Tensor, sr: int, xf_sec: float = 1.25) -> torch.Tensor:
    """
    src, gen: (C,T) on CPU; returns (C, T_src + T_gen - Nxf).
    Equal-power curves; slightly shorter default to reduce summed energy.
    """
    assert src.dim() == 2 and gen.dim() == 2, "Expected (C,T) tensors"
    C1, T1 = src.shape
    C2, T2 = gen.shape
    assert C1 == C2, "Channel mismatch between src and gen"
    if T1 == 0:
        return gen
    if T2 == 0:
        return src

    nxf = max(1, min(int(sr * float(xf_sec)), T1, T2))
    t = torch.linspace(0.0, 1.0, nxf)
    a = torch.cos(0.5 * math.pi * t)   # fades src 1→0
    b = torch.sin(0.5 * math.pi * t)   # fades gen 0→1
    a = a.view(1, -1)
    b = b.view(1, -1)

    src_keep = src[:, : T1 - nxf]
    mixed = src[:, T1 - nxf :] * a + gen[:, : nxf] * b
    out = torch.cat([src_keep, mixed, gen[:, nxf:]], dim=1)
    return out.contiguous()

# ---------- MusicGen scoring helpers [NEW] ----------

def _time_stretch_to_grid(wav: torch.Tensor, seconds: float, sr: int, bpm: float) -> torch.Tensor:
    """If close to a beat grid, stretch by <2% to align."""
    beat = 60.0 / max(1.0, float(bpm))
    target = round(float(seconds) / beat) * beat
    cur = wav.shape[-1] / sr
    if target <= 0:
        return wav
    diff = abs(cur - target) / target
    if diff <= 0.02:
        new_len = int(target * sr)
        wav = F.interpolate(wav.unsqueeze(0), size=new_len, mode="linear", align_corners=False).squeeze(0)
    return wav


def _score_candidate(wav: torch.Tensor, sr: int, bpm: float) -> float:
    """Lightweight heuristic scoring for auto-selection."""
    wav = wav.detach().cpu()
    rms = _rms(wav)
    peak = wav.abs().max().item()
    crest = peak / (rms + 1e-9)
    mix = max(0.0, 1.0 - abs(crest - 10.0) / 10.0)  # crest ~10 preferred
    spec = torch.fft.rfft(wav, dim=-1)
    freqs = torch.fft.rfftfreq(wav.shape[-1], 1.0 / sr)
    edges = [0, 200, 400, 800, 1600, 3200, 6400, 12800, sr / 2]
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            bands.append(spec[..., mask].abs().pow(2).mean())
    balance = torch.stack(bands)
    spectral = 1.0 - balance.std().item() / (balance.mean().item() + 1e-9)
    tempo_score = 1.0  # placeholder
    key_score = 1.0    # placeholder
    prompt_score = 1.0 # placeholder
    total = 0.3 * tempo_score + 0.2 * key_score + 0.2 * mix + 0.15 * spectral + 0.15 * prompt_score
    return float(total)


MAX_SECTIONS = 8
STYLE_SECTIONS = {"Intro", "Bed", "Break", "Outro"}


def add_section(n):
    n = int(n) + 1
    n = min(MAX_SECTIONS, n)
    updates = [gr.update(visible=i < n) for i in range(MAX_SECTIONS)]
    return [n] + updates


def remove_section(n):
    """Hide the last visible section row."""
    n = int(n) - 1
    n = max(1, n)
    updates = [gr.update(visible=i < n) for i in range(MAX_SECTIONS)]
    return [n] + updates


def add_queue_item(queue, item):
    """Append ``item`` to an in-memory queue list."""
    queue = list(queue or [])
    if item:
        queue.append(item)
    return queue, gr.update(choices=queue, value=item), ""


def delete_queue_item(queue, selected):
    """Remove ``selected`` item from the queue list."""
    queue = list(queue or [])
    if selected in queue:
        queue.remove(selected)
    return queue, gr.update(choices=queue, value=None)


def compose_sections(
    sections: list[dict],
    init_audio,
    bpm: float = 120.0,
    xf_beats: float = 1.0,
    decoder: str = "Default",
    out_trim_db: float = -3.0,
    candidates: int = 2,
): 
    """Compose sequential sections with model chaining."""
    if not sections:
        raise gr.Error("No sections provided.")
    # Start with all heavy models on CPU to free up GPU memory.
    _offload_style()
    _offload_musicgen(MEDIUM_MODEL)
    _offload_musicgen(LARGE_MODEL)

    sr = TARGET_SR
    xf_sec = float(xf_beats) * 60.0 / max(1.0, float(bpm))
    xf_sec = max(0.8, min(1.2, xf_sec))
    assembled = None
    if init_audio is not None:
        try:
            sr_in, data = init_audio
            wav = torch.tensor(data).float().T
            wav = convert_audio(wav, sr_in, sr, TARGET_AC)
            assembled = wav
        except Exception:
            pass
    active = None
    for sec in sections:
        if sec["type"] in STYLE_SECTIONS:
            style_load_model()
            if decoder == "MultiBand_Diffusion":
                style_load_diffusion()
            model = STYLE_MODEL
            device = STYLE_DEVICE
            key = "style"
        else:
            medium_load_model()
            model = MEDIUM_MODEL
            device = MEDIUM_DEVICE
            key = "medium"

        if key != active:
            if active == "style":
                _offload_style()
            elif active == "medium":
                _offload_musicgen(MEDIUM_MODEL)
            _move_musicgen(model, device)
            if key == "style" and decoder == "MultiBand_Diffusion" and STYLE_MBD is not None:
                _move_to_device(STYLE_MBD, DIFFUSION_DEVICE)
                STYLE_MBD.device = DIFFUSION_DEVICE
            torch.cuda.set_device(device)
            active = key
        else:
            torch.cuda.set_device(device)

        model.set_generation_params(duration=int(sec["length"]))
        best_wav = None
        best_score = -1e9
        for _ in range(int(candidates)):
            with _no_grad():
                out = model.generate([sec["prompt"]], return_tokens=(decoder == "MultiBand_Diffusion"))
            if decoder == "MultiBand_Diffusion" and STYLE_MBD is not None:
                tokens = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else out[0]
                tokens = tokens.to(STYLE_MBD.device)
                torch.cuda.set_device(STYLE_MBD.device)
                wav = STYLE_MBD.tokens_to_wav(tokens)[0]
            else:
                wav = out[0]
            wav = wav.detach().cpu().float()
            score = _score_candidate(wav, sr, bpm)
            if score > best_score:
                best_score = score
                best_wav = wav
        if best_wav is None:
            raise gr.Error("Section generation failed.")
        best_wav = _time_stretch_to_grid(best_wav, sec["length"], sr, bpm)
        assembled = best_wav if assembled is None else _crossfade_concat(assembled, best_wav, sr, xf_sec=xf_sec)
        torch.cuda.empty_cache()

    if active == "style":
        _offload_style()
    elif active == "medium":
        _offload_musicgen(MEDIUM_MODEL)

    return _write_wav(assembled, sr, stem="sections", trim_db=float(out_trim_db))


def compose_sections_ui(init_audio, count, *vals):
    bpm = vals[-4]
    xf_beats = vals[-3]
    decoder = vals[-2]
    out_trim_db = vals[-1]
    vals = vals[:-4]
    sections = []
    count = int(count)
    for i in range(count):
        typ = vals[3 * i]
        prompt = vals[3 * i + 1]
        length = vals[3 * i + 2]
        if prompt and length:
            sections.append({"type": typ, "prompt": prompt, "length": float(length)})
    return compose_sections(sections, init_audio, bpm, xf_beats, decoder, out_trim_db)

# ============================================================================
# SECTION COMBINER [NEW]
# ============================================================================

def _load_audio_file(path: str) -> torch.Tensor:
    """Load ``path`` as mono 32k tensor."""
    if not path:
        raise gr.Error("Internal error: missing audio path.")
    if LIBROSA_AVAILABLE:  # pragma: no branch - simple runtime check
        wav, sr = librosa.load(path, sr=None, mono=True)
        wav_t = torch.from_numpy(wav).unsqueeze(0)
    elif SOUNDFILE_AVAILABLE:
        wav, sr = sf.read(path)
        wav_t = torch.from_numpy(wav.T)
    else:  # pragma: no cover - optional dependency
        raise gr.Error("Please install librosa or soundfile to combine sections.")
    wav_t = _ensure_2d(wav_t)
    if sr != TARGET_SR or wav_t.shape[0] != TARGET_AC:
        wav_t = convert_audio(wav_t, sr, TARGET_SR, TARGET_AC)
    return wav_t


def section_generate_transition(model, intro_audio, prompt, bpm, beats, decoder, out_trim_db):
    """Generate a transition segment from ``intro_audio`` using ``model``."""
    duration = float(beats) * 60.0 / float(bpm)
    if model == "Style":
        return style_predict(
            prompt,
            intro_audio,
            duration=duration,
            decoder=decoder,
            out_trim_db=float(out_trim_db),
        )
    if model == "Melody-Large":
        return melody_generate_transition(intro_audio, prompt, duration, out_trim_db)
    # Default to AudioGen continuation
    lookback = min(6.0, duration)
    return audiogen_continuation(
        intro_audio,
        prompt,
        lookback_sec=lookback,
        duration=int(duration),
        out_trim_db=float(out_trim_db),
    )


def section_generate_and_combine(
    model,
    intro_audio,
    next_audio,
    prompt,
    bpm,
    beats,
    xf_beats,
    decoder,
    out_trim_db,
):
    """Generate transition and overlap with both input clips."""
    if intro_audio is None or next_audio is None:
        raise gr.Error("Please provide both input audio clips.")
    trans_path = section_generate_transition(
        model,
        intro_audio,
        prompt,
        bpm,
        beats,
        decoder,
        out_trim_db,
    )
    intro = _prep_to_32k(intro_audio).cpu()
    outro = _prep_to_32k(next_audio).cpu()
    transition = _load_audio_file(trans_path)
    # ``xf_beats`` represents the total desired overlap in beats.  Split the
    # value across both boundaries so that an 8‑beat setting behaves like a DJ
    # style 8‑beat transition (4 beats into the bridge and 4 beats out) rather
    # than two full 8‑beat fades that leave a gap between clips.
    xf_total = float(xf_beats) * 60.0 / float(bpm)
    half_xf = xf_total / 2.0
    assembled = _crossfade_concat(intro, transition, TARGET_SR, xf_sec=half_xf)
    assembled = _crossfade_concat(assembled, outro, TARGET_SR, xf_sec=half_xf)
    return _write_wav(
        assembled,
        TARGET_SR,
        stem="section_combined",
        trim_db=float(out_trim_db),
    )

# ============================================================================
# STYLE TAB (MusicGen-Style) [UNCHANGED intent; better writer]
# ============================================================================
def style_load_model():
    global STYLE_MODEL
    if STYLE_MODEL is None:
        print("[Style] Loading facebook/musicgen-style")
        STYLE_MODEL = MusicGen.get_pretrained("facebook/musicgen-style")
        _move_musicgen(STYLE_MODEL, torch.device("cpu"))
    
def style_load_diffusion():
    global STYLE_MBD
    if STYLE_MBD is None:
        print("[Style] Loading MultiBandDiffusion...")
        STYLE_MBD = MultiBandDiffusion.get_mbd_musicgen()
        # ``MultiBandDiffusion`` isn't a regular ``nn.Module`` so a naive
        # ``.to(device)`` call can leave internal tensors (e.g. quantizer
        # codebooks) on their original device.  Use the recursive helper to
        # ensure *everything* lives on ``STYLE_DEVICE``.
        _move_to_device(STYLE_MBD, torch.device("cpu"))
        STYLE_MBD.device = torch.device("cpu")


def _offload_style():
    _offload_musicgen(STYLE_MODEL)
    if STYLE_MBD is not None:
        prev_dev = getattr(STYLE_MBD, "device", None)
        _move_to_device(STYLE_MBD, torch.device("cpu"))
        STYLE_MBD.device = torch.device("cpu")
        if prev_dev is not None and prev_dev.type == "cuda":
            with torch.cuda.device(prev_dev):
                torch.cuda.empty_cache()


def medium_load_model():
    """Lazy-load facebook/musicgen-medium on MEDIUM_DEVICE."""
    global MEDIUM_MODEL
    if MEDIUM_MODEL is None:
        print("[Medium] Loading facebook/musicgen-medium")
        MEDIUM_MODEL = MusicGen.get_pretrained("facebook/musicgen-medium")
        _move_musicgen(MEDIUM_MODEL, torch.device("cpu"))
    return MEDIUM_MODEL


def large_load_model():
    """Lazy-load facebook/musicgen-large on LARGE_DEVICE."""
    global LARGE_MODEL
    if LARGE_MODEL is None:
        print("[Large] Loading facebook/musicgen-large")
        LARGE_MODEL = MusicGen.get_pretrained("facebook/musicgen-large")
        _move_musicgen(LARGE_MODEL, torch.device("cpu"))
    return LARGE_MODEL


def melody_load_model():
    """Lazy-load facebook/musicgen-melody-large on LARGE_DEVICE."""
    global MELODY_MODEL
    if MELODY_MODEL is None:
        print("[Melody] Loading facebook/musicgen-melody-large")
        MELODY_MODEL = MusicGen.get_pretrained("facebook/musicgen-melody")
        _move_musicgen(MELODY_MODEL, torch.device("cpu"))
    return MELODY_MODEL


def style_predict(text, melody, duration=10, topk=200, topp=50.0, temperature=1.0,
                  cfg_coef=3.0, double_cfg="Yes", cfg_coef_beta=5.0,
                  eval_q=3, excerpt_length=3.0, decoder="Default",
                  out_trim_db=-3.0):
    """Generate with MusicGen-Style, optional melody excerpt."""
    style_load_model()
    try:
        _move_musicgen(STYLE_MODEL, STYLE_DEVICE)
        STYLE_MODEL.set_generation_params(
            duration=int(duration),
            top_k=int(topk),
            top_p=float(topp),
            temperature=float(temperature),
            cfg_coef=float(cfg_coef),
            cfg_coef_beta=float(cfg_coef_beta) if double_cfg == "Yes" else None,
        )
        STYLE_MODEL.set_style_conditioner_params(
            eval_q=int(eval_q), excerpt_length=float(excerpt_length)
        )

        # Diffusion toggle
        global STYLE_USE_DIFFUSION
        if decoder == "MultiBand_Diffusion":
            STYLE_USE_DIFFUSION = True
            style_load_diffusion()
            if STYLE_MBD is not None:
                _move_to_device(STYLE_MBD, DIFFUSION_DEVICE)
                STYLE_MBD.device = DIFFUSION_DEVICE
        else:
            STYLE_USE_DIFFUSION = False

        melody_tensor = None
        if melody:
            sr, arr = melody
            mel = torch.from_numpy(arr).float().t().to(STYLE_DEVICE)
            mel = _ensure_2d(mel)
            mel = convert_audio(mel, sr, TARGET_SR, TARGET_AC).to(STYLE_DEVICE)
            melody_tensor = [mel]

        if melody_tensor:
            with _no_grad():
                outputs = STYLE_MODEL.generate_with_chroma(
                    descriptions=[text or "style generation"],
                    melody_wavs=melody_tensor,
                    melody_sample_rate=TARGET_SR,
                    return_tokens=STYLE_USE_DIFFUSION,
                )
        else:
            with _no_grad():
                outputs = STYLE_MODEL.generate(
                    [text or "style generation"], return_tokens=STYLE_USE_DIFFUSION
                )

        if STYLE_USE_DIFFUSION:
            tokens = outputs[1]
            if tokens.device != STYLE_MBD.device:
                tokens = tokens.to(STYLE_MBD.device)
            torch.cuda.set_device(STYLE_MBD.device)
            wavs = STYLE_MBD.tokens_to_wav(tokens)
            wav = wavs.detach().cpu().float()[0]  # (C,T)
            sr_out = TARGET_SR
        else:
            wav = outputs[0].detach().cpu().float()[0]
            sr_out = STYLE_MODEL.sample_rate

        return _write_wav(wav, sr_out, stem="style", trim_db=float(out_trim_db))
    finally:
        _offload_style()

# ============================================================================
# AUDIOGEN CONTINUATION TAB [ALTERED]
# ============================================================================
def audiogen_load_model(name: str = "facebook/audiogen-medium"):
    global AUDIOGEN_MODEL
    if AUDIOGEN_MODEL is None:
        print(f"[AudioGen] Loading {name} ...")
        AUDIOGEN_MODEL = AudioGen.get_pretrained(name)
    _move_musicgen(AUDIOGEN_MODEL, AUDIOGEN_DEVICE)
    return AUDIOGEN_MODEL

def audiogen_continuation(
    audio_input,
    text_prompt: str = "",
    lookback_sec: float = 6.0,
    duration: int = 12,
    topk: int = 200,
    topp: float = 50.0,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
    crossfade_sec: float = 1.25,   # slightly shorter to avoid hot sums
    out_trim_db: float = -3.0,
): 
    """Emulate continuation by generating fresh audio and crossfading onto the input tail."""
    if audio_input is None:
        raise gr.Error("Please provide an input audio clip.")

    prompt = text_prompt or "Continuation"
    lookback_sec = float(max(0.5, min(30.0, lookback_sec)))

    # 1) Prepare tail @32k mono on UTILITY_DEVICE (cuda:3 if available)
    tail = _prep_to_32k(audio_input, take_last_seconds=lookback_sec, device=UTILITY_DEVICE).cpu()  # (1,T)

    # 2) Load and configure AudioGen (cuda:0 per your layout)
    model = audiogen_load_model("facebook/audiogen-medium")
    try:
        model.set_generation_params(
            duration=int(duration),
            top_k=int(topk),
            top_p=float(topp),
            temperature=float(temperature),
            cfg_coef=float(cfg_coef),
        )

        # 3) Emulate continuation with crossfade
        print("[AudioGen] Generating new segment for crossfade continuation.")
        with _no_grad():
            gen_out = model.generate([prompt])             # -> Tensor[B,C,T] or similar
        gen_batch = _extract_audio_batch(gen_out)
        gen = gen_batch.detach().cpu().float()[0]      # (C,T)

        if _rms(gen) < 1e-6:
            raise gr.Error("Generated segment is near-silent. Try a richer prompt or higher temperature/top_k.")

        tail_cpu = tail.detach().cpu().float()
        gen_cpu  = gen.detach().cpu().float()

        xf = float(max(0.25, min(crossfade_sec, lookback_sec * 0.5, 3.0)))
        glued = _crossfade_concat(tail_cpu, gen_cpu, sr=TARGET_SR, xf_sec=xf)  # (C, T)

        return _write_wav(glued, TARGET_SR, stem="audiogen_emul", trim_db=float(out_trim_db))
    finally:
        _offload_musicgen(model)


def melody_continuation(
    audio_input,
    text_prompt: str = "",
    lookback_sec: float = 6.0,
    duration: int = 12,
    topk: int = 200,
    topp: float = 50.0,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
    crossfade_sec: float = 1.25,
    out_trim_db: float = -3.0,
):
    """Continuation using MusicGen Melody Large with crossfade."""
    if audio_input is None:
        raise gr.Error("Please provide an input audio clip.")

    prompt = text_prompt or "Continuation"
    lookback_sec = float(max(0.5, min(30.0, lookback_sec)))
    tail = _prep_to_32k(audio_input, take_last_seconds=lookback_sec, device=UTILITY_DEVICE).cpu()
    model = melody_load_model()
    try:
        _move_musicgen(model, LARGE_DEVICE)
        model.set_generation_params(
            duration=int(duration),
            top_k=int(topk),
            top_p=float(topp),
            temperature=float(temperature),
            cfg_coef=float(cfg_coef),
        )

        with _no_grad():
            gen_out = model.generate_with_chroma(
                [prompt], melody_wavs=[tail], melody_sample_rate=TARGET_SR
            )
        gen = _extract_audio_batch(gen_out).detach().cpu().float()[0]

        if _rms(gen) < 1e-6:
            raise gr.Error(
                "Generated segment is near-silent. Try a richer prompt or higher temperature/top_k."
            )

        tail_cpu = tail.detach().cpu().float()
        xf = float(max(0.25, min(crossfade_sec, lookback_sec * 0.5, 3.0)))
        glued = _crossfade_concat(tail_cpu, gen, sr=TARGET_SR, xf_sec=xf)
        return _write_wav(glued, TARGET_SR, stem="melody_emul", trim_db=float(out_trim_db))
    finally:
        _offload_musicgen(model)


def melody_generate_transition(intro_audio, prompt, duration, out_trim_db):
    """Generate a transition segment using MusicGen Melody Large."""
    if intro_audio is None:
        raise gr.Error("Please provide intro audio.")
    mel = _prep_to_32k(intro_audio).cpu()
    model = melody_load_model()
    try:
        _move_musicgen(model, LARGE_DEVICE)
        model.set_generation_params(duration=int(duration))
        with _no_grad():
            out = model.generate_with_chroma(
                [prompt], melody_wavs=[mel], melody_sample_rate=TARGET_SR
            )
        gen = _extract_audio_batch(out)[0]
        return _write_wav(gen, TARGET_SR, stem="melody_trans", trim_db=float(out_trim_db))
    finally:
        _offload_musicgen(model)


def continuation(
    model,
    audio_input,
    text_prompt="",
    lookback_sec=6.0,
    duration=12,
    topk=200,
    topp=50.0,
    temperature=1.0,
    cfg_coef=3.0,
    out_trim_db=-3.0,
):
    """Dispatch continuation to AudioGen or Melody model."""
    if model == "Melody-Large":
        return melody_continuation(
            audio_input,
            text_prompt,
            lookback_sec,
            duration,
            topk,
            topp,
            temperature,
            cfg_coef,
            out_trim_db=out_trim_db,
        )
    return audiogen_continuation(
        audio_input,
        text_prompt,
        lookback_sec,
        duration,
        topk,
        topp,
        temperature,
        cfg_coef,
        out_trim_db=out_trim_db,
    )

# ============================================================================
# STEMS (DEMUCS) TAB [UNCHANGED intent]
# ============================================================================
def separate_stems(audio_input):
    if not DEMUCS_AVAILABLE:
        raise gr.Error("Demucs not installed. `pip install demucs`")
    try:
        sr, wav_np = audio_input
    except Exception:
        raise gr.Error("Please provide an audio clip.")
    in_path = TMP_DIR / f"sep_{uuid.uuid4().hex}.wav"
    audio_write(str(in_path), torch.from_numpy(wav_np).float().t(), sr, add_suffix=False)

    out_dir = TMP_DIR / "separated"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer running on UTILITY_DEVICE cuda:3 when available
    if UTILITY_DEVICE.type == "cuda":
        demucs_args = ["-n", "htdemucs", "-d", "cuda", "-o", str(out_dir), str(in_path)]
    else:
        demucs_args = ["-n", "htdemucs", "-d", "cpu", "-o", str(out_dir), str(in_path)]

    demucs.separate.main(demucs_args)
    stem_dir = out_dir / "htdemucs" / in_path.stem
    return (
        str(stem_dir / "drums.wav"),
        str(stem_dir / "vocals.wav"),
        str(stem_dir / "bass.wav"),
        str(stem_dir / "other.wav"),
    )


# ============================================================================
# STEM COMBINER + HARMONIZER [NEW]
# ============================================================================

def _freq_to_midi(freq: float) -> float:
    return 69 + 12 * math.log2(freq / 440.0)


def _midi_to_freq(midi: float) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12))


def _build_scale_freqs(scale: str) -> list[float]:
    allowed = SCALE_NOTES.get(scale, SCALE_NOTES["C Major"])
    freqs = []
    for midi in range(21, 109):  # piano range
        if midi % 12 in allowed:
            freqs.append(_midi_to_freq(midi))
    return freqs


def _harmonic_distance(f_in: float, f_c: float) -> float:
    dist = 0.0
    for h in (1, 2, 3):
        target = h * f_in
        nearest = round(target / f_c) * f_c
        dist += abs(target - nearest)
    return dist


def _ftha_align(freq: float, scale: str) -> float:
    if freq <= 0:
        return freq
    candidates = _build_scale_freqs(scale)
    return min(candidates, key=lambda f_c: _harmonic_distance(freq, f_c))


def detect_bpm(src) -> float:
    """Return estimated BPM of a file or raw audio using librosa."""
    if not LIBROSA_AVAILABLE or src is None:  # pragma: no cover - optional
        return 0.0
    try:
        if isinstance(src, str):
            y, sr = librosa.load(src, sr=None, mono=True)
        elif isinstance(src, (list, tuple)) and len(src) == 2:
            sr, y = src
            if y.ndim > 1:
                y = y.mean(axis=0)
        else:
            return 0.0
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)
    except Exception:
        return 0.0


def detect_scale(src) -> str:
    """Return estimated musical key of a file or raw audio using librosa."""
    if not LIBROSA_AVAILABLE or src is None:  # pragma: no cover - optional
        return "Unknown"
    try:
        if isinstance(src, str):
            y, sr = librosa.load(src, sr=None, mono=True)
        elif isinstance(src, (list, tuple)) and len(src) == 2:
            sr, y = src
            if y.ndim > 1:
                y = y.mean(axis=0)
        else:
            return "Unknown"
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        major_profile = np.array(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
        minor_profile = np.array(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )
        profiles = np.vstack(
            [np.roll(major_profile, i) for i in range(12)]
            + [np.roll(minor_profile, i) for i in range(12)]
        )
        correlation = profiles @ chroma_mean
        key_idx = int(np.argmax(correlation))
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        if key_idx < 12:
            return f"{notes[key_idx]} Major"
        return f"{notes[key_idx - 12]} Minor"
    except Exception:
        return "Unknown"


def detect_scale_bpm(src) -> tuple[str, float]:
    """Return detected scale and BPM for ``src``."""
    return detect_scale(src), detect_bpm(src)


def _scale_bpm_wrap(p):
    """Gradio helper to return scale and BPM for uploads."""
    if isinstance(p, dict):
        p = p.get("name") or p.get("path") or p.get("data")
    return detect_scale_bpm(p)


def _scale_bpm_wrap_slider(p):
    """Return scale and update a BPM slider for uploads."""
    scale, bpm_val = _scale_bpm_wrap(p)
    return scale, gr.update(value=bpm_val)


def _scale_bpm_wrap_full(p):
    """Return scale, update a BPM slider and provide numeric readout."""
    scale, bpm_val = _scale_bpm_wrap(p)
    return scale, gr.update(value=bpm_val), bpm_val


def _bpm_wrap(p):
    """Gradio helper to auto-populate a BPM slider from uploaded audio."""
    if isinstance(p, dict):
        p = p.get("name") or p.get("path") or p.get("data")
    bpm_val = detect_bpm(p)
    return gr.update(value=bpm_val)


def _bpm_wrap_with_readout(p):
    """Update BPM slider and provide numeric readout for uploads."""
    if isinstance(p, dict):
        p = p.get("name") or p.get("path") or p.get("data")
    bpm_val = detect_bpm(p)
    return gr.update(value=bpm_val), bpm_val


def _waveform_plot(p):
    """Return a simple waveform visualization for ``p``."""
    if not (MATPLOTLIB_AVAILABLE and LIBROSA_AVAILABLE):  # pragma: no cover
        return None
    try:
        if isinstance(p, dict):
            p = p.get("name") or p.get("path") or p.get("data")
        if isinstance(p, str):
            y, sr = librosa.load(p, sr=None, mono=True)
        elif isinstance(p, (list, tuple)) and len(p) == 2:
            sr, y = p
            if y.ndim > 1:
                y = y.mean(axis=0)
        else:
            return None
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(np.linspace(0, len(y) / sr, num=len(y)), y, color="#43FF7E")
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        return fig
    except Exception:
        return None


def _apply_gain_file(path: str | dict | None, gain_db: float):
    """Return a new filepath with gain applied for interactive preview."""
    if isinstance(path, dict):
        path = path.get("name") or path.get("path")
    if not path or not SOUNDFILE_AVAILABLE:
        return path
    try:
        if LIBROSA_AVAILABLE:
            y, sr = librosa.load(path, sr=None, mono=True)
        else:
            y, sr = sf.read(path)
            if y.ndim > 1:
                y = y.mean(axis=1)
        y = y * _db_to_amp(gain_db)
        out_path = TMP_DIR / f"gain_{uuid.uuid4().hex}.wav"
        sf.write(out_path, y, sr)
        return str(out_path)
    except Exception:
        return path


def _looper(audio: np.ndarray, sr: int, bpm: float, duration_measures: float) -> np.ndarray:
    """Loop ``audio`` to ``duration_measures`` or pad with silence."""
    if duration_measures <= 0 or bpm <= 0:
        return audio
    measure_sec = 60.0 / bpm * 4
    measure_samples = int(measure_sec * sr)
    total_measures = int(len(audio) / measure_samples)
    trimmed = audio[: total_measures * measure_samples]
    if total_measures == 0:
        trimmed = np.zeros(0)
    if duration_measures <= total_measures:
        return trimmed[: int(duration_measures * measure_samples)]
    loops = duration_measures // total_measures if total_measures else 0
    out = np.tile(trimmed, int(loops))
    remaining = int((duration_measures - loops * total_measures) * measure_samples)
    if remaining > 0:
        out = np.concatenate([out, np.zeros(remaining, dtype=audio.dtype)])
    return out


def harmonize_file(path: str, scale: str) -> str:
    """Pitch-align audio to ``scale`` using FTHA and export MIDI."""
    if not (LIBROSA_AVAILABLE and SOUNDFILE_AVAILABLE and path):  # pragma: no cover
        raise gr.Error("librosa and soundfile are required for harmonize")

    y, sr = librosa.load(path, sr=44100, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    hop = 512
    frame = 2048
    f0 = librosa.yin(y.mean(axis=0), fmin=50, fmax=1000, frame_length=frame, hop_length=hop)

    window = np.hanning(frame)
    harmonized = np.zeros_like(y)
    weight = np.zeros(y.shape[1])

    mid = None
    track = None
    frame_ticks = 0
    if MIDO_AVAILABLE:
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        frame_ticks = int((hop / sr) * mid.ticks_per_beat)

    last_note = None
    dur_ticks = 0

    for i, freq in enumerate(f0):
        start = i * hop
        end = min(start + frame, y.shape[1])
        segment = y[:, start:end]
        if freq <= 0 or end - start <= 0:
            harmonized[:, start:end] += segment * window[: end - start]
            weight[start:end] += window[: end - start]
            if MIDO_AVAILABLE and last_note is not None:
                track.append(mido.Message("note_off", note=last_note, time=dur_ticks))
                last_note = None
                dur_ticks = 0
            continue

        target = _ftha_align(freq, scale)
        shift = 12 * math.log2(target / freq)
        for ch in range(y.shape[0]):
            shifted = librosa.effects.pitch_shift(segment[ch], sr, shift)
            if np.max(np.abs(shifted)) > 0:
                rms_orig = np.sqrt(np.mean(segment[ch]**2) + 1e-9)
                rms_new = np.sqrt(np.mean(shifted**2) + 1e-9)
                shifted *= rms_orig / rms_new
            length = min(len(shifted), len(window))
            harmonized[ch, start:start + length] += shifted[:length] * window[:length]
        weight[start:start + length] += window[:length]

        if MIDO_AVAILABLE:
            note = int(round(_freq_to_midi(target)))
            if note != last_note:
                if last_note is not None:
                    track.append(mido.Message("note_off", note=last_note, time=dur_ticks))
                track.append(mido.Message("note_on", note=note, velocity=64, time=0))
                last_note = note
                dur_ticks = frame_ticks
            else:
                dur_ticks += frame_ticks

    if MIDO_AVAILABLE and last_note is not None:
        track.append(mido.Message("note_off", note=last_note, time=dur_ticks))

    harmonized = np.divide(
        harmonized,
        weight[np.newaxis, :],
        out=np.zeros_like(harmonized),
        where=weight[np.newaxis, :] > 0,
    )
    if np.max(np.abs(harmonized)) > 0:
        harmonized /= np.max(np.abs(harmonized))

    out_file = TMP_DIR / f"harm_{uuid.uuid4().hex}.wav"
    sf.write(out_file, harmonized.T, sr)

    if MIDO_AVAILABLE:
        out_midi = TMP_DIR / f"harm_{uuid.uuid4().hex}.mid"
        mid.save(out_midi)

    return str(out_file)


def _harm_wrap(audio, scale: str):
    """Harmonize ``audio`` (filepath or numpy tuple) to ``scale``."""
    if isinstance(audio, str):
        return harmonize_file(audio, scale)
    if isinstance(audio, tuple) and len(audio) == 2 and SOUNDFILE_AVAILABLE:
        sr, y = audio
        tmp = TMP_DIR / f"harm_{uuid.uuid4().hex}.wav"
        sf.write(tmp, y, sr)
        out_path = harmonize_file(str(tmp), scale)
        if LIBROSA_AVAILABLE:
            y2, sr2 = librosa.load(out_path, sr=sr, mono=False)
            y2 = y2.T if y2.ndim > 1 else y2
        else:
            y2, sr2 = sf.read(out_path)
        return (sr2, y2)
    return audio


def _apply_pedalboard(audio: np.ndarray, sr: int, reverb: float, dist: float, gate: float) -> np.ndarray:
    if not (PEDALBOARD_AVAILABLE and len(audio)):
        return audio
    board = Pedalboard()
    if reverb > 0:
        board.append(Reverb(room_size=reverb))
    if dist > 0:
        board.append(Distortion(drive=dist))
    if gate > 0:
        board.append(NoiseGate(threshold_db=-gate))
    audio = board(audio, sr)
    return np.asarray(audio)


def _envelope_follower(signal: np.ndarray, attack_ms: float, release_ms: float, sr: int) -> np.ndarray:
    """Return the envelope of ``signal`` using attack/release times in milliseconds."""
    if attack_ms == 0:
        attack_ms = 1e-9
    attack = attack_ms / 1000.0
    release = release_ms / 1000.0
    attack_coef = math.exp(-1.0 / (attack * sr))
    release_coef = math.exp(-1.0 / (release * sr))
    env = np.zeros_like(signal)
    for i in range(1, len(signal)):
        x = abs(signal[i])
        if env[i - 1] < x:
            env[i] = attack_coef * env[i - 1] + (1 - attack_coef) * x
        else:
            env[i] = release_coef * env[i - 1] + (1 - release_coef) * x
    return env


def _compressor(signal: np.ndarray, sr: int, threshold: float, ratio: float, attack: float, release: float) -> np.ndarray:
    env = _envelope_follower(signal, attack, release, sr)
    out = np.zeros_like(signal)
    for i in range(len(signal)):
        e = env[i]
        if e > threshold:
            amt = (e - threshold) / ratio
            factor = (threshold + amt) / e
            out[i] = signal[i] * factor
        else:
            out[i] = signal[i]
    return out


def _cheby_filter(signal: np.ndarray, sr: int, cutoff: float, order: int, btype: str) -> np.ndarray:
    if not SCIPY_AVAILABLE:
        return signal
    rp = 0.5
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = cheby1(order, rp, normal_cutoff, btype=btype, analog=False)
    return lfilter(b, a, signal)


def _band_generator(signal: np.ndarray, sr: int, freq: float) -> tuple[np.ndarray, np.ndarray]:
    if freq < 200 or freq > 18000:
        return signal, np.zeros_like(signal)
    x = 4 * freq / 100.0
    low = _cheby_filter(signal, sr, freq - x, 6, "low")
    high = _cheby_filter(signal, sr, freq + x, 6, "high")
    return np.nan_to_num(low), np.nan_to_num(high)


def _divide_signal(signal: np.ndarray, sr: int, crossovers: np.ndarray) -> list[np.ndarray]:
    crossovers = np.sort(np.asarray(crossovers))
    aux = signal
    bands = []
    for freq in crossovers:
        low, aux = _band_generator(aux, sr, freq)
        bands.append(low)
    bands.append(aux)
    return bands


def _merge_signals(signals: list[np.ndarray]) -> np.ndarray:
    merged = signals[0]
    for s in signals[1:]:
        merged = merged + s
    return merged


def _multiband_compress(signal: np.ndarray, sr: int) -> np.ndarray:
    """Apply a simple multiband compressor with fixed parameters."""
    if not len(signal):
        return signal
    if not SCIPY_AVAILABLE:
        # Fallback: apply a naive static compressor without filtering.
        threshold = 0.1
        ratio = 4.0
        out = signal.copy()
        over = np.abs(out) > threshold
        out[over] = np.sign(out[over]) * (
            threshold + (np.abs(out[over]) - threshold) / ratio
        )
        return out
    cross = np.array([400, 1000, 6000])
    thresholds = np.array([0.1, 0.2, 0.18, 0.05])
    ratios = np.array([2, 2, 2, 3])
    attacks = np.array([10, 0, 10, 0])
    releases = np.array([60, 60, 60, 60])
    bands = _divide_signal(signal, sr, cross)
    processed: list[np.ndarray] = []
    for b, th, ra, at, rel in zip(bands, thresholds, ratios, attacks, releases):
        processed.append(_compressor(b, sr, th, ra, at, rel))
    return _merge_signals(processed)


def _rhythmic_gate(
    audio: np.ndarray,
    sr: int,
    bpm: float,
    freq: str,
    dur: float,
    loc: str,
    pattern: str,
) -> np.ndarray:
    """Gate audio on/off according to musical timing parameters.

    Parameters mirror the user's description from the prompt:

    - ``freq``: string like ``"1/4"`` or ``"1"`` indicating the length of the
      mute window in beats.  Values are clamped to ``1/32`` .. ``4`` beats.
    - ``dur``: fraction ``0..1`` of each window that is silenced.
    - ``loc``: where the silence sits in the window – ``start``, ``middle`` or
      ``end``.
    - ``pattern``: one of ``flat``, ``trance``, ``build_single``,
      ``build_double``, ``slow`` or ``aggro``.  Patterns modulate the window
      frequency over time using simple heuristics.
    """

    if bpm <= 0 or dur <= 0:
        return audio

    beat_dur = 60.0 / bpm
    try:
        num, den = freq.split("/") if "/" in freq else (freq, "1")
        base_beats = float(num) / float(den)
    except Exception:
        base_beats = 1.0
    base_beats = float(np.clip(base_beats, 1 / 32.0, 4.0))

    total_beats = len(audio) / sr / beat_dur
    # Pre-compute the gating schedule per beat to keep implementation simple
    beat_freqs = []
    for beat in range(int(math.ceil(total_beats))):
        mult = 1.0
        if pattern == "trance":
            mult = 2.0 if beat % 2 else 1.0
        elif pattern == "build_single":
            mult = 2.0 ** beat
        elif pattern == "build_double":
            mult = 2.0 ** (beat * 2)
        elif pattern == "slow":
            mult = 0.5 ** beat
        elif pattern == "aggro" and (beat + 1) % 4 == 0:
            mult = 4.0
        beat_freqs.append(base_beats / mult)

    out = audio.copy()
    cur = 0
    for beat_len_beats in beat_freqs:
        beat_len_sec = beat_len_beats * beat_dur
        start_idx = int(cur * sr)
        end_idx = int((cur + beat_len_beats) * sr)
        if end_idx <= start_idx:
            break
        section = out[start_idx:end_idx]
        phase = np.linspace(0, 1, len(section), endpoint=False)
        offset = {"start": 0.0, "middle": 0.5 - dur / 2.0, "end": 1.0 - dur}.get(loc, 0.0)
        mask = (phase < offset) | (phase > offset + dur)
        section *= mask.astype(section.dtype)
        out[start_idx:end_idx] = section
        cur += beat_len_beats
    return out


def _glitch_audio(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Reverse random short segments to create a simple glitch effect."""
    if amount <= 0:
        return audio
    seg_len = max(1, int(0.1 * sr))
    audio = audio.copy()
    for start in range(0, len(audio) - seg_len, seg_len):
        if np.random.rand() < amount:
            audio[start : start + seg_len] = audio[start : start + seg_len][::-1]
    return audio


def _apply_stem_fx(
    audio: np.ndarray,
    sr: int,
    reverb: float,
    dist: float,
    gate: float,
    glitch: float,
    rhythm_gate: dict | None,
    bpm: float,
    loop_measures: float,
    multiband: float,
) -> np.ndarray:
    """Apply optional looper, glitch, rhythmic gate, and pedalboard effects."""
    audio = _looper(audio, sr, bpm, loop_measures)
    audio = _glitch_audio(audio, sr, glitch)
    if rhythm_gate:
        audio = _rhythmic_gate(audio, sr, bpm, **rhythm_gate)
    audio = _apply_pedalboard(audio, sr, reverb, dist, gate)
    if multiband > 0:
        comp = _multiband_compress(audio, sr)
        audio = (1 - multiband) * audio + multiband * comp
    return audio


def combine_stems(
    drums_path: str | None,
    drums_fx: list[str] | None,
    drums_bpm: float,
    drums_gain: float,
    vocals_path: str | None,
    vocals_fx: list[str] | None,
    vocals_bpm: float,
    vocals_gain: float,
    bass_path: str | None,
    bass_fx: list[str] | None,
    bass_bpm: float,
    bass_gain: float,
    other_path: str | None,
    other_fx: list[str] | None,
    other_bpm: float,
    other_gain: float,
    out_dir: str,
    prompt: str,
    reverb: float,
    dist: float,
    gate: float,
    glitch: float,
    multiband: float,
    rhythm_gate_freq: str,
    rhythm_gate_dur: float,
    rhythm_gate_loc: str,
    rhythm_gate_pattern: str,
    loop_measures: float,
):
    """Load stems, apply selected effects, then mix."""

    if not SOUNDFILE_AVAILABLE and not LIBROSA_AVAILABLE:
        raise gr.Error("soundfile or librosa required")

    paths = [drums_path, vocals_path, bass_path, other_path]
    gains = [drums_gain, vocals_gain, bass_gain, other_gain]
    bpms = [drums_bpm, vocals_bpm, bass_bpm, other_bpm]
    fxs = [drums_fx, vocals_fx, bass_fx, other_fx]

    stems: list[np.ndarray | None] = []
    sr = TARGET_SR
    target_bpm = drums_bpm if drums_bpm > 0 else None
    for p, g_db, bpm, fx_list in zip(paths, gains, bpms, fxs):
        if p and Path(p).exists():
            if LIBROSA_AVAILABLE:
                y, sr = librosa.load(p, sr=sr, mono=True)
            else:
                y, sr = sf.read(p)
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if sr != TARGET_SR:
                    y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
                    sr = TARGET_SR
            if target_bpm and LIBROSA_AVAILABLE and bpm > 0:
                rate = target_bpm / bpm
                if rate != 1.0:
                    y = librosa.effects.time_stretch(y, rate)
                    bpm = target_bpm
            fx_set = set(fx_list or [])
            rg_cfg = (
                {
                    "freq": rhythm_gate_freq,
                    "dur": rhythm_gate_dur,
                    "loc": rhythm_gate_loc,
                    "pattern": rhythm_gate_pattern,
                }
                if "Rhythmic Gate" in fx_set
                else None
            )
            y = _apply_stem_fx(
                y,
                sr,
                reverb if "Reverb" in fx_set else 0.0,
                dist if "Distortion" in fx_set else 0.0,
                gate if "Gate" in fx_set else 0.0,
                glitch if "Glitch" in fx_set else 0.0,
                rg_cfg,
                bpm,
                loop_measures if "Looper" in fx_set else 0.0,
                multiband if "Multiband Compressor" in fx_set else 0.0,
            )
            y = y * _db_to_amp(g_db)
            stems.append(y)
        else:
            stems.append(None)

    max_len = max(len(s) if s is not None else 0 for s in stems)
    mix = np.zeros(max_len)
    for s in stems:
        if s is None:
            continue
        if len(s) < max_len:
            s = np.pad(s, (0, max_len - len(s)))
        mix += s

    out_dir = Path(out_dir) if out_dir else TMP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = (prompt or "output")[:10]
    orig = Path(drums_path or vocals_path or bass_path or other_path or "mix.wav").name
    fname = f"{prefix}_combine_{orig}"
    out_path = out_dir / fname
    sf.write(out_path, mix, sr)
    return str(out_path)


def preview_stem_fx(
    audio_path: str | None,
    reverb: float,
    dist: float,
    gate: float,
    glitch: float,
    multiband: float,
    rhythm_gate_freq: str,
    rhythm_gate_dur: float,
    rhythm_gate_loc: str,
    rhythm_gate_pattern: str,
    loop_measures: float,
    bpm: float,
    gain: float,
):
    """Load a stem, apply effects, and return a preview filepath."""
    if not audio_path or not Path(audio_path).exists():
        raise gr.Error("no audio provided")
    if not SOUNDFILE_AVAILABLE and not LIBROSA_AVAILABLE:
        raise gr.Error("soundfile or librosa required")

    sr = 44100
    if LIBROSA_AVAILABLE:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
    elif SOUNDFILE_AVAILABLE:
        y, sr = sf.read(audio_path)
        if y.ndim > 1:
            y = y.mean(axis=1)
    else:  # pragma: no cover
        raise gr.Error("librosa or soundfile required to load audio")
    rg_cfg = {
        "freq": rhythm_gate_freq,
        "dur": rhythm_gate_dur,
        "loc": rhythm_gate_loc,
        "pattern": rhythm_gate_pattern,
    }
    y = _apply_stem_fx(
        y,
        sr,
        reverb,
        dist,
        gate,
        glitch,
        rg_cfg,
        bpm,
        loop_measures,
        multiband,
    )
    y = y * _db_to_amp(gain)
    out_path = TMP_DIR / f"preview_{uuid.uuid4().hex}.wav"
    sf.write(out_path, y, sr)
    return str(out_path)


def _preview_wrap(
    audio_path: str | None,
    fx_list: list[str] | None,
    bpm: float,
    gain: float,
    reverb: float,
    dist: float,
    gate: float,
    glitch: float,
    multiband: float,
    rg_freq: str,
    rg_dur: float,
    rg_loc: str,
    rg_pattern: str,
    loop_measures: float,
):
    """Wrapper around :func:`preview_stem_fx` that respects ``fx_list``.

    Returns the preview filepath alongside the original path so callers can
    revert the operation.
    """

    fx_set = set(fx_list or [])
    out = preview_stem_fx(
        audio_path,
        reverb if "Reverb" in fx_set else 0.0,
        dist if "Distortion" in fx_set else 0.0,
        gate if "Gate" in fx_set else 0.0,
        glitch if "Glitch" in fx_set else 0.0,
        multiband if "Multiband Compressor" in fx_set else 0.0,
        rg_freq,
        rg_dur if "Rhythmic Gate" in fx_set else 0.0,
        rg_loc,
        rg_pattern,
        loop_measures if "Looper" in fx_set else 0.0,
        bpm,
        gain,
    )
    return out, audio_path


def _harmonize_wrap(path: str | None, scale: str) -> tuple[str | None, str | None]:
    """Harmonize ``path`` and return the new/old paths for undo."""

    if not path:
        return None, None
    out = _harm_wrap(path, scale)
    return out if isinstance(out, str) else path, path


def _revert_wrap(prev: str | None, current: str | None) -> tuple[str | None, None]:
    """Return ``prev`` if available, otherwise ``current``."""

    return (prev or current), None

def _matchering_match(target: str, reference: str, output: str) -> None:
    """Call Matchering's matching via API fallbacks or CLI.

    Some Matchering releases expose a simple ``match`` helper while
    others only bundle the lower level ``process`` API or a CLI script.
    This wrapper tries the high level helper first, then the low level
    API, and finally falls back to invoking a CLI entry point if one is
    available.
    """
    if mg is not None:
        # Some Matchering releases expose a high level ``match`` helper. If
        # available, use it directly.
        if hasattr(mg, "match"):
            mg.match(target=target, reference=reference, output=output)
            return

        # Some versions ship without a convenient ``match`` wrapper or CLI
        # entry point but still expose the lower level ``process`` API and the
        # ``Result`` container.  Emulate the CLI by calling ``process``
        # directly so we don't require an external executable.
        if hasattr(mg, "process") and hasattr(mg, "Result"):
            res = mg.Result(output, "PCM_16")
            mg.process(target=target, reference=reference, results=[res])
            return

    # Fall back to a CLI invocation.  Different Matchering versions expose
    # different entry points, so try a couple of common variants before giving
    # up.
    cmd_candidates = [
        [sys.executable, "-m", "matchering.cli", target, reference, output],
        [sys.executable, "-m", "matchering", target, reference, output],
    ]
    exe = shutil.which("matchering")
    if exe:
        cmd_candidates.append([exe, target, reference, output])

    for cmd in cmd_candidates:
        try:
            sp.run(cmd, check=True)
            return
        except (FileNotFoundError, sp.CalledProcessError):
            continue
    raise gr.Error("Matchering CLI invocation failed; ensure it is installed")


def rvc_convert_vocals(
    vocal_path: Path,
    model_path: Path,
    output_path: Path,
    pitch_shift: float = 0.0,
    index_ratio: float = 0.5,
    filter_radius: int = 3,
    algorithm: str = "harvest",
) -> Path:
    """Convert a vocal track to a target timbre using an RVC model.

    Parameters
    ----------
    vocal_path:
        Path to the isolated vocal audio to convert. High-quality, clean vocals
        yield the best results; tools like UVR can help isolate vocals.
    model_path:
        Location of the pretrained RVC model weights.
    output_path:
        Destination where the converted audio will be written.
    pitch_shift:
        Semitone adjustment applied before conversion to better match the
        target voice.
    index_ratio:
        Blend ratio between the model's learned characteristics and the source
        signal. Values between 0.3 and 0.8 usually provide a natural balance.
    filter_radius:
        Smoothing radius used by the converter. Higher values preserve more of
        the original vocal; a range of 3–7 is generally effective.
    algorithm:
        Pitch extraction algorithm. ``"harvest"`` often yields the most
        natural-sounding results.

    Notes
    -----
    Match the source audio's EQ and compression to the training data when
    possible, and apply subtle reverb/EQ/compression after conversion to blend
    with the instrumental.
    """
    if not RVC_AVAILABLE:
        raise RuntimeError("RVC library is not available")

    converter = VoiceConverter(model_path)
    converter.convert(
        str(vocal_path),
        str(output_path),
        pitch_shift=pitch_shift,
        index_ratio=index_ratio,
        filter_radius=filter_radius,
        algorithm=algorithm,
    )
    return output_path


def _apply_stereo_space(in_path: Path, out_path: Path, sr: int, width: float = 1.5, pan: float = 0.0) -> None:
    """Use ffmpeg to widen stereo image and apply gentle panning.

    ``stereotools`` uses ``slev`` to control stereo width; earlier versions of
    this script tried to pass a non-existent ``width`` option which caused
    ffmpeg to fail. Mapping ``width`` to ``slev`` keeps the caller API stable
    while using the correct ffmpeg parameter.

    Some ffmpeg builds may lack support for ``slev`` altogether. If the
    ``stereotools`` invocation fails we fall back to the simpler
    ``stereowiden`` filter as a best-effort approximation.
    """
    width_s = 1.0 + (width - 1.0) / 10.0
    pan_s = pan / 10.0
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_path),
        "-af",
        f"stereotools=slev={width_s:.6f}:balance_out={pan_s:.6f}",
        "-ar",
        str(sr),
        str(out_path),
    ]
    try:
        sp.run(cmd, check=True)
    except sp.CalledProcessError:
        crossfeed = max(0.0, width_s - 1.0)
        fallback = [
            "ffmpeg",
            "-y",
            "-i",
            str(in_path),
            "-af",
            f"stereowiden=delay=20:feedback=0.3:crossfeed={crossfeed:.6f}:drymix=0.8",
            "-ar",
            str(sr),
            str(out_path),
        ]
        sp.run(fallback, check=True)


def _apply_bass_boost(in_path: Path, out_path: Path, sr: int, gain_db: float) -> None:
    """Apply a simple low shelf boost using ffmpeg's bass filter."""
    if abs(gain_db) < 1e-6:
        shutil.copyfile(in_path, out_path)
        return
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_path),
        "-af",
        f"bass=g={gain_db}",
        "-ar",
        str(sr),
        str(out_path),
    ]
    sp.run(cmd, check=True)


# Bands that can be attenuated/boosted via the frequency cut UI. Each tuple is
# (label, low_hz, high_hz, description).
FREQ_BANDS = [
    (
        "Infrasound",
        1,
        20,
        "Anxiety, nausea, chest pressure, hallucinations; vibrates organs/eyeballs",
    ),
    (
        "Eyeball resonance",
        16,
        18,
        "Visual disturbance, discomfort; matches eyeball natural frequency",
    ),
    (
        "Sub-rumble conflict",
        30,
        80,
        "Muddy, boomy, disorienting; overlaps room modes and sub-harmonics",
    ),
    (
        "Beating tones",
        0,
        10,
        "Pulsing, wobble, unsteady feeling; interference between close frequencies",
    ),
    (
        "Roughness region",
        15,
        30,
        "Metallic, buzzy, stressful; fast beating equals tension",
    ),
    (
        "Dissonant intervals",
        600,
        800,
        "Tension, unresolved, horror vibes; unstable harmonic ratios",
    ),
    (
        "Midrange overload",
        250,
        800,
        "Muddiness, fatigue, crowding; conflicts with human vocal range",
    ),
    (
        "Harsh upper mids",
        2500,
        4500,
        "Shrillness, ear pain, baby-cry level alert; peak auditory sensitivity",
    ),
    (
        "Digital aliasing artifacts",
        10000,
        20000,
        "Grainy, robotic, unnatural; math artifacts from poor sampling",
    ),
]


def _apply_bass_narrow(in_path: Path, out_path: Path, sr: int, width: float) -> None:
    """Reduce stereo width of low frequencies via ffmpeg."""
    if width >= 0.999:
        shutil.copyfile(in_path, out_path)
        return
    w = max(0.0, min(1.0, width))
    if w <= 1e-6:
        pan_expr = "c0=0.5*c0+0.5*c1|c1=0.5*c0+0.5*c1"
    else:
        mono_mix = 0.5 * (1.0 - w)
        base = w + mono_mix
        pan_expr = (
            f"c0={base:.6f}*c0+{mono_mix:.6f}*c1|"
            f"c1={base:.6f}*c1+{mono_mix:.6f}*c0"
        )
    filter_expr = (
        f"asplit=2[low][high];"
        f"[low]lowpass=f=150,pan=stereo|{pan_expr}[l];"
        f"[high]highpass=f=150[h];"
        f"[l][h]amix=inputs=2:dropout_transition=0"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_path),
        "-af",
        filter_expr,
        "-ar",
        str(sr),
        str(out_path),
    ]
    sp.run(cmd, check=True)


def _apply_frequency_cuts(
    in_path: Path, out_path: Path, sr: int, gains: list[float]
) -> None:
    """Apply per-band gain adjustments using ffmpeg's equalizer filter."""
    filters = []
    for gain_db, (label, low, high, _desc) in zip(gains, FREQ_BANDS):
        if abs(gain_db) < 1e-6:
            continue
        center = (low + high) / 2.0
        width = max(high - low, 1)
        filters.append(f"equalizer=f={center}:t=h:w={width}:g={gain_db}")
    if not filters:
        shutil.copyfile(in_path, out_path)
        return
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_path),
        "-af",
        ",".join(filters),
        "-ar",
        str(sr),
        str(out_path),
    ]
    sp.run(cmd, check=True)


def _master_simple(
    audio_input,
    reference_audio: str | None = None,
    out_trim_db: float = -1.0,
    width: float = 1.5,
    pan: float = 0.0,
    bass_boost_db: float = 0.0,
    bass_width: float = 0.0,
    freq_gains: list[float] | None = None,
):
    if not MATCHERING_AVAILABLE:
        raise gr.Error("Matchering not installed. `pip install matchering`")
    try:
        sr, wav_np = audio_input
    except Exception:
        raise gr.Error("Please provide an audio clip.")
    in_path = TMP_DIR / f"master_in_{uuid.uuid4().hex}.wav"
    audio_write(str(in_path), torch.from_numpy(wav_np).float().t(), sr, add_suffix=False)
    if reference_audio:
        ref = Path(reference_audio)
    else:
        ref = Path("/references/reference.wav")
    if not ref.exists():
        raise gr.Error(f"Reference file missing: {ref}")
    matched_path = TMP_DIR / f"mastered_simple_{uuid.uuid4().hex}.wav"
    _matchering_match(target=str(in_path), reference=str(ref), output=str(matched_path))
    bass_path = TMP_DIR / f"mastered_simple_bass_{uuid.uuid4().hex}.wav"
    _apply_bass_boost(matched_path, bass_path, sr, bass_boost_db)
    widened_path = TMP_DIR / f"mastered_simple_wide_{uuid.uuid4().hex}.wav"
    _apply_stereo_space(bass_path, widened_path, sr, width=width, pan=pan)
    narrowed_path = TMP_DIR / f"mastered_simple_narrow_{uuid.uuid4().hex}.wav"
    _apply_bass_narrow(widened_path, narrowed_path, sr, bass_width)
    final_path = TMP_DIR / f"mastered_simple_final_{uuid.uuid4().hex}.wav"
    _apply_frequency_cuts(
        narrowed_path,
        final_path,
        sr,
        freq_gains or [0.0] * len(FREQ_BANDS),
    )
    return str(final_path)


def audiosr_load_model(device=None):
    """Load and cache an ``AudioSR`` model on ``device``.

    When no ``device`` is specified, models are distributed across
    ``AUDIOSR_DEVICES`` in a round‑robin fashion so that multiple upscales can
    run concurrently on separate GPUs.
    """
    global AUDIOSR_MODELS, _AUDIOSR_NEXT_DEVICE

    if not AUDIOSR_AVAILABLE:
        raise gr.Error(
            "AudioSR not installed. Add the 'versatile_audio_super_resolution' directory."
        )

    if device is None:
        device = AUDIOSR_DEVICES[_AUDIOSR_NEXT_DEVICE]
        _AUDIOSR_NEXT_DEVICE = (_AUDIOSR_NEXT_DEVICE + 1) % len(AUDIOSR_DEVICES)

    model = AUDIOSR_MODELS.get(device)
    if model is None:
        AUDIOSR_MODELS[device] = audiosr_build_model(device=str(device))
        model = AUDIOSR_MODELS[device]
    return model


def _audiosr_process(audio_input):
    """Upscale audio using AudioSR to 48kHz."""
    if not AUDIOSR_AVAILABLE:
        raise gr.Error("AudioSR not installed. Add the 'versatile_audio_super_resolution' directory.")
    try:
        sr, wav_np = audio_input
    except Exception:
        raise gr.Error("Please provide an audio clip.")
    model = audiosr_load_model()
    chunk_samples = int(sr * 10)
    chunks = []
    for start in range(0, len(wav_np), chunk_samples):
        end = start + chunk_samples
        chunk = wav_np[start:end]
        in_path = TMP_DIR / f"audiosr_in_{uuid.uuid4().hex}.wav"
        audio_write(str(in_path), torch.from_numpy(chunk).float().t(), sr, add_suffix=False)
        out = audiosr_super_resolution(model, str(in_path))
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
        out_arr = out[0]
        if out_arr.ndim == 1:
            out_arr = out_arr[None, :]
        chunks.append(out_arr)
    combined = np.concatenate(chunks, axis=1) if chunks else np.empty((1, 0))
    out_path = TMP_DIR / f"audiosr_out_{uuid.uuid4().hex}.wav"
    audio_write(str(out_path), torch.from_numpy(combined).float(), 48000, add_suffix=False)
    return str(out_path)


def master_track(
    audio_input,
    reference_audio: str | None,
    out_trim_db: float = -1.0,
    width: float = 1.5,
    pan: float = 0.0,
    bass_boost_db: float = 0.0,
    bass_width: float = 0.0,
    infrasound_db: float = 0.0,
    eyeball_db: float = 0.0,
    sub_rumble_db: float = 0.0,
    beating_db: float = 0.0,
    roughness_db: float = 0.0,
    dissonant_db: float = 0.0,
    midrange_db: float = 0.0,
    harsh_db: float = 0.0,
    alias_db: float = 0.0,
    method: str = "Matchering",
):
    if method == "AudioSR":
        return _audiosr_process(audio_input)
    gains = [
        infrasound_db,
        eyeball_db,
        sub_rumble_db,
        beating_db,
        roughness_db,
        dissonant_db,
        midrange_db,
        harsh_db,
        alias_db,
    ]
    return _master_simple(
        audio_input,
        reference_audio,
        out_trim_db,
        width,
        pan,
        bass_boost_db,
        bass_width,
        gains,
    )


def analyze_and_rename(audio_path: str) -> tuple[str, str, float, str]:
    """Caption ``audio_path`` with five words, detect key/BPM and rename it.

    Parameters
    ----------
    audio_path:
        Path to the audio sample that should be analyzed.

    Returns
    -------
    tuple
        ``(description, key, bpm, new_path)`` where ``new_path`` is the
        potentially renamed file.  When analysis fails or renaming is not
        possible the original path is returned.
    """

    if not audio_path:
        return "", "", 0.0, ""

    if WAN2AUDIO_AVAILABLE:
        desc, key, bpm = wan2audio.analyze(audio_path)
    else:  # pragma: no cover - executed when WAN2Audio is unavailable
        desc, key, bpm = ("analysis unavailable", "C", 120.0)

    # Replace any non alphanumeric character with an underscore so that the
    # resulting filename is portable across filesystems.
    safe_desc = re.sub(r"[^0-9A-Za-z]+", "_", desc).strip("_")
    new_name = f"{safe_desc}_{key}_{int(round(bpm))}{Path(audio_path).suffix}"
    new_path = str(Path(audio_path).with_name(new_name))

    try:
        os.rename(audio_path, new_path)
    except OSError:  # pragma: no cover - depends on filesystem permissions
        new_path = audio_path

    return desc, key, float(bpm), new_path


def analyze_and_rename_batch(audio_paths: Iterable[str]) -> list[list]:
    """Analyze and rename multiple audio files.

    Parameters
    ----------
    audio_paths:
        Iterable of paths to audio samples.

    Returns
    -------
    list of lists
        Each inner list corresponds to ``[description, key, bpm, new_path]``.
    """

    results: list[list] = []
    if not audio_paths:
        return results
    for path in audio_paths:
        if not path:
            continue
        desc, key, bpm, new_path = analyze_and_rename(path)
        results.append([desc, key, bpm, new_path])
    return results


def _analyze_and_rename_batch_files(
    files: Iterable[str | tempfile._TemporaryFileWrapper] | None,
) -> list[list]:
    """Wrapper to accept uploaded files from ``gr.Files``.

    Depending on the Gradio version, ``gr.Files`` may yield either a list of
    ``tempfile.NamedTemporaryFile`` objects (when ``type="file"``) or plain file
    paths as strings (when ``type="filepath"``). This helper normalizes both
    forms to paths so the existing :func:`analyze_and_rename_batch` function can
    operate unchanged.

    Parameters
    ----------
    files:
        Iterable of file paths or temporary files as provided by ``gr.Files``.
        May be ``None`` if no files were uploaded.

    Returns
    -------
    list of lists
        Direct passthrough of :func:`analyze_and_rename_batch` results.
    """

    paths = []
    if files:
        for f in files:
            paths.append(f if isinstance(f, str) else f.name)
    return analyze_and_rename_batch(paths)

# ============================================================================
# UI (tabs, all Enqueue) [ALTERED]
# ============================================================================
def ui_full(launch_kwargs):
    # Ensure the output directory is always allowed for file serving
    launch_kwargs = dict(launch_kwargs)
    allowed_paths = list(launch_kwargs.get("allowed_paths", []))
    if str(TMP_DIR) not in allowed_paths:
        allowed_paths.append(str(TMP_DIR))
    launch_kwargs["allowed_paths"] = allowed_paths

    analyze_queue = gr.Queue(concurrency_count=1) if hasattr(gr, "Queue") else True
    with gr.Blocks(css=CUSTOM_CSS) as demo:
        demo.title = "Fortheye"
        gr.Image(
            "assets/banner.png",
            show_label=False,
            show_download_button=False,
            elem_id="top-banner",
        )
        queue_items = gr.State([])
        output_folder = gr.State(str(TMP_DIR))
        # Harmonization controls removed due to fidelity concerns

        # ----- QUEUE MANAGER -----
        with gr.Tab("Queue"):
            gr.Markdown("Manage items that will be processed sequentially.")
            queue_display = gr.Dropdown(label="Queued Items", choices=[], interactive=True)
            with gr.Row():
                new_item = gr.Textbox(label="New Queue Item")
                btn_add_q = gr.Button("Queue Item")
                btn_del_q = gr.Button("Delete Selected")
            btn_add_q.click(
                add_queue_item,
                inputs=[queue_items, new_item],
                outputs=[queue_items, queue_display, new_item],
                queue=False,
            )
            btn_del_q.click(
                delete_queue_item,
                inputs=[queue_items, queue_display],
                outputs=[queue_items, queue_display],
                queue=False,
            )

        # ----- STYLE -----
        with gr.Tab("Style (MusicGen-Style, GPU2)"):
            with gr.Row():
                with gr.Column():
                    text = gr.Textbox(label="Text Prompt", placeholder="e.g., glossy synthwave with gated drums")
                    with gr.Row():
                        btn_save_text = gr.Button("", elem_classes=["icon-btn", "icon-save"])
                        btn_view_text = gr.Button("", elem_classes=["icon-btn", "icon-view"])
                    notebook_text = gr.Dropdown(
                        label="Prompt Notebook", choices=[], visible=False
                    )
                melody = gr.Audio(label="Style Excerpt (optional)", type="numpy")
                btn_detect_mel = gr.Button("Detect Scale+BPM")
            with gr.Row():
                scale_mel = gr.Textbox(label="Detected Scale", interactive=False)
                bpm_mel = gr.Number(label="Detected BPM", value=0, interactive=False)
            btn_detect_mel.click(_scale_bpm_wrap, melody, [scale_mel, bpm_mel])
            btn_save_text.click(save_prompt, inputs=text, outputs=None)
            btn_view_text.click(show_notebook, outputs=notebook_text)
            notebook_text.change(load_prompt, inputs=notebook_text, outputs=text)
            with gr.Row():
                dur = gr.Slider(1, 60, value=10, step=1, label="Duration (s)")
                eval_q = gr.Slider(1, 6, value=3, step=1, label="Style RVQ")
                excerpt = gr.Slider(0.5, 4.5, value=3.0, step=0.5, label="Excerpt length (s)")
            with gr.Row():
                topk = gr.Number(label="Top-k", value=200)
                topp = gr.Number(label="Top-p", value=50.0)
                temp = gr.Number(label="Temperature", value=1.0)
                cfg = gr.Number(label="CFG α", value=3.0)
                double_cfg = gr.Radio(["Yes", "No"], value="Yes", label="Double CFG")
                cfg_beta = gr.Number(label="CFG β (double)", value=5.0)
            out_trim_style = gr.Slider(-24.0, 0.0, value=-3.0, step=0.5, label="Output Trim (dB)")
            decoder = gr.Radio(["Default", "MultiBand_Diffusion"], value="Default", label="Decoder")

            out_style = gr.Audio(label="Output", type="filepath")
            btn_style = gr.Button("Enqueue", variant="primary")
            btn_style.click(
                style_predict,
                inputs=[text, melody, dur, topk, topp, temp, cfg, double_cfg, cfg_beta, eval_q, excerpt, decoder, out_trim_style],
                outputs=out_style,
                queue=True,
            )

        # ----- SECTION COMBINER -----
        with gr.Tab("Section Combiner"):
            with gr.Row():
                intro_audio = gr.Audio(label="Intro Audio", type="numpy")
                btn_detect_intro = gr.Button("Detect Scale+BPM")
            with gr.Row():
                next_audio = gr.Audio(label="Next Audio", type="numpy")
                btn_detect_next = gr.Button("Detect Scale+BPM")
            with gr.Row():
                prompt = gr.Textbox(label="Prompt")
                btn_save_comb = gr.Button("", elem_classes=["icon-btn", "icon-save"])
                btn_view_comb = gr.Button("", elem_classes=["icon-btn", "icon-view"])
            notebook_comb = gr.Dropdown(label="Prompt Notebook", choices=[], visible=False)
            model_choice = gr.Radio(["AudioGen", "Style", "Melody-Large"], value="AudioGen", label="Generator")
            with gr.Row():
                bpm = gr.Slider(40, 240, value=120, step=1, label="Tempo (BPM)")
                beats = gr.Slider(2, 32, value=8, step=1, label="Transition Length (beats)")
                xf_beats = gr.Slider(0.0, 8.0, value=1.0, step=0.25, label="Crossfade (beats)")
            scale_comb = gr.Textbox(label="Detected Scale", interactive=False)
            bpm_detect = gr.Number(value=0, label="Detected BPM", interactive=False)
            btn_detect_intro.click(
                _scale_bpm_wrap_full, intro_audio, [scale_comb, bpm, bpm_detect]
            )
            btn_detect_next.click(
                _scale_bpm_wrap_full, next_audio, [scale_comb, bpm, bpm_detect]
            )
            btn_save_comb.click(save_prompt, inputs=prompt, outputs=None)
            btn_view_comb.click(show_notebook, outputs=notebook_comb)
            notebook_comb.change(load_prompt, inputs=notebook_comb, outputs=prompt)
            decoder = gr.Radio(["Default", "MultiBand_Diffusion"], value="Default", label="Decoder")
            out_trim_comb = gr.Slider(-24.0, 0.0, value=-3.0, step=0.5, label="Output Trim (dB)")
            out_transition = gr.Audio(label="Transition", type="filepath")
            out_full = gr.Audio(label="Combined Output", type="filepath")
            btn_trans = gr.Button("Generate Transition")
            btn_full = gr.Button("Generate & Combine", variant="primary")
            btn_trans.click(
                section_generate_transition,
                inputs=[model_choice, intro_audio, prompt, bpm, beats, decoder, out_trim_comb],
                outputs=out_transition,
                queue=True,
            )
            btn_full.click(
                section_generate_and_combine,
                inputs=[model_choice, intro_audio, next_audio, prompt, bpm, beats, xf_beats, decoder, out_trim_comb],
                outputs=out_full,
                queue=True,
            )

        # ----- AUDIOGEN CONTINUATION -----
        with gr.Tab("AudioGen Continuation (GPU0)"):
            with gr.Row():
                audio_in = gr.Audio(label="Input Clip", type="numpy")
                btn_detect_ag = gr.Button("Detect Scale+BPM")
            with gr.Row():
                scale_ag = gr.Textbox(label="Detected Scale", interactive=False)
                bpm_ag = gr.Number(label="Detected BPM", value=0, interactive=False)
            btn_detect_ag.click(_scale_bpm_wrap, audio_in, [scale_ag, bpm_ag])
            with gr.Row():
                prompt = gr.Textbox(label="Prompt (optional)", placeholder="e.g., keep the groove, add arps")
                btn_save_cont = gr.Button("", elem_classes=["icon-btn", "icon-save"])
                btn_view_cont = gr.Button("", elem_classes=["icon-btn", "icon-view"])
            notebook_cont = gr.Dropdown(label="Prompt Notebook", choices=[], visible=False)
            btn_save_cont.click(save_prompt, inputs=prompt, outputs=None)
            btn_view_cont.click(show_notebook, outputs=notebook_cont)
            notebook_cont.change(load_prompt, inputs=notebook_cont, outputs=prompt)
            model_ag = gr.Radio(["AudioGen", "Melody-Large"], value="AudioGen", label="Model")
            with gr.Row():
                lookback = gr.Slider(0.5, 30.0, value=6.0, step=0.5, label="Lookback (s)")
                cont_len = gr.Slider(1, 60, value=12, step=1, label="Continuation Length (s)")
                ag_topk = gr.Number(label="Top-k", value=200)
                ag_topp = gr.Number(label="Top-p", value=50.0)
                ag_temp = gr.Number(label="Temperature", value=1.0)
                ag_cfg = gr.Number(label="CFG α", value=3.0)
            out_trim_ag = gr.Slider(-24.0, 0.0, value=-3.0, step=0.5, label="Output Trim (dB)")

            with gr.Row():
                out_ag = gr.Audio(label="Output", type="filepath")
            btn_ag = gr.Button("Enqueue", variant="primary")
            btn_ag.click(
                continuation,
                inputs=[model_ag, audio_in, prompt, lookback, cont_len, ag_topk, ag_topp, ag_temp, ag_cfg, out_trim_ag],
                outputs=out_ag,
                queue=True,
            )

        # ----- STEMS -----
        with gr.Tab("Stems (Demucs on GPU3 when available)"):
            if DEMUCS_AVAILABLE:
                with gr.Row():
                    audio_in2 = gr.Audio(label="Input Track", type="numpy")
                    btn_detect_in2 = gr.Button("Detect Scale+BPM")
                with gr.Row():
                    scale_in2 = gr.Textbox(label="Detected Scale", interactive=False)
                    bpm_in2 = gr.Number(label="Detected BPM", value=0, interactive=False)
                btn_detect_in2.click(_scale_bpm_wrap, audio_in2, [scale_in2, bpm_in2])
                with gr.Row():
                    drums = gr.Audio(label="Drums", type="filepath")
                with gr.Row():
                    vocals = gr.Audio(label="Vocals", type="filepath")
                with gr.Row():
                    bass = gr.Audio(label="Bass", type="filepath")
                with gr.Row():
                    other = gr.Audio(label="Other", type="filepath")
                btn_sep = gr.Button("Enqueue", variant="primary")
                btn_sep.click(separate_stems, inputs=audio_in2, outputs=[drums, vocals, bass, other], queue=True)
            else:
                gr.Markdown("⚠️ Demucs not installed. `pip install demucs` to enable stems.")

        # ----- COMBINE -----
        with gr.Tab("Combine Stems"):
            with gr.Row():
                prompt_name = gr.Textbox(label="Prompt / Name", value="")
                btn_save_name = gr.Button("", elem_classes=["icon-btn", "icon-save"])
                btn_view_name = gr.Button("", elem_classes=["icon-btn", "icon-view"])
            notebook_name = gr.Dropdown(label="Prompt Notebook", choices=[], visible=False)
            btn_save_name.click(save_prompt, inputs=prompt_name, outputs=None)
            btn_view_name.click(show_notebook, outputs=notebook_name)
            notebook_name.change(load_prompt, inputs=notebook_name, outputs=prompt_name)
            harm_scale = gr.Dropdown(SCALE_NAMES, value="C Major", label="Harmonize Scale")
            with gr.Row():
                with gr.Column():
                    drums_c = gr.Audio(label="Drums", type="filepath")
                    drums_vis = gr.Plot(label="Drums Visual")
                    bpm_d = gr.Slider(40, 220, label="BPM")
                    gain_d = gr.Slider(-60.0, 6.0, value=-6.0, step=0.5, label="Gain (dB)")
                    drums_c.upload(_waveform_plot, drums_c, drums_vis)
                    drums_c.upload(_apply_gain_file, [drums_c, gain_d], drums_c)
                    gain_d.change(_apply_gain_file, [drums_c, gain_d], drums_c)
                    scale_d = gr.Textbox(label="Scale", interactive=False)
                    btn_detect_d = gr.Button("Detect Scale+BPM")
                    btn_detect_d.click(_scale_bpm_wrap_slider, drums_c, [scale_d, bpm_d])
                    drums_fx = gr.CheckboxGroup(
                        [
                            "Reverb",
                            "Distortion",
                            "Gate",
                            "Glitch",
                            "Multiband Compressor",
                            "Rhythmic Gate",
                            "Looper",
                        ],
                        label="FX",
                    )
                    drums_prev = gr.State()
                    with gr.Row():
                        btn_prev_d = gr.Button("", elem_classes=["icon-btn", "icon-preview"])
                        btn_harm_d = gr.Button("", elem_classes=["icon-btn", "icon-harmonize"])
                        btn_rev_d = gr.Button("", elem_classes=["icon-btn", "icon-revert"])
                with gr.Column():
                    vocals_c = gr.Audio(label="Vocals", type="filepath")
                    vocals_vis = gr.Plot(label="Vocals Visual")
                    bpm_v = gr.Slider(40, 220, label="BPM")
                    gain_v = gr.Slider(-60.0, 6.0, value=-6.0, step=0.5, label="Gain (dB)")
                    vocals_c.upload(_waveform_plot, vocals_c, vocals_vis)
                    vocals_c.upload(_apply_gain_file, [vocals_c, gain_v], vocals_c)
                    gain_v.change(_apply_gain_file, [vocals_c, gain_v], vocals_c)
                    scale_v = gr.Textbox(label="Scale", interactive=False)
                    btn_detect_v = gr.Button("Detect Scale+BPM")
                    btn_detect_v.click(_scale_bpm_wrap_slider, vocals_c, [scale_v, bpm_v])
                    vocals_fx = gr.CheckboxGroup(
                        [
                            "Reverb",
                            "Distortion",
                            "Gate",
                            "Glitch",
                            "Multiband Compressor",
                            "Rhythmic Gate",
                            "Looper",
                        ],
                        label="FX",
                    )
                    vocals_prev = gr.State()
                    with gr.Row():
                        btn_prev_v = gr.Button("", elem_classes=["icon-btn", "icon-preview"])
                        btn_harm_v = gr.Button("", elem_classes=["icon-btn", "icon-harmonize"])
                        btn_rev_v = gr.Button("", elem_classes=["icon-btn", "icon-revert"])
                with gr.Column():
                    bass_c = gr.Audio(label="Bass", type="filepath")
                    bass_vis = gr.Plot(label="Bass Visual")
                    bpm_b = gr.Slider(40, 220, label="BPM")
                    gain_b = gr.Slider(-60.0, 6.0, value=-6.0, step=0.5, label="Gain (dB)")
                    bass_c.upload(_waveform_plot, bass_c, bass_vis)
                    bass_c.upload(_apply_gain_file, [bass_c, gain_b], bass_c)
                    gain_b.change(_apply_gain_file, [bass_c, gain_b], bass_c)
                    scale_b = gr.Textbox(label="Scale", interactive=False)
                    btn_detect_b = gr.Button("Detect Scale+BPM")
                    btn_detect_b.click(_scale_bpm_wrap_slider, bass_c, [scale_b, bpm_b])
                    bass_fx = gr.CheckboxGroup(
                        [
                            "Reverb",
                            "Distortion",
                            "Gate",
                            "Glitch",
                            "Multiband Compressor",
                            "Rhythmic Gate",
                            "Looper",
                        ],
                        label="FX",
                    )
                    bass_prev = gr.State()
                    with gr.Row():
                        btn_prev_b = gr.Button("", elem_classes=["icon-btn", "icon-preview"])
                        btn_harm_b = gr.Button("", elem_classes=["icon-btn", "icon-harmonize"])
                        btn_rev_b = gr.Button("", elem_classes=["icon-btn", "icon-revert"])
                with gr.Column():
                    other_c = gr.Audio(label="Other", type="filepath")
                    other_vis = gr.Plot(label="Other Visual")
                    bpm_o = gr.Slider(40, 220, label="BPM")
                    gain_o = gr.Slider(-60.0, 6.0, value=-6.0, step=0.5, label="Gain (dB)")
                    other_c.upload(_waveform_plot, other_c, other_vis)
                    other_c.upload(_apply_gain_file, [other_c, gain_o], other_c)
                    gain_o.change(_apply_gain_file, [other_c, gain_o], other_c)
                    scale_o = gr.Textbox(label="Scale", interactive=False)
                    btn_detect_o = gr.Button("Detect Scale+BPM")
                    btn_detect_o.click(_scale_bpm_wrap_slider, other_c, [scale_o, bpm_o])
                    other_fx = gr.CheckboxGroup(
                        [
                            "Reverb",
                            "Distortion",
                            "Gate",
                            "Glitch",
                            "Multiband Compressor",
                            "Rhythmic Gate",
                            "Looper",
                        ],
                        label="FX",
                    )
                    other_prev = gr.State()
                    with gr.Row():
                        btn_prev_o = gr.Button("", elem_classes=["icon-btn", "icon-preview"])
                        btn_harm_o = gr.Button("", elem_classes=["icon-btn", "icon-harmonize"])
                        btn_rev_o = gr.Button("", elem_classes=["icon-btn", "icon-revert"])
            with gr.Accordion("Effect Settings", open=False):
                reverb_amt = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Reverb")
                dist_amt = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Distortion")
                gate_amt = gr.Slider(0.0, 60.0, value=0.0, step=1.0, label="Gate Threshold (dB)")
                glitch_amt = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Glitch Intensity")
                comp_amt = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Multiband Compressor Mix")
                rg_freq = gr.Dropdown(
                    ["1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4"],
                    value="1/4",
                    label="Gate Frequency (beats)",
                )
                rg_dur = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Gate Duration")
                rg_loc = gr.Dropdown(
                    ["start", "middle", "end"], value="start", label="Gate Location"
                )
                rg_pattern = gr.Dropdown(
                    ["flat", "trance", "build_single", "build_double", "slow", "aggro"],
                    value="flat",
                    label="Gate Pattern",
                )
                loop_dur = gr.Slider(1, 64, value=4, step=1, label="Loop Duration (measures)")
            btn_prev_d.click(
                _preview_wrap,
                inputs=[
                    drums_c,
                    drums_fx,
                    bpm_d,
                    gain_d,
                    reverb_amt,
                    dist_amt,
                    gate_amt,
                    glitch_amt,
                    comp_amt,
                    rg_freq,
                    rg_dur,
                    rg_loc,
                    rg_pattern,
                    loop_dur,
                ],
                outputs=[drums_c, drums_prev],
            )
            btn_harm_d.click(
                _harmonize_wrap,
                inputs=[drums_c, harm_scale],
                outputs=[drums_c, drums_prev],
            )
            btn_rev_d.click(
                _revert_wrap,
                inputs=[drums_prev, drums_c],
                outputs=[drums_c, drums_prev],
            )
            btn_prev_v.click(
                _preview_wrap,
                inputs=[
                    vocals_c,
                    vocals_fx,
                    bpm_v,
                    gain_v,
                    reverb_amt,
                    dist_amt,
                    gate_amt,
                    glitch_amt,
                    comp_amt,
                    rg_freq,
                    rg_dur,
                    rg_loc,
                    rg_pattern,
                    loop_dur,
                ],
                outputs=[vocals_c, vocals_prev],
            )
            btn_harm_v.click(
                _harmonize_wrap,
                inputs=[vocals_c, harm_scale],
                outputs=[vocals_c, vocals_prev],
            )
            btn_rev_v.click(
                _revert_wrap,
                inputs=[vocals_prev, vocals_c],
                outputs=[vocals_c, vocals_prev],
            )
            btn_prev_b.click(
                _preview_wrap,
                inputs=[
                    bass_c,
                    bass_fx,
                    bpm_b,
                    gain_b,
                    reverb_amt,
                    dist_amt,
                    gate_amt,
                    glitch_amt,
                    comp_amt,
                    rg_freq,
                    rg_dur,
                    rg_loc,
                    rg_pattern,
                    loop_dur,
                ],
                outputs=[bass_c, bass_prev],
            )
            btn_harm_b.click(
                _harmonize_wrap,
                inputs=[bass_c, harm_scale],
                outputs=[bass_c, bass_prev],
            )
            btn_rev_b.click(
                _revert_wrap,
                inputs=[bass_prev, bass_c],
                outputs=[bass_c, bass_prev],
            )
            btn_prev_o.click(
                _preview_wrap,
                inputs=[
                    other_c,
                    other_fx,
                    bpm_o,
                    gain_o,
                    reverb_amt,
                    dist_amt,
                    gate_amt,
                    glitch_amt,
                    comp_amt,
                    rg_freq,
                    rg_dur,
                    rg_loc,
                    rg_pattern,
                    loop_dur,
                ],
                outputs=[other_c, other_prev],
            )
            btn_harm_o.click(
                _harmonize_wrap,
                inputs=[other_c, harm_scale],
                outputs=[other_c, other_prev],
            )
            btn_rev_o.click(
                _revert_wrap,
                inputs=[other_prev, other_c],
                outputs=[other_c, other_prev],
            )
            out_mix = gr.Audio(label="Output Mix", type="filepath")
            btn_combine = gr.Button("Combine", variant="primary")
            btn_combine.click(
                combine_stems,
                inputs=[
                    drums_c,
                    drums_fx,
                    bpm_d,
                    gain_d,
                    vocals_c,
                    vocals_fx,
                    bpm_v,
                    gain_v,
                    bass_c,
                    bass_fx,
                    bpm_b,
                    gain_b,
                    other_c,
                    other_fx,
                    bpm_o,
                    gain_o,
                    output_folder,
                    prompt_name,
                    reverb_amt,
                    dist_amt,
                    gate_amt,
                    glitch_amt,
                    comp_amt,
                    rg_freq,
                    rg_dur,
                    rg_loc,
                    rg_pattern,
                    loop_dur,
                ],
                outputs=out_mix,
            )
        # ----- ANALYZE -----
        with gr.Tab("Analyze"):
            analyze_in = gr.Files(
                label="Samples", type="filepath", file_types=["audio"]
            )
            analyze_out = gr.Dataframe(
                headers=["Description", "Key", "BPM", "Renamed File"],
                datatype=["str", "str", "number", "str"],
                label="Analysis",
                interactive=False,
            )
            btn_analyze = gr.Button("Analyze")
            btn_analyze.click(
                _analyze_and_rename_batch_files,
                inputs=analyze_in,
                outputs=analyze_out,
                queue=analyze_queue,
            )

        # ----- MASTERING -----
        with gr.Tab("Mastering"):
            with gr.Row():
                audio_in3 = gr.Audio(label="Input Track", type="numpy")
                btn_detect_mst = gr.Button("Detect Scale+BPM")
            with gr.Row():
                ref_master = gr.Audio(label="Reference Track", type="filepath")
                btn_detect_ref = gr.Button("Detect Scale+BPM")
            scale_master = gr.Textbox(label="Detected Scale", interactive=False)
            bpm_master = gr.Number(label="Detected BPM", value=0, interactive=False)
            btn_detect_mst.click(_scale_bpm_wrap, audio_in3, [scale_master, bpm_master])
            btn_detect_ref.click(_scale_bpm_wrap, ref_master, [scale_master, bpm_master])
            out_trim_master = gr.Slider(-24.0, 0.0, value=-1.0, step=0.5, label="Output Trim (dB)")
            width_master = gr.Slider(0.5, 3.0, value=1.5, step=0.1, label="Stereo Width")
            pan_master = gr.Slider(-1.0, 1.0, value=0.0, step=0.1, label="Stereo Pan")
            bass_master = gr.Slider(0.0, 12.0, value=0.0, step=0.5, label="Bass Boost (dB)")
            bass_width = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Bass Width", info="0=mono")
            freq_sliders = []
            with gr.Accordion("Frequency Cuts", open=False):
                eq_defaults = USER_SETTINGS.get("eq_gains", [0.0] * len(FREQ_BANDS))
                for idx, (label, low, high, desc) in enumerate(FREQ_BANDS):
                    rng = f"{low}-{high} Hz" if low != high else f"{low} Hz"
                    freq_sliders.append(
                        gr.Slider(
                            -12.0,
                            12.0,
                            value=eq_defaults[idx] if idx < len(eq_defaults) else 0.0,
                            step=0.5,
                            label=f"{label} ({rng})",
                            info=desc,
                        )
                    )
            master_method = gr.Radio(["Matchering", "AudioSR"], value="Matchering", label="Method")
            out_master = gr.Audio(label="Output", type="filepath")
            btn_master = gr.Button("Enqueue", variant="primary")
            btn_master.click(
                master_track,
                inputs=[
                    audio_in3,
                    ref_master,
                    out_trim_master,
                    width_master,
                    pan_master,
                    bass_master,
                    bass_width,
                    *freq_sliders,
                    master_method,
                ],
                outputs=out_master,
                queue=True,
            )

        with gr.Accordion("Settings", open=False):
            out_box = gr.Textbox(value=str(TMP_DIR), label="Output Folder")
            set_btn = gr.Button("Set")
            set_btn.click(lambda x: x, inputs=out_box, outputs=output_folder)

            gpu_opts = [f"cuda:{i}" for i in range(max(1, torch.cuda.device_count()))]

            def _gpu_row(label, value):
                with gr.Row():
                    gr.Markdown(label)
                    return gr.Radio(gpu_opts, value=value, label="", interactive=True)

            gr.Markdown("**GPU Selection**")
            style_gpu = _gpu_row("Style", str(STYLE_DEVICE))
            mg_large_gpu = _gpu_row("MusicGen Large", str(LARGE_DEVICE))
            ag_gpu = _gpu_row("AudioGen Medium", str(AUDIOGEN_DEVICE))
            audiosr_gpu = _gpu_row("AudioSR", str(AUDIOSR_DEVICE))
            diffusion_gpu = _gpu_row("Diffusion", str(DIFFUSION_DEVICE))

            apply_gpu = gr.Button("Apply GPU Assignments")
            gpu_status = gr.Markdown("")

            apply_gpu.click(
                _apply_gpus,
                [style_gpu, mg_large_gpu, ag_gpu, audiosr_gpu, diffusion_gpu],
                gpu_status,
            )

            shard_box = gr.Textbox(
                value=CUSTOM_SHARD_RAW,
                label="Custom Shard Devices",
                placeholder="cuda:0,cuda:1",
            )
            model_opts = gr.Dropdown(
                ["Style", "Medium", "Large", "AudioGen"],
                value=MODEL_OPTIONS,
                label="Models",
                multiselect=True,
            )

            save_btn = gr.Button("Save Settings")
            save_status = gr.Markdown("", elem_id="save_status")

            def _save_settings(shard_str, models, *gains):
                global CUSTOM_SHARD_RAW, CUSTOM_SHARD_DEVICES, MODEL_OPTIONS
                CUSTOM_SHARD_RAW = shard_str
                CUSTOM_SHARD_DEVICES = [
                    torch.device(d.strip()) for d in shard_str.split(",") if d.strip()
                ]
                MODEL_OPTIONS = models
                cfg = {
                    "eq_gains": list(gains),
                    "shard_devices": shard_str,
                    "model_options": models,
                }
                save_settings(cfg)
                return "✅ Settings saved"

            save_btn.click(_save_settings, [shard_box, model_opts, *freq_sliders], save_status)

        # Global queue
        demo.queue(default_concurrency_limit=1, max_size=32).launch(**launch_kwargs)

# ---------- Main [UNCHANGED] ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.INFO)
    ui_full(
        {
            "server_name": args.listen,
            "server_port": args.port,
            "allowed_paths": [str(TMP_DIR)],
        }
    )



