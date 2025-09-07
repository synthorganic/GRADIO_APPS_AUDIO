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
import warnings
import uuid
from pathlib import Path
import subprocess as sp
import sys
import shutil
import numpy as np

import torch
import torch.nn.functional as F
import gradio as gr

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen, MultiBandDiffusion, AudioGen

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

# ---------- Devices [ALTERED] ----------
# High‑VRAM GPUs 0 & 1 host the heavy generation models.  Smaller GPUs 2 & 3
# are reserved for diffusion/utility work so that section composer can always
# offload to MultiBandDiffusion without exhausting memory.
STYLE_DEVICE = torch.device("cuda:0")        # Style + Large on GPU0
MEDIUM_DEVICE = (
    torch.device("cuda:1")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1
    else torch.device("cuda:0")
)
LARGE_DEVICE = STYLE_DEVICE                  # Large shares GPU0
AUDIOGEN_DEVICE = MEDIUM_DEVICE              # AudioGen on GPU1
DIFFUSION_DEVICE = (
    torch.device("cuda:2")
    if torch.cuda.is_available() and torch.cuda.device_count() > 2
    else torch.device("cpu")
)
UTILITY_DEVICE = (
    torch.device("cuda:3")
    if torch.cuda.is_available() and torch.cuda.device_count() > 3
    else DIFFUSION_DEVICE
)
# ``AudioSR`` is lightweight enough to duplicate across two GPUs. Keep a list
# of devices to round‑robin requests between them so that long upscales don't
# monopolise a single card.  When only one GPU is available we fall back to the
# utility device.
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
TMP_DIR = Path("/home/archway/music/n-Track")  # writable, persistent
TMP_DIR.mkdir(parents=True, exist_ok=True)

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
# Cache one ``AudioSR`` model per device and keep an index for load balancing
# across ``AUDIOSR_DEVICES``.
AUDIOSR_MODELS = {}
_AUDIOSR_NEXT_DEVICE = 0

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
    """Push ``model`` back to CPU and free the current CUDA cache."""
    _move_musicgen(model, torch.device("cpu"))
    torch.cuda.empty_cache()

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
    return queue, gr.update(choices=queue, value=None), ""


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
        _move_to_device(STYLE_MBD, torch.device("cpu"))
        STYLE_MBD.device = torch.device("cpu")


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


def style_predict(text, melody, duration=10, topk=200, topp=50.0, temperature=1.0,
                  cfg_coef=3.0, double_cfg="Yes", cfg_coef_beta=5.0,
                  eval_q=3, excerpt_length=3.0, decoder="Default",
                  out_trim_db=-3.0):
    """Generate with MusicGen-Style, optional melody excerpt."""
    style_load_model()
    _move_musicgen(STYLE_MODEL, STYLE_DEVICE)
    STYLE_MODEL.set_generation_params(duration=int(duration),
                                      top_k=int(topk), top_p=float(topp),
                                      temperature=float(temperature),
                                      cfg_coef=float(cfg_coef),
                                      cfg_coef_beta=float(cfg_coef_beta) if double_cfg == "Yes" else None)
    STYLE_MODEL.set_style_conditioner_params(eval_q=int(eval_q),
                                             excerpt_length=float(excerpt_length))

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
        outputs = STYLE_MODEL.generate_with_chroma(
            descriptions=[text or "style generation"],
            melody_wavs=melody_tensor,
            melody_sample_rate=TARGET_SR,
            return_tokens=STYLE_USE_DIFFUSION
        )
    else:
        outputs = STYLE_MODEL.generate([text or "style generation"],
                                       return_tokens=STYLE_USE_DIFFUSION)

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

# ============================================================================
# AUDIOGEN CONTINUATION TAB [ALTERED]
# ============================================================================
def audiogen_load_model(name: str = "facebook/audiogen-medium"):
    global AUDIOGEN_MODEL
    if AUDIOGEN_MODEL is None:
        print(f"[AudioGen] Loading {name} on {AUDIOGEN_DEVICE} ...")
        AUDIOGEN_MODEL = AudioGen.get_pretrained(name)
        AUDIOGEN_MODEL.device = AUDIOGEN_DEVICE
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
    model.set_generation_params(
        duration=int(duration),
        top_k=int(topk),
        top_p=float(topp),
        temperature=float(temperature),
        cfg_coef=float(cfg_coef),
    )

    # 3) Emulate continuation with crossfade
    print("[AudioGen] Generating new segment for crossfade continuation.")
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


def _nearest_scale_pitch(freq: float, scale: str) -> float:
    if freq <= 0:
        return freq
    midi = round(_freq_to_midi(freq))
    allowed = SCALE_NOTES.get(scale, SCALE_NOTES["C Major"])
    while (midi % 12) not in allowed:
        midi += 1
    return _midi_to_freq(midi)


def detect_bpm(path: str) -> float:
    """Return estimated BPM of file using librosa."""
    if not (LIBROSA_AVAILABLE and path):  # pragma: no cover - optional
        return 0.0
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)
    except Exception:
        return 0.0


def harmonize_file(path: str, scale: str) -> str:
    """Apply simple harmonization to ``path`` in-place and return new path."""
    if not (LIBROSA_AVAILABLE and SOUNDFILE_AVAILABLE and path):  # pragma: no cover
        raise gr.Error("librosa and soundfile are required for harmonize")

    y, sr = librosa.load(path, sr=44100, mono=True)
    f0 = librosa.yin(y, fmin=50, fmax=1000, frame_length=2048, hop_length=512)

    harmonized = np.zeros_like(y)
    for i, freq in enumerate(f0):
        if freq <= 0:
            continue
        root = _nearest_scale_pitch(freq, scale)
        start = i * 512
        end = min(start + 2048, len(y))
        t = np.arange(start, end) / sr
        root_wave = np.sin(2 * np.pi * root * t)
        harm_waves = []
        for interval in [4, 7]:  # major 3rd, perfect 5th
            new_freq = _midi_to_freq(_freq_to_midi(root) + interval)
            harm_waves.append(np.sin(2 * np.pi * new_freq * t))
        frame_wave = root_wave + sum(harm_waves)
        frame_wave /= (1 + len(harm_waves))
        harmonized[start:end] += frame_wave[: end - start]

    if np.max(np.abs(harmonized)) > 0:
        harmonized /= np.max(np.abs(harmonized))

    out_path = Path(path).with_suffix("")
    out_file = TMP_DIR / f"harm_{uuid.uuid4().hex}.wav"
    sf.write(out_file, harmonized, sr)
    return str(out_file)


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


def combine_stems(
    drums_path: str | None,
    vocals_path: str | None,
    bass_path: str | None,
    other_path: str | None,
    scale: str,
    sidechain: float,
    out_dir: str,
    prompt: str,
    reverb: float,
    dist: float,
    gate: float,
):
    """Load stems, optionally harmonize/sidechain/effect, then mix."""

    if not SOUNDFILE_AVAILABLE and not LIBROSA_AVAILABLE:
        raise gr.Error("soundfile or librosa required")

    paths = [drums_path, vocals_path, bass_path, other_path]
    stems = []
    sr = 44100
    for p in paths:
        if p and Path(p).exists():
            if LIBROSA_AVAILABLE:
                y, sr = librosa.load(p, sr=sr, mono=True)
            elif SOUNDFILE_AVAILABLE:
                y, sr = sf.read(p)
                if y.ndim > 1:
                    y = y.mean(axis=1)
            else:  # pragma: no cover
                raise gr.Error("librosa or soundfile required to load audio")
            stems.append(y)
        else:
            stems.append(None)

    max_len = max(len(s) if s is not None else 0 for s in stems)
    mix = np.zeros(max_len)

    # Basic sidechain: reduce others using drum amplitude envelope
    drum_env = None
    if LIBROSA_AVAILABLE and stems[0] is not None and sidechain > 0:
        env = np.abs(stems[0])
        env = librosa.util.normalize(env)
        drum_env = env[:max_len]

    for idx, s in enumerate(stems):
        if s is None:
            continue
        if len(s) < max_len:
            s = np.pad(s, (0, max_len - len(s)))
        if idx != 0 and drum_env is not None:
            s = s * (1 - sidechain * drum_env)
        mix += s

    mix = _apply_pedalboard(mix, sr, reverb, dist, gate)

    out_dir = Path(out_dir) if out_dir else TMP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = (prompt or "output")[:10]
    orig = Path(drums_path or vocals_path or bass_path or other_path or "mix.wav").name
    fname = f"{prefix}_combine_{orig}"
    out_path = out_dir / fname
    sf.write(out_path, mix, sr)
    return str(out_path)

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
    in_path = TMP_DIR / f"audiosr_in_{uuid.uuid4().hex}.wav"
    audio_write(str(in_path), torch.from_numpy(wav_np).float().t(), sr, add_suffix=False)
    model = audiosr_load_model()
    out = audiosr_super_resolution(model, str(in_path))
    if isinstance(out, torch.Tensor):
        out = out.detach().cpu().numpy()
    out_arr = out[0]
    if out_arr.ndim == 1:
        out_arr = out_arr[None, :]
    out_path = TMP_DIR / f"audiosr_out_{uuid.uuid4().hex}.wav"
    audio_write(str(out_path), torch.from_numpy(out_arr).float(), 48000, add_suffix=False)
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

# ============================================================================
# UI (tabs, all Enqueue) [ALTERED]
# ============================================================================
def ui_full(launch_kwargs):
    with gr.Blocks() as demo:
        gr.Markdown("# 🎛️ Music Suite — Style • AudioGen Continuation • Stems  \n*Enqueue buttons; global queue enabled*")
        queue_items = gr.State([])
        output_folder = gr.State(str(TMP_DIR))

        # ----- QUEUE MANAGER -----
        with gr.Tab("Queue"):
            queue_display = gr.Dropdown(label="Queued Items", choices=[], interactive=True)
            with gr.Row():
                new_item = gr.Textbox(label="New Item")
                btn_add_q = gr.Button("Add")
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
                text = gr.Textbox(label="Text Prompt", placeholder="e.g., glossy synthwave with gated drums")
                melody = gr.Audio(label="Style Excerpt (optional)", type="numpy")
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

        # ----- SECTION COMPOSER -----
        with gr.Tab("Section Composer"):
            init_audio = gr.Audio(label="Initial Audio (optional)", type="numpy")
            section_count = gr.State(1)
            section_rows = []
            section_inputs = []
            for i in range(MAX_SECTIONS):
                with gr.Row(visible=(i == 0)) as row:
                    sec_type = gr.Dropdown(
                        ["Intro", "Build", "Break", "Drop", "Bridge", "Bed", "Outro"],
                        value="Intro",
                        label=f"Section {i+1}",
                    )
                    sec_prompt = gr.Textbox(label="Prompt")
                    sec_length = gr.Number(label="Length (s)", value=8)
                section_rows.append(row)
                section_inputs.extend([sec_type, sec_prompt, sec_length])
            with gr.Row():
                btn_add = gr.Button("Add Section")
                btn_del = gr.Button("Delete Section")
            btn_add.click(add_section, inputs=section_count, outputs=[section_count] + section_rows, queue=False)
            btn_del.click(remove_section, inputs=section_count, outputs=[section_count] + section_rows, queue=False)
            with gr.Row():
                bpm = gr.Slider(40, 240, value=120, step=1, label="Tempo (BPM)")
                xf_beats = gr.Slider(0.0, 8.0, value=1.0, step=0.25, label="Crossfade (beats)")
            decoder = gr.Radio(["Default", "MultiBand_Diffusion"], value="Default", label="Decoder")
            out_trim_sections = gr.Slider(-24.0, 0.0, value=-3.0, step=0.5, label="Output Trim (dB)")
            out_sections = gr.Audio(label="Output", type="filepath")
            btn_sections = gr.Button("Enqueue", variant="primary")
            btn_sections.click(
                compose_sections_ui,
                inputs=[init_audio, section_count] + section_inputs + [bpm, xf_beats, decoder, out_trim_sections],
                outputs=out_sections,
                queue=True,
            )

        # ----- AUDIOGEN CONTINUATION -----
        with gr.Tab("AudioGen Continuation (GPU0)"):
            audio_in = gr.Audio(label="Input Clip", type="numpy")
            with gr.Row():
                prompt = gr.Textbox(label="Prompt (optional)", placeholder="e.g., keep the groove, add arps")
            with gr.Row():
                lookback = gr.Slider(0.5, 30.0, value=6.0, step=0.5, label="Lookback (s)")
                cont_len = gr.Slider(1, 60, value=12, step=1, label="Continuation Length (s)")
                ag_topk = gr.Number(label="Top-k", value=200)
                ag_topp = gr.Number(label="Top-p", value=50.0)
                ag_temp = gr.Number(label="Temperature", value=1.0)
                ag_cfg = gr.Number(label="CFG α", value=3.0)
            out_trim_ag = gr.Slider(-24.0, 0.0, value=-3.0, step=0.5, label="Output Trim (dB)")

            out_ag = gr.Audio(label="Output", type="filepath")
            btn_ag = gr.Button("Enqueue", variant="primary")
            btn_ag.click(
                audiogen_continuation,
                inputs=[audio_in, prompt, lookback, cont_len, ag_topk, ag_topp, ag_temp, ag_cfg, out_trim_ag],
                outputs=out_ag,
                queue=True,
            )

        # ----- STEMS -----
        with gr.Tab("Stems (Demucs on GPU3 when available)"):
            if DEMUCS_AVAILABLE:
                audio_in2 = gr.Audio(label="Input Track", type="numpy")
                drums = gr.Audio(label="Drums", type="filepath")
                vocals = gr.Audio(label="Vocals", type="filepath")
                bass = gr.Audio(label="Bass", type="filepath")
                other = gr.Audio(label="Other", type="filepath")
                btn_sep = gr.Button("Enqueue", variant="primary")
                btn_sep.click(separate_stems, inputs=audio_in2, outputs=[drums, vocals, bass, other], queue=True)
            else:
                gr.Markdown("⚠️ Demucs not installed. `pip install demucs` to enable stems.")

        # ----- COMBINE -----
        with gr.Tab("Combine Stems"):
            scale_sel = gr.Dropdown(SCALE_NAMES, value="C Major", label="Harmonize Scale")
            sidechain_amt = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Sidechain Drums → Others")
            reverb_amt = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Reverb")
            dist_amt = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Distortion")
            gate_amt = gr.Slider(0.0, 60.0, value=0.0, step=1.0, label="Gate Threshold (dB)")
            prompt_name = gr.Textbox(label="Prompt / Name", value="")

            def _bpm_wrap(p):
                return detect_bpm(p)

            def _harm_wrap(p, scale):
                return harmonize_file(p, scale)

            with gr.Row():
                drums_c = gr.Audio(label="Drums", type="filepath")
                bpm_d = gr.Slider(40, 220, label="Drums BPM")
                drums_c.change(_bpm_wrap, drums_c, bpm_d)
                btn_hd = gr.Button("Harmonize")
                btn_hd.click(_harm_wrap, [drums_c, scale_sel], drums_c)

            with gr.Row():
                vocals_c = gr.Audio(label="Vocals", type="filepath")
                bpm_v = gr.Slider(40, 220, label="Vocals BPM")
                vocals_c.change(_bpm_wrap, vocals_c, bpm_v)
                btn_hv = gr.Button("Harmonize")
                btn_hv.click(_harm_wrap, [vocals_c, scale_sel], vocals_c)

            with gr.Row():
                bass_c = gr.Audio(label="Bass", type="filepath")
                bpm_b = gr.Slider(40, 220, label="Bass BPM")
                bass_c.change(_bpm_wrap, bass_c, bpm_b)
                btn_hb = gr.Button("Harmonize")
                btn_hb.click(_harm_wrap, [bass_c, scale_sel], bass_c)

            with gr.Row():
                other_c = gr.Audio(label="Other", type="filepath")
                bpm_o = gr.Slider(40, 220, label="Other BPM")
                other_c.change(_bpm_wrap, other_c, bpm_o)
                btn_ho = gr.Button("Harmonize")
                btn_ho.click(_harm_wrap, [other_c, scale_sel], other_c)

            out_mix = gr.Audio(label="Output Mix", type="filepath")
            btn_combine = gr.Button("Combine", variant="primary")
            btn_combine.click(
                combine_stems,
                inputs=[drums_c, vocals_c, bass_c, other_c, scale_sel, sidechain_amt, output_folder, prompt_name, reverb_amt, dist_amt, gate_amt],
                outputs=out_mix,
            )

        # ----- MASTERING -----
        with gr.Tab("Mastering"):
            audio_in3 = gr.Audio(label="Input Track", type="numpy")
            ref_master = gr.Audio(label="Reference Track", type="filepath")
            out_trim_master = gr.Slider(-24.0, 0.0, value=-1.0, step=0.5, label="Output Trim (dB)")
            width_master = gr.Slider(0.5, 3.0, value=1.5, step=0.1, label="Stereo Width")
            pan_master = gr.Slider(-1.0, 1.0, value=0.0, step=0.1, label="Stereo Pan")
            bass_master = gr.Slider(0.0, 12.0, value=0.0, step=0.5, label="Bass Boost (dB)")
            bass_width = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Bass Width", info="0=mono")
            freq_sliders = []
            with gr.Accordion("Frequency Cuts", open=False):
                for label, low, high, desc in FREQ_BANDS:
                    rng = f"{low}-{high} Hz" if low != high else f"{low} Hz"
                    freq_sliders.append(
                        gr.Slider(
                            -12.0,
                            12.0,
                            value=0.0,
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

        # Global queue
        demo.queue(concurrency_count=1, max_size=32).launch(**launch_kwargs)

# ---------- Main [UNCHANGED] ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.INFO)
    ui_full({"server_name": args.listen, "server_port": args.port})



