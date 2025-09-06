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

# ---------- Devices [ALTERED] ----------
# Push section-composer workloads onto GPUs 2 & 3 so that heavy MusicGen
# models can be swapped in and out without exhausting memory.  Style and
# large models share GPU2 while the medium model lives on GPU3 when
# available.  AudioGen remains on GPU1.
STYLE_DEVICE = torch.device("cuda:2")        # Style model on GPU2
MEDIUM_DEVICE = (
    torch.device("cuda:3")
    if torch.cuda.is_available() and torch.cuda.device_count() > 3
    else torch.device("cuda:2")
)
LARGE_DEVICE = STYLE_DEVICE                  # Large shares GPU2
AUDIOGEN_DEVICE = torch.device("cuda:1")     # AudioGen on GPU1
UTILITY_DEVICE = (
    torch.device("cuda:3")
    if torch.cuda.is_available() and torch.cuda.device_count() > 3
    else torch.device("cpu")
)

print(
    f"[Boot] STYLE: {STYLE_DEVICE} | MEDIUM: {MEDIUM_DEVICE} | "
    f"LARGE: {LARGE_DEVICE} | AUDIOGEN: {AUDIOGEN_DEVICE} | "
    f"UTILITY(preproc/demucs): {UTILITY_DEVICE}"
)

# ---------- Constants & paths [ALTERED] ----------
TARGET_SR = 32000
TARGET_AC = 1
TMP_DIR = Path("/home/archway/music/n-Track")  # writable, persistent
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Caches [UNCHANGED] ----------
STYLE_MODEL = None
STYLE_MBD = None
STYLE_USE_DIFFUSION = False
AUDIOGEN_MODEL = None
MEDIUM_MODEL = None
LARGE_MODEL = None

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
    model.device = device


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
                _move_to_device(STYLE_MBD, device)
                STYLE_MBD.device = device
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


def style_predict(text, melody, duration=10, topk=250, topp=0.0, temperature=1.0,
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
            _move_to_device(STYLE_MBD, STYLE_DEVICE)
            STYLE_MBD.device = STYLE_DEVICE
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
    topk: int = 250,
    topp: float = 0.0,
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
# MASTERING TAB UTILITIES [NEW]
# ============================================================================

def _ffmpeg_basic_cleanup(in_path: Path, out_path: Path, sr: int) -> None:
    """Apply basic corrective EQ via ffmpeg to mitigate common artefacts."""
    hums = [50, 60, 100, 120, 150, 180]
    filters = ["highpass=f=30", "highpass=f=40"]
    # ``equalizer`` is widely available in ffmpeg whereas ``anequalizer``
    # lacks the "f" option on older builds.  Using ``equalizer`` avoids
    # "Option 'f' not found" runtime failures.  Make reductions 10x subtler.
    filters += [f"equalizer=f={h}:t=o:w=2:g=-2.5" for h in hums]
    if 15600 < TARGET_SR / 2:
        filters.append("equalizer=f=15600:t=o:w=2:g=-2.5")
    filters.append("lowpass=f=18000")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_path),
        "-af",
        ",".join(filters),
        "-ar",
        str(sr),
        "-sample_fmt",
        "s32",
        str(out_path),
    ]
    sp.run(cmd, check=True)


def _apply_harmonic_exciter(in_path: Path, out_path: Path, freq: float, sr: int) -> None:
    """Light harmonic excitement using ffmpeg's aexciter filter."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_path),
        "-af",
        f"aexciter=freq={freq}:amount=0.1",
        "-ar",
        str(sr),
        "-sample_fmt",
        "s32",
        str(out_path),
    ]
    sp.run(cmd, check=True)


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
        f"stereotools=slev={width_s}:balance_out={pan_s}",
        "-ar",
        str(sr),
        "-sample_fmt",
        "s32",
        str(out_path),
    ]
    try:
        sp.run(cmd, check=True)
    except sp.CalledProcessError:
        fallback = [
            "ffmpeg",
            "-y",
            "-i",
            str(in_path),
            "-af",
            f"stereowiden=delay=20:feedback=0.3:crossfeed={max(0.0, width_s - 1.0)}:drymix=0.8",
            "-ar",
            str(sr),
            "-sample_fmt",
            "s32",
            str(out_path),
        ]
        sp.run(fallback, check=True)


def _beat_match_and_harmonize(stem_paths: list[Path], out_path: Path, sr: int) -> None:
    """Recombine stems with basic beat matching and harmonization.

    Uses ffmpeg's ``amix`` to sum the stems and ``aresample=async`` to keep
    streams aligned. Acts as a lightweight placeholder for more advanced
    alignment or key detection.
    """
    cmd = ["ffmpeg", "-y"]
    for p in stem_paths:
        cmd.extend(["-i", str(p)])
    filter_complex = f"amix=inputs={len(stem_paths)}:normalize=0,aresample=async=1"
    cmd.extend(["-filter_complex", filter_complex, "-ar", str(sr), "-sample_fmt", "s32", str(out_path)])
    sp.run(cmd, check=True)


def _master_simple(audio_input, out_trim_db: float = -1.0):
    if not MATCHERING_AVAILABLE:
        raise gr.Error("Matchering not installed. `pip install matchering`")
    try:
        sr, wav_np = audio_input
    except Exception:
        raise gr.Error("Please provide an audio clip.")
    in_path = TMP_DIR / f"master_in_{uuid.uuid4().hex}.wav"
    audio_write(str(in_path), torch.from_numpy(wav_np).float().t(), sr, add_suffix=False)
    ref = Path.home() / "references" / "reference.wav"
    if not ref.exists():
        raise gr.Error(f"Reference file missing: {ref}")
    matched_path = TMP_DIR / f"mastered_simple_{uuid.uuid4().hex}.wav"
    _matchering_match(target=str(in_path), reference=str(ref), output=str(matched_path))
    widened_path = TMP_DIR / f"mastered_simple_wide_{uuid.uuid4().hex}.wav"
    _apply_stereo_space(matched_path, widened_path, sr)
    return str(widened_path)


def _master_complex(audio_input, out_trim_db: float = -1.0):
    """Complex mastering pipeline with per-stem Matchering and recombination."""
    if not MATCHERING_AVAILABLE:
        raise gr.Error("Matchering not installed. `pip install matchering`")
    if not DEMUCS_AVAILABLE:
        raise gr.Error("Demucs not installed. `pip install demucs`")
    try:
        sr, _ = audio_input
    except Exception:
        raise gr.Error("Please provide an audio clip.")
    stems = separate_stems(audio_input)
    ref = Path.home() / "references" / "reference.wav"
    if not ref.exists():
        raise gr.Error(f"Reference file missing: {ref}")

    processed = []
    for stem_path in stems:
        stem_p = Path(stem_path)
        p_path = TMP_DIR / f"proc_{stem_p.stem}_{uuid.uuid4().hex}.wav"
        _ffmpeg_basic_cleanup(stem_p, p_path, sr)

        if stem_p.stem in {"vocals", "other"}:
            freq = 4000 if stem_p.stem == "vocals" else 8000
            e_path = TMP_DIR / f"excite_{stem_p.stem}_{uuid.uuid4().hex}.wav"
            _apply_harmonic_exciter(p_path, e_path, freq, sr)
            p_path = e_path

        m_path = TMP_DIR / f"master_{stem_p.stem}_{uuid.uuid4().hex}.wav"
        _matchering_match(target=str(p_path), reference=str(ref), output=str(m_path))

        w_path = TMP_DIR / f"stereo_{stem_p.stem}_{uuid.uuid4().hex}.wav"
        _apply_stereo_space(m_path, w_path, sr)
        processed.append(w_path)

    if len(processed) != 4:
        raise gr.Error(f"Expected 4 stems (drums, vocals, bass, other); got {len(processed)}")

    mix_path = TMP_DIR / f"mix_{uuid.uuid4().hex}.wav"
    _beat_match_and_harmonize(processed, mix_path, sr)

    final_path = TMP_DIR / f"mastered_complex_{uuid.uuid4().hex}.wav"
    _matchering_match(target=str(mix_path), reference=str(ref), output=str(final_path))
    resamp_path = TMP_DIR / f"mastered_complex_sr_{uuid.uuid4().hex}.wav"
    sp.run(["ffmpeg", "-y", "-i", str(final_path), "-ar", str(sr), "-sample_fmt", "s32", str(resamp_path)], check=True)
    return str(resamp_path)


def master_track(audio_input, pathway: str, out_trim_db: float = -1.0):
    if pathway == "Simple":
        return _master_simple(audio_input, out_trim_db)
    return _master_complex(audio_input, out_trim_db)

# ============================================================================
# UI (three tabs, all Enqueue) [ALTERED]
# ============================================================================
def ui_full(launch_kwargs):
    with gr.Blocks() as demo:
        gr.Markdown("# 🎛️ Music Suite — Style • AudioGen Continuation • Stems  \n*Enqueue buttons; global queue enabled*")

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
                topk = gr.Number(label="Top-k", value=250)
                topp = gr.Number(label="Top-p", value=0.0)
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
            btn_add = gr.Button("Add Section")
            btn_add.click(add_section, inputs=section_count, outputs=[section_count] + section_rows, queue=False)
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
                ag_topk = gr.Number(label="Top-k", value=250)
                ag_topp = gr.Number(label="Top-p", value=0.0)
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

        # ----- MASTERING -----
        with gr.Tab("Mastering"):
            audio_in3 = gr.Audio(label="Input Track", type="numpy")
            pathway = gr.Radio(["Simple", "Complex"], value="Simple", label="Pathway")
            out_trim_master = gr.Slider(-24.0, 0.0, value=-1.0, step=0.5, label="Output Trim (dB)")
            out_master = gr.Audio(label="Output", type="filepath")
            btn_master = gr.Button("Enqueue", variant="primary")
            btn_master.click(master_track, inputs=[audio_in3, pathway, out_trim_master], outputs=out_master, queue=True)

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


