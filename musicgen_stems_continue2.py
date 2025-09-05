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

import torch
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

# ---------- Devices [ALTERED] ----------
STYLE_DEVICE = torch.device("cuda:1")       # Style pinned to GPU1
AUDIOGEN_DEVICE = torch.device("cuda:0")    # AudioGen on GPU0
UTILITY_DEVICE = torch.device("cuda:3") if (torch.cuda.is_available() and torch.cuda.device_count() > 3) else torch.device("cpu")

print(f"[Boot] STYLE: {STYLE_DEVICE} | AUDIOGEN: {AUDIOGEN_DEVICE} | UTILITY(preproc/demucs): {UTILITY_DEVICE}")

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

def _audiogen_continue_compat(model, prompt: str, tail_32k_mono: torch.Tensor, sr: int):
    """
    Try AudioGen continuation across Audiocraft versions, using keyword variants only.
    Always pass a TENSOR for the audio prompt; never wrap in a list.
    If no compatible signature exists, return None (caller will emulate).
    """
    tail = tail_32k_mono.detach().to(model.device)
    if tail.dim() == 3 and tail.size(0) == 1:
        tail = tail[0]           # (1, T)
    elif tail.dim() == 1:
        tail = tail.unsqueeze(0) # (1, T)
    elif tail.dim() != 2:
        raise gr.Error(f"tail must be (C,T) or (1,C,T); got {tuple(tail.shape)}")

    if hasattr(model, "generate_continuation"):
        # Newer-ish
        try:
            return model.generate_continuation(
                descriptions=[prompt],
                audio_wavs=tail,
                audio_sample_rate=sr,
            )
        except TypeError:
            pass
        # Variant naming
        try:
            return model.generate_continuation(
                descriptions=[prompt],
                audio=tail,
                audio_sample_rate=sr,
            )
        except TypeError:
            pass
        # Older-ish
        try:
            return model.generate_continuation(
                descriptions=[prompt],
                wavs=tail,
                sample_rate=sr,
            )
        except TypeError:
            pass
        # Some builds use "prompts" instead of "descriptions"
        try:
            return model.generate_continuation(
                prompts=[prompt],
                audio=tail,
                audio_sample_rate=sr,
            )
        except TypeError:
            pass

    return None  # let caller emulate

# ============================================================================
# STYLE TAB (MusicGen-Style) [UNCHANGED intent; better writer]
# ============================================================================
def style_load_model():
    global STYLE_MODEL
    if STYLE_MODEL is None:
        print(f"[Style] Loading facebook/musicgen-style on {STYLE_DEVICE}")
        STYLE_MODEL = MusicGen.get_pretrained("facebook/musicgen-style")
        STYLE_MODEL.device = STYLE_DEVICE  # MusicGen isn't nn.Module; don't call .to()

def style_load_diffusion():
    global STYLE_MBD
    if STYLE_MBD is None:
        print("[Style] Loading MultiBandDiffusion...")
        STYLE_MBD = MultiBandDiffusion.get_mbd_musicgen().to(STYLE_DEVICE)


def style_predict(text, melody, duration=10, topk=250, topp=0.0, temperature=1.0,
                  cfg_coef=3.0, double_cfg="Yes", cfg_coef_beta=5.0,
                  eval_q=3, excerpt_length=3.0, decoder="Default",
                  out_trim_db=-3.0):
    """Generate with MusicGen-Style, optional melody excerpt."""
    style_load_model()
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
    """
    Use AudioGen’s continuation if available; otherwise emulate continuation by
    generating fresh audio and equal-power crossfading onto the input tail.
    """
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

    # 3) Try real continuation via compatibility shim
    out = _audiogen_continue_compat(model, prompt, tail, TARGET_SR)
    if out is not None:
        batch = _extract_audio_batch(out)          # (B,C,T) or similar
        wav = batch.detach().cpu().float()[0]      # (C,T)
        if _rms(wav) < 1e-6:
            raise gr.Error("AudioGen continuation returned near-silence.")
        return _write_wav(wav, TARGET_SR, stem="audiogen_cont", trim_db=float(out_trim_db))

    # 4) Fallback: emulate continuation with crossfade
    print("[AudioGen] Continuation API unavailable – emulating via crossfade.")
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
# UI (three tabs, all Enqueue) [ALTERED]
# ============================================================================
def ui_full(launch_kwargs):
    with gr.Blocks() as demo:
        gr.Markdown("# 🎛️ Music Suite — Style • AudioGen Continuation • Stems  \n*Enqueue buttons; global queue enabled*")

        # ----- STYLE -----
        with gr.Tab("Style (MusicGen-Style, GPU1)"):
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

