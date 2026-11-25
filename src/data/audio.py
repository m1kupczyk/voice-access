"""Audio loading: read, mono, resample, peak-normalize, energy-based VAD trim."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def to_mono(wav: np.ndarray) -> np.ndarray:
    """Average channels. Accepts (n,) or (n, channels); returns (n,)."""
    if wav.ndim == 1:
        return wav
    return wav.mean(axis=1)


def resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling of a 1-D signal from sr_in to sr_out."""
    if sr_in == sr_out:
        return wav
    ratio = Fraction(sr_out, sr_in)
    return resample_poly(wav, ratio.numerator, ratio.denominator).astype(np.float32)


def peak_normalize(wav: np.ndarray, peak: float, eps: float) -> np.ndarray:
    """Scale so that max |x| equals `peak`. Silent input is returned unchanged."""
    m = float(np.max(np.abs(wav))) if wav.size else 0.0
    if m < eps:
        return wav
    return (wav * (peak / m)).astype(np.float32)


def vad_trim(wav: np.ndarray, sr: int, vad_cfg: dict[str, Any], eps: float = 1e-8) -> np.ndarray:
    """Trim leading/trailing silence with a frame-energy VAD.

    A frame is speech if its energy is within `threshold_db` of the loudest
    frame. Speech runs shorter than `min_speech_ms` are dropped; the signal is
    cut from the first to the last remaining run, padded by `pad_ms`.
    Returns an empty array if no speech is found.
    """
    frame = int(sr * vad_cfg["frame_ms"] / 1000)
    hop = int(sr * vad_cfg["hop_ms"] / 1000)
    if wav.size < frame:
        return wav[:0]
    n_frames = 1 + (wav.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    energy = np.mean(wav[idx].astype(np.float64) ** 2, axis=1)
    if energy.max() < eps:  # digital silence
        return wav[:0]
    db = 10.0 * np.log10(energy + eps)
    is_speech = db >= db.max() + vad_cfg["threshold_db"]

    # Find runs of speech frames and keep those long enough.
    edges = np.diff(np.concatenate(([0], is_speech.astype(np.int8), [0])))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]  # ends exclusive
    min_frames = max(1, int(round(vad_cfg["min_speech_ms"] / vad_cfg["hop_ms"])))
    keep = (ends - starts) >= min_frames
    if not keep.any():
        return wav[:0]
    first, last = starts[keep][0], ends[keep][-1]
    pad = int(sr * vad_cfg["pad_ms"] / 1000)
    begin = max(0, first * hop - pad)
    end = min(wav.size, (last - 1) * hop + frame + pad)
    return wav[begin:end]


def load_audio(path: str | Path, cfg: dict[str, Any], vad: bool | None = None) -> np.ndarray:
    """Load an audio file as float32 mono at cfg['audio']['sample_rate'].

    Steps follow the config: mono, resample, peak normalization, VAD trim.
    `vad` overrides cfg['audio']['vad']['enabled'] when given.
    """
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return process_audio(wav, sr, cfg, vad=vad)


def process_audio(
    wav: np.ndarray, sr: int, cfg: dict[str, Any], vad: bool | None = None
) -> np.ndarray:
    """Apply the loader pipeline to an in-memory array (n,) or (n, channels)."""
    a = cfg["audio"]
    if a["mono"]:
        wav = to_mono(wav)
    wav = resample(wav.astype(np.float32), sr, int(a["sample_rate"]))
    if a["normalize"]:
        wav = peak_normalize(wav, a["peak_target"], a["eps"])
    use_vad = a["vad"]["enabled"] if vad is None else vad
    if use_vad:
        wav = vad_trim(wav, int(a["sample_rate"]), a["vad"], a["eps"])
    return wav
