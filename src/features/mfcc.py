"""MFCC + delta + delta-delta features with per-utterance CMVN."""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np


def feature_dim(cfg: dict[str, Any]) -> int:
    """Dimension of the feature vector produced by `extract_mfcc`."""
    m = cfg["features"]["mfcc"]
    return int(m["n_mfcc"]) * (3 if m["deltas"] else 1)


def cmvn(feats: np.ndarray, mean: bool, var: bool, eps: float) -> np.ndarray:
    """Per-utterance cepstral mean (and variance) normalization over time axis 0."""
    out = feats - feats.mean(axis=0, keepdims=True) if mean else feats
    if var:
        out = out / (out.std(axis=0, keepdims=True) + eps)
    return out


def extract_mfcc(wav: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Compute MFCC features of a mono waveform at the configured sample rate.

    Returns float32 array (n_frames, feature_dim): static MFCCs, optionally
    followed by deltas and delta-deltas, then CMVN. Empty input gives (0, dim).
    """
    sr = int(cfg["audio"]["sample_rate"])
    m = cfg["features"]["mfcc"]
    dim = feature_dim(cfg)
    n_fft = int(sr * m["frame_ms"] / 1000)
    if wav.size < n_fft:
        return np.zeros((0, dim), dtype=np.float32)
    fmax = m["fmax_hz"] if m["fmax_hz"] is not None else sr / 2
    mfcc = librosa.feature.mfcc(
        y=wav.astype(np.float32), sr=sr, n_mfcc=int(m["n_mfcc"]), n_mels=int(m["n_mels"]),
        n_fft=n_fft, hop_length=int(sr * m["hop_ms"] / 1000),
        fmin=float(m["fmin_hz"]), fmax=float(fmax),
    )
    if m["deltas"]:
        w = int(m["delta_width"])
        d1 = librosa.feature.delta(mfcc, width=w, order=1, mode="nearest")
        d2 = librosa.feature.delta(mfcc, width=w, order=2, mode="nearest")
        mfcc = np.concatenate([mfcc, d1, d2], axis=0)
    feats = mfcc.T
    feats = cmvn(feats, bool(m["cmvn_mean"]), bool(m["cmvn_var"]), float(cfg["audio"]["eps"]))
    return feats.astype(np.float32)
