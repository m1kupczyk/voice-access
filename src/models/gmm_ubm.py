"""Diagonal-covariance GMM-UBM: training, MAP mean adaptation, save/load."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

from src.data.audio import load_audio
from src.features.mfcc import extract_mfcc


@dataclass
class UBM:
    """Universal background model: weights (K,), means (K, D), variances (K, D)."""

    weights: np.ndarray
    means: np.ndarray
    variances: np.ndarray

    @property
    def n_components(self) -> int:
        return int(self.weights.shape[0])


def component_loglik(
    x: np.ndarray, weights: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    """Per-frame, per-component log(w_k * N(x | mu_k, diag var_k)); shape (T, K)."""
    d = means.shape[1]
    prec = 1.0 / variances
    const = np.log(weights) - 0.5 * (
        d * np.log(2 * np.pi) + np.log(variances).sum(1) + (means**2 * prec).sum(1)
    )
    x = x.astype(np.float64)
    return const[None, :] + (x**2) @ (-0.5 * prec).T + x @ (means * prec).T


def frame_loglik_sum(
    x: np.ndarray, weights: np.ndarray, means: np.ndarray, variances: np.ndarray, chunk: int
) -> float:
    """Sum over frames of log p(x_t), computed in blocks of `chunk` frames."""
    total = 0.0
    for i in range(0, x.shape[0], chunk):
        total += float(logsumexp(component_loglik(x[i:i + chunk], weights, means, variances), 1).sum())
    return total


def list_ubm_files(cfg: dict[str, Any]) -> list[Path]:
    """Seeded, shuffled list of background-audio files, capped by utts and hours."""
    g = cfg["gmm"]
    if g["ubm_source"] != "librispeech_train_clean_100":
        raise ValueError(f"gmm.ubm_source '{g['ubm_source']}' is not implemented")
    files = sorted(Path(g["librispeech_root"]).rglob(f"*{g['audio_extension']}"))
    if not files:
        raise FileNotFoundError(f"no {g['audio_extension']} files under {g['librispeech_root']}")
    random.Random(cfg["seed"]).shuffle(files)
    max_utts, max_hours = g["ubm_max_utts"], g["ubm_max_hours"]
    out: list[Path] = []
    seconds = 0.0
    for f in files:
        if max_utts is not None and len(out) >= int(max_utts):
            break
        if max_hours is not None and seconds >= float(max_hours) * 3600:
            break
        info = sf.info(str(f))
        seconds += info.frames / info.samplerate
        out.append(f)
    return out


def collect_ubm_frames(files: list[Path], cfg: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Extract MFCCs and sample up to `ubm_frames_per_utt` random frames per file.

    Returns (frames (N, D) float32, number of utterances that yielded features).
    """
    from tqdm import tqdm

    rng = np.random.default_rng(cfg["seed"])
    cap = int(cfg["gmm"]["ubm_frames_per_utt"])
    chunks: list[np.ndarray] = []
    for f in tqdm(files, desc="UBM features"):
        feats = extract_mfcc(load_audio(f, cfg), cfg)
        if feats.shape[0] == 0:
            continue
        if feats.shape[0] > cap:
            feats = feats[np.sort(rng.choice(feats.shape[0], cap, replace=False))]
        chunks.append(feats)
    return np.concatenate(chunks, axis=0), len(chunks)


def train_ubm(frames: np.ndarray, cfg: dict[str, Any]) -> UBM:
    """Fit a diagonal GMM with EM on pooled background frames (seeded)."""
    g = cfg["gmm"]
    gm = GaussianMixture(
        n_components=int(g["n_components"]), covariance_type=g["covariance_type"],
        max_iter=int(g["max_iter"]), tol=float(g["em_tol"]), reg_covar=float(g["reg_covar"]),
        init_params=g["init_params"], random_state=int(cfg["seed"]), verbose=2,
    )
    gm.fit(frames)
    return UBM(gm.weights_.astype(np.float64), gm.means_.astype(np.float64),
               gm.covariances_.astype(np.float64))


def map_adapt(ubm: UBM, x: np.ndarray, relevance_factor: float, n_iters: int) -> np.ndarray:
    """MAP-adapt UBM means to frames x (T, D); weights and variances stay fixed.

    mu_k = a_k * E_k[x] + (1 - a_k) * mu_ubm_k, with a_k = n_k / (n_k + r).
    Returns adapted means (K, D).
    """
    means = ubm.means.copy()
    x64 = x.astype(np.float64)
    for _ in range(n_iters):
        post = np.exp(component_loglik(x64, ubm.weights, means, ubm.variances))
        post /= post.sum(1, keepdims=True) + 1e-300
        n_k = post.sum(0)
        ex = (post.T @ x64) / np.maximum(n_k, 1e-10)[:, None]
        alpha = (n_k / (n_k + relevance_factor))[:, None]
        means = alpha * ex + (1.0 - alpha) * ubm.means
    return means


def save_ubm(ubm: UBM, path: str | Path) -> None:
    """Save the UBM as a compressed .npz (no audio, parameters only)."""
    np.savez_compressed(path, weights=ubm.weights, means=ubm.means, variances=ubm.variances)


def load_ubm(path: str | Path) -> UBM:
    """Load a UBM saved by `save_ubm`."""
    z = np.load(path)
    return UBM(z["weights"], z["means"], z["variances"])
