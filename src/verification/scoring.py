"""GMM-UBM log-likelihood ratio scoring."""

from __future__ import annotations

import numpy as np

from src.models.gmm_ubm import UBM, frame_loglik_sum


def mean_loglik(
    x: np.ndarray, weights: np.ndarray, means: np.ndarray, variances: np.ndarray, chunk: int
) -> float:
    """Average per-frame log-likelihood of x (T, D) under a diagonal GMM."""
    return frame_loglik_sum(x, weights, means, variances, chunk) / max(x.shape[0], 1)


def llr(
    x: np.ndarray, ubm: UBM, adapted_means: np.ndarray, chunk: int, ubm_mean_ll: float | None = None
) -> float:
    """Frame-averaged LLR: mean log p(x | speaker) - mean log p(x | UBM).

    `ubm_mean_ll` lets callers pass a cached UBM term for the same frames.
    """
    spk = mean_loglik(x, ubm.weights, adapted_means, ubm.variances, chunk)
    bg = ubm_mean_ll if ubm_mean_ll is not None else mean_loglik(
        x, ubm.weights, ubm.means, ubm.variances, chunk)
    return spk - bg


def cosine_score(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity of two embedding vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), eps))
