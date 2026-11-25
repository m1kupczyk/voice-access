"""Tests for MFCC features, GMM-UBM, MAP adaptation and EER on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config
from src.eval.metrics import compute_eer
from src.features.mfcc import extract_mfcc, feature_dim
from src.models.gmm_ubm import map_adapt, train_ubm
from src.verification.verifier import GmmUbmVerifier

SEED = 0


@pytest.fixture(scope="module")
def cfg() -> dict:
    c = load_config()
    c["gmm"].update({"n_components": 4, "em_max_iter": 20, "map_iters": 2, "init_params": "kmeans"})
    return c


def _voice(f0: float, seconds: float, rng: np.random.Generator, sr: int = 16000) -> np.ndarray:
    """Synthetic 'speaker': harmonic stack with speaker-specific f0 plus noise."""
    t = np.arange(int(sr * seconds)) / sr
    f0 = f0 * (1 + 0.02 * rng.standard_normal())
    w = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    return (w + 0.05 * rng.standard_normal(t.size)).astype(np.float32)


def test_mfcc_shape_and_cmvn(cfg: dict) -> None:
    rng = np.random.default_rng(SEED)
    x = extract_mfcc(_voice(150, 1.0, rng), cfg)
    assert x.shape == (101, feature_dim(cfg))
    assert np.allclose(x.mean(0), 0, atol=1e-4)
    assert np.allclose(x.std(0), 1, atol=1e-2)
    assert extract_mfcc(np.zeros(10, np.float32), cfg).shape == (0, feature_dim(cfg))


def test_map_adapt_moves_towards_data_and_respects_relevance() -> None:
    from src.models.gmm_ubm import UBM

    ubm = UBM(np.array([1.0]), np.zeros((1, 2)), np.ones((1, 2)))
    x = np.full((100, 2), 2.0)
    small_r = map_adapt(ubm, x, 1.0, 1)[0, 0]
    large_r = map_adapt(ubm, x, 1e4, 1)[0, 0]
    assert 1.9 < small_r < 2.0
    assert 0.0 < large_r < 0.05


def test_verifier_separates_synthetic_speakers(cfg: dict) -> None:
    rng = np.random.default_rng(SEED)
    f0s = {"a": 120.0, "b": 210.0}
    bg = np.concatenate([extract_mfcc(_voice(f, 2.0, rng), cfg)
                         for f in (100, 140, 180, 220, 260) for _ in range(2)])
    ubm = train_ubm(bg, cfg)
    ver = GmmUbmVerifier(ubm, cfg)
    tmpl_a = ver.enroll([_voice(f0s["a"], 3.0, rng)])
    target = ver.score(tmpl_a, _voice(f0s["a"], 2.0, rng))
    impostor = ver.score(tmpl_a, _voice(f0s["b"], 2.0, rng))
    assert target > impostor


def test_eer_known_values() -> None:
    y = np.array([1, 1, 1, 0, 0, 0])
    assert compute_eer(y, np.array([3, 4, 5, 0, 1, 2.0])) == 0.0
    assert compute_eer(y, np.array([0, 1, 2, 3, 4, 5.0])) == 1.0
    assert compute_eer(np.array([1, 0]), np.array([1.0, 1.0])) == 0.5
