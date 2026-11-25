"""ECAPA wrapper, cache and cosine scoring with a mocked encoder (no model download)."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from src.config import load_config
from src.models.ecapa import EcapaEmbedder
from src.verification.scoring import cosine_score
from src.verification.verifier import EcapaVerifier


@pytest.fixture()
def cfg(tmp_path) -> dict:
    c = load_config()
    c["paths"]["embeddings_dir"] = str(tmp_path / "emb")
    return c


class FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.array([audio.mean(), audio.std(), np.abs(audio).max()], dtype=np.float32)


def _wav(path, seed: int, sr: int = 16000) -> None:
    sf.write(path, np.random.default_rng(seed).standard_normal(sr).astype(np.float32) * 0.1, sr)


def test_cosine_known_values() -> None:
    assert cosine_score(np.array([1.0, 0]), np.array([2.0, 0])) == pytest.approx(1.0)
    assert cosine_score(np.array([1.0, 0]), np.array([0, 3.0])) == pytest.approx(0.0)
    assert cosine_score(np.array([1.0, 0]), np.array([-1.0, 0])) == pytest.approx(-1.0)


def test_cache_hit_and_persistence(cfg, tmp_path) -> None:
    p = tmp_path / "a.wav"
    _wav(p, 0)
    enc = FakeEncoder()
    emb = EcapaEmbedder(cfg, encoder=enc)
    e1 = emb.embed_file(p)
    e2 = emb.embed_file(p)
    assert enc.calls == 1 and np.array_equal(e1, e2)
    emb.cache.flush()
    enc2 = FakeEncoder()
    emb2 = EcapaEmbedder(cfg, encoder=enc2)  # fresh object reads the npz
    assert np.array_equal(emb2.embed_file(p), e1) and enc2.calls == 0


def test_cache_keyed_by_model_name(cfg, tmp_path) -> None:
    p = tmp_path / "a.wav"
    _wav(p, 0)
    a = EcapaEmbedder(cfg, encoder=FakeEncoder())
    a.embed_file(p)
    a.cache.flush()
    cfg2 = {**cfg, "ecapa": {**cfg["ecapa"], "model_source": "other/model"}}
    enc = FakeEncoder()
    EcapaEmbedder(cfg2, encoder=enc).embed_file(p)
    assert enc.calls == 1


def test_verifier_enroll_score(cfg) -> None:
    v = EcapaVerifier(EcapaEmbedder(cfg, encoder=FakeEncoder()))
    t = v.enroll_embeddings([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
    assert np.allclose(t, [0.5, 0.5])
    assert v.score_embedding(t, np.array([1.0, 1.0])) == pytest.approx(1.0)
    a = np.random.default_rng(1).standard_normal(1600).astype(np.float32)
    assert v.score(v.enroll([a]), a) == pytest.approx(1.0)
