"""Common enroll/score interface for speaker-verification systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from src.features.mfcc import extract_mfcc
from src.models.gmm_ubm import UBM, map_adapt
from src.verification.scoring import cosine_score, llr


class Verifier(Protocol):
    """enroll(audio_list) -> template, score(template, audio) -> float."""

    def enroll(self, audio_list: Sequence[np.ndarray]) -> Any: ...

    def score(self, template: Any, audio: np.ndarray) -> float: ...


@dataclass
class GmmTemplate:
    """Speaker template: MAP-adapted means (K, D)."""

    means: np.ndarray


class GmmUbmVerifier:
    """MFCC + GMM-UBM verifier. Audio is mono float32 at the configured sample rate."""

    def __init__(self, ubm: UBM, cfg: dict[str, Any]) -> None:
        self.ubm = ubm
        self.cfg = cfg
        self._g = cfg["gmm"]

    def enroll_features(self, feats_list: Sequence[np.ndarray]) -> GmmTemplate:
        """MAP-adapt the UBM to the pooled frames of all enrollment utterances."""
        x = np.concatenate(list(feats_list), axis=0)
        means = map_adapt(self.ubm, x, float(self._g["relevance_factor"]), int(self._g["map_iters"]))
        return GmmTemplate(means)

    def score_features(
        self, template: GmmTemplate, feats: np.ndarray, ubm_mean_ll: float | None = None
    ) -> float:
        """LLR of precomputed features against a template."""
        return llr(feats, self.ubm, template.means, int(self._g["score_chunk_frames"]), ubm_mean_ll)

    def enroll(self, audio_list: Sequence[np.ndarray]) -> GmmTemplate:
        return self.enroll_features([extract_mfcc(a, self.cfg) for a in audio_list])

    def score(self, template: GmmTemplate, audio: np.ndarray) -> float:
        return self.score_features(template, extract_mfcc(audio, self.cfg))


class EcapaVerifier:
    """ECAPA-TDNN + cosine verifier. Template = mean of enrollment embeddings."""

    def __init__(self, embedder: Any) -> None:
        self.embedder = embedder

    def enroll_embeddings(self, embs: Sequence[np.ndarray]) -> np.ndarray:
        return np.mean(np.stack(list(embs)), axis=0)

    def score_embedding(self, template: np.ndarray, emb: np.ndarray) -> float:
        return cosine_score(template, emb)

    def enroll(self, audio_list: Sequence[np.ndarray]) -> np.ndarray:
        return self.enroll_embeddings([self.embedder.embed(a) for a in audio_list])

    def score(self, template: np.ndarray, audio: np.ndarray) -> float:
        return self.score_embedding(template, self.embedder.embed(audio))
