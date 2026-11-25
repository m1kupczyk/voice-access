"""Combined decision: transcribed digits match the challenge AND speaker score >= threshold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.challenge.asr import WhisperASR
from src.verification.scoring import cosine_score


@dataclass(frozen=True)
class ChallengeDecision:
    """Outcome of one challenge-response verification attempt."""

    recognized: str
    digits_ok: bool
    score: float
    score_ok: bool
    accepted: bool


def decide(challenge: str, recognized: str, score: float, threshold: float) -> ChallengeDecision:
    """Accept only if digits match exactly and score >= threshold."""
    digits_ok = recognized == challenge
    score_ok = score >= threshold
    return ChallengeDecision(recognized, digits_ok, float(score), score_ok, digits_ok and score_ok)


REJECT_REASONS = ("speaker", "digits", "both", "expired", "replay_cm")


def rejection_reason(digits_ok: bool, score_ok: bool, cm_ok: bool = True) -> str | None:
    """Category of a scored attempt: None if accepted; a failed replay CM overrides the rest."""
    if not cm_ok:
        return "replay_cm"
    if not digits_ok and not score_ok:
        return "both"
    if not digits_ok:
        return "digits"
    if not score_ok:
        return "speaker"
    return None


class ChallengePipeline:
    """ASR + speaker embedding + combined decision (replay CM is added in stage 5)."""

    def __init__(self, embedder: Any, asr: WhisperASR, threshold: float) -> None:
        self.embedder = embedder
        self.asr = asr
        self.threshold = float(threshold)

    def verify(self, audio: np.ndarray, challenge: str, template: np.ndarray) -> ChallengeDecision:
        _, recognized, _ = self.asr.recognize_digits(audio)
        score = cosine_score(template, self.embedder.embed(audio))
        return decide(challenge, recognized, score, self.threshold)
