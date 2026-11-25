"""Pydantic response models for the API."""

from __future__ import annotations

from pydantic import BaseModel


class EnrollResponse(BaseModel):
    user_id: int
    name: str
    model_name: str
    n_embeddings: int
    audio_seconds: float


class ChallengeResponse(BaseModel):
    user_id: int
    nonce: str
    digits: str
    expires_at: float
    ttl_seconds: float


class AsrResult(BaseModel):
    text: str
    recognized: str
    expected: str
    digits_ok: bool
    latency_s: float


class SpeakerResult(BaseModel):
    model_name: str
    score: float
    threshold: float
    score_ok: bool
    latency_s: float


class ReplayResult(BaseModel):
    enabled: bool
    score: float | None = None
    threshold: float | None = None
    cm_ok: bool = True


class VerifyResponse(BaseModel):
    user_id: int
    accepted: bool
    decision: str
    scenario: str
    reject_reason: str | None = None
    asr: AsrResult
    speaker: SpeakerResult
    replay: ReplayResult
    attempt_id: int


class DeleteResponse(BaseModel):
    user_id: int
    deleted: dict[str, int]


class ScenarioStats(BaseModel):
    scenario: str
    n_attempts: int
    n_accepted: int
    accept_rate: float | None = None
    n_scored: int
    score_mean: float | None = None
    score_median: float | None = None
    latency_mean_s: float | None = None
    reject_reasons: dict[str, int]


class StatsResponse(BaseModel):
    user_id: int | None = None
    n_attempts: int
    scenarios: list[ScenarioStats]


class HealthResponse(BaseModel):
    status: str
    speaker_model: str
    threshold: float
    threshold_source: str
    asr_model: str
    replay_enabled: bool
