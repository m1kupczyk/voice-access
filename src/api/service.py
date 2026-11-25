"""Model loading and wiring: in-memory audio decoding, enroll / challenge / verify / delete."""

from __future__ import annotations

import io
import json
import secrets
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf

from src.api.db import Database
from src.challenge.asr import WhisperASR
from src.challenge.digits import challenge_from_config
from src.challenge.pipeline import decide, rejection_reason
from src.data.audio import process_audio
from src.eval.live_stats import aggregate, attempts_csv
from src.models.ecapa import EcapaEmbedder
from src.verification.verifier import EcapaVerifier

SUPPORTED_SPEAKER_MODELS = ("ecapa",)


class ApiError(Exception):
    """Error with an HTTP status code, raised by the service and mapped by the app."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# Audio decoding (memory only)

def _ffmpeg_decode(data: bytes, sr: int, ffmpeg_bin: str, timeout: float) -> np.ndarray:
    """Decode any ffmpeg-readable container via stdin/stdout pipes to float32 mono at `sr`."""
    cmd = [ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "pipe:1"]
    try:
        proc = subprocess.run(cmd, input=data, capture_output=True, timeout=timeout, check=False)
    except FileNotFoundError:
        raise ApiError(500, f"ffmpeg not found: {ffmpeg_bin}") from None
    except subprocess.TimeoutExpired:
        raise ApiError(400, "audio decoding timed out") from None
    if proc.returncode != 0 or not proc.stdout:
        msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise ApiError(400, f"cannot decode audio: {msg[-1] if msg else 'empty output'}")
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)


def decode_audio(data: bytes, cfg: dict[str, Any]) -> np.ndarray:
    """Upload bytes -> float32 mono at audio.sample_rate, preprocessed like the stage 4 runs."""
    if not data:
        raise ApiError(400, "empty audio upload")
    sr_out = int(cfg["audio"]["sample_rate"])
    try:  # WAV / FLAC / OGG directly from memory
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, TypeError):  # browser formats: webm/opus, mp4/aac
        a = cfg["api"]
        wav, sr = _ffmpeg_decode(data, sr_out, a["ffmpeg_bin"], float(a["ffmpeg_timeout_seconds"])), sr_out
    audio = process_audio(wav, sr, cfg, vad=bool(cfg["ecapa"]["vad"]))
    min_s = float(cfg["api"]["min_audio_seconds"])
    if audio.size < min_s * sr_out:
        raise ApiError(400, f"audio too short: {audio.size / sr_out:.2f} s < {min_s} s")
    return audio


# Threshold

def load_threshold(cfg: dict[str, Any]) -> tuple[float, str]:
    """Read the speaker threshold from api.threshold_file at the dotted api.threshold_key."""
    a = cfg["api"]
    path = Path(a["threshold_file"])
    with open(path, encoding="utf-8") as f:
        node: Any = json.load(f)
    for part in str(a["threshold_key"]).split("."):
        node = node[part]
    return float(node), f"{path}:{a['threshold_key']}"


# Stage 5 hook

class ReplayDetector(Protocol):
    """Replay countermeasure: audio -> score, higher = more likely bona fide."""

    def score(self, audio: np.ndarray) -> float: ...


def load_replay_detector(cfg: dict[str, Any]) -> tuple[ReplayDetector | None, float | None]:
    """Return (detector, threshold), or (None, None) when api.replay.enabled is false.

    Stage 5 hook: load the trained countermeasure from api.replay.model_path here.
    """
    r = cfg["api"]["replay"]
    if not r["enabled"]:
        return None, None
    raise NotImplementedError("replay countermeasure not implemented yet (stage 5)")


# Service

class VoiceAccessService:
    """Enroll / challenge / verify / delete on top of the database and injected models."""

    def __init__(
        self,
        cfg: dict[str, Any],
        db: Database,
        embedder: Any,
        asr: WhisperASR,
        threshold: float,
        threshold_source: str = "",
        replay: ReplayDetector | None = None,
        replay_threshold: float | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        model = cfg["api"]["speaker_model"]
        if model not in SUPPORTED_SPEAKER_MODELS:
            raise ValueError(f"api.speaker_model must be one of {SUPPORTED_SPEAKER_MODELS}")
        self.cfg = cfg
        self.db = db
        self.embedder = embedder
        self.verifier = EcapaVerifier(embedder)
        self.model_name = str(getattr(embedder, "name", cfg["ecapa"]["model_source"]))
        self.asr = asr
        self.threshold = float(threshold)
        self.threshold_source = threshold_source
        self.replay = replay
        self.replay_threshold = replay_threshold
        self.clock = clock
        self._infer_lock = threading.Lock()  # one inference at a time on the shared models

    def _require_user(self, user_id: int) -> Any:
        user = self.db.get_user(user_id)
        if user is None:
            raise ApiError(404, f"unknown user {user_id}")
        return user

    def enroll(self, name: str, data: bytes) -> dict[str, Any]:
        """Embed free speech; create the user or add an embedding to an existing one."""
        name = name.strip()
        if not name or len(name) > int(self.cfg["api"]["max_name_length"]):
            raise ApiError(422, "name must be 1..max_name_length characters")
        audio = decode_audio(data, self.cfg)
        with self._infer_lock:
            emb = self.embedder.embed(audio)
        now = self.clock()
        user = self.db.get_user_by_name(name)
        user_id = int(user["id"]) if user is not None else self.db.create_user(name, now)
        self.db.add_embedding(user_id, self.model_name, emb, now)
        sr = int(self.cfg["audio"]["sample_rate"])
        return {"user_id": user_id, "name": name, "model_name": self.model_name,
                "n_embeddings": len(self.db.get_embeddings(user_id, self.model_name)),
                "audio_seconds": audio.size / sr}

    def challenge(self, user_id: int) -> dict[str, Any]:
        """Random digits (secrets) and a one-time nonce with expiry."""
        self._require_user(user_id)
        a = self.cfg["api"]
        ttl = float(a["challenge_ttl_seconds"])
        nonce = secrets.token_urlsafe(int(a["nonce_bytes"]))
        digits = challenge_from_config(self.cfg)
        now = self.clock()
        self.db.create_challenge(user_id, nonce, digits, now, now + ttl)
        return {"user_id": user_id, "nonce": nonce, "digits": digits, "expires_at": now + ttl,
                "ttl_seconds": ttl}

    def _reject_challenge(self, user_id: int, challenge_id: int | None, expected: str | None,
                          status: int, reason: str, scenario: str,
                          reject_reason: str | None = None) -> None:
        self.db.log_attempt(user_id=user_id, challenge_id=challenge_id, created_at=self.clock(),
                            expected=expected, accepted=0, reason=reason, scenario=scenario,
                            reject_reason=reject_reason)
        raise ApiError(status, reason)

    def _scenario(self, scenario: str | None) -> str:
        a = self.cfg["api"]
        s = a["default_scenario"] if scenario is None else scenario
        if s not in a["scenarios"]:
            raise ApiError(422, f"scenario must be one of {list(a['scenarios'])}")
        return str(s)

    def verify(self, user_id: int, nonce: str, data: bytes,
               scenario: str | None = None) -> dict[str, Any]:
        """CM (if enabled) AND Whisper digits == challenge AND ECAPA cosine >= threshold.

        `scenario` is an operator label stored for live statistics; it never enters the decision.
        """
        scenario = self._scenario(scenario)
        self._require_user(user_id)
        ch = self.db.get_challenge(nonce)
        if ch is None:
            raise ApiError(404, "unknown challenge nonce")
        if ch.user_id != user_id:
            self._reject_challenge(user_id, None, None, 403, "challenge issued for another user",
                                   scenario)
        if not self.db.consume_challenge(nonce, self.clock()):
            self._reject_challenge(user_id, ch.id, ch.digits, 409, "challenge already used",
                                   scenario)
        if self.clock() > ch.expires_at:
            self._reject_challenge(user_id, ch.id, ch.digits, 410, "challenge expired", scenario,
                                   "expired")
        embs = self.db.get_embeddings(user_id, self.model_name)
        if not embs:
            self._reject_challenge(user_id, ch.id, ch.digits, 409,
                                   f"user has no enrollment for {self.model_name}", scenario)

        t0 = time.perf_counter()
        audio = decode_audio(data, self.cfg)

        cm_score: float | None = None
        cm_ok = True
        with self._infer_lock:
            if self.replay is not None:
                cm_score = float(self.replay.score(audio))
                cm_ok = cm_score >= float(self.replay_threshold)  # type: ignore[arg-type]
            text, recognized, t_asr = self.asr.recognize_digits(audio)
            t1 = time.perf_counter()
            score = self.verifier.score(self.verifier.enroll_embeddings(embs), audio)
            t_sv = time.perf_counter() - t1
        dec = decide(ch.digits, recognized, score, self.threshold)
        accepted = dec.accepted and cm_ok
        reasons = [r for r, bad in (("replay", not cm_ok), ("digits", not dec.digits_ok),
                                    ("speaker", not dec.score_ok)) if bad]
        reject_reason = rejection_reason(dec.digits_ok, dec.score_ok, cm_ok)
        attempt_id = self.db.log_attempt(
            user_id=user_id, challenge_id=ch.id, created_at=self.clock(), expected=ch.digits,
            asr_text=text, recognized=recognized, digits_ok=int(dec.digits_ok), score=dec.score,
            threshold=self.threshold, score_ok=int(dec.score_ok), cm_score=cm_score,
            cm_ok=int(cm_ok), accepted=int(accepted), reason=",".join(reasons) or None,
            latency_s=time.perf_counter() - t0, scenario=scenario, reject_reason=reject_reason)
        return {
            "user_id": user_id, "accepted": accepted,
            "decision": "accepted" if accepted else "rejected",
            "scenario": scenario, "reject_reason": reject_reason,
            "asr": {"text": text, "recognized": recognized, "expected": ch.digits,
                    "digits_ok": dec.digits_ok, "latency_s": t_asr},
            "speaker": {"model_name": self.model_name, "score": dec.score,
                        "threshold": self.threshold, "score_ok": dec.score_ok, "latency_s": t_sv},
            "replay": {"enabled": self.replay is not None, "score": cm_score,
                       "threshold": self.replay_threshold, "cm_ok": cm_ok},
            "attempt_id": attempt_id,
        }

    def delete_user(self, user_id: int) -> dict[str, Any]:
        """GDPR withdrawal: remove the user with all embeddings, challenges and attempts."""
        self._require_user(user_id)
        return {"user_id": user_id, "deleted": self.db.delete_user(user_id)}

    def stats(self, user_id: int | None = None) -> dict[str, Any]:
        """Live per-scenario statistics over the attempt log, optionally for one user."""
        if user_id is not None:
            self._require_user(user_id)
        return {"user_id": user_id,
                **aggregate(self.db.list_attempts(user_id), self.cfg["api"]["scenarios"])}

    def stats_csv(self, user_id: int | None = None) -> str:
        if user_id is not None:
            self._require_user(user_id)
        return attempts_csv(self.db.list_attempts(user_id))

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "speaker_model": self.model_name, "threshold": self.threshold,
                "threshold_source": self.threshold_source, "asr_model": self.cfg["asr"]["model"],
                "replay_enabled": self.replay is not None}


def build_service(cfg: dict[str, Any]) -> VoiceAccessService:
    """Open the database and load ECAPA and Whisper once (eagerly, on cfg['device'])."""
    threshold, source = load_threshold(cfg)
    embedder = EcapaEmbedder(cfg)
    _ = embedder.encoder
    asr = WhisperASR(cfg)
    _ = asr.transcriber
    replay, replay_thr = load_replay_detector(cfg)
    return VoiceAccessService(cfg, Database(cfg["api"]["db_path"]), embedder, asr, threshold,
                              source, replay, replay_thr)
