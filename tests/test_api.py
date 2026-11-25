"""Stage 6 tests: FastAPI + SQLite with mocked ASR and ECAPA."""

from __future__ import annotations

import builtins
import copy
import io
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from src.api.db import Database
from src.api.main import create_app
from src.api.service import (
    ApiError,
    VoiceAccessService,
    decode_audio,
    load_replay_detector,
)
from src.challenge.asr import WhisperASR
from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]
PL_WORDS = ["zero", "jeden", "dwa", "trzy", "cztery", "pięć", "sześć", "siedem", "osiem", "dziewięć"]
THRESHOLD = 0.5
SPEAKER_A_HZ, SPEAKER_B_HZ = 220.0, 440.0


class FakeEmbedder:
    """Embedding = unit vector at an angle given by the dominant frequency (one 'speaker' per tone)."""

    name = "fake-ecapa"

    def embed(self, audio: np.ndarray) -> np.ndarray:
        spec = np.abs(np.fft.rfft(audio))
        hz = float(np.argmax(spec)) * 16000 / audio.size
        return np.array([np.cos(hz / 100), np.sin(hz / 100), 0.0], dtype=np.float32)


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class Env:
    def __init__(self, client: TestClient, service: VoiceAccessService, clock: Clock,
                 asr_reply: dict[str, str]) -> None:
        self.client, self.service, self.clock, self.asr_reply = client, service, clock, asr_reply

    def say(self, digits: str) -> None:
        self.asr_reply["text"] = " ".join(PL_WORDS[int(d)] for d in digits)

    def enroll(self, name: str, hz: float = SPEAKER_A_HZ) -> int:
        r = self.client.post(f"/enroll?name={name}", content=wav_bytes(hz))
        assert r.status_code == 200, r.text
        return int(r.json()["user_id"])

    def challenge(self, user_id: int) -> dict[str, Any]:
        r = self.client.get(f"/challenge?user_id={user_id}")
        assert r.status_code == 200, r.text
        return r.json()

    def verify(self, user_id: int, nonce: str, hz: float = SPEAKER_A_HZ, data: bytes | None = None,
               scenario: str | None = None):
        q = "" if scenario is None else f"&scenario={scenario}"
        return self.client.post(f"/verify?user_id={user_id}&nonce={nonce}{q}",
                                content=wav_bytes(hz) if data is None else data)


def wav_bytes(hz: float, seconds: float = 2.0, sr: int = 44100, channels: int = 2) -> bytes:
    """In-memory WAV of a tone (stereo, 44.1 kHz) to exercise mono + resampling."""
    t = np.arange(int(seconds * sr)) / sr
    x = 0.5 * np.sin(2 * np.pi * hz * t)
    buf = io.BytesIO()
    sf.write(buf, np.stack([x] * channels, axis=1), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def cfg(tmp_path: Path) -> dict[str, Any]:
    c = copy.deepcopy(load_config())
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"threshold_transfer": {"threshold": THRESHOLD}}))
    c["api"].update(db_path=str(tmp_path / "api.db"), threshold_file=str(metrics))
    c["paths"]["embeddings_dir"] = str(tmp_path / "emb")
    return c


@pytest.fixture
def env(cfg: dict[str, Any]) -> Iterator[Env]:
    from src.api.service import load_threshold

    clock, reply = Clock(), {"text": ""}
    asr = WhisperASR(cfg, transcriber=lambda audio: reply["text"])
    thr, src = load_threshold(cfg)
    service = VoiceAccessService(cfg, Database(cfg["api"]["db_path"]), FakeEmbedder(), asr, thr,
                                 src, clock=clock)
    with TestClient(create_app(cfg, service)) as client:
        yield Env(client, service, clock, reply)


def test_full_flow(env: Env) -> None:
    uid = env.enroll("spk01")
    assert env.client.get("/health").json()["threshold"] == THRESHOLD
    ch = env.challenge(uid)
    assert len(ch["digits"]) == 4 and ch["digits"].isdigit()
    env.say(ch["digits"])
    r = env.verify(uid, ch["nonce"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted"] and body["decision"] == "accepted"
    assert body["asr"]["recognized"] == ch["digits"] and body["asr"]["digits_ok"]
    assert body["speaker"]["score"] == pytest.approx(1.0, abs=1e-6)
    assert body["speaker"]["threshold"] == THRESHOLD and body["speaker"]["score_ok"]
    assert body["replay"]["enabled"] is False
    db = env.service.db
    assert db.count("attempts") == 1 and db.count("embeddings") == 1
    assert db.get_embeddings(uid, "fake-ecapa")[0].shape == (3,)


def test_impostor_rejected(env: Env) -> None:
    uid = env.enroll("spk01", SPEAKER_A_HZ)
    ch = env.challenge(uid)
    env.say(ch["digits"])
    body = env.verify(uid, ch["nonce"], hz=SPEAKER_B_HZ).json()
    assert not body["accepted"] and body["asr"]["digits_ok"] and not body["speaker"]["score_ok"]


def test_wrong_digits(env: Env) -> None:
    uid = env.enroll("spk01")
    ch = env.challenge(uid)
    env.say(ch["digits"][::-1])
    body = env.verify(uid, ch["nonce"]).json()
    assert not body["accepted"] and body["decision"] == "rejected"
    assert not body["asr"]["digits_ok"] and body["speaker"]["score_ok"]


def test_challenge_expiry(env: Env) -> None:
    uid = env.enroll("spk01")
    ch = env.challenge(uid)
    env.say(ch["digits"])
    env.clock.t += float(env.service.cfg["api"]["challenge_ttl_seconds"]) + 1
    r = env.verify(uid, ch["nonce"])
    assert r.status_code == 410
    assert env.verify(uid, ch["nonce"]).status_code == 409  # expired nonce is burned too


def test_challenge_reuse(env: Env) -> None:
    uid = env.enroll("spk01")
    ch = env.challenge(uid)
    env.say(ch["digits"])
    assert env.verify(uid, ch["nonce"]).json()["accepted"]
    r = env.verify(uid, ch["nonce"])
    assert r.status_code == 409


def test_challenge_bound_to_user(env: Env) -> None:
    a, b = env.enroll("spk01"), env.enroll("spk02", SPEAKER_B_HZ)
    ch = env.challenge(a)
    env.say(ch["digits"])
    assert env.verify(b, ch["nonce"], hz=SPEAKER_B_HZ).status_code == 403
    assert env.verify(a, "no-such-nonce").status_code == 404


def test_unknown_user(env: Env) -> None:
    assert env.client.get("/challenge?user_id=999").status_code == 404
    assert env.verify(999, "x").status_code == 404
    assert env.client.delete("/users/999").status_code == 404


def test_delete_user(env: Env) -> None:
    uid = env.enroll("spk01")
    env.enroll("spk01")  # second enrollment of the same pseudonym adds an embedding
    other = env.enroll("spk02", SPEAKER_B_HZ)
    ch = env.challenge(uid)
    env.say(ch["digits"])
    env.verify(uid, ch["nonce"])
    env.challenge(uid)
    r = env.client.delete(f"/users/{uid}")
    assert r.status_code == 200
    assert r.json()["deleted"] == {"users": 1, "embeddings": 2, "challenges": 2, "attempts": 1}
    db = env.service.db
    assert db.get_user(uid) is None and db.get_embeddings(uid, "fake-ecapa") == []
    assert db.count("challenges") == 0 and db.count("attempts") == 0
    assert db.count("users") == 1 and db.get_user(other) is not None
    assert env.client.get(f"/challenge?user_id={uid}").status_code == 404


def test_rejects_bad_uploads(env: Env) -> None:
    assert env.client.post("/enroll?name=x", content=b"").status_code == 400
    assert env.client.post("/enroll?name=x", content=b"not audio at all").status_code == 400
    short = wav_bytes(SPEAKER_A_HZ, seconds=0.2)
    assert env.client.post("/enroll?name=x", content=short).status_code == 400
    big = b"\0" * (int(env.service.cfg["api"]["max_upload_bytes"]) + 1)
    assert env.client.post("/enroll?name=x", content=big).status_code == 413
    assert env.service.db.count("users") == 0


def test_index_page(env: Env) -> None:
    r = env.client.get("/")
    assert r.status_code == 200 and "Rejestracja" in r.text and "Weryfikacja" in r.text
    assert "Statystyki" in r.text and 'id="verify-scenario"' in r.text


def _snapshot(dirs: list[Path]) -> set[Path]:
    skip = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
    out: set[Path] = set()
    for d in dirs:
        for p in d.rglob("*"):
            if not skip.intersection(p.parts) and p.is_file():
                out.add(p)
    return out


def test_no_audio_written(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    big = wav_bytes(SPEAKER_A_HZ, seconds=8.0)  # > 1 MiB: would spool to disk in multipart
    assert len(big) > 1024 * 1024
    webm = _encode_webm(wav_bytes(SPEAKER_A_HZ)) if shutil.which("ffmpeg") else None
    dirs = [tmp_path, ROOT]
    before = _snapshot(dirs)

    writes: list[str] = []
    real_open, real_os_open = builtins.open, os.open

    def spy_open(file: Any, mode: str = "r", *a: Any, **k: Any) -> Any:
        if not isinstance(file, int) and any(m in mode for m in "wax+"):  # int = pipe fd
            writes.append(str(file))
        return real_open(file, mode, *a, **k)

    def spy_os_open(path: Any, flags: int, *a: Any, **k: Any) -> int:
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            writes.append(str(path))
        return real_os_open(path, flags, *a, **k)

    def forbid(*a: Any, **k: Any) -> Any:
        raise AssertionError("audio must not be written to disk")

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(io, "open", spy_open)
    monkeypatch.setattr(os, "open", spy_os_open)
    for name in ("NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile", "mkstemp"):
        monkeypatch.setattr(tempfile, name, forbid)
    monkeypatch.setattr(sf, "write", forbid)

    uid = int(env.client.post("/enroll?name=spk01", content=big).json()["user_id"])
    ch = env.challenge(uid)
    env.say(ch["digits"])
    assert env.verify(uid, ch["nonce"], data=big).json()["accepted"]
    if webm is not None:
        ch = env.challenge(uid)
        env.say(ch["digits"])
        assert env.verify(uid, ch["nonce"], data=webm).json()["accepted"]

    assert writes == []
    new = _snapshot(dirs) - before
    db_files = {Path(env.service.db.path + s) for s in ("", "-journal", "-wal", "-shm")}
    assert new - db_files == set(), new


def _encode_webm(wav: bytes) -> bytes:
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0", "-c:a", "libopus",
           "-f", "webm", "pipe:1"]
    return subprocess.run(cmd, input=wav, capture_output=True, check=True).stdout


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_decode_browser_webm(cfg: dict[str, Any]) -> None:
    audio = decode_audio(_encode_webm(wav_bytes(SPEAKER_A_HZ, seconds=2.0)), cfg)
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert abs(audio.size / 16000 - 2.0) < 0.1


def test_replay_hook(cfg: dict[str, Any]) -> None:
    assert load_replay_detector(cfg) == (None, None)
    cfg["api"]["replay"]["enabled"] = True
    with pytest.raises(NotImplementedError):
        load_replay_detector(cfg)


def test_replay_detector_rejects(cfg: dict[str, Any]) -> None:
    class Spoofed:
        def score(self, audio: np.ndarray) -> float:
            return -1.0

    reply = {"text": ""}
    asr = WhisperASR(cfg, transcriber=lambda audio: reply["text"])
    svc = VoiceAccessService(cfg, Database(":memory:"), FakeEmbedder(), asr, THRESHOLD,
                             replay=Spoofed(), replay_threshold=0.0)
    uid = svc.enroll("spk01", wav_bytes(SPEAKER_A_HZ))["user_id"]
    ch = svc.challenge(uid)
    reply["text"] = " ".join(PL_WORDS[int(d)] for d in ch["digits"])
    r = svc.verify(uid, ch["nonce"], wav_bytes(SPEAKER_A_HZ))
    assert not r["accepted"] and r["asr"]["digits_ok"] and r["speaker"]["score_ok"]
    assert r["replay"] == {"enabled": True, "score": -1.0, "threshold": 0.0, "cm_ok": False}


def _attempt(env: Env, uid: int, hz: float, wrong_digits: bool, scenario: str | None) -> Any:
    ch = env.challenge(uid)
    env.say(ch["digits"][::-1] if wrong_digits else ch["digits"])
    return env.verify(uid, ch["nonce"], hz=hz, scenario=scenario)


def test_scenario_label_does_not_change_decision(env: Env) -> None:
    uid = env.enroll("spk01", SPEAKER_A_HZ)
    cases = [(SPEAKER_A_HZ, False), (SPEAKER_A_HZ, True), (SPEAKER_B_HZ, False),
             (SPEAKER_B_HZ, True)]
    for hz, wrong in cases:
        outcomes = set()
        for scenario in [None, *env.service.cfg["api"]["scenarios"]]:
            body = _attempt(env, uid, hz, wrong, scenario).json()
            assert body["scenario"] == (scenario or env.service.cfg["api"]["default_scenario"])
            outcomes.add((body["accepted"], body["reject_reason"], body["asr"]["digits_ok"],
                          body["speaker"]["score"], body["speaker"]["score_ok"]))
        assert len(outcomes) == 1, outcomes
    stored = [r["scenario"] for r in env.service.db.list_attempts(uid)]
    assert stored == [env.service.cfg["api"]["default_scenario"],
                      *env.service.cfg["api"]["scenarios"]] * len(cases)


def test_invalid_scenario_keeps_challenge(env: Env) -> None:
    uid = env.enroll("spk01")
    ch = env.challenge(uid)
    env.say(ch["digits"])
    assert env.verify(uid, ch["nonce"], scenario="bogus").status_code == 422
    assert env.service.db.count("attempts") == 0
    assert env.verify(uid, ch["nonce"]).json()["accepted"]


def test_reject_reasons_logged(env: Env, cfg: dict[str, Any]) -> None:
    uid = env.enroll("spk01", SPEAKER_A_HZ)
    assert _attempt(env, uid, SPEAKER_A_HZ, False, "genuine").json()["reject_reason"] is None
    assert _attempt(env, uid, SPEAKER_A_HZ, True, "wrong_digits").json()["reject_reason"] == "digits"
    assert _attempt(env, uid, SPEAKER_B_HZ, False, "impostor").json()["reject_reason"] == "speaker"
    assert _attempt(env, uid, SPEAKER_B_HZ, True, "impostor").json()["reject_reason"] == "both"
    ch = env.challenge(uid)
    env.clock.t += float(cfg["api"]["challenge_ttl_seconds"]) + 1
    assert env.verify(uid, ch["nonce"], scenario="replay").status_code == 410
    assert env.verify(uid, ch["nonce"], scenario="replay").status_code == 409
    got = [(r["scenario"], r["reject_reason"]) for r in env.service.db.list_attempts(uid)]
    assert got == [("genuine", None), ("wrong_digits", "digits"), ("impostor", "speaker"),
                   ("impostor", "both"), ("replay", "expired"), ("replay", None)]


def test_reject_reason_replay_cm(cfg: dict[str, Any]) -> None:
    class Spoofed:
        def score(self, audio: np.ndarray) -> float:
            return -1.0

    reply = {"text": ""}
    asr = WhisperASR(cfg, transcriber=lambda audio: reply["text"])
    svc = VoiceAccessService(cfg, Database(":memory:"), FakeEmbedder(), asr, THRESHOLD,
                             replay=Spoofed(), replay_threshold=0.0)
    uid = svc.enroll("spk01", wav_bytes(SPEAKER_A_HZ))["user_id"]
    ch = svc.challenge(uid)
    reply["text"] = " ".join(PL_WORDS[int(d)] for d in ch["digits"])
    r = svc.verify(uid, ch["nonce"], wav_bytes(SPEAKER_A_HZ), "replay")
    assert r["reject_reason"] == "replay_cm"
    assert svc.db.list_attempts(uid)[0]["reject_reason"] == "replay_cm"


def test_stats_endpoints(env: Env) -> None:
    a, b = env.enroll("spk01", SPEAKER_A_HZ), env.enroll("spk02", SPEAKER_B_HZ)
    _attempt(env, a, SPEAKER_A_HZ, False, "genuine")
    _attempt(env, a, SPEAKER_B_HZ, False, "impostor")
    _attempt(env, b, SPEAKER_B_HZ, True, "wrong_digits")

    body = env.client.get("/stats").json()
    assert body["n_attempts"] == 3 and body["user_id"] is None
    by = {s["scenario"]: s for s in body["scenarios"]}
    assert list(by) == env.service.cfg["api"]["scenarios"]
    assert by["genuine"]["accept_rate"] == 1.0
    assert by["genuine"]["score_mean"] == pytest.approx(1.0, abs=1e-6)
    assert by["impostor"]["accept_rate"] == 0.0 and by["impostor"]["reject_reasons"]["speaker"] == 1
    assert by["wrong_digits"]["reject_reasons"]["digits"] == 1
    assert by["replay"]["n_attempts"] == 0 and by["replay"]["accept_rate"] is None

    only_a = env.client.get(f"/stats?user_id={a}").json()
    assert only_a["n_attempts"] == 2 and only_a["user_id"] == a
    assert env.client.get("/stats?user_id=999").status_code == 404

    r = env.client.get("/stats.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    lines = r.text.strip().splitlines()
    assert len(lines) == 4 and lines[0].startswith("id,user_id,")
    assert len(env.client.get(f"/stats.csv?user_id={b}").text.strip().splitlines()) == 2


def test_unsupported_speaker_model(cfg: dict[str, Any]) -> None:
    cfg["api"]["speaker_model"] = "gmm"
    with pytest.raises(ValueError):
        VoiceAccessService(cfg, Database(":memory:"), FakeEmbedder(), WhisperASR(cfg), THRESHOLD)
    with pytest.raises(ApiError):
        decode_audio(b"", cfg)
