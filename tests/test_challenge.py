"""Stage 4 tests: normalization, challenge generation, synthetic strings, pipeline (mocked ASR)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.challenge.asr import WhisperASR
from src.challenge.digits import different_challenge, generate_challenge
from src.challenge.normalize import edit_distance, normalize_digits
from src.challenge.pipeline import ChallengePipeline, decide
from src.data import common_voice as cv
from src.eval.run_challenge import asr_metrics, build_trials, scenario_rates

CFG = {
    "seed": 7, "audio": {"sample_rate": 16000},
    "challenge": {"digit_string_length": 4, "unique_digits": True},
    "common_voice": {"min_digits_per_speaker": 9, "enroll_clips": 5, "strings_per_speaker": 5,
                     "silence_ms": [100, 200]},
}


@pytest.mark.parametrize("text,expected", [
    ("jeden dwa trzy cztery", "1234"),
    ("Pięć, sześć, siedem, osiem.", "5678"),
    ("piec szesc dziewiec zero", "5690"),
    ("1 2 3 4", "1234"),
    ("1-2-3-4", "1234"),
    ("one two three nine", "1239"),
    ("dwie jedna", "21"),
    ("dwadzieścia", "20"),
    ("to jest 12 test", "12"),
    ("", ""),
])
def test_normalize_digits(text: str, expected: str) -> None:
    assert normalize_digits(text) == expected


def test_edit_distance() -> None:
    assert edit_distance("1234", "1234") == 0
    assert edit_distance("1234", "1244") == 1
    assert edit_distance("1234", "134") == 1
    assert edit_distance("", "12") == 2


def test_challenge_generation() -> None:
    a = generate_challenge(6, unique=True, rng=np.random.default_rng(1))
    b = generate_challenge(6, unique=True, rng=np.random.default_rng(1))
    assert a == b and len(set(a)) == 6
    s = generate_challenge(5)
    assert len(s) == 5 and s.isdigit()
    assert different_challenge("1234", 4, True, np.random.default_rng(0)) != "1234"
    with pytest.raises(ValueError):
        generate_challenge(11, unique=True)


def _clips(n_speakers: int = 3) -> pd.DataFrame:
    rows = [{"speaker": f"cv{s:03d}", "digit": str(d), "path": f"{s}_{d}.mp3"}
            for s in range(n_speakers) for d in range(10 if s else 9)]
    rows.append({"speaker": "cv099", "digit": "1", "path": "x.mp3"})  # too few digits
    return pd.DataFrame(rows)


def test_plan_seeded_and_disjoint() -> None:
    clips = _clips()
    kept, excluded = cv.eligible_speakers(clips, CFG)
    assert kept == ["cv000", "cv001", "cv002"] and excluded == 1
    p1, p2 = cv.build_plan(clips, CFG), cv.build_plan(clips, CFG)
    assert p1.equals(p2)
    for _, r in p1.iterrows():
        pool = set(clips.loc[clips["speaker"] == r["speaker"], "digit"]) - set(r["enroll_digits"])
        assert len(r["enroll_digits"]) == 5
        assert len(set(r["test_strings"])) == len(r["test_strings"]) > 0
        for s in r["test_strings"]:
            assert len(s) == 4 and set(s) <= pool  # test clips disjoint from enrollment clips
    CFG2 = {**CFG, "seed": 8}
    assert not cv.build_plan(clips, CFG2).equals(p1)


def test_concat_clips_deterministic() -> None:
    wavs = [np.ones(100, np.float32), np.ones(50, np.float32)]
    a = cv.concat_clips(wavs, 16000, (100, 200), np.random.default_rng(0))
    b = cv.concat_clips(wavs, 16000, (100, 200), np.random.default_rng(0))
    assert np.array_equal(a, b)
    assert 150 + 1600 <= len(a) <= 150 + 3200


def test_decide_and_pipeline() -> None:
    assert decide("1234", "1234", 0.8, 0.5).accepted
    assert not decide("1234", "1235", 0.8, 0.5).accepted  # wrong digits
    assert not decide("1234", "1234", 0.4, 0.5).accepted  # low score
    class Emb:
        def embed(self, audio: np.ndarray) -> np.ndarray:
            return np.array([1.0, 0.0])
    asr = WhisperASR({}, transcriber=lambda a: "Jeden, dwa, trzy, cztery")
    pipe = ChallengePipeline(Emb(), asr, threshold=0.5)
    assert pipe.verify(np.zeros(10, np.float32), "1234", np.array([1.0, 0.1])).accepted
    assert not pipe.verify(np.zeros(10, np.float32), "1235", np.array([1.0, 0.1])).accepted
    assert not pipe.verify(np.zeros(10, np.float32), "1234", np.array([0.0, 1.0])).accepted


def test_trials_and_rates() -> None:
    utts = pd.DataFrame({
        "speaker": ["a", "a", "b", "b"], "challenge": ["1234", "5678", "1234", "9012"],
        "recognized": ["1234", "5678", "1234", "9019"],
        "asr_seconds": [1.0] * 4, "ecapa_seconds": [0.1] * 4, "audio_seconds": [5.0] * 4})
    embs = np.array([[1, 0], [1, 0.1], [0, 1], [0.1, 1]], dtype=float)
    templates = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    t = build_trials(utts, embs, templates, CFG, np.random.default_rng(0))
    counts = t["scenario"].value_counts().to_dict()
    assert counts == {"genuine": 4, "impostor": 4, "wrong_digits": 4, "replay": 4}
    assert (t.loc[t["scenario"] == "replay", "challenge"] != "").all()
    r = scenario_rates(t, 0.5)
    assert r["genuine"]["accept_speaker_only"] == 1.0
    assert r["genuine"]["accept_combined"] == 0.75  # one misrecognized utterance
    assert r["impostor"]["accept_speaker_only"] == 0.0
    assert r["wrong_digits"]["accept_combined"] == 0.0
    m = asr_metrics(utts)
    assert m["digit_string_accuracy"] == 0.75 and m["digit_error_rate"] == 1 / 16
