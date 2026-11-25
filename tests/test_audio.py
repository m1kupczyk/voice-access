"""Tests for the audio loader: resampling, mono conversion, VAD trimming."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from src.config import load_config
from src.data.audio import load_audio, resample, to_mono, vad_trim
from src.data.voxceleb import build_manifest, check_trials, load_trials

SEED = 0


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


def _tone(freq: float, sr: int, seconds: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_resample_length_and_frequency() -> None:
    x = _tone(440.0, 48000, 1.0)
    y = resample(x, 48000, 16000)
    assert len(y) == 16000
    peak_bin = np.argmax(np.abs(np.fft.rfft(y)))  # 1 s -> bin index == Hz
    assert abs(peak_bin - 440) <= 1


def test_resample_noop_same_rate() -> None:
    x = _tone(440.0, 16000, 0.1)
    assert resample(x, 16000, 16000) is x


def test_to_mono_averages_channels() -> None:
    stereo = np.stack([np.full(100, 0.2), np.full(100, 0.6)], axis=1).astype(np.float32)
    mono = to_mono(stereo)
    assert mono.shape == (100,)
    assert np.allclose(mono, 0.4)
    assert to_mono(mono) is mono


def test_vad_trims_silence(cfg: dict) -> None:
    sr = cfg["audio"]["sample_rate"]
    rng = np.random.default_rng(SEED)
    noise = lambda s: (1e-4 * rng.standard_normal(int(sr * s))).astype(np.float32)
    speech = _tone(300.0, sr, 1.0)
    x = np.concatenate([noise(1.0), speech, noise(1.0)])
    y = vad_trim(x, sr, cfg["audio"]["vad"], cfg["audio"]["eps"])
    pad = cfg["audio"]["vad"]["pad_ms"] / 1000
    assert len(speech) <= len(y) <= len(speech) + 2 * pad * sr + 2 * 0.025 * sr
    assert len(y) < len(x) / 2


def test_vad_returns_empty_for_silence(cfg: dict) -> None:
    sr = cfg["audio"]["sample_rate"]
    y = vad_trim(np.zeros(sr, dtype=np.float32), sr, cfg["audio"]["vad"], cfg["audio"]["eps"])
    assert y.size == 0


def test_vad_drops_short_click(cfg: dict) -> None:
    sr = cfg["audio"]["sample_rate"]
    x = np.zeros(2 * sr, dtype=np.float32)
    x[sr : sr + int(0.02 * sr)] = 0.5  # 20 ms click < min_speech_ms
    assert vad_trim(x, sr, cfg["audio"]["vad"], cfg["audio"]["eps"]).size == 0


def test_load_audio_end_to_end(cfg: dict, tmp_path) -> None:
    sr_in = 44100
    left = _tone(300.0, sr_in, 1.0, 0.1)
    stereo = np.stack([left, left], axis=1)
    silence = np.zeros((sr_in, 2), dtype=np.float32)
    path = tmp_path / "a.wav"
    sf.write(path, np.concatenate([silence, stereo, silence]), sr_in)
    y = load_audio(path, cfg)
    assert y.ndim == 1 and y.dtype == np.float32
    assert 0.9 * 16000 < len(y) < 1.3 * 16000
    assert np.isclose(np.max(np.abs(y)), cfg["audio"]["peak_target"], atol=1e-3)


def test_trial_check_reports_missing(cfg: dict, tmp_path) -> None:
    wav_dir = tmp_path / cfg["voxceleb1"]["wav_subdir"] / "id1" / "v"
    wav_dir.mkdir(parents=True)
    (wav_dir / "1.wav").write_bytes(b"")
    trials_file = tmp_path / "t.txt"
    trials_file.write_text("1 id1/v/1.wav id1/v/2.wav\n0 id1/v/1.wav id1/v/1.wav\n")
    missing, n = check_trials(load_trials(trials_file), tmp_path, cfg)
    assert missing == ["id1/v/2.wav"] and n == 2


def test_voxceleb_flat_layout_found(cfg: dict, tmp_path) -> None:
    """Regression: root/idXXXXX/video/NNNNN.wav is found with the default config."""
    utts = ["id10270/x6uYqmx31kE/00001.wav", "id10270/8jEAjG6SegY/00008.wav"]
    for u in utts:
        f = tmp_path / u
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"")
    trials_file = tmp_path / "t.txt"
    trials_file.write_text(f"1 {utts[0]} {utts[1]}\n")
    missing, n = check_trials(load_trials(trials_file), tmp_path, cfg)
    assert missing == [] and n == 2
    manifest = build_manifest(tmp_path, cfg)
    assert sorted(manifest["utt_id"]) == sorted(utts)
