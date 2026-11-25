"""Whisper ASR wrapper (CPU) returning raw text and the digit string."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from src.challenge.normalize import normalize_digits


def load_whisper_transcriber(cfg: dict[str, Any]) -> Callable[[np.ndarray], str]:
    """Load openai-whisper on cfg['device']; return audio (16 kHz float32) -> text."""
    import torch
    import whisper

    a = cfg["asr"]
    torch.set_num_threads(int((cfg.get("runtime") or {}).get("num_threads", 1)))
    model = whisper.load_model(a["model"], device=cfg["device"])

    def transcribe(audio: np.ndarray) -> str:
        res = model.transcribe(
            np.asarray(audio, dtype=np.float32), language=a["language"], fp16=bool(a["fp16"]),
            temperature=float(a["temperature"]), initial_prompt=a["initial_prompt"],
            condition_on_previous_text=bool(a["condition_on_previous_text"]), verbose=None)
        return str(res["text"])

    return transcribe


class WhisperASR:
    """audio -> text / digit string; the transcriber is injectable (mocked in tests)."""

    def __init__(
        self, cfg: dict[str, Any], transcriber: Callable[[np.ndarray], str] | None = None
    ) -> None:
        self.cfg = cfg
        self._transcriber = transcriber

    @property
    def transcriber(self) -> Callable[[np.ndarray], str]:
        if self._transcriber is None:  # lazy model load
            self._transcriber = load_whisper_transcriber(self.cfg)
        return self._transcriber

    def transcribe(self, audio: np.ndarray) -> str:
        return self.transcriber(audio)

    def recognize_digits(self, audio: np.ndarray) -> tuple[str, str, float]:
        """Return (raw text, digit string, latency in seconds)."""
        t0 = time.perf_counter()
        text = self.transcribe(audio)
        return text, normalize_digits(text), time.perf_counter() - t0

    def warmup(self) -> None:
        """Load the model and run one dummy pass so latency stats exclude start-up cost."""
        self.transcribe(np.zeros(int(self.cfg["audio"]["sample_rate"]), dtype=np.float32))
