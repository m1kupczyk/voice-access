"""SpeechBrain ECAPA-TDNN wrapper with an on-disk embedding cache."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np


def load_speechbrain_encoder(cfg: dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    """Load the pretrained model on cfg['device']; return audio (n,) -> embedding (D,)."""
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    e = cfg["ecapa"]
    torch.set_num_threads(int((cfg.get("runtime") or {}).get("num_threads", 1)))
    clf = EncoderClassifier.from_hparams(
        source=e["model_source"], savedir=e["savedir"], run_opts={"device": cfg["device"]})
    clf.eval()

    def encode(audio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            emb = clf.encode_batch(torch.from_numpy(np.asarray(audio, dtype=np.float32))[None, :])
        return emb.squeeze().cpu().numpy().astype(np.float32)

    return encode


class EmbeddingCache:
    """Embeddings keyed by (model name, file path); one .npz file per model name."""

    def __init__(self, cache_dir: str | Path, model_name: str) -> None:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
        self.path = Path(cache_dir) / f"{slug}.npz"
        self._data: dict[str, np.ndarray] = {}
        if self.path.exists():
            z = np.load(self.path, allow_pickle=False)
            self._data = {str(k): v for k, v in zip(z["keys"], z["values"])}
        self._dirty = 0

    @staticmethod
    def key(path: str | Path) -> str:
        return hashlib.sha1(str(path).encode("utf-8")).hexdigest()

    def get(self, path: str | Path) -> np.ndarray | None:
        return self._data.get(self.key(path))

    def put(self, path: str | Path, emb: np.ndarray) -> None:
        self._data[self.key(path)] = emb
        self._dirty += 1

    @property
    def dirty(self) -> int:
        return self._dirty

    def flush(self) -> None:
        """Write the cache atomically (embeddings only, no audio)."""
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self._data.keys()))
        values = np.stack(list(self._data.values())) if self._data else np.zeros((0, 0))
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=keys, values=values)
        tmp.replace(self.path)
        self._dirty = 0

    def __len__(self) -> int:
        return len(self._data)


class EcapaEmbedder:
    """audio -> L2-unnormalized embedding; `embed_file` adds disk caching."""

    def __init__(
        self, cfg: dict[str, Any], encoder: Callable[[np.ndarray], np.ndarray] | None = None
    ) -> None:
        self.cfg = cfg
        self.name = cfg["ecapa"]["model_source"]
        self._encoder = encoder
        self.cache = EmbeddingCache(cfg["paths"]["embeddings_dir"], self.name)
        self.n_computed = 0

    @property
    def encoder(self) -> Callable[[np.ndarray], np.ndarray]:
        if self._encoder is None:  # lazy: fully cached runs never load the model
            self._encoder = load_speechbrain_encoder(self.cfg)
        return self._encoder

    def embed(self, audio: np.ndarray) -> np.ndarray:
        return self.encoder(audio)

    def embed_file(self, path: str | Path) -> np.ndarray:
        """Cached embedding of an audio file (loaded via src.data.audio.load_audio)."""
        emb = self.cache.get(path)
        if emb is None:
            from src.data.audio import load_audio

            emb = self.embed(load_audio(path, self.cfg, vad=bool(self.cfg["ecapa"]["vad"])))
            self.cache.put(path, emb)
            self.n_computed += 1
            if self.cache.dirty >= int(self.cfg["ecapa"]["cache_flush_every"]):
                self.cache.flush()
        return emb
