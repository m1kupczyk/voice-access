"""Load and validate config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

# Top-level sections and keys that must be present.
_REQUIRED: dict[str, list[str]] = {
    "": ["seed", "device", "paths", "audio", "voxceleb1", "own_corpus"],
    "paths": ["voxceleb1_root", "voxceleb1_trials", "own_root", "manifests_dir", "results_dir"],
    "audio": ["sample_rate", "mono", "normalize", "peak_target", "eps", "vad"],
    "audio.vad": ["enabled", "frame_ms", "hop_ms", "threshold_db", "min_speech_ms", "pad_ms"],
}


def _get(cfg: dict[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in filter(None, dotted.split(".")):
        node = node[part]
    return node


def validate_config(cfg: dict[str, Any]) -> None:
    """Raise ValueError if required keys are missing or values are invalid."""
    for section, keys in _REQUIRED.items():
        try:
            node = _get(cfg, section)
        except (KeyError, TypeError):
            raise ValueError(f"config: missing section '{section}'") from None
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                raise ValueError(f"config: missing key '{section + '.' if section else ''}{key}'")
    if cfg["device"] not in ("cpu", "mps"):
        raise ValueError("config: device must be 'cpu' or 'mps'")
    if int(cfg["audio"]["sample_rate"]) <= 0:
        raise ValueError("config: audio.sample_rate must be positive")
    vad = cfg["audio"]["vad"]
    if vad["hop_ms"] <= 0 or vad["frame_ms"] < vad["hop_ms"]:
        raise ValueError("config: audio.vad requires 0 < hop_ms <= frame_ms")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read and validate the YAML config."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config: {path} is not a mapping")
    validate_config(cfg)
    return cfg
