"""Common Voice Single Word (pl): digit-clip manifest and seeded synthetic digit strings.

Each speaker has at most one clip per digit, so the digits of a speaker are split into
disjoint enrollment digits and test-pool digits; test strings are built only from the
test pool (no clip is shared between enrollment and test). Speaker IDs are pseudonymized
(`cv000`, ...) and never leave the local manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.challenge.normalize import digit_word_to_digit
from src.data.audio import load_audio

MANIFEST_COLUMNS = ["speaker", "digit", "path"]


def load_digit_clips(cfg: dict[str, Any]) -> pd.DataFrame:
    """Validated single-digit clips: one row per (speaker, digit), pseudonymized speakers.

    Duplicate takes of a digit are reduced to one (most up-votes, then path). Clips missing
    on disk are dropped.
    """
    c = cfg["common_voice"]
    root = Path(c["root"])
    df = pd.read_csv(root / c["tsv"], sep="\t", quoting=3, dtype=str)
    df["digit"] = df["sentence"].map(lambda s: digit_word_to_digit(str(s)))
    df = df[df["digit"].notna()].copy()
    df["up"] = df["up_votes"].astype(int)
    df["path"] = df["path"].map(lambda p: str(root / c["clips_subdir"] / p))
    df = df[df["path"].map(lambda p: Path(p).is_file())]
    df = df.sort_values(["client_id", "digit", "up", "path"], ascending=[True, True, False, True])
    df = df.drop_duplicates(["client_id", "digit"])
    ids = sorted(df["client_id"].unique())
    pseudo = {cid: f"cv{i:03d}" for i, cid in enumerate(ids)}
    df["speaker"] = df["client_id"].map(pseudo)
    return df[MANIFEST_COLUMNS].sort_values(["speaker", "digit"]).reset_index(drop=True)


def eligible_speakers(clips: pd.DataFrame, cfg: dict[str, Any]) -> tuple[list[str], int]:
    """Speakers with enough digits for enrollment + at least one test string; (kept, n_excluded)."""
    c = cfg["common_voice"]
    n_min = max(int(c["min_digits_per_speaker"]),
                int(c["enroll_clips"]) + int(cfg["challenge"]["digit_string_length"]))
    counts = clips.groupby("speaker")["digit"].nunique()
    kept = sorted(counts[counts >= n_min].index)
    return kept, int(len(counts) - len(kept))


def speaker_plan(
    digits: list[str], cfg: dict[str, Any], seed: int, index: int
) -> dict[str, Any]:
    """Seeded split of a speaker's digits into enrollment digits and distinct test strings."""
    c, length = cfg["common_voice"], int(cfg["challenge"]["digit_string_length"])
    rng = np.random.default_rng([seed, index])
    perm = [digits[i] for i in rng.permutation(len(digits))]
    n_enroll = int(c["enroll_clips"])
    enroll, pool = sorted(perm[:n_enroll]), sorted(perm[n_enroll:])
    if len(pool) < length:
        raise ValueError("test pool smaller than the digit string length")
    strings: list[str] = []
    n_target = int(c["strings_per_speaker"])
    for _ in range(100 * n_target):  # bounded: pools can hold fewer than n_target strings
        if len(strings) == n_target:
            break
        s = "".join(pool[i] for i in rng.permutation(len(pool))[:length])
        if s not in strings:
            strings.append(s)
    return {"enroll_digits": enroll, "test_strings": strings}


def build_plan(clips: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Plan for all eligible speakers: columns speaker, enroll_digits, test_strings (lists)."""
    speakers, _ = eligible_speakers(clips, cfg)
    seed = int(cfg["seed"])
    rows = []
    for i, spk in enumerate(speakers):
        digits = sorted(clips.loc[clips["speaker"] == spk, "digit"])
        rows.append({"speaker": spk, **speaker_plan(digits, cfg, seed, i)})
    return pd.DataFrame(rows)


def concat_clips(
    clips: list[np.ndarray], sr: int, silence_ms: tuple[float, float], rng: np.random.Generator
) -> np.ndarray:
    """Concatenate clips with seeded random silences (uniform in silence_ms) between them."""
    parts: list[np.ndarray] = []
    for k, w in enumerate(clips):
        if k:
            n = int(sr * rng.uniform(silence_ms[0], silence_ms[1]) / 1000)
            parts.append(np.zeros(n, dtype=np.float32))
        parts.append(np.asarray(w, dtype=np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def load_digit_clip(path: str | Path, cfg: dict[str, Any]) -> np.ndarray:
    """Load one clip (16 kHz mono, peak-normalized, VAD-trimmed so silences are controlled)."""
    return load_audio(path, cfg, vad=True)


def build_utterance(
    digit_string: str, wavs: dict[str, np.ndarray], cfg: dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    """Audio of `digit_string` from the speaker's per-digit clips `wavs`."""
    lo, hi = cfg["common_voice"]["silence_ms"]
    return concat_clips([wavs[d] for d in digit_string], int(cfg["audio"]["sample_rate"]),
                        (float(lo), float(hi)), rng)
