"""CLI recorder for own-corpus collection.

Example:
    python -m src.data.record --speaker spk01 --session 1 --device laptop_mic --consent
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from src.config import load_config
from src.data.own import INDEX_COLUMNS, index_path


def make_digit_strings(seed: Any, n: int, length: int) -> list[str]:
    """Reproducible digit prompts for one (speaker, session)."""
    rng = random.Random(str(seed))
    return ["".join(rng.choice("0123456789") for _ in range(length)) for _ in range(n)]


def record_clip(seconds: float, sample_rate: int) -> np.ndarray:
    """Record `seconds` of mono audio from the default input device."""
    import sounddevice as sd  # imported lazily: not needed for tests

    data = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return data[:, 0]


def append_index(cfg: dict[str, Any], row: dict[str, Any]) -> None:
    """Append one row to the recordings index, writing the header if new."""
    idx = index_path(cfg)
    idx.parent.mkdir(parents=True, exist_ok=True)
    new = not idx.is_file()
    with open(idx, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def _save(cfg, speaker, session, kind, n, prompt, device, wav) -> None:
    root = Path(cfg["paths"]["own_root"])
    rel = Path(speaker) / f"s{session}" / f"{kind}_{n:02d}.wav"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    sf.write(root / rel, wav, int(cfg["audio"]["sample_rate"]))
    append_index(cfg, {"speaker_id": speaker, "session": session, "date": date.today().isoformat(),
                       "kind": kind, "prompt": prompt, "device": device, "path": str(rel)})


def _record_with_retry(seconds: float, sr: int, countdown: int) -> np.ndarray | None:
    """Record one clip; Enter accepts, 'r' re-records, 's' skips."""
    while True:
        for i in range(countdown, 0, -1):
            print(f"  {i}...", end=" ", flush=True)
            time.sleep(1)
        print("SPEAK")
        wav = record_clip(seconds, sr)
        ans = input("  [Enter] keep, [r] redo, [s] skip: ").strip().lower()
        if ans == "r":
            continue
        return None if ans == "s" else wav


def main() -> int:
    ap = argparse.ArgumentParser(description="Record one own-corpus session.")
    ap.add_argument("--speaker", required=True, help="Pseudonymized ID, e.g. spk01")
    ap.add_argument("--session", required=True, type=int)
    ap.add_argument("--device", default=None, help="Recording device label for the manifest")
    ap.add_argument("--consent", action="store_true",
                    help="Confirm a signed consent form exists for this speaker")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    if not args.consent:
        print("Refusing to record: pass --consent after the speaker signed the consent form.")
        return 2

    cfg = load_config(args.config) if args.config else load_config()
    oc, sr = cfg["own_corpus"], int(cfg["audio"]["sample_rate"])
    rec = oc["recording"]
    device = args.device or rec["devices"][0]
    if not args.speaker.startswith(oc["speaker_id_prefix"]):
        print(f"Speaker ID must start with '{oc['speaker_id_prefix']}' (pseudonymized).")
        return 2

    print(f"Speaker {args.speaker}, session {args.session}, device {device}")
    print("Free speech: talk about anything for "
          f"{rec['seconds_free_speech']} s (enrollment material).")
    wav = _record_with_retry(rec["seconds_free_speech"], sr, rec["countdown_seconds"])
    if wav is not None:
        _save(cfg, args.speaker, args.session, "free", 1, "", device, wav)

    prompts = make_digit_strings(f"{cfg['seed']}-{args.speaker}-{args.session}",
                                 rec["n_digit_strings"], rec["digit_string_length"])
    for n, p in enumerate(prompts, 1):
        print(f"Digits {n}/{len(prompts)}: read aloud in Polish:  {' '.join(p)}")
        secs = len(p) * rec["seconds_per_digit"] + rec["digit_extra_seconds"]
        wav = _record_with_retry(secs, sr, rec["countdown_seconds"])
        if wav is not None:
            _save(cfg, args.speaker, args.session, "digits", n, p, device, wav)
    print(f"Done. Index: {index_path(cfg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
