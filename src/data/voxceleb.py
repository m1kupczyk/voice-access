"""VoxCeleb1 manifest and VoxCeleb1-O trial list (no automatic download).

Expected layout: <root>/<wav_subdir>/idXXXXX/<video_id>/<n>.wav.
The official cleaned trial list (veri_test2.txt) has lines "label path1 path2".
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config


def build_manifest(root: str | Path, cfg: dict[str, Any]) -> pd.DataFrame:
    """Scan the VoxCeleb1 test speakers and return a manifest DataFrame.

    Columns: speaker_id, video_id, utt_id (relative path like trial lists), path.
    """
    vc = cfg["voxceleb1"]
    lo, hi = vc["test_speaker_id_range"]
    wav_root = Path(root) / vc["wav_subdir"]
    rows: list[dict[str, str]] = []
    for spk in range(lo, hi + 1):
        spk_id = f"id{spk}"
        for f in sorted((wav_root / spk_id).glob(f"*/*{vc['audio_extension']}")):
            rows.append({
                "speaker_id": spk_id,
                "video_id": f.parent.name,
                "utt_id": f"{spk_id}/{f.parent.name}/{f.name}",
                "path": str(f),
            })
    return pd.DataFrame(rows, columns=["speaker_id", "video_id", "utt_id", "path"])


def load_trials(trials_path: str | Path) -> pd.DataFrame:
    """Parse a trial list. Columns: label (1 target / 0 non-target), enroll, test."""
    df = pd.read_csv(trials_path, sep=r"\s+", header=None, names=["label", "enroll", "test"])
    if not df["label"].isin([0, 1]).all():
        raise ValueError(f"{trials_path}: labels must be 0 or 1")
    return df


def check_trials(
    trials: pd.DataFrame, root: str | Path, cfg: dict[str, Any]
) -> tuple[list[str], int]:
    """Check that every file referenced by the trials exists.

    Returns (sorted missing utt_ids, number of unique referenced files).
    """
    wav_root = Path(root) / cfg["voxceleb1"]["wav_subdir"]
    needed = sorted(set(trials["enroll"]) | set(trials["test"]))
    missing = [u for u in needed if not (wav_root / u).is_file()]
    return missing, len(needed)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate VoxCeleb1-O files and build manifests.")
    ap.add_argument("--root", help="VoxCeleb1 root (default: paths.voxceleb1_root)")
    ap.add_argument("--trials", help="Official trial list (default: paths.voxceleb1_trials)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    root = Path(args.root or cfg["paths"]["voxceleb1_root"])
    trials_path = Path(args.trials or cfg["paths"]["voxceleb1_trials"])
    if not trials_path.is_file():
        print(f"Trial list not found: {trials_path}")
        return 2

    trials = load_trials(trials_path)
    missing, n_needed = check_trials(trials, root, cfg)
    out_dir = Path(cfg["paths"]["manifests_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    vc = cfg["voxceleb1"]

    report = out_dir / vc["missing_report_file"]
    report.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    trials.to_csv(out_dir / vc["trials_file"], index=False)
    manifest = build_manifest(root, cfg)
    manifest.to_csv(out_dir / vc["manifest_file"], index=False)

    print(f"Trials: {len(trials)}; unique files referenced: {n_needed}")
    print(f"Files found: {n_needed - len(missing)}; missing: {len(missing)}")
    print(f"Manifest rows (test speakers): {len(manifest)}")
    print(f"Missing list: {report}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
