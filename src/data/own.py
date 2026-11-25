"""Own-corpus index loader and validator.

Recordings live in <own_root>/<speaker_id>/s<session>/<kind>_<nn>.wav and are
listed in <own_root>/<index_file>, which record.py appends to.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config

INDEX_COLUMNS = ["speaker_id", "session", "date", "kind", "prompt", "device", "path"]
KINDS = ("free", "digits")


def index_path(cfg: dict[str, Any]) -> Path:
    """Location of the recordings index CSV."""
    return Path(cfg["paths"]["own_root"]) / cfg["own_corpus"]["index_file"]


def load_own_manifest(cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Load the own-corpus index.

    Returns (manifest with absolute `path`, list of indexed files missing on disk).
    Rows whose file is missing are dropped from the manifest.
    """
    idx = index_path(cfg)
    if not idx.is_file():
        return pd.DataFrame(columns=INDEX_COLUMNS), []
    df = pd.read_csv(idx, dtype={"speaker_id": str, "prompt": str})
    bad = set(df["kind"]) - set(KINDS)
    if bad:
        raise ValueError(f"{idx}: unknown kind(s) {sorted(bad)}")
    root = Path(cfg["paths"]["own_root"])
    full = df["path"].map(lambda p: root / p)
    exists = full.map(Path.is_file)
    missing = df.loc[~exists, "path"].tolist()
    df = df[exists].copy()
    df["path"] = full[exists].map(str)
    return df.reset_index(drop=True), missing


def corpus_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Recording counts per speaker and session (for checking the >= 2 sessions rule)."""
    return df.groupby(["speaker_id", "session"]).size().rename("n_recordings").reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the own corpus and write its manifest.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    df, missing = load_own_manifest(cfg)
    out = Path(cfg["paths"]["manifests_dir"]) / cfg["own_corpus"]["manifest_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Recordings: {len(df)}; indexed but missing on disk: {len(missing)}")
    for p in missing:
        print(f"  missing: {p}")
    if len(df):
        summ = corpus_summary(df)
        print(summ.to_string(index=False))
        n_sess = summ.groupby("speaker_id")["session"].nunique()
        short = n_sess[n_sess < cfg["own_corpus"]["sessions_min"]]
        for spk in short.index:
            print(f"  warning: {spk} has {short[spk]} session(s), "
                  f"need {cfg['own_corpus']['sessions_min']}")
    print(f"Manifest: {out}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
