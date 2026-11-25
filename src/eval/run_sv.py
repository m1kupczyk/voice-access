"""Speaker-verification experiments on VoxCeleb1-O (GMM-UBM and ECAPA + cosine).

Usage:
    python -m src.eval.run_sv --system gmm                       # full run
    python -m src.eval.run_sv --system ecapa --max-trials 200       # smoke test
    python -m src.eval.run_sv --system gmm --max-utts 20 --max-trials 50   # smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

# Thread env vars must be set BEFORE numpy/sklearn are imported (reproducibility).
from src.threads import (  # noqa: E402
    apply_num_threads, limit_threads, num_threads_from_config, peek_num_threads,
)

if __name__ == "__main__":
    from src.config import DEFAULT_CONFIG_PATH  # yaml only, no numpy

    apply_num_threads(peek_num_threads(DEFAULT_CONFIG_PATH))

import numpy as np  # noqa: E402
import pandas as pd
from tqdm import tqdm

from src.config import load_config
from src.data.audio import load_audio
from src.eval.metrics import compute_eer
from src.features.mfcc import extract_mfcc
from src.models.ecapa import EcapaEmbedder
from src.models.gmm_ubm import (
    collect_ubm_frames, list_ubm_files, load_ubm, save_ubm, train_ubm,
)
from src.repro import make_run_dir, set_seed
from src.verification.scoring import mean_loglik
from src.verification.verifier import EcapaVerifier, GmmUbmVerifier


def run_gmm(
    cfg: dict[str, Any], max_trials: int | None, ubm_path: str | None, run_id: str | None
) -> Path:
    """Train (or load) the UBM, score VoxCeleb1-O trials, save results. Returns run dir."""
    n_threads = num_threads_from_config(cfg)
    apply_num_threads(n_threads)
    tp_applied = limit_threads(n_threads)
    set_seed(int(cfg["seed"]))
    g = cfg["gmm"]
    run_dir = make_run_dir(cfg, g["results_experiment"], run_id)
    metrics: dict[str, Any] = {
        "ubm_source": g["ubm_source"], "n_components": int(g["n_components"]),
        "seed": int(cfg["seed"]), "max_trials": max_trials,
        "num_threads": n_threads, "threadpoolctl_applied": tp_applied,
    }

    # --- UBM ---
    if ubm_path:
        ubm = load_ubm(ubm_path)
        metrics["ubm_loaded_from"] = ubm_path
    else:
        t0 = time.perf_counter()
        files = list_ubm_files(cfg)
        frames, n_utts = collect_ubm_frames(files, cfg)
        t1 = time.perf_counter()
        ubm = train_ubm(frames, cfg)
        t2 = time.perf_counter()
        metrics.update({
            "ubm_n_utts": n_utts, "ubm_n_frames": int(frames.shape[0]),
            "ubm_feature_seconds": t1 - t0, "ubm_train_seconds": t2 - t1,
        })
        save_ubm(ubm, run_dir / "ubm.npz")
    verifier = GmmUbmVerifier(ubm, cfg)

    # --- Trials ---
    mdir = Path(cfg["paths"]["manifests_dir"])
    manifest = pd.read_csv(mdir / cfg["voxceleb1"]["manifest_file"])
    path_of = dict(zip(manifest["utt_id"], manifest["path"]))
    trials = pd.read_csv(mdir / cfg["voxceleb1"]["trials_file"])
    if max_trials is not None and max_trials < len(trials):
        trials = trials.sample(n=max_trials, random_state=int(cfg["seed"])).sort_index()
    trials = trials.reset_index(drop=True)

    chunk = int(g["score_chunk_frames"])
    feats_cache: dict[str, np.ndarray] = {}
    ubm_ll_cache: dict[str, float] = {}
    tmpl_cache: dict[str, Any] = {}

    def feats(utt: str) -> np.ndarray:
        if utt not in feats_cache:
            feats_cache[utt] = extract_mfcc(load_audio(path_of[utt], cfg), cfg)
        return feats_cache[utt]

    t3 = time.perf_counter()
    scores = np.empty(len(trials), dtype=np.float64)
    for i, row in enumerate(tqdm(trials.itertuples(index=False), total=len(trials), desc="Scoring")):
        if row.enroll not in tmpl_cache:
            tmpl_cache[row.enroll] = verifier.enroll_features([feats(row.enroll)])
        if row.test not in ubm_ll_cache:
            ubm_ll_cache[row.test] = mean_loglik(
                feats(row.test), ubm.weights, ubm.means, ubm.variances, chunk)
        scores[i] = verifier.score_features(
            tmpl_cache[row.enroll], feats(row.test), ubm_ll_cache[row.test])
    t4 = time.perf_counter()

    trials = trials.assign(score=scores)
    trials.to_csv(run_dir / "scores.csv", index=False, float_format="%.10g")
    labels = trials["label"].to_numpy()
    metrics.update({
        "n_trials": len(trials), "n_target": int((labels == 1).sum()),
        "n_nontarget": int((labels == 0).sum()),
        "eer": compute_eer(labels, scores) if len(set(labels)) == 2 else None,
        "scoring_seconds": t4 - t3,
        "note": "LibriSpeech UBM: domain mismatch vs VoxCeleb1 (see PLAN.md)",
    })
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return run_dir


def run_ecapa(cfg: dict[str, Any], max_trials: int | None, run_id: str | None) -> Path:
    """Score VoxCeleb1-O trials with ECAPA embeddings + cosine. Returns run dir."""
    n_threads = num_threads_from_config(cfg)
    apply_num_threads(n_threads)
    tp_applied = limit_threads(n_threads)
    set_seed(int(cfg["seed"]))
    import torch

    torch.set_num_threads(n_threads)
    e = cfg["ecapa"]
    run_dir = make_run_dir(cfg, e["results_experiment"], run_id)
    mdir = Path(cfg["paths"]["manifests_dir"])
    manifest = pd.read_csv(mdir / cfg["voxceleb1"]["manifest_file"])
    path_of = dict(zip(manifest["utt_id"], manifest["path"]))
    trials = pd.read_csv(mdir / cfg["voxceleb1"]["trials_file"])
    if max_trials is not None and max_trials < len(trials):
        trials = trials.sample(n=max_trials, random_state=int(cfg["seed"])).sort_index()
    trials = trials.reset_index(drop=True)

    embedder = EcapaEmbedder(cfg)
    verifier = EcapaVerifier(embedder)
    n_cached_before = len(embedder.cache)
    tmpl_cache: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    scores = np.empty(len(trials), dtype=np.float64)
    for i, row in enumerate(tqdm(trials.itertuples(index=False), total=len(trials), desc="Scoring")):
        if row.enroll not in tmpl_cache:
            tmpl_cache[row.enroll] = verifier.enroll_embeddings(
                [embedder.embed_file(path_of[row.enroll])])
        scores[i] = verifier.score_embedding(
            tmpl_cache[row.enroll], embedder.embed_file(path_of[row.test]))
    embedder.cache.flush()
    t1 = time.perf_counter()

    trials = trials.assign(score=scores)
    trials.to_csv(run_dir / "scores.csv", index=False, float_format="%.10g")
    labels = trials["label"].to_numpy()
    metrics = {
        "model": embedder.name, "seed": int(cfg["seed"]), "max_trials": max_trials,
        "num_threads": n_threads, "threadpoolctl_applied": tp_applied,
        "device": cfg["device"], "n_trials": len(trials),
        "n_target": int((labels == 1).sum()), "n_nontarget": int((labels == 0).sum()),
        "eer": compute_eer(labels, scores) if len(set(labels)) == 2 else None,
        "n_embeddings_computed": embedder.n_computed,
        "n_embeddings_cached_at_start": n_cached_before,
        "scoring_seconds": t1 - t0,
        "seconds_per_computed_embedding":
            (t1 - t0) / embedder.n_computed if embedder.n_computed else None,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return run_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", choices=["gmm", "ecapa"], default="gmm")
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-trials", type=int, default=None, help="random subset of trials")
    ap.add_argument("--max-utts", type=int, default=None, help="override gmm.ubm_max_utts")
    ap.add_argument("--ubm", default=None, help="reuse a saved ubm.npz, skip training")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=None, help="override config seed")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.max_utts is not None:
        cfg["gmm"]["ubm_max_utts"] = args.max_utts
    if args.system == "ecapa":
        run_ecapa(cfg, args.max_trials, args.run_id)
    else:
        run_gmm(cfg, args.max_trials, args.ubm, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
