"""Stage 4: text-dependent verification (random digits + Whisper ASR + ECAPA).

Usage:
    python -m src.eval.run_challenge --corpus cv                         # Common Voice SW, full
    python -m src.eval.run_challenge --corpus cv --max-speakers 4 --run-id smoke
    python -m src.eval.run_challenge --corpus own --threshold-from results/challenge/<run>/metrics.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.threads import (  # noqa: E402
    apply_num_threads, limit_threads, num_threads_from_config, peek_num_threads,
)

if __name__ == "__main__":
    from src.config import DEFAULT_CONFIG_PATH

    apply_num_threads(peek_num_threads(DEFAULT_CONFIG_PATH))

import numpy as np  # noqa: E402
import pandas as pd
from tqdm import tqdm

from src.challenge.asr import WhisperASR
from src.challenge.digits import different_challenge
from src.challenge.normalize import edit_distance
from src.challenge.pipeline import decide
from src.config import load_config
from src.data import common_voice as cv
from src.data.audio import load_audio
from src.eval.metrics import (
    compute_eer, percentile_ci, speaker_bootstrap_eer, speaker_disjoint_split,
    transfer_threshold_from_config,
)
from src.models.ecapa import EcapaEmbedder
from src.repro import make_run_dir, set_seed

SCENARIOS = ("genuine", "impostor", "wrong_digits", "replay")
LIMITATION = ("Synthetic strings are concatenated single-word clips (no coarticulation, "
              "artificial silences, one session/device per speaker); not equivalent to "
              "naturally read digit strings.")


# --------------------------------------------------------------------------- utterances
def _timed(fn, *args):  # noqa: ANN001
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def process_cv_speakers(
    cfg: dict[str, Any], embedder: EcapaEmbedder, asr: WhisperASR, speakers: list[str],
    clips: pd.DataFrame, plan: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    """Build enrollment/test audio per speaker; ASR + ECAPA. Returns (utts, embs, templates)."""
    seed = int(cfg["seed"])
    rows: list[dict[str, Any]] = []
    embs: list[np.ndarray] = []
    templates: dict[str, np.ndarray] = {}
    index = {s: i for i, s in enumerate(plan["speaker"])}
    for spk in tqdm(speakers, desc="Speakers"):
        p = plan.iloc[index[spk]]
        sub = clips[clips["speaker"] == spk].set_index("digit")["path"]
        wavs = {d: cv.load_digit_clip(sub[d], cfg) for d in sub.index}
        rng = np.random.default_rng([seed, 1, index[spk]])
        enroll = cv.build_utterance("".join(p["enroll_digits"]), wavs, cfg, rng)
        templates[spk] = embedder.embed(enroll)
        for s in p["test_strings"]:
            audio = cv.build_utterance(s, wavs, cfg, rng)
            emb, t_emb = _timed(embedder.embed, audio)
            text, rec, t_asr = asr.recognize_digits(audio)
            rows.append({"speaker": spk, "challenge": s, "recognized": rec, "raw_text": text,
                         "asr_seconds": t_asr, "ecapa_seconds": t_emb,
                         "audio_seconds": len(audio) / int(cfg["audio"]["sample_rate"])})
            embs.append(emb)
    return pd.DataFrame(rows), np.stack(embs), templates


def process_own(
    cfg: dict[str, Any], embedder: EcapaEmbedder, asr: WhisperASR
) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Own corpus: enroll on free speech of the first session, test on digit strings of the rest."""
    from src.data.own import load_own_manifest

    df, _ = load_own_manifest(cfg)
    e = cfg["challenge_eval"]
    rows, embs, templates = [], [], {}
    sessions_used: dict[str, Any] = {}
    vad = bool(cfg["ecapa"]["vad"])
    for spk, g in df.groupby("speaker_id"):
        first = int(g["session"].min())
        enr = g[(g["kind"] == e["own_enroll_kind"]) & (g["session"] == first)]
        tst = g[(g["kind"] == e["own_test_kind"]) & (g["session"] != first)]
        if enr.empty or tst.empty:
            continue
        templates[spk] = np.mean([embedder.embed(load_audio(p, cfg, vad=vad)) for p in enr["path"]],
                                 axis=0)
        sessions_used[spk] = {"enroll_session": first,
                              "test_sessions": sorted(int(x) for x in tst["session"].unique())}
        for r in tst.itertuples(index=False):
            audio = load_audio(r.path, cfg, vad=vad)
            emb, t_emb = _timed(embedder.embed, audio)
            text, rec, t_asr = asr.recognize_digits(audio)
            rows.append({"speaker": spk, "challenge": "".join(c for c in str(r.prompt) if c.isdigit()),
                         "recognized": rec, "raw_text": text, "asr_seconds": t_asr,
                         "ecapa_seconds": t_emb,
                         "audio_seconds": len(audio) / int(cfg["audio"]["sample_rate"])})
            embs.append(emb)
    if not rows:
        raise SystemExit("own corpus: no enrollment + test recordings in different sessions")
    return pd.DataFrame(rows), np.stack(embs), templates, sessions_used


# --------------------------------------------------------------------------- trials
def _unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def build_trials(
    utts: pd.DataFrame, embs: np.ndarray, templates: dict[str, np.ndarray],
    cfg: dict[str, Any], rng: np.random.Generator,
) -> pd.DataFrame:
    """Trials for the four scenarios (claimed = enrolled identity, audio_speaker = who spoke).

    genuine: owner, challenge = spoken digits. impostor: other speaker saying the challenge
    digits (zero-effort, correct digits). wrong_digits: owner, challenge is a random other
    string. replay: owner's audio played for a different challenge (the digits of another
    utterance of the same owner), i.e. a pre-recorded genuine utterance with other digits.
    """
    claims = sorted(templates)
    T = _unit(np.stack([templates[c] for c in claims]))
    S = _unit(embs) @ T.T  # (n_utts, n_claims)
    length, unique = int(cfg["challenge"]["digit_string_length"]), bool(cfg["challenge"]["unique_digits"])
    rows: list[dict[str, Any]] = []
    by_spk = {s: list(g.index) for s, g in utts.groupby("speaker")}

    def add(scn: str, claim: str, i: int, challenge: str, score: float) -> None:
        u = utts.iloc[i]
        rows.append({"scenario": scn, "label": int(scn == "genuine"), "claimed": claim,
                     "audio_speaker": u["speaker"], "challenge": challenge,
                     "recognized": u["recognized"], "score": float(score)})

    for i in range(len(utts)):
        u = utts.iloc[i]
        for k, c in enumerate(claims):
            if c == u["speaker"]:
                add("genuine", c, i, u["challenge"], S[i, k])
                add("wrong_digits", c, i,
                    different_challenge(u["challenge"], length, unique, rng), S[i, k])
                others = [j for j in by_spk[c] if utts.at[j, "challenge"] != u["challenge"]]
                if others:
                    j = others[int(rng.integers(len(others)))]
                    add("replay", c, i, utts.at[j, "challenge"], S[i, k])
            else:
                add("impostor", c, i, u["challenge"], S[i, k])
    return pd.DataFrame(rows)


def scenario_rates(trials: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Per scenario: trial count and acceptance rate by speaker check, digits check, combined."""
    out: dict[str, Any] = {}
    for scn in SCENARIOS:
        t = trials[trials["scenario"] == scn]
        if t.empty:
            out[scn] = {"n": 0}
            continue
        dec = [decide(r.challenge, r.recognized, r.score, threshold) for r in t.itertuples()]
        out[scn] = {"n": len(t),
                    "accept_speaker_only": float(np.mean([d.score_ok for d in dec])),
                    "accept_digits_only": float(np.mean([d.digits_ok for d in dec])),
                    "accept_combined": float(np.mean([d.accepted for d in dec]))}
    return out


def asr_metrics(utts: pd.DataFrame) -> dict[str, Any]:
    """Digit-string accuracy, digit error rate (edit distance / reference length), latency."""
    ref, hyp = utts["challenge"].tolist(), utts["recognized"].tolist()
    errs = sum(edit_distance(r, h) for r, h in zip(ref, hyp))
    asr_s = utts["asr_seconds"].to_numpy()
    return {"n_utterances": len(utts),
            "digit_string_accuracy": float(np.mean([r == h for r, h in zip(ref, hyp)])),
            "digit_error_rate": errs / max(1, sum(len(r) for r in ref)),
            "asr_latency_mean_s": float(asr_s.mean()),
            "asr_latency_median_s": float(np.median(asr_s)),
            "asr_latency_p95_s": float(np.quantile(asr_s, 0.95)),
            "ecapa_latency_mean_s": float(utts["ecapa_seconds"].mean()),
            "audio_mean_s": float(utts["audio_seconds"].mean())}


def _speaker_eer_block(trials: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    sv = trials[trials["scenario"].isin(["genuine", "impostor"])]
    y, s = sv["label"].to_numpy(), sv["score"].to_numpy()
    b = cfg["eval"]["bootstrap"]
    boot = speaker_bootstrap_eer(y, s, sv["claimed"].to_numpy(), int(b["n_resamples"]),
                                 int(cfg["seed"]))[:, 0]
    lo, hi = percentile_ci(boot, float(b["ci_level"]))
    return {"n_target": int(y.sum()), "n_nontarget": int((1 - y).sum()),
            "eer": compute_eer(y, s), "eer_ci": [float(lo), float(hi)],
            "ci_level": float(b["ci_level"])}


# --------------------------------------------------------------------------- runs
def run_cv(cfg: dict[str, Any], max_speakers: int | None, run_id: str | None) -> Path:
    """Common Voice SW synthetic-string experiment with speaker-disjoint dev/eval."""
    n_threads = num_threads_from_config(cfg)
    apply_num_threads(n_threads)
    limit_threads(n_threads)
    seed = int(cfg["seed"])
    set_seed(seed)
    run_dir = make_run_dir(cfg, cfg["challenge_eval"]["results_experiment"], run_id)
    clips = cv.load_digit_clips(cfg)
    plan = cv.build_plan(clips, cfg)
    all_speakers = list(plan["speaker"])
    _, n_excluded = cv.eligible_speakers(clips, cfg)
    speakers = all_speakers
    if max_speakers is not None:
        pick = np.random.default_rng(seed).permutation(len(all_speakers))[:max_speakers]
        speakers = [all_speakers[i] for i in sorted(pick)]
    embedder, asr = EcapaEmbedder(cfg), WhisperASR(cfg)
    asr.warmup()
    utts, embs, templates = process_cv_speakers(cfg, embedder, asr, speakers, clips, plan)
    trials = build_trials(utts, embs, templates, cfg, np.random.default_rng([seed, 2]))

    dev_m, eval_m = speaker_disjoint_split(
        trials["claimed"], trials["audio_speaker"],
        float(cfg["eval"]["dev_eval_split"]["dev_fraction"]), seed)
    trials["split"] = np.where(dev_m, "dev", np.where(eval_m, "eval", "dropped"))
    sv = trials["scenario"].isin(["genuine", "impostor"])
    parts = {n: trials[sv & (trials["split"] == n)] for n in ("dev", "eval")}
    transfer = transfer_threshold_from_config(
        parts["dev"]["label"], parts["dev"]["score"], parts["eval"]["label"],
        parts["eval"]["score"], cfg)
    thr = transfer["threshold"]
    metrics: dict[str, Any] = {
        "corpus": "common_voice_sw_pl_digits", "split_mode": "speaker_disjoint",
        "n_speakers": len(speakers), "n_speakers_excluded_too_few_digits": n_excluded,
        "seed": seed, "device": cfg["device"], "num_threads": n_threads,
        "asr_model": cfg["asr"]["model"], "asr_language": cfg["asr"]["language"],
        "digit_string_length": int(cfg["challenge"]["digit_string_length"]),
        "asr": asr_metrics(utts), "speaker_overall": _speaker_eer_block(trials, cfg),
        "threshold_transfer": transfer,
        "scenarios_dev": scenario_rates(trials[trials["split"] == "dev"], thr),
        "scenarios_eval": scenario_rates(trials[trials["split"] == "eval"], thr),
        "limitation": LIMITATION,
    }
    _save(run_dir, trials, utts, metrics)
    return run_dir


def run_own(
    cfg: dict[str, Any], threshold_from: str, run_id: str | None, max_impostor: int | None
) -> Path:
    """Own-corpus run: threshold from a Common Voice run, CV speakers as impostors."""
    n_threads = num_threads_from_config(cfg)
    apply_num_threads(n_threads)
    limit_threads(n_threads)
    seed = int(cfg["seed"])
    set_seed(seed)
    thr = float(json.load(open(threshold_from, encoding="utf-8"))["threshold_transfer"]["threshold"])
    run_dir = make_run_dir(cfg, cfg["challenge_eval"]["results_experiment"], run_id)
    embedder, asr = EcapaEmbedder(cfg), WhisperASR(cfg)
    asr.warmup()
    own_u, own_e, templates, sessions = process_own(cfg, embedder, asr)

    clips = cv.load_digit_clips(cfg)
    plan = cv.build_plan(clips, cfg)
    n_imp = max_impostor if max_impostor is not None else int(
        cfg["common_voice"]["impostor_speakers_for_own"])
    pick = np.random.default_rng(seed).permutation(len(plan))[:n_imp]
    imp_speakers = [plan["speaker"].iloc[i] for i in sorted(pick)]
    imp_u, imp_e, _ = process_cv_speakers(cfg, embedder, asr, imp_speakers, clips, plan)
    utts = pd.concat([own_u, imp_u], ignore_index=True)
    trials = build_trials(utts, np.vstack([own_e, imp_e]), templates, cfg,
                          np.random.default_rng([seed, 2]))
    trials["split"] = "none"
    metrics: dict[str, Any] = {
        "corpus": "own", "split_mode": "single_speaker" if len(templates) < 3 else "pooled",
        "n_speakers": len(templates), "sessions": sessions, "seed": seed,
        "n_impostor_speakers_common_voice": len(imp_speakers),
        "threshold_source": threshold_from, "threshold": thr,
        "asr": asr_metrics(own_u), "speaker_overall": _speaker_eer_block(trials, cfg),
        "scenarios": scenario_rates(trials, thr),
        "note": ("Single-speaker setup: no population-level claim. Impostors are Common Voice "
                 "synthetic strings (different domain); threshold comes from the Common Voice dev "
                 "split. Own-corpus genuine trials are naturally read digit strings."),
    }
    _save(run_dir, trials, utts, metrics)
    return run_dir


def _save(run_dir: Path, trials: pd.DataFrame, utts: pd.DataFrame, metrics: dict[str, Any]) -> None:
    trials.to_csv(run_dir / "scores.csv", index=False, float_format="%.10g")
    utts.drop(columns=["raw_text"]).to_csv(run_dir / "utterances.csv", index=False,
                                           float_format="%.6g")
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=["cv", "own"], default="cv")
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-speakers", type=int, default=None, help="smoke test: speaker subset")
    ap.add_argument("--threshold-from", default=None, help="own: metrics.json of a cv run")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.corpus == "cv":
        run_cv(cfg, args.max_speakers, args.run_id)
    else:
        if not args.threshold_from:
            ap.error("--corpus own needs --threshold-from")
        run_own(cfg, args.threshold_from, args.run_id, args.max_speakers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
