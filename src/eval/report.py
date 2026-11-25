"""Stage 3 report: EER, minDCF and speaker-level bootstrap CIs per system on VoxCeleb1-O,
plus joint DET and ROC figures. Runs are listed in config.yaml (eval.report.systems).

Usage:
    python -m src.eval.report [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from src.threads import (
    apply_num_threads,
    limit_threads,
    num_threads_from_config,
    peek_num_threads,
)

if __name__ == "__main__":
    from src.config import DEFAULT_CONFIG_PATH  # yaml only, no numpy

    apply_num_threads(peek_num_threads(DEFAULT_CONFIG_PATH))

import numpy as np
import pandas as pd

from src.config import load_config
from src.eval.metrics import (
    compute_eer,
    det_curve,
    min_dcf_from_config,
    percentile_ci,
    speaker_bootstrap_eer,
)
from src.eval.plots import LANGUAGES, Curve, plot_det, plot_roc, plt
from src.repro import make_run_dir

TRIAL_KEYS = ["label", "enroll", "test"]
EER_TOLERANCE = 1e-9  # recomputed vs stored metrics.json EER


def speaker_of(utt: str) -> str:
    """VoxCeleb utterance ids look like idXXXXX/<video>/<n>.wav."""
    return utt.split("/", 1)[0]


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    trials = pd.read_csv(run_dir / "scores.csv")
    with open(run_dir / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    return trials, metrics


def _colors(n: int) -> list[str]:
    """First n colors of the default matplotlib cycle."""
    return plt.rcParams["axes.prop_cycle"].by_key()["color"][:n]


def _runs_word(n: int, language: str) -> str:
    """'runs' in the figure legend; Polish genitive/nominative plural by count."""
    if language != "pl":
        return "runs"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return "przebiegi"
    return "przebiegów"


def _fmt_mean_std(mean: float, std: float, scale: float, dec: int) -> str:
    if np.isnan(std):
        return f"{scale * mean:.{dec}f}"
    return f"{scale * mean:.{dec}f} \u00b1 {scale * std:.{dec}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing report dir")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    limit_threads(num_threads_from_config(cfg))
    ev = cfg["eval"]
    rep, boot, dcf_p = ev["report"], ev["bootstrap"], ev["min_dcf"]
    seed = int(cfg["seed"])
    language = str(rep.get("language", "pl"))
    if language not in LANGUAGES:
        raise ValueError(f"report.language must be one of {LANGUAGES}, got {language!r}")

    out = Path(cfg["paths"]["results_dir"]) / rep["experiment"] / rep["run_id"]
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out = make_run_dir(cfg, rep["experiment"], rep["run_id"])

    ref: pd.DataFrame | None = None
    run_rows, sum_rows, curves_by_sys = [], [], {}
    for system in rep["systems"]:
        name, run_dirs = system["name"], [Path(p) for p in system["runs"]]
        S, run_metrics = [], []
        for rd in run_dirs:
            trials, m = load_run(rd)
            if ref is None:
                ref = trials[TRIAL_KEYS].copy()
            elif not trials[TRIAL_KEYS].equals(ref):
                raise ValueError(f"{rd}: trial list differs from the first run")
            S.append(trials["score"].to_numpy(np.float64))
            run_metrics.append(m)
        S = np.vstack(S)
        labels = ref["label"].to_numpy()
        clusters = ref["enroll"].map(speaker_of).to_numpy()

        samples = speaker_bootstrap_eer(labels, S, clusters, int(boot["n_resamples"]), seed)
        run_lo, run_hi = percentile_ci(samples, float(boot["ci_level"]))
        sys_lo, sys_hi = percentile_ci(samples.mean(axis=1), float(boot["ci_level"]))

        eers, dcfs, curves = [], [], []
        for j, (rd, m) in enumerate(zip(run_dirs, run_metrics)):
            eer = compute_eer(labels, S[j])
            if m.get("eer") is not None and abs(eer - m["eer"]) > EER_TOLERANCE:
                raise ValueError(f"{rd}: EER {eer} differs from metrics.json {m['eer']}")
            dcf, dcf_thr = min_dcf_from_config(labels, S[j], cfg)
            eers.append(eer)
            dcfs.append(dcf)
            run_rows.append({
                "system": name, "run": str(rd), "seed": m.get("seed"),
                "n_trials": len(labels), "eer": eer,
                "eer_ci_lo": float(run_lo[j]), "eer_ci_hi": float(run_hi[j]),
                "min_dcf": dcf, "min_dcf_threshold": dcf_thr,
            })
            _, far, frr = det_curve(labels, S[j])
            curves.append((far, frr, eer))
        ddof = 1 if len(eers) > 1 else 0
        sum_rows.append({
            "system": name, "n_runs": len(eers),
            "eer_mean": float(np.mean(eers)),
            "eer_std": float(np.std(eers, ddof=ddof)) if ddof else float("nan"),
            "eer_ci_lo": float(sys_lo), "eer_ci_hi": float(sys_hi),
            "min_dcf_mean": float(np.mean(dcfs)),
            "min_dcf_std": float(np.std(dcfs, ddof=ddof)) if ddof else float("nan"),
        })
        curves_by_sys[name] = curves

    runs_df, sum_df = pd.DataFrame(run_rows), pd.DataFrame(sum_rows)
    runs_df.to_csv(out / "sv_runs.csv", index=False, float_format="%.10g")
    sum_df.to_csv(out / "sv_summary.csv", index=False, float_format="%.10g")

    pd_, dd = int(rep["pct_decimals"]), int(rep["dcf_decimals"])
    ci_pct = f"{100 * float(boot['ci_level']):g}"
    lines = [
        f"| System | Runs | EER (%) | {ci_pct}% CI of EER (%) | minDCF |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in sum_rows:
        lines.append(
            f"| {r['system']} | {r['n_runs']} "
            f"| {_fmt_mean_std(r['eer_mean'], r['eer_std'], 100, pd_)} "
            f"| [{100 * r['eer_ci_lo']:.{pd_}f}, {100 * r['eer_ci_hi']:.{pd_}f}] "
            f"| {_fmt_mean_std(r['min_dcf_mean'], r['min_dcf_std'], 1, dd)} |"
        )
    n_spk = len(np.unique(ref["enroll"].map(speaker_of)))
    lines += [
        "",
        (f"VoxCeleb1-O (cleaned), {len(ref)} trials, {n_spk} enrollment speakers. "
        "Multi-run systems: mean \u00b1 sample std (ddof=1) over runs."),
        (f"minDCF: normalized, p_target={dcf_p['p_target']}, c_miss={dcf_p['c_miss']}, "
        f"c_fa={dcf_p['c_fa']}."),
        (f"CI: speaker-level percentile bootstrap, {boot['n_resamples']} resamples of "
        f"enrollment speakers, seed {seed}; for multi-run systems the CI is of the "
        "run-mean EER under shared resamples."),
    ]
    (out / "sv_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    colors = _colors(len(curves_by_sys))
    curves: list[Curve] = []
    for color, (name, cs) in zip(colors, curves_by_sys.items()):
        r = next(r for r in sum_rows if r["system"] == name)
        eer_txt = _fmt_mean_std(r["eer_mean"], r["eer_std"], 100, pd_)
        if language == "pl":
            eer_txt = eer_txt.replace(".", ",")
        runs_txt = f", {len(cs)} {_runs_word(len(cs), language)}" if len(cs) > 1 else ""
        for k, (far, frr, eer) in enumerate(cs):
            curves.append(Curve(
                far=far, frr=frr, color=color, eer=eer,
                alpha=1.0 if len(cs) == 1 else 0.6,
                label=f"{name} (EER {eer_txt}%{runs_txt})" if k == 0 else None,
            ))
    stem = out / rep["figure_stem"]
    figs = plot_det(curves, stem.with_name(f"det_{stem.name}"), ev["plots"], language)
    figs += plot_roc(curves, stem.with_name(f"roc_{stem.name}"), ev["plots"], language)

    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump({
            "systems": rep["systems"], "seed": seed, "bootstrap": boot, "min_dcf": dcf_p,
            "n_trials": len(ref), "n_enroll_speakers": n_spk,
            "figures": [p.name for p in figs],
        }, f, indent=2)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
