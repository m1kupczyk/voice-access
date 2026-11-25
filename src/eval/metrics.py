"""Verification metrics: FAR/FRR, EER, minDCF, threshold selection, dev-to-eval
threshold transfer and speaker-level bootstrap confidence intervals.

Conventions: labels 1 = target, 0 = non-target; higher score = more target;
a trial is accepted if score >= threshold.
"""

from __future__ import annotations

from typing import Any

import numpy as np

ArrayLike = np.ndarray | list | tuple


def _prepare(
    labels: ArrayLike, scores: ArrayLike, weights: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(labels).astype(int).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError("labels and scores must have the same length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 (non-target) or 1 (target)")
    w = np.ones_like(s) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != s.shape or (w < 0).any():
        raise ValueError("weights must be non-negative and match scores")
    return y, s, w


def _det_sorted(
    s: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Operating points for scores already sorted in decreasing order."""
    wt = w * y
    tp = np.cumsum(wt)
    fp = np.cumsum(w - wt)
    n_t, n_n = tp[-1], fp[-1]
    if n_t <= 0 or n_n <= 0:
        raise ValueError("metrics need both target and non-target trials")
    # Only evaluate at the last index of each run of tied scores.
    last = np.r_[s[1:] != s[:-1], True]
    thr = np.r_[np.inf, s[last]]
    far = np.r_[0.0, fp[last] / n_n]
    frr = np.r_[1.0, 1.0 - tp[last] / n_t]
    return thr, far, frr


def det_curve(
    labels: ArrayLike, scores: ArrayLike, weights: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (thresholds, FAR, FRR) at every distinct threshold.

    Thresholds decrease from +inf (reject all: FAR 0, FRR 1) to the minimum score
    (accept all: FAR 1, FRR 0). Optional per-trial weights (e.g. bootstrap
    multiplicities) replace trial counts.
    """
    y, s, w = _prepare(labels, scores, weights)
    order = np.argsort(-s, kind="mergesort")
    return _det_sorted(s[order], y[order], w[order])


def roc_curve_points(
    labels: ArrayLike, scores: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (FAR, true acceptance rate = 1 - FRR, thresholds)."""
    thr, far, frr = det_curve(labels, scores)
    return far, 1.0 - frr, thr


def _eer_index(far: np.ndarray, frr: np.ndarray) -> tuple[float, int]:
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), i


def compute_eer(
    labels: ArrayLike, scores: ArrayLike, weights: ArrayLike | None = None
) -> float:
    """Equal error rate: mean of FAR and FRR at the operating point where they are closest."""
    _, far, frr = det_curve(labels, scores, weights)
    return _eer_index(far, frr)[0]


def far_frr_at_threshold(
    labels: ArrayLike, scores: ArrayLike, threshold: float
) -> tuple[float, float]:
    """FAR and FRR when accepting trials with score >= threshold."""
    y, s, _ = _prepare(labels, scores)
    if not ((y == 1).any() and (y == 0).any()):
        raise ValueError("metrics need both target and non-target trials")
    accept = s >= threshold
    return float(accept[y == 0].mean()), float(1.0 - accept[y == 1].mean())


def min_dcf(
    labels: ArrayLike, scores: ArrayLike, p_target: float, c_miss: float, c_fa: float
) -> tuple[float, float]:
    """Normalized minimum detection cost and the threshold that attains it.

    DCF(t) = c_miss * p_target * FRR(t) + c_fa * (1 - p_target) * FAR(t), divided by
    min(c_miss * p_target, c_fa * (1 - p_target)), the cost of the best trivial system.
    """
    if not 0.0 < p_target < 1.0:
        raise ValueError("p_target must be in (0, 1)")
    thr, far, frr = det_curve(labels, scores)
    dcf = c_miss * p_target * frr + c_fa * (1.0 - p_target) * far
    i = int(np.argmin(dcf))
    norm = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return float(dcf[i] / norm), float(thr[i])


def min_dcf_from_config(
    labels: ArrayLike, scores: ArrayLike, cfg: dict[str, Any]
) -> tuple[float, float]:
    """`min_dcf` with cost parameters from cfg['eval']['min_dcf']."""
    p = cfg["eval"]["min_dcf"]
    return min_dcf(labels, scores, float(p["p_target"]), float(p["c_miss"]), float(p["c_fa"]))


def threshold_at_eer(labels: ArrayLike, scores: ArrayLike) -> float:
    """Threshold of the EER operating point."""
    thr, far, frr = det_curve(labels, scores)
    return float(thr[_eer_index(far, frr)[1]])


def threshold_at_far(labels: ArrayLike, scores: ArrayLike, far_target: float) -> float:
    """Lowest threshold with FAR <= far_target (minimum FRR under the FAR constraint)."""
    if not 0.0 <= far_target <= 1.0:
        raise ValueError("far_target must be in [0, 1]")
    thr, far, _ = det_curve(labels, scores)
    return float(thr[np.flatnonzero(far <= far_target)[-1]])


def select_threshold(
    labels: ArrayLike, scores: ArrayLike, method: str, far_target: float | None = None
) -> float:
    """Select a decision threshold: method 'eer' or 'far_target'."""
    if method == "eer":
        return threshold_at_eer(labels, scores)
    if method == "far_target":
        if far_target is None:
            raise ValueError("method 'far_target' needs far_target")
        return threshold_at_far(labels, scores, far_target)
    raise ValueError(f"unknown threshold method '{method}'")


def speaker_disjoint_split(
    enroll_speakers: ArrayLike, test_speakers: ArrayLike, dev_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Random speaker-disjoint split of trials; returns boolean masks (dev, eval).

    Speakers are assigned to dev or eval; a trial belongs to a split only if both its
    enrollment and test speakers do. Trials crossing the split are dropped.
    """
    enroll = np.asarray(enroll_speakers)
    test = np.asarray(test_speakers)
    speakers = np.unique(np.r_[enroll, test])
    n_dev = round(dev_fraction * len(speakers))
    if not 0 < n_dev < len(speakers):
        raise ValueError("dev_fraction leaves dev or eval without speakers")
    dev_spk = np.random.default_rng(seed).permutation(speakers)[:n_dev]
    e_dev, t_dev = np.isin(enroll, dev_spk), np.isin(test, dev_spk)
    return e_dev & t_dev, ~e_dev & ~t_dev


def transfer_threshold(
    dev_labels: ArrayLike, dev_scores: ArrayLike,
    eval_labels: ArrayLike, eval_scores: ArrayLike,
    method: str, far_target: float | None = None,
) -> dict[str, Any]:
    """Select the threshold on dev and report the resulting FAR/FRR on dev and eval."""
    thr = select_threshold(dev_labels, dev_scores, method, far_target)
    dev_far, dev_frr = far_frr_at_threshold(dev_labels, dev_scores, thr)
    eval_far, eval_frr = far_frr_at_threshold(eval_labels, eval_scores, thr)
    return {
        "method": method, "far_target": far_target if method == "far_target" else None,
        "threshold": thr,
        "n_dev": int(np.size(dev_labels)), "n_eval": int(np.size(eval_labels)),
        "dev_far": dev_far, "dev_frr": dev_frr, "dev_eer": compute_eer(dev_labels, dev_scores),
        "eval_far": eval_far, "eval_frr": eval_frr,
        "eval_eer": compute_eer(eval_labels, eval_scores),
    }


def transfer_threshold_from_config(
    dev_labels: ArrayLike, dev_scores: ArrayLike,
    eval_labels: ArrayLike, eval_scores: ArrayLike, cfg: dict[str, Any],
) -> dict[str, Any]:
    """`transfer_threshold` with method and FAR target from cfg['eval']['threshold']."""
    t = cfg["eval"]["threshold"]
    return transfer_threshold(dev_labels, dev_scores, eval_labels, eval_scores,
                              t["method"], float(t["far_target"]))


def speaker_bootstrap_eer(
    labels: ArrayLike, scores: ArrayLike, clusters: ArrayLike, n_resamples: int, seed: int
) -> np.ndarray:
    """Speaker-level (cluster) bootstrap samples of the EER.

    Clusters (e.g. enrollment speakers) are drawn with replacement; every trial is
    weighted by the multiplicity of its cluster. `scores` is (n_trials,) or
    (n_systems, n_trials); all systems share the same resamples (paired bootstrap).
    Returns an array of shape (n_resamples, n_systems).
    """
    y = np.asarray(labels).astype(int).ravel()
    S = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    if S.shape[1] != y.size:
        raise ValueError("scores must have n_trials columns")
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    if inv.size != y.size:
        raise ValueError("clusters must have one entry per trial")
    k = int(inv.max()) + 1
    sorted_sys = []
    for s in S:
        order = np.argsort(-s, kind="mergesort")
        sorted_sys.append((s[order], y[order], inv[order]))
    rng = np.random.default_rng(seed)
    out = np.empty((n_resamples, len(S)))
    for b in range(n_resamples):
        counts = np.bincount(rng.integers(0, k, size=k), minlength=k).astype(np.float64)
        for j, (s, ys, cs) in enumerate(sorted_sys):
            _, far, frr = _det_sorted(s, ys, counts[cs])
            out[b, j] = _eer_index(far, frr)[0]
    return out


def percentile_ci(samples: np.ndarray, ci_level: float) -> tuple[Any, Any]:
    """Two-sided percentile confidence interval along axis 0."""
    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(np.asarray(samples), [alpha, 1.0 - alpha], axis=0)
    return lo, hi
