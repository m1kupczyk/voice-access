"""Tests for evaluation metrics and plots on synthetic scores with known values."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.metrics import roc_curve

from src.eval.metrics import (
    compute_eer,
    det_curve,
    far_frr_at_threshold,
    min_dcf,
    percentile_ci,
    select_threshold,
    speaker_bootstrap_eer,
    speaker_disjoint_split,
    threshold_at_eer,
    threshold_at_far,
    transfer_threshold,
)
from src.eval.plots import Curve, plot_det, plot_roc

SEED = 0
N_GAUSS = 200_000   # trials per class for asymptotic checks
GAUSS_TOL = 0.005   # sampling tolerance for N_GAUSS trials


def _gauss(mu: float, n: int = N_GAUSS, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Targets ~ N(mu, 1), non-targets ~ N(0, 1)."""
    rng = np.random.default_rng(seed)
    scores = np.r_[rng.normal(mu, 1.0, n), rng.normal(0.0, 1.0, n)]
    return np.r_[np.ones(n, int), np.zeros(n, int)], scores


def _sklearn_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def test_hand_computed_example() -> None:
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.4, 0.7, 0.3, 0.2])
    assert compute_eer(labels, scores) == pytest.approx(1 / 3)
    assert threshold_at_eer(labels, scores) == pytest.approx(0.7)
    assert far_frr_at_threshold(labels, scores, 0.7) == pytest.approx((1 / 3, 1 / 3))
    assert threshold_at_far(labels, scores, 0.0) == pytest.approx(0.8)


def test_perfect_reversed_and_identical() -> None:
    labels = np.r_[np.ones(50, int), np.zeros(50, int)]
    sep = np.r_[np.linspace(1, 2, 50), np.linspace(-2, -1, 50)]
    assert compute_eer(labels, sep) == 0.0
    assert compute_eer(labels, -sep) == 1.0
    assert min_dcf(labels, sep, 0.01, 1.0, 1.0)[0] == 0.0
    assert far_frr_at_threshold(labels, sep, threshold_at_far(labels, sep, 0.0)) == (0.0, 0.0)
    same = np.zeros(100)
    assert compute_eer(labels, same) == 0.5
    assert min_dcf(labels, same, 0.01, 1.0, 1.0)[0] == pytest.approx(1.0)
    assert min_dcf(labels, same, 0.7, 1.0, 1.0)[0] == pytest.approx(1.0)


def test_curve_endpoints() -> None:
    labels, scores = _gauss(1.0, n=100)
    thr, far, frr = det_curve(labels, scores)
    assert thr[0] == np.inf and (far[0], frr[0]) == (0.0, 1.0)
    assert (far[-1], frr[-1]) == (1.0, 0.0)
    assert np.all(np.diff(far) >= 0) and np.all(np.diff(frr) <= 0)


@pytest.mark.parametrize("seed", range(5))
def test_eer_matches_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 2000
    labels = rng.integers(0, 2, n)
    scores = np.round(rng.normal(labels * 1.5, 1.0), 1)  # rounding creates ties
    assert compute_eer(labels, scores) == pytest.approx(_sklearn_eer(labels, scores), abs=1e-12)
    fpr, tpr, thr_sk = roc_curve(labels, scores, drop_intermediate=False)
    thr, far, frr = det_curve(labels, scores)
    np.testing.assert_allclose(far, fpr, atol=1e-12)
    np.testing.assert_allclose(1.0 - frr, tpr, atol=1e-12)
    np.testing.assert_allclose(thr[1:], thr_sk[1:])


@pytest.mark.parametrize("mu", [1.0, 2.0, 3.0])
def test_gaussian_eer_and_rates(mu: float) -> None:
    labels, scores = _gauss(mu)
    assert compute_eer(labels, scores) == pytest.approx(norm.cdf(-mu / 2), abs=GAUSS_TOL)
    t = mu / 3
    far, frr = far_frr_at_threshold(labels, scores, t)
    assert far == pytest.approx(norm.sf(t), abs=GAUSS_TOL)
    assert frr == pytest.approx(norm.cdf(t - mu), abs=GAUSS_TOL)


def test_gaussian_identical_distributions() -> None:
    labels, scores = _gauss(0.0)
    assert compute_eer(labels, scores) == pytest.approx(0.5, abs=GAUSS_TOL)


def test_gaussian_min_dcf() -> None:
    mu, p, cm, cf = 2.0, 0.01, 1.0, 1.0
    labels, scores = _gauss(mu)
    res = minimize_scalar(lambda t: cm * p * norm.cdf(t - mu) + cf * (1 - p) * norm.sf(t),
                          bounds=(-5, 10), method="bounded")
    expected = res.fun / min(cm * p, cf * (1 - p))
    got, thr = min_dcf(labels, scores, p, cm, cf)
    assert got == pytest.approx(expected, abs=0.01)
    # Analytic optimum: t* = (ln((1-p) cf / (p cm)) + mu^2 / 2) / mu.
    assert thr == pytest.approx((np.log((1 - p) * cf / (p * cm)) + mu ** 2 / 2) / mu, abs=0.2)


def test_far_target_threshold() -> None:
    labels, scores = _gauss(2.0)
    for target in (0.001, 0.01, 0.1):
        far, frr = far_frr_at_threshold(labels, scores, threshold_at_far(labels, scores, target))
        assert far <= target
        assert far == pytest.approx(target, abs=1 / N_GAUSS)
        assert frr == pytest.approx(norm.cdf(norm.isf(target) - 2.0), abs=GAUSS_TOL)
    with pytest.raises(ValueError):
        select_threshold(labels, scores, "bogus")


def test_weights_equal_duplication() -> None:
    rng = np.random.default_rng(SEED)
    labels = rng.integers(0, 2, 300)
    scores = rng.normal(labels, 1.0)
    w = rng.integers(0, 4, 300)
    dup = compute_eer(np.repeat(labels, w), np.repeat(scores, w))
    assert compute_eer(labels, scores, weights=w) == pytest.approx(dup, abs=1e-12)


def _synthetic_trials(n_spk: int = 40, per_spk: int = 200, mu: float = 2.0, seed: int = SEED):
    """Trials with speaker ids: target = same speaker, scores ~ N(mu,1) vs N(0,1)."""
    rng = np.random.default_rng(seed)
    enroll = np.repeat(np.arange(n_spk), per_spk)
    is_t = rng.random(enroll.size) < 0.5
    other = (enroll + rng.integers(1, n_spk, enroll.size)) % n_spk
    test = np.where(is_t, enroll, other)
    labels = is_t.astype(int)
    scores = rng.normal(mu * labels, 1.0)
    return labels, scores, enroll, test


def test_speaker_disjoint_split() -> None:
    _, _, enroll, test = _synthetic_trials()
    dev, ev = speaker_disjoint_split(enroll, test, 0.5, SEED)
    dev_spk = set(enroll[dev]) | set(test[dev])
    ev_spk = set(enroll[ev]) | set(test[ev])
    assert dev_spk.isdisjoint(ev_spk) and len(dev_spk) == len(ev_spk) == 20
    assert not (dev & ev).any()
    dev2, _ = speaker_disjoint_split(enroll, test, 0.5, SEED)
    assert np.array_equal(dev, dev2)
    with pytest.raises(ValueError):
        speaker_disjoint_split(enroll, test, 0.0, SEED)


def test_threshold_transfer_synthetic() -> None:
    labels, scores, enroll, test = _synthetic_trials(per_spk=2000)
    dev, ev = speaker_disjoint_split(enroll, test, 0.5, SEED)
    r = transfer_threshold(labels[dev], scores[dev], labels[ev], scores[ev], "far_target", 0.05)
    assert r["dev_far"] <= 0.05
    assert r["eval_far"] == pytest.approx(0.05, abs=0.01)
    assert r["eval_frr"] == pytest.approx(norm.cdf(norm.isf(0.05) - 2.0), abs=0.02)
    r = transfer_threshold(labels[dev], scores[dev], labels[ev], scores[ev], "eer")
    assert r["far_target"] is None
    assert r["dev_far"] == pytest.approx(r["dev_frr"], abs=0.005)
    assert r["eval_far"] == pytest.approx(norm.cdf(-1.0), abs=0.02)
    assert r["eval_frr"] == pytest.approx(norm.cdf(-1.0), abs=0.02)


def test_speaker_bootstrap() -> None:
    labels, scores, enroll, _ = _synthetic_trials()
    a = speaker_bootstrap_eer(labels, scores, enroll, 200, SEED)
    b = speaker_bootstrap_eer(labels, scores, enroll, 200, SEED)
    assert a.shape == (200, 1) and np.array_equal(a, b)
    lo, hi = percentile_ci(a[:, 0], 0.95)
    assert lo < compute_eer(labels, scores) < hi
    assert lo < norm.cdf(-1.0) < hi
    paired = speaker_bootstrap_eer(labels, np.vstack([scores, scores]), enroll, 50, SEED)
    assert np.array_equal(paired[:, 0], paired[:, 1])
    sep = np.where(labels == 1, 1.0, -1.0)
    assert np.all(speaker_bootstrap_eer(labels, sep, enroll, 20, SEED) == 0.0)


def test_plots_written(tmp_path) -> None:
    plot_cfg = {"dpi": 72, "formats": ["png", "pdf"], "figsize_in": [3, 3], "font_size": 8,
                "line_width": 1.0, "det_limits": [0.001, 0.5], "det_ticks": [0.01, 0.1]}
    labels, scores = _gauss(2.0, n=500)
    _, far, frr = det_curve(labels, scores)
    curves = [Curve(far=far, frr=frr, label="sys", eer=compute_eer(labels, scores))]
    paths = plot_det(curves, tmp_path / "det", plot_cfg) + plot_roc(curves, tmp_path / "roc",
                                                                    plot_cfg)
    assert sorted(p.name for p in paths) == ["det.pdf", "det.png", "roc.pdf", "roc.png"]
    assert all(p.stat().st_size > 0 for p in paths)
