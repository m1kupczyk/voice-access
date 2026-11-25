"""Publication-ready DET (probit axes) and ROC plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

_PROB_EPS = np.finfo(np.float64).eps  # keeps probit finite at FAR/FRR = 0 or 1


@dataclass
class Curve:
    """One operating-point curve. Set `label=None` to hide it from the legend."""

    far: np.ndarray
    frr: np.ndarray
    label: str | None
    color: str | None = None
    linestyle: str = "-"
    alpha: float = 1.0
    eer: float | None = None  # if set, the EER point is marked


def probit(p: np.ndarray | float) -> np.ndarray:
    """Standard normal quantile, clipped to stay finite."""
    return norm.ppf(np.clip(p, _PROB_EPS, 1.0 - _PROB_EPS))


def _rc(plot_cfg: dict[str, Any]) -> dict[str, Any]:
    fs = plot_cfg["font_size"]
    return {
        "font.size": fs, "axes.labelsize": fs, "legend.fontsize": fs,
        "xtick.labelsize": fs, "ytick.labelsize": fs,
        "lines.linewidth": plot_cfg["line_width"], "pdf.fonttype": 42, "ps.fonttype": 42,
    }


def save_figure(fig: plt.Figure, stem: Path, plot_cfg: dict[str, Any]) -> list[Path]:
    """Save `fig` as <stem>.<fmt> for every configured format; returns the paths."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in plot_cfg["formats"]:
        p = stem.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=plot_cfg["dpi"], bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


LANGUAGES = ("pl", "en")

# Figure labels per language.
_LABELS: dict[str, dict[str, str | None]] = {
    "en": {
        "far_det": "False acceptance rate, FAR (%)",
        "frr_det": "False rejection rate, FRR (%)",
        "eer_line": "EER line",
        "far_roc": "False acceptance rate, FAR",
        "tar_roc": "True acceptance rate, 1 \u2212 FRR",
        "chance": "Chance",
    },
    "pl": {
        "far_det": "Współczynnik fałszywych akceptacji, FAR (%)",
        "frr_det": "Współczynnik fałszywych odrzuceń, FRR (%)",
        "eer_line": "linia EER",
        "far_roc": "Współczynnik fałszywych akceptacji, FAR",
        "tar_roc": "Współczynnik prawdziwych akceptacji, 1 \u2212 FRR",
        "chance": "losowo",
    },
}


def _labels(language: str) -> dict[str, str | None]:
    if language not in _LABELS:
        raise ValueError(f"report.language must be one of {LANGUAGES}, got {language!r}")
    return _LABELS[language]


def format_number(x: float, language: str) -> str:
    """Format a number for display; pl uses a decimal comma."""
    s = f"{x:g}"
    return s.replace(".", ",") if language == "pl" else s


def _pct(p: float, language: str = "en") -> str:
    return format_number(100 * p, language)


def plot_det(curves: list[Curve], stem: Path, plot_cfg: dict[str, Any],
             language: str = "en") -> list[Path]:
    """DET plot (FRR vs FAR on normal-deviate axes); returns saved file paths."""
    lab = _labels(language)
    lo, hi = plot_cfg["det_limits"]
    ticks = [t for t in plot_cfg["det_ticks"] if lo <= t <= hi]
    with plt.rc_context(_rc(plot_cfg)):
        fig, ax = plt.subplots(figsize=plot_cfg["figsize_in"])
        for c in curves:
            line, = ax.plot(probit(c.far), probit(c.frr), color=c.color, ls=c.linestyle,
                            alpha=c.alpha, label=c.label)
            if c.eer is not None:
                ax.plot(probit(c.eer), probit(c.eer), "o", color=line.get_color(),
                        alpha=c.alpha, ms=3)
        ax.plot(probit([lo, hi]), probit([lo, hi]), color="0.6", ls=":", lw=0.8,
                label=lab["eer_line"])
        ax.set_xlim(probit(lo), probit(hi))
        ax.set_ylim(probit(lo), probit(hi))
        ax.set_xticks(probit(ticks), [_pct(t, language) for t in ticks])
        ax.set_yticks(probit(ticks), [_pct(t, language) for t in ticks])
        ax.set_xlabel(lab["far_det"])
        ax.set_ylabel(lab["frr_det"])
        ax.set_aspect("equal")
        ax.grid(True, color="0.85", lw=0.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
        return save_figure(fig, stem, plot_cfg)


def plot_roc(curves: list[Curve], stem: Path, plot_cfg: dict[str, Any],
             language: str = "en") -> list[Path]:
    """ROC plot (1 - FRR vs FAR, linear axes); returns saved file paths."""
    lab = _labels(language)
    with plt.rc_context(_rc(plot_cfg)):
        fig, ax = plt.subplots(figsize=plot_cfg["figsize_in"])
        for c in curves:
            ax.plot(c.far, 1.0 - c.frr, color=c.color, ls=c.linestyle, alpha=c.alpha,
                    label=c.label)
        ax.plot([0, 1], [0, 1], color="0.6", ls=":", lw=0.8, label=lab["chance"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(lab["far_roc"])
        ax.set_ylabel(lab["tar_roc"])
        ax.set_aspect("equal")
        ax.grid(True, color="0.85", lw=0.5)
        ax.legend(loc="lower right", frameon=True)
        return save_figure(fig, stem, plot_cfg)
