"""Reproducibility helpers: seeding, run directory, git hash, config snapshot."""

from __future__ import annotations

import json
import os
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def git_commit_hash(repo: Path = REPO_ROOT) -> str:
    """Return the current git commit hash, or 'unknown' if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def make_run_dir(cfg: dict[str, Any], experiment: str, run_id: str | None = None) -> Path:
    """Create results/<experiment>/<run_id>/ and write config snapshot + run_info.json.

    run_id defaults to a UTC timestamp. The snapshot holds the config used,
    the seed and the git commit hash.
    """
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    run_dir = Path(cfg["paths"]["results_dir"]) / experiment / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    info = {"experiment": experiment, "run_id": run_id, "seed": cfg["seed"],
            "num_threads": int((cfg.get("runtime") or {}).get("num_threads", 1)),
            "git_commit": git_commit_hash()}
    with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return run_dir
