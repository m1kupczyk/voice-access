"""Thread-count control for deterministic numerics. Must not import numpy/sklearn.

Call `apply_num_threads` before numpy is imported so BLAS/OpenMP libraries pick
up the limits; `limit_threads` additionally caps already-loaded pools.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
)
DEFAULT_NUM_THREADS = 1


def num_threads_from_config(cfg: dict[str, Any]) -> int:
    """Return runtime.num_threads from a config dict (default 1)."""
    return int((cfg.get("runtime") or {}).get("num_threads", DEFAULT_NUM_THREADS))


def peek_num_threads(config_path: str | Path, argv: list[str] | None = None) -> int:
    """Read runtime.num_threads straight from YAML (no numpy); honours --config in argv."""
    argv = sys.argv[1:] if argv is None else argv
    path = config_path
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            path = argv[i + 1]
        elif a.startswith("--config="):
            path = a.split("=", 1)[1]
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return num_threads_from_config(cfg if isinstance(cfg, dict) else {})


def apply_num_threads(n: int) -> None:
    """Force thread env vars to `n` (overrides any inherited values)."""
    if n < 1:
        raise ValueError("runtime.num_threads must be >= 1")
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(n)


def limit_threads(n: int) -> bool:
    """Cap already-loaded BLAS/OpenMP pools via threadpoolctl. Returns True if applied."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return False
    threadpool_limits(limits=n)
    return True
