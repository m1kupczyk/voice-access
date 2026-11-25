"""Pytest setup: point numba's cache at a writable temp dir before librosa is imported."""

from __future__ import annotations

import os
import tempfile

# librosa's numba-jitted functions need a writable cache locator; sandboxes may lack one.
os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.mkdtemp(prefix="numba-cache-"))
