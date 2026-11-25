"""Tests for thread-count configuration."""

from __future__ import annotations

import os

import pytest

from src.threads import (
    THREAD_ENV_VARS, apply_num_threads, num_threads_from_config, peek_num_threads,
)


def test_default_and_config_value() -> None:
    assert num_threads_from_config({}) == 1
    assert num_threads_from_config({"runtime": {"num_threads": 3}}) == 3


def test_apply_sets_all_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    for v in THREAD_ENV_VARS:
        monkeypatch.setenv(v, "8")
    apply_num_threads(1)
    assert all(os.environ[v] == "1" for v in THREAD_ENV_VARS)
    with pytest.raises(ValueError):
        apply_num_threads(0)


def test_peek_reads_yaml(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("runtime:\n  num_threads: 2\n")
    assert peek_num_threads("unused.yaml", ["--config", str(p)]) == 2
