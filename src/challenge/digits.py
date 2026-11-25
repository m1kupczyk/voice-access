"""Random digit challenge generator."""

from __future__ import annotations

import secrets
from typing import Any

import numpy as np

DIGITS = "0123456789"


def generate_challenge(
    length: int, unique: bool = False, rng: np.random.Generator | None = None
) -> str:
    """Random digit string. Uses `secrets` unless a seeded `rng` is given (experiments)."""
    if unique and length > len(DIGITS):
        raise ValueError("unique digits: length must be <= 10")
    if rng is not None:
        if unique:
            return "".join(DIGITS[i] for i in rng.permutation(len(DIGITS))[:length])
        return "".join(DIGITS[i] for i in rng.integers(0, len(DIGITS), size=length))
    if unique:
        pool = list(DIGITS)
        return "".join(pool.pop(secrets.randbelow(len(pool))) for _ in range(length))
    return "".join(secrets.choice(DIGITS) for _ in range(length))


def challenge_from_config(cfg: dict[str, Any], rng: np.random.Generator | None = None) -> str:
    """Challenge with length and uniqueness from cfg['challenge']."""
    c = cfg["challenge"]
    return generate_challenge(int(c["digit_string_length"]), bool(c["unique_digits"]), rng)


def different_challenge(
    ref: str, length: int, unique: bool, rng: np.random.Generator
) -> str:
    """Seeded random challenge that differs from `ref`."""
    while True:
        c = generate_challenge(length, unique, rng)
        if c != ref:
            return c
