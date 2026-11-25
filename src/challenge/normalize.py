"""Normalize an ASR transcript (Polish/English number words, digit formats) to a digit string."""

from __future__ import annotations

import re
import unicodedata

_UNITS_PL = {"zero": 0, "jeden": 1, "jedna": 1, "jedno": 1, "dwa": 2, "dwie": 2, "trzy": 3,
             "cztery": 4, "piec": 5, "szesc": 6, "siedem": 7, "osiem": 8, "dziewiec": 9}
_UNITS_EN = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9}
_LARGER = {  # numbers >= 10 are expanded digit by digit (no composition of "twenty three")
    "dziesiec": 10, "jedenascie": 11, "dwanascie": 12, "trzynascie": 13, "czternascie": 14,
    "pietnascie": 15, "szesnascie": 16, "siedemnascie": 17, "osiemnascie": 18,
    "dziewietnascie": 19, "dwadziescia": 20, "trzydziesci": 30, "czterdziesci": 40,
    "piecdziesiat": 50, "szescdziesiat": 60, "siedemdziesiat": 70, "osiemdziesiat": 80,
    "dziewiecdziesiat": 90,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}
WORD_TO_DIGITS: dict[str, str] = {
    w: str(n) for d in (_UNITS_PL, _UNITS_EN, _LARGER) for w, n in d.items()}
_TOKEN = re.compile(r"[0-9]+|[^\W\d_]+")


def fold(text: str) -> str:
    """Lowercase and strip Polish diacritics (ą->a, ł->l, ...)."""
    text = text.lower().replace("ł", "l")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def digit_word_to_digit(word: str) -> str | None:
    """Single-digit word (Polish or English) -> '0'..'9', else None."""
    d = WORD_TO_DIGITS.get(fold(word.strip()))
    return d if d is not None and len(d) == 1 else None


def normalize_digits(text: str) -> str:
    """Transcript -> digit string. Digit runs are kept, number words are mapped, rest dropped."""
    out: list[str] = []
    for tok in _TOKEN.findall(fold(text)):
        if tok.isdigit():
            out.append(tok)
        elif tok in WORD_TO_DIGITS:
            out.append(WORD_TO_DIGITS[tok])
    return "".join(out)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two sequences."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
