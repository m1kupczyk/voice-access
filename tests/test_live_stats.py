"""Live attempt statistics: reason categories, aggregation, CSV export, DB migration, CLI."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from src.api.db import Database
from src.challenge.pipeline import REJECT_REASONS, rejection_reason
from src.eval.live_stats import (
    CSV_COLUMNS,
    OTHER_REASON,
    UNLABELED,
    aggregate,
    attempts_csv,
    main,
)

SCENARIOS = ["genuine", "wrong_digits", "impostor", "replay"]


def row(scenario: str | None, accepted: bool, score: float | None = None,
        latency: float | None = None, reason: str | None = None, **kw: Any) -> dict[str, Any]:
    return {"scenario": scenario, "accepted": int(accepted), "score": score,
            "latency_s": latency, "reject_reason": reason, **kw}


@pytest.mark.parametrize("digits_ok,score_ok,cm_ok,expected", [
    (True, True, True, None),
    (True, False, True, "speaker"),
    (False, True, True, "digits"),
    (False, False, True, "both"),
    (True, True, False, "replay_cm"),
    (False, False, False, "replay_cm"),
])
def test_rejection_reason(digits_ok: bool, score_ok: bool, cm_ok: bool,
                          expected: str | None) -> None:
    assert rejection_reason(digits_ok, score_ok, cm_ok) == expected
    assert expected is None or expected in REJECT_REASONS


def test_aggregate() -> None:
    rows = [
        row("genuine", True, 0.8, 1.0),
        row("genuine", True, 0.6, 2.0),
        row("genuine", False, 0.1, 3.0, "speaker"),
        row("genuine", False, None, None, "expired"),
        row("impostor", False, 0.2, 1.0, "speaker"),
        row("impostor", False, 0.3, 2.0, "both"),
        row("wrong_digits", False, 0.9, 4.0, "digits"),
        row(None, False, None, None, None),
    ]
    out = aggregate(rows, SCENARIOS)
    assert out["n_attempts"] == len(rows)
    by = {s["scenario"]: s for s in out["scenarios"]}
    assert list(by) == [*SCENARIOS, UNLABELED]

    g = by["genuine"]
    assert (g["n_attempts"], g["n_accepted"], g["n_scored"]) == (4, 2, 3)
    assert g["accept_rate"] == pytest.approx(0.5)
    assert g["score_mean"] == pytest.approx(0.5)
    assert g["score_median"] == pytest.approx(0.6)
    assert g["latency_mean_s"] == pytest.approx(2.0)
    assert g["reject_reasons"] == {"speaker": 1, "digits": 0, "both": 0, "expired": 1,
                                   "replay_cm": 0, OTHER_REASON: 0}

    imp = by["impostor"]
    assert imp["accept_rate"] == 0.0 and imp["score_median"] == pytest.approx(0.25)
    assert imp["reject_reasons"]["speaker"] == 1 and imp["reject_reasons"]["both"] == 1

    r = by["replay"]
    assert r["n_attempts"] == 0 and r["accept_rate"] is None and r["score_mean"] is None
    assert by[UNLABELED]["reject_reasons"][OTHER_REASON] == 1


def test_attempts_csv() -> None:
    rows = [row("genuine", True, 0.7, 1.5, None, id=1, user_id=1, created_at=0.0,
                asr_text="jeden dwa", vector=b"\0")]
    text = attempts_csv(rows)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert tuple(parsed[0]) == CSV_COLUMNS
    assert parsed[0]["decision"] == "accepted" and parsed[0]["scenario"] == "genuine"
    assert parsed[0]["created_at_iso"].startswith("1970-01-01T00:00:00")
    assert "vector" not in parsed[0] and not any("audio" in c for c in CSV_COLUMNS)


OLD_ATTEMPTS = """
CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE,
                    created_at REAL NOT NULL);
CREATE TABLE attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
    challenge_id INTEGER, created_at REAL NOT NULL, expected TEXT, asr_text TEXT,
    recognized TEXT, digits_ok INTEGER, score REAL, threshold REAL, score_ok INTEGER,
    cm_score REAL, cm_ok INTEGER, accepted INTEGER NOT NULL, reason TEXT, latency_s REAL);
INSERT INTO users (name, created_at) VALUES ('spk01', 0);
INSERT INTO attempts (user_id, created_at, accepted, score) VALUES (1, 0, 1, 0.9);
"""


def test_migrates_existing_db(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as c:
        c.executescript(OLD_ATTEMPTS)
    for _ in range(2):  # idempotent
        db = Database(path)
        rows = db.list_attempts()
        db.close()
    assert len(rows) == 1 and rows[0]["score"] == 0.9
    assert rows[0]["scenario"] is None and rows[0]["reject_reason"] is None
    by = {s["scenario"]: s for s in aggregate(rows, SCENARIOS)["scenarios"]}
    assert by[UNLABELED]["n_accepted"] == 1


def test_cli(tmp_path: Path) -> None:
    path = tmp_path / "api.db"
    db = Database(path)
    uid = db.create_user("spk01", 0.0)
    db.log_attempt(user_id=uid, created_at=1.0, accepted=0, score=0.1, latency_s=1.0,
                   scenario="impostor", reject_reason="speaker", asr_text="jeden")
    db.close()
    out = tmp_path / "live" / "run1"
    main(["--out", str(out), "--db", str(path)])
    stats = json.loads((out / "stats.json").read_text())
    assert stats["kind"] == "live_acceptance_test" and stats["n_attempts"] == 1
    imp = next(s for s in stats["scenarios"] if s["scenario"] == "impostor")
    assert imp["reject_reasons"]["speaker"] == 1
    lines = (out / "attempts.csv").read_text().splitlines()
    assert len(lines) == 2 and "impostor" in lines[1]
    with pytest.raises(FileExistsError):
        main(["--out", str(out), "--db", str(path)])
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "x"), "--db", str(tmp_path / "missing.db")])
