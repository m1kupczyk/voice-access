"""Relabel CLI: label validation, audit row, decision fields untouched."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.db import Database
from src.api.relabel import main


def _db(tmp_path: Path) -> tuple[Path, int]:
    path = tmp_path / "api.db"
    db = Database(path)
    uid = db.create_user("spk01", 0.0)
    aid = db.log_attempt(user_id=uid, created_at=1.0, expected="1259", accepted=0, score=0.235,
                         scenario="wrong_digits", reject_reason="speaker")
    db.close()
    return path, aid


def test_relabel(tmp_path: Path) -> None:
    path, aid = _db(tmp_path)
    assert main(["--attempt-id", str(aid), "--scenario", "impostor", "--db", str(path),
                 "--note", "other person"]) == 0
    db = Database(path)
    r = db.get_attempt(aid)
    audit = db._exec("SELECT * FROM relabels").fetchall()
    db.close()
    assert r["scenario"] == "impostor"
    assert (r["accepted"], r["score"], r["reject_reason"]) == (0, 0.235, "speaker")
    assert len(audit) == 1
    assert (audit[0]["attempt_id"], audit[0]["old_scenario"], audit[0]["new_scenario"],
            audit[0]["note"]) == (aid, "wrong_digits", "impostor", "other person")


def test_relabel_rejects_bad_input(tmp_path: Path) -> None:
    path, aid = _db(tmp_path)
    with pytest.raises(SystemExit):
        main(["--attempt-id", str(aid), "--scenario", "bogus", "--db", str(path)])
    with pytest.raises(SystemExit):
        main(["--attempt-id", "999", "--scenario", "impostor", "--db", str(path)])
    with pytest.raises(SystemExit):
        main(["--attempt-id", str(aid), "--scenario", "impostor",
              "--db", str(tmp_path / "missing.db")])
    db = Database(path)
    assert db.get_attempt(aid)["scenario"] == "wrong_digits"
    assert db.count("attempts") == 1
    db.close()
