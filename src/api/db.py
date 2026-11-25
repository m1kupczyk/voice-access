"""SQLite schema and queries. Stores embeddings and attempt logs only, never audio."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    model_name  TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS challenges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    nonce       TEXT NOT NULL UNIQUE,
    digits      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL,
    used_at     REAL
);
CREATE TABLE IF NOT EXISTS attempts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    challenge_id INTEGER REFERENCES challenges(id) ON DELETE CASCADE,
    created_at   REAL NOT NULL,
    expected     TEXT,
    asr_text     TEXT,
    recognized   TEXT,
    digits_ok    INTEGER,
    score        REAL,
    threshold    REAL,
    score_ok     INTEGER,
    cm_score     REAL,
    cm_ok        INTEGER,
    accepted     INTEGER NOT NULL,
    reason       TEXT,
    latency_s    REAL,
    scenario     TEXT,
    reject_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_embeddings_user ON embeddings(user_id, model_name);
CREATE INDEX IF NOT EXISTS idx_attempts_user ON attempts(user_id);
"""

# Columns added after the first schema version; ALTER TABLE adds them to existing databases.
MIGRATIONS: dict[str, dict[str, str]] = {
    "attempts": {"scenario": "TEXT", "reject_reason": "TEXT"},
}


@dataclass(frozen=True)
class ChallengeRow:
    id: int
    user_id: int
    nonce: str
    digits: str
    created_at: float
    expires_at: float
    used_at: float | None


def vector_to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype="<f4").tobytes()


def blob_to_vector(b: bytes, dim: int) -> np.ndarray:
    v = np.frombuffer(b, dtype="<f4")
    if v.size != dim:
        raise ValueError(f"embedding blob has {v.size} values, expected {dim}")
    return v.astype(np.float32)


class Database:
    """Thread-safe wrapper around one SQLite connection (FastAPI runs sync routes in a pool)."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        for table, cols in MIGRATIONS.items():
            have = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")}
            for col, decl in cols.items():
                if col not in have:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _exec(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    # users
    def get_user_by_name(self, name: str) -> sqlite3.Row | None:
        return self._exec("SELECT * FROM users WHERE name = ?", (name,)).fetchone()

    def get_user(self, user_id: int) -> sqlite3.Row | None:
        return self._exec("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

    def create_user(self, name: str, now: float) -> int:
        cur = self._exec("INSERT INTO users (name, created_at) VALUES (?, ?)", (name, now))
        return int(cur.lastrowid)

    def delete_user(self, user_id: int) -> dict[str, int]:
        """Delete a user and (via ON DELETE CASCADE) all dependent rows; return deleted counts."""
        with self._lock:
            c = self._conn
            counts = {t: int(c.execute(f"SELECT COUNT(*) FROM {t} WHERE user_id = ?",
                                       (user_id,)).fetchone()[0])
                      for t in ("embeddings", "challenges", "attempts")}
            counts["users"] = c.execute("DELETE FROM users WHERE id = ?", (user_id,)).rowcount
        return counts

    # embeddings
    def add_embedding(self, user_id: int, model_name: str, vector: np.ndarray, now: float) -> int:
        v = np.asarray(vector, dtype=np.float32).ravel()
        cur = self._exec(
            "INSERT INTO embeddings (user_id, model_name, dim, vector, created_at) "
            "VALUES (?, ?, ?, ?, ?)", (user_id, model_name, int(v.size), vector_to_blob(v), now))
        return int(cur.lastrowid)

    def get_embeddings(self, user_id: int, model_name: str) -> list[np.ndarray]:
        rows = self._exec(
            "SELECT dim, vector FROM embeddings WHERE user_id = ? AND model_name = ? ORDER BY id",
            (user_id, model_name)).fetchall()
        return [blob_to_vector(r["vector"], int(r["dim"])) for r in rows]

    # challenges
    def create_challenge(
        self, user_id: int, nonce: str, digits: str, now: float, expires_at: float
    ) -> int:
        cur = self._exec(
            "INSERT INTO challenges (user_id, nonce, digits, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?)", (user_id, nonce, digits, now, expires_at))
        return int(cur.lastrowid)

    def get_challenge(self, nonce: str) -> ChallengeRow | None:
        r = self._exec("SELECT * FROM challenges WHERE nonce = ?", (nonce,)).fetchone()
        return None if r is None else ChallengeRow(
            int(r["id"]), int(r["user_id"]), r["nonce"], r["digits"], float(r["created_at"]),
            float(r["expires_at"]), None if r["used_at"] is None else float(r["used_at"]))

    def consume_challenge(self, nonce: str, now: float) -> bool:
        """Atomically mark an unused challenge as used; False if it was already used."""
        cur = self._exec(
            "UPDATE challenges SET used_at = ? WHERE nonce = ? AND used_at IS NULL", (now, nonce))
        return cur.rowcount == 1

    # attempts
    def log_attempt(self, **fields: Any) -> int:
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        cur = self._exec(f"INSERT INTO attempts ({cols}) VALUES ({marks})", tuple(fields.values()))
        return int(cur.lastrowid)

    def list_attempts(self, user_id: int | None = None) -> list[sqlite3.Row]:
        if user_id is None:
            return self._exec("SELECT * FROM attempts ORDER BY id").fetchall()
        return self._exec("SELECT * FROM attempts WHERE user_id = ? ORDER BY id",
                          (user_id,)).fetchall()

    def count(self, table: str) -> int:
        if table not in ("users", "embeddings", "challenges", "attempts"):
            raise ValueError(table)
        return int(self._exec(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
