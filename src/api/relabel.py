"""Correct the operator scenario label of one logged attempt (live statistics only).

The decision fields (accepted, scores, reject_reason) are never touched; every change is
logged and recorded in the `relabels` table with the old and new label.

Run: python -m src.api.relabel --attempt-id N --scenario impostor [--note TEXT] [--db PATH]
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from src.api.db import Database
from src.config import load_config

log = logging.getLogger("relabel")


def main(argv: Sequence[str] | None = None) -> int:
    cfg = load_config()
    scenarios = list(cfg["api"]["scenarios"])
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--attempt-id", type=int, required=True)
    p.add_argument("--scenario", required=True, choices=scenarios)
    p.add_argument("--note", default=None, help="reason for the correction (stored in relabels)")
    p.add_argument("--db", default=cfg["api"]["db_path"], help="SQLite path (default: api.db_path)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if not Path(args.db).is_file():
        raise SystemExit(f"database not found: {args.db}")
    db = Database(args.db)
    try:
        row = db.get_attempt(args.attempt_id)
        if row is None:
            raise SystemExit(f"attempt {args.attempt_id} not found")
        if row["scenario"] == args.scenario:
            log.info("attempt %d already labelled %r, nothing to do", args.attempt_id, args.scenario)
            return 0
        old = db.relabel_attempt(args.attempt_id, args.scenario, time.time(), args.note)
    finally:
        db.close()
    log.info("attempt %d (user %d, expected %s, score %s): scenario %r -> %r",
             args.attempt_id, row["user_id"], row["expected"], row["score"], old, args.scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
