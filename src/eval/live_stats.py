"""Live attempt statistics from the API database.

An acceptance test of the running prototype, not a replacement for the offline evaluation:
scenario labels are set by the operator and the attempts are not a controlled trial list.

Run: python -m src.eval.live_stats --out results/live/<run-id>/ [--user-id N] [--db PATH]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.challenge.pipeline import REJECT_REASONS

UNLABELED = "unlabeled"     # attempts logged before the scenario column existed
OTHER_REASON = "other"      # rejected without a category (reused / foreign challenge, no enrollment)

CSV_COLUMNS = (
    "id", "user_id", "challenge_id", "created_at", "created_at_iso", "scenario", "expected",
    "asr_text", "recognized", "digits_ok", "score", "threshold", "score_ok", "cm_score", "cm_ok",
    "accepted", "decision", "reject_reason", "reason", "latency_s",
)


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def _median(xs: list[float]) -> float | None:
    return float(np.median(xs)) if xs else None


def scenario_stats(scenario: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts, accept rate, speaker score mean/median, mean latency and rejection reasons."""
    scores = [float(r["score"]) for r in rows if r["score"] is not None]
    latencies = [float(r["latency_s"]) for r in rows if r["latency_s"] is not None]
    n_acc = sum(int(r["accepted"]) for r in rows)
    reasons = dict.fromkeys((*REJECT_REASONS, OTHER_REASON), 0)
    for r in rows:
        if not r["accepted"]:
            key = r["reject_reason"] or OTHER_REASON
            reasons[key] = reasons.get(key, 0) + 1
    return {
        "scenario": scenario, "n_attempts": len(rows), "n_accepted": n_acc,
        "accept_rate": n_acc / len(rows) if rows else None, "n_scored": len(scores),
        "score_mean": _mean(scores), "score_median": _median(scores),
        "latency_mean_s": _mean(latencies), "reject_reasons": reasons,
    }


def aggregate(rows: Iterable[Mapping[str, Any]], scenarios: Sequence[str]) -> dict[str, Any]:
    """Per-scenario statistics; configured scenarios always appear, unknown labels are appended."""
    groups: dict[str, list[Mapping[str, Any]]] = {s: [] for s in scenarios}
    for r in rows:
        groups.setdefault(r["scenario"] or UNLABELED, []).append(r)
    per = [scenario_stats(s, rs) for s, rs in groups.items()]
    return {"n_attempts": sum(p["n_attempts"] for p in per), "scenarios": per}


def attempts_csv(rows: Iterable[Mapping[str, Any]]) -> str:
    """Attempt metadata, scores, ASR text and decision as CSV (the DB never holds audio)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        d = dict(r)
        d["created_at_iso"] = datetime.fromtimestamp(float(d["created_at"]), UTC).isoformat()
        d["decision"] = "accepted" if d["accepted"] else "rejected"
        w.writerow(d)
    return buf.getvalue()


def main(argv: Sequence[str] | None = None) -> None:
    from src.api.db import Database
    from src.config import load_config
    from src.repro import git_commit_hash

    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, type=Path, help="output dir, e.g. results/live/<run-id>/")
    p.add_argument("--db", default=cfg["api"]["db_path"], help="SQLite path (default: api.db_path)")
    p.add_argument("--user-id", type=int, default=None, help="only attempts of this user")
    args = p.parse_args(argv)

    if not Path(args.db).is_file():
        raise SystemExit(f"database not found: {args.db}")
    db = Database(args.db)
    try:
        rows = db.list_attempts(args.user_id)
    finally:
        db.close()

    args.out.mkdir(parents=True, exist_ok=False)
    stats = {
        "kind": "live_acceptance_test",
        "note": "operator-labelled live attempts; not a replacement for the offline evaluation",
        "generated_at": datetime.now(UTC).isoformat(), "db_path": str(args.db),
        "user_id": args.user_id, "git_commit": git_commit_hash(),
        **aggregate(rows, cfg["api"]["scenarios"]),
    }
    with open(args.out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    with open(args.out / "attempts.csv", "w", encoding="utf-8", newline="") as f:
        f.write(attempts_csv(rows))
    print(f"{stats['n_attempts']} attempts -> {args.out}")


if __name__ == "__main__":
    main()
