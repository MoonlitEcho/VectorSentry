"""Durable run history in SQLite: what was planned, what shipped, and the
original vector of every document a run changed, so any run can be rolled
back exactly. One file, no server -- swap for Postgres when several workers
share it."""

from __future__ import annotations

import datetime
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started TEXT NOT NULL,
    finished TEXT,
    collection TEXT,
    mode TEXT NOT NULL,              -- scan | dry-run | apply | rollback
    status TEXT NOT NULL,            -- running | done | failed | rolled-back
    n_documents INTEGER,
    n_flagged INTEGER,
    budget REAL,
    cost_planned REAL,
    embed_cost REAL,
    n_full INTEGER,
    n_patch INTEGER,
    n_rolled_back INTEGER,
    health_before REAL,
    health_after REAL,
    protected_share REAL,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS actions (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    doc_id TEXT NOT NULL,
    action TEXT NOT NULL,            -- full_reembed | patch | defer
    p_broken REAL,
    traffic REAL,
    rollout_rank INTEGER,
    applied INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS snapshots (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    doc_id TEXT NOT NULL,
    vector BLOB NOT NULL,
    dim INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS actions_run ON actions(run_id);
CREATE INDEX IF NOT EXISTS snapshots_run ON snapshots(run_id);
"""

_RUN_FIELDS = (
    "collection", "mode", "status", "n_documents", "n_flagged", "budget", "cost_planned", "embed_cost", "n_full",
    "n_patch", "n_rolled_back", "health_before", "health_after", "protected_share", "notes", "finished",
)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class StateStore:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def start_run(self, mode: str, collection: str = "", **fields) -> int:
        cur = self.conn.execute("INSERT INTO runs (started, mode, status, collection) VALUES (?, ?, 'running', ?)",
                                (_now(), mode, collection))
        assert cur.lastrowid is not None  # always set after an INSERT
        run_id = int(cur.lastrowid)
        if fields:
            self.update_run(run_id, **fields)
        self.conn.commit()
        return run_id

    def update_run(self, run_id: int, **fields) -> None:
        unknown = set(fields) - set(_RUN_FIELDS)
        if unknown:
            raise ValueError(f"unknown run fields: {sorted(unknown)}")
        if fields:
            assignments = ", ".join(f"{k} = ?" for k in fields)
            self.conn.execute(f"UPDATE runs SET {assignments} WHERE id = ?", (*fields.values(), run_id))
            self.conn.commit()

    def finish_run(self, run_id: int, status: str = "done", **fields) -> None:
        self.update_run(run_id, status=status, finished=_now(), **fields)

    def record_actions(self, run_id: int, rows: Sequence[dict]) -> None:
        self.conn.executemany(
            "INSERT INTO actions (run_id, doc_id, action, p_broken, traffic, rollout_rank, applied) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(run_id, r["doc_id"], r["action"], r.get("p_broken"), r.get("traffic"), r.get("rollout_rank"),
              int(bool(r.get("applied")))) for r in rows],
        )
        self.conn.commit()

    def save_snapshot(self, run_id: int, doc_ids: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        self.conn.executemany(
            "INSERT INTO snapshots (run_id, doc_id, vector, dim) VALUES (?, ?, ?, ?)",
            [(run_id, d, v.tobytes(), v.shape[0]) for d, v in zip(doc_ids, vectors)],
        )
        self.conn.commit()

    def load_snapshot(self, run_id: int) -> tuple[list[str], np.ndarray]:
        rows = self.conn.execute("SELECT doc_id, vector, dim FROM snapshots WHERE run_id = ?", (run_id,)).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        return [r["doc_id"] for r in rows], np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])

    def run(self, run_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def runs(self, limit: int = 20) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))]

    def actions(self, run_id: int) -> list[dict]:
        # Planned fixes in rollout order first; deferred documents (no rank) last.
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM actions WHERE run_id = ? ORDER BY rollout_rank IS NULL, rollout_rank, traffic DESC", (run_id,))]
