"""Repair-pair flywheel — every repair/retry/self-correction becomes a training pair.

Storage decision: a dedicated ``~/.prometheus/data/training.db``, NOT
telemetry.db. Telemetry is the hot, dashboard-read path this sprint just made
honest; training data has a different lifecycle (accumulate → export → prune
→ eventually LoRA), different growth profile, and later joins gym/teacher
data. Keeping them apart means a pair-capture bug can never contend with or
distort live observability.

Capture contract (fail-loud-but-non-blocking): a capture failure logs an
error AND writes a ``silent_failures`` telemetry row when a tracker is
available, and never breaks the turn.

Pair sources (DPO/KTO-shaped rows):
  levenshtein_repair — adapter fuzzy-renamed the tool; rejected=as-emitted,
                       chosen=repaired call. (Reachable at tier light since
                       the invariants-vs-policy split, 511ed2f.)
  schema_repair      — adapter repaired args (coercion/strip/json-extract).
  retry_success      — a call failed validation/execution, the model retried
                       the SAME tool within the turn and succeeded.
  self_correction    — same shape, captured from execution-level failures
                       (input_validation / mode-misuse tool errors).
  malformed_recovery — provider dropped empty envelopes, loop fed guidance
                       back, the model then produced a working call.
  cloud_golden       — config-gated, DEFAULT OFF: tool calls from cloud-
                       routed turns stored as chosen-only exemplars
                       (rejected NULL; paired later against local attempts).
                       Never generates cloud traffic — harvests only what
                       the owner already routes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_TRAINING_DB = "~/.prometheus/data/training.db"

PAIR_SOURCES = (
    "levenshtein_repair",
    "schema_repair",
    "retry_success",
    "self_correction",
    "malformed_recovery",
    "cloud_golden",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS training_pairs (
    id            TEXT PRIMARY KEY,
    timestamp     REAL NOT NULL,
    pair_source   TEXT NOT NULL,
    model_id      TEXT NOT NULL,
    tool_name     TEXT NOT NULL,
    context       TEXT,              -- JSON (see context kinds below)
    rejected      TEXT,              -- JSON {"name":..., "input":...}; NULL for cloud_golden
    chosen        TEXT NOT NULL,     -- JSON {"name":..., "input":...}
    meta          TEXT,              -- JSON: error feedback, repair log, latency
    context_hash  TEXT NOT NULL UNIQUE  -- sha256(context + rejected) — dedupe
);
CREATE INDEX IF NOT EXISTS idx_pairs_source ON training_pairs (pair_source);
CREATE INDEX IF NOT EXISTS idx_pairs_tool ON training_pairs (tool_name);
CREATE INDEX IF NOT EXISTS idx_pairs_ts ON training_pairs (timestamp);
"""


def _call_json(name: str, input_: Any) -> str:
    try:
        return json.dumps({"name": name, "input": input_}, default=str, sort_keys=True)
    except Exception:
        return json.dumps({"name": name, "input": repr(input_)})


class PairStore:
    """SQLite store for training pairs."""

    def __init__(self, db_path: str | Path = DEFAULT_TRAINING_DB) -> None:
        p = Path(db_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(p), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add_pair(
        self,
        *,
        pair_source: str,
        model_id: str,
        tool_name: str,
        context: dict[str, Any] | None,
        rejected: dict[str, Any] | None,   # {"name":..., "input":...}
        chosen: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> bool:
        """Insert a pair; returns False on dedupe-hit. Raises on bad source."""
        if pair_source not in PAIR_SOURCES:
            raise ValueError(f"unknown pair_source {pair_source!r}")
        context_json = json.dumps(context or {}, default=str, sort_keys=True)
        rejected_json = (
            _call_json(rejected["name"], rejected.get("input", {}))
            if rejected is not None else None
        )
        chosen_json = _call_json(chosen["name"], chosen.get("input", {}))
        h = hashlib.sha256(
            (context_json + "\x1e" + (rejected_json or chosen_json)).encode()
        ).hexdigest()
        try:
            self._conn.execute(
                "INSERT INTO training_pairs "
                "(id, timestamp, pair_source, model_id, tool_name, context, "
                " rejected, chosen, meta, context_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    hashlib.sha256(f"{h}{time.time()}".encode()).hexdigest()[:32],
                    time.time(),
                    pair_source,
                    model_id,
                    tool_name,
                    context_json,
                    rejected_json,
                    chosen_json,
                    json.dumps(meta or {}, default=str),
                    h,
                ),
            )
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # duplicate (context, rejected) — by design

    def stats(self) -> dict[str, Any]:
        by_source = dict(self._conn.execute(
            "SELECT pair_source, COUNT(*) FROM training_pairs GROUP BY pair_source"
        ).fetchall())
        by_tool = dict(self._conn.execute(
            "SELECT tool_name, COUNT(*) FROM training_pairs "
            "GROUP BY tool_name ORDER BY COUNT(*) DESC LIMIT 15"
        ).fetchall())
        total = self._conn.execute(
            "SELECT COUNT(*) FROM training_pairs"
        ).fetchone()[0]
        week_ago = time.time() - 7 * 86400
        last_7d = self._conn.execute(
            "SELECT COUNT(*) FROM training_pairs WHERE timestamp >= ?",
            (week_ago,),
        ).fetchone()[0]
        return {
            "total": total,
            "by_source": by_source,
            "by_tool": by_tool,
            "last_7d": last_7d,
            "per_day_7d": round(last_7d / 7.0, 2),
        }

    def rows_since(self, since: float | None = None) -> list[sqlite3.Row]:
        if since is None:
            return self._conn.execute(
                "SELECT * FROM training_pairs ORDER BY timestamp"
            ).fetchall()
        return self._conn.execute(
            "SELECT * FROM training_pairs WHERE timestamp >= ? ORDER BY timestamp",
            (since,),
        ).fetchall()

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Module singleton + capture API (the loop calls these; never raises)
# ---------------------------------------------------------------------------

_store: PairStore | None = None
_cloud_golden_enabled = False


def configure(config: dict[str, Any] | None = None) -> None:
    """Wire the capture singleton. Called once at daemon startup.

    config (the ``training:`` block of prometheus.yaml):
        capture_enabled:      bool, default True  (local pair sources)
        cloud_golden_capture: bool, default False (spec: default OFF)
        db_path:              str, default ~/.prometheus/data/training.db
    """
    global _store, _cloud_golden_enabled
    cfg = config or {}
    if not cfg.get("capture_enabled", True):
        _store = None
        _cloud_golden_enabled = False
        log.info("pair capture disabled by config")
        return
    _store = PairStore(cfg.get("db_path", DEFAULT_TRAINING_DB))
    _cloud_golden_enabled = bool(cfg.get("cloud_golden_capture", False))
    log.info(
        "pair capture enabled (cloud_golden=%s)",
        "on" if _cloud_golden_enabled else "off",
    )


def get_store() -> PairStore | None:
    return _store


def cloud_golden_enabled() -> bool:
    return _cloud_golden_enabled


def capture_pair(
    *,
    pair_source: str,
    model_id: str,
    tool_name: str,
    context: dict[str, Any] | None,
    rejected: dict[str, Any] | None,
    chosen: dict[str, Any],
    meta: dict[str, Any] | None = None,
    telemetry: Any = None,
) -> None:
    """Fail-loud-but-non-blocking capture. Never raises into the turn."""
    if _store is None:
        return
    try:
        _store.add_pair(
            pair_source=pair_source,
            model_id=model_id,
            tool_name=tool_name,
            context=context,
            rejected=rejected,
            chosen=chosen,
            meta=meta,
        )
    except Exception as exc:
        log.error("pair capture failed (%s/%s): %s", pair_source, tool_name, exc,
                  exc_info=True)
        try:
            if telemetry is not None and hasattr(telemetry, "record_silent_failure"):
                telemetry.record_silent_failure(
                    subsystem="pair_capture",
                    operation=pair_source,
                    exc=exc,
                    context={"tool_name": tool_name, "model_id": model_id},
                )
        except Exception:
            log.exception("pair capture: silent-failure recording also failed")
