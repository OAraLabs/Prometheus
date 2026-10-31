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

from prometheus.adapter import markup_guard

log = logging.getLogger(__name__)

DEFAULT_TRAINING_DB = "~/.prometheus/data/training.db"

# Marks a row whose `chosen` side is not fit to train on. Set by
# scripts/flag_corrupt_pairs.py on rows banked before the gate existed, and
# honoured by scripts/export_training_pairs.py. Flagged, never deleted — these
# rows are the only evidence of the upstream leak.
QUALITY_KEY = "quality"
QUALITY_TEMPLATE_MARKUP = "template_markup"


class CorruptPairRejected(ValueError):
    """A pair whose ``chosen`` side carries leaked chat-template markup.

    Raised rather than silently dropped so the rejection is counted: the
    existing ``capture_pair`` handler writes a ``silent_failures`` telemetry
    row, which is what makes the rate measurable instead of assumed.
    """

PAIR_SOURCES = (
    "levenshtein_repair",
    "schema_repair",
    "retry_success",
    "self_correction",
    "malformed_recovery",
    "cloud_golden",
    "gym_harvest",       # series-2: stamped on pairs captured during gym
                         # harvest runs (separable from organic for training-
                         # day weighting). The original repair type is kept in
                         # meta["origin_source"]. Set via configure(source_override).
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

def _trace_label_columns() -> list[tuple[str, str]]:
    from prometheus.telemetry.trace_labels import LABEL_COLUMNS

    return list(LABEL_COLUMNS)


# Columns added after the table first shipped, added on open when missing
# (PRAGMA-guarded ADD COLUMN, as telemetry.db's _EXPECTED_COLUMNS). Nullable,
# never backfilled.
#
# TELEMETRY V2 (WP-X.54 T-1): where a pair came from in telemetry terms.
# None of these is part of `context_hash`: the miner re-runs with no cursor
# and relies on that hash to dedupe, so a field inside it would make a re-run
# write every pair twice (audit Q10). The pair KIND stays `pair_source`.
_EXPECTED_COLUMNS: list[tuple[str, str]] = [
    ("turn_id", "TEXT"),          # telemetry.db turns.turn_id
    ("round_index", "INTEGER"),   # the round the rejected/chosen call came from
    ("repair_kind", "TEXT"),      # the adapter repair, as tool_calls.repair_kind
    # Declared as "the turn's outcome, copied at mine time", but nothing has
    # ever written it: NULL on every row. The turn's end state is now
    # `turn_outcome` below; the user signal stays on telemetry's turns row.
    ("outcome", "TEXT"),
    # TRACE LABELS (telemetry/trace_labels.py), as on tool_calls: `tool_ok`
    # (the chosen call ran without error; NULL where the source banks it
    # before it runs), `turn_outcome` (how its turn ended, stamped at turn
    # end by `label_turn`), `judged_ok` / `judged_by` / `judged_at`. Not in
    # `context_hash`, for the reason above.
    *_trace_label_columns(),
]


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
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add any ``_EXPECTED_COLUMNS`` missing from an older training.db."""
        try:
            existing = {r[1] for r in self._conn.execute("PRAGMA table_info(training_pairs)")}
        except sqlite3.DatabaseError:
            return
        for name, sql in _EXPECTED_COLUMNS:
            if name in existing:
                continue
            try:
                self._conn.execute(f"ALTER TABLE training_pairs ADD COLUMN {name} {sql}")
            except sqlite3.DatabaseError:
                # Read-only, or added by a concurrent opener. Best-effort, as
                # telemetry's migration is; writes then fail loudly instead.
                log.warning("training.db: could not add column %s", name, exc_info=True)

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
        turn_id: str | None = None,
        round_index: int | None = None,
        repair_kind: str | None = None,
        outcome: str | None = None,
        tool_ok: bool | None = None,
    ) -> bool:
        """Insert a pair; returns False on dedupe-hit.

        ``turn_id``, ``round_index``, ``repair_kind``, ``outcome`` and
        ``tool_ok`` are stored beside the pair and are NOT part of the dedupe
        hash (see ``_EXPECTED_COLUMNS``): the same (context, rejected) with
        different telemetry fields is still the same pair. ``tool_ok`` None
        means the chosen call has not run yet (a repair banked before it
        executes), not that it failed.

        Raises on a bad source, and on a ``chosen`` side carrying leaked
        chat-template markup (``CorruptPairRejected``). ``capture_pair`` turns
        either into a ``silent_failures`` row without breaking the turn.
        """
        if pair_source not in PAIR_SOURCES:
            raise ValueError(f"unknown pair_source {pair_source!r}")
        # The `chosen` side is the POSITIVE label — what training points
        # toward. It reached here having passed structure, syntax and type
        # checks (see adapter/markup_guard), none of which inspect content, so
        # a string of decoder artifacts can arrive as a perfectly valid `str`.
        # 10 of 415 banked pairs were exactly that, all self_correction, whose
        # bar for `chosen` is "the retry succeeded" == "pydantic accepted it".
        # Guard the whole store rather than one call site: the gym harness and
        # any future writer use PairStore directly.
        corrupt = markup_guard.scan_arguments((chosen or {}).get("input"))
        if corrupt:
            raise CorruptPairRejected(
                f"refusing to bank {pair_source}/{tool_name}: chosen side "
                f"carries chat-template markup — {markup_guard.describe(corrupt)}"
            )
        # Capture-time redaction: a pair is training data — the one place a
        # retained credential is guaranteed to be copied, exported and read
        # back into a model. Redact every side (context, both calls, meta)
        # before hashing, so the dedupe key is over what is actually stored.
        from prometheus.security.log_redaction import redact_capture
        context = redact_capture(context)
        rejected = redact_capture(rejected)
        chosen = redact_capture(chosen)
        meta = redact_capture(meta)
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
                " rejected, chosen, meta, context_hash, "
                " turn_id, round_index, repair_kind, outcome, tool_ok) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    turn_id,
                    round_index,
                    repair_kind,
                    outcome,
                    None if tool_ok is None else (1 if tool_ok else 0),
                ),
            )
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # duplicate (context, rejected) — by design

    def stats(self) -> dict[str, Any]:
        """Counts for ``/pairs`` and ``/api/pairs``.

        ``total`` is every row, as before. It is NOT a count of pairs:
        ``cloud_golden`` rows are chosen-only copies of cloud tool calls that
        telemetry.db also holds (``tool_calls.is_golden``) and the golden
        export writes out again, so adding ``total`` to either counts the same
        calls twice (2026-10-30: 11,207 here + 11,152 exported read as ~22K).
        ``pairs`` is rows with a rejected side; ``exemplars`` the chosen-only
        rows, with ``exemplars_note`` saying where their canonical count is.
        ``turn_outcome`` counts rows by how their turn ended
        (telemetry/trace_labels.py); none of these is a success rate.
        """
        by_source = dict(self._conn.execute(
            "SELECT pair_source, COUNT(*) FROM training_pairs GROUP BY pair_source"
        ).fetchall())
        pairs = self._conn.execute(
            "SELECT COUNT(*) FROM training_pairs WHERE rejected IS NOT NULL"
        ).fetchone()[0]
        try:
            by_outcome = {
                (k if k is not None else "unlabelled"): n
                for k, n in self._conn.execute(
                    "SELECT turn_outcome, COUNT(*) FROM training_pairs GROUP BY turn_outcome")
            }
        except sqlite3.DatabaseError:
            by_outcome = {}
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
            "pairs": pairs,
            "exemplars": total - pairs,
            "exemplars_note": (
                "chosen-only cloud_golden copies of telemetry golden calls; "
                "count teacher traces in telemetry.db, not here"),
            "by_turn_outcome": by_outcome,
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

    def label_turn(self, turn_id: str, outcome: str) -> int:
        """Stamp ``turn_outcome`` on this turn's pairs that have none. Returns rows changed."""
        cur = self._conn.execute(
            "UPDATE training_pairs SET turn_outcome = ? WHERE turn_id = ? AND turn_outcome IS NULL",
            (outcome, turn_id))
        self._conn.commit()
        return cur.rowcount or 0

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Module singleton + capture API (the loop calls these; never raises)
# ---------------------------------------------------------------------------

_store: PairStore | None = None
_cloud_golden_enabled = False
_source_override: str | None = None


def configure(config: dict[str, Any] | None = None) -> None:
    """Wire the capture singleton. Called once at daemon startup.

    config (the ``training:`` block of prometheus.yaml):
        capture_enabled:      bool, default True  (local pair sources)
        cloud_golden_capture: bool, default False (spec: default OFF)
        db_path:              str, default ~/.prometheus/data/training.db
        source_override:      str | None, default None — when set, every
                              captured pair is stamped with this pair_source
                              (the gym harvest path sets "gym_harvest"); the
                              original repair type is preserved in
                              meta["origin_source"]. Never set by the daemon.
    """
    global _store, _cloud_golden_enabled, _source_override
    cfg = config or {}
    if not cfg.get("capture_enabled", True):
        _store = None
        _cloud_golden_enabled = False
        _source_override = None
        log.info("pair capture disabled by config")
        return
    _store = PairStore(cfg.get("db_path", DEFAULT_TRAINING_DB))
    _cloud_golden_enabled = bool(cfg.get("cloud_golden_capture", False))
    _source_override = cfg.get("source_override") or None
    log.info(
        "pair capture enabled (cloud_golden=%s%s)",
        "on" if _cloud_golden_enabled else "off",
        f", source_override={_source_override}" if _source_override else "",
    )


def get_store() -> PairStore | None:
    return _store


def cloud_golden_enabled() -> bool:
    return _cloud_golden_enabled


def label_turn(turn_id: str, outcome: str) -> None:
    """Turn end: label the turn's pairs with how it ended. Never raises into the turn."""
    if _store is None or not turn_id:
        return
    try:
        _store.label_turn(turn_id, outcome)
    except Exception:
        log.warning("pair capture: could not label turn %s", turn_id, exc_info=True)


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
    repair_kind: str | None = None,
    turn_id: str | None = None,
    round_index: int | None = None,
    tool_ok: bool | None = None,
) -> None:
    """Fail-loud-but-non-blocking capture. Never raises into the turn.

    ``repair_kind`` is the adapter repair's kind (WP-X.54 T-2's vocabulary,
    folded by the loop when a call took several). ``turn_id`` and
    ``round_index`` are the telemetry v2 turn and round the pair was captured
    in, joining it to its ``tool_calls`` and ``responses`` rows. All three
    are stored in their own columns, outside the dedupe hash, as ``add_pair``
    stores them. ``tool_ok`` True when the chosen call has already run without
    error; None when it is banked before it runs.
    """
    if _store is None:
        return
    effective_source = pair_source
    if _source_override:
        # Series-2 harvest provenance: stamp the override as the source so
        # harvest pairs are separable from organic. Preserve the original
        # repair type in meta (the "Replace with gym_harvest" decision keeps
        # the distinction recoverable rather than truly losing it).
        meta = dict(meta or {})
        meta.setdefault("origin_source", pair_source)
        effective_source = _source_override
    try:
        _store.add_pair(
            pair_source=effective_source,
            model_id=model_id,
            tool_name=tool_name,
            context=context,
            rejected=rejected,
            chosen=chosen,
            meta=meta,
            repair_kind=repair_kind,
            turn_id=turn_id,
            round_index=round_index,
            tool_ok=tool_ok,
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
