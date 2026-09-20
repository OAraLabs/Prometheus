"""The candidate-table corpus — captured evidence, and human judgments about it.

WHERE IT LANDS, AND WHY NOT ``telemetry.db``
---------------------------------------------
A dedicated ``<data dir>/computer_corpus.db``, resolved once by
``config.paths.get_computer_corpus_db_path`` and pinned by
``tests/test_computer_corpus_path_resolution.py``.

Not ``telemetry.db``. Measured on the operator box, that file is 58 MB with
28 080 ``tool_calls`` rows and 46 616 ``subsystem_runs`` rows, WAL, written by
the whole daemon while it runs, and pruned by age. A corpus is small, cold,
permanent and hand-annotated — every one of those is the opposite. The repo has
already made this exact call twice and written down why both times:
``gym/store.py`` keeps gym runs out of ``telemetry.db`` because they contain
induced failures that would distort live observability, and
``learning/pair_capture.py`` gives training pairs their own ``training.db``.
A harvested chooser corpus is the same kind of thing.

WHY A DATABASE AND NOT A JSONL FILE
------------------------------------
The first draft of this module was JSONL, on the reasoning that a human has to
hand-annotate it and a text file is the friendlier thing to edit. Two facts
overturned that:

* ``jobs/db_snapshot.discover_databases`` walks the config dir and snapshots
  **every** ``*.db`` it finds, with no registration step. A ``.db`` gets nightly
  ``VACUUM INTO`` backups for free. **There is no rotation or backup primitive
  for JSONL anywhere in this codebase.** A corpus costs human hours to annotate;
  an artifact that expensive with no backup is a bad artifact.
* Table size is the column the harvest spec reports on, and it has to be
  queryable. Blobbed inside a JSON line it is not.

The annotation ergonomics argument survives, but it was an argument about the
*interface*, not the *storage* — and it turned out not to favour JSONL anyway:
this repo has no post-hoc human-annotation path of any kind, so one has to be
built regardless of format, and a small CLI over SQLite is no harder to write
than one over a text file.

TWO TABLES, TWO OWNERS
-----------------------
``tables``       machine-written, append-only, never updated.
``annotations``  human-written, keyed by ``record_id``, append-only.

Separate because they have different owners and different mutability.
Re-harvesting must never clobber a judgment, and revising a judgment must never
touch the evidence it was made about. Appending a second annotation for the same
record supersedes the first on read, so a revision is visible rather than
destructive — the same discipline as the vault's immutable ``raw/`` beside a
human-only ``notes/``.

THE FOUR ANNOTATION STATES
---------------------------
``correct_candidate_id`` must distinguish four things, and collapsing any two of
them corrupts every score computed afterwards:

============================  =============================================
state                          meaning
============================  =============================================
*no annotation row at all*     nobody has reviewed this yet. THE DEFAULT.
a candidate id                 that candidate was the right action
``none_correct``               the table was fine; no entry in it was right,
                               so ``abstain`` was the correct answer
``table_unusable``             the observation should not have produced a
                               table; excluded from scoring entirely
============================  =============================================

The dangerous collapse is *unannotated* into ``none_correct``. Absence of a row
means "unknown"; if it were stored as "abstain was right", a chooser that always
abstains would score 100% on an unreviewed corpus. So unannotated is represented
by ABSENCE — ``record_annotation`` refuses a null answer, which means there is no
way to write the state down at all. That is the point.

``table_unusable`` is separate from ``none_correct`` because they score
differently: on ``none_correct`` an abstaining chooser is RIGHT, while a
``table_unusable`` row is a fault upstream of the chooser and must be dropped
rather than counted as a loss.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prometheus.permissions.audit import AuditLogger

log = logging.getLogger(__name__)

#: Bumped when a stored row's shape changes incompatibly. A reader that meets a
#: version it does not know REFUSES the row rather than guessing — ``training.db``
#: has no version column at all and that is a shortfall, not a precedent.
CORPUS_SCHEMA_VERSION = 1

#: The table was usable and no candidate in it was right — ``abstain`` was the
#: correct answer. A chooser that abstained on this row scores a WIN.
ANNOTATION_NONE_CORRECT = "none_correct"

#: The observation should never have produced a table. The fault is upstream of
#: the chooser, so this row is EXCLUDED from scoring rather than counted lost.
ANNOTATION_TABLE_UNUSABLE = "table_unusable"

SENTINEL_ANNOTATIONS = frozenset(
    {ANNOTATION_NONE_CORRECT, ANNOTATION_TABLE_UNUSABLE}
)

#: Descriptions come from accessibility labels scraped off a live desktop, so
#: they are bounded as well as redacted. Larger than the approval prompt's 160
#: (a corpus row is read by a scoring harness, not squeezed into a chat message)
#: but not unbounded — a document-text node can carry a whole paragraph.
MAX_DESCRIPTION_CHARS = 300
MAX_GOAL_CHARS = 300
MAX_REASON_CHARS = 300

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Machine-written. Append-only: nothing in this module ever UPDATEs a row here.
CREATE TABLE IF NOT EXISTS tables (
    record_id             TEXT PRIMARY KEY,
    captured_at           REAL NOT NULL,
    schema_version        INTEGER NOT NULL,
    goal                  TEXT NOT NULL,
    target                TEXT NOT NULL,
    app                   TEXT NOT NULL,
    window_id             INTEGER NOT NULL,
    pid                   INTEGER NOT NULL DEFAULT 0,
    -- PROVENANCE. tests/fixtures/divergence_traces.py states the rule this
    -- implements: "a calibration round that cannot tell the two apart is
    -- calibrating against its own author." A FixtureDriver row and a real Cua
    -- row must never be indistinguishable, or the chooser is being scored
    -- against tables this repo's own test fixtures invented.
    driver_kind           TEXT NOT NULL DEFAULT 'unknown',
    -- "human" | "derived" | "unknown". docs/computer-use-corpus.md measures
    -- that a description-derived goal makes 93% of rows trivial, so goal
    -- provenance IS the experiment and must be on the row, not inferred.
    goal_source           TEXT NOT NULL DEFAULT 'unknown',
    harvest_session       TEXT NOT NULL DEFAULT '',
    -- sha over the table itself, for spotting a re-captured identical window.
    table_fingerprint     TEXT NOT NULL DEFAULT '',
    -- sha over the constants that DEFINE what a table is. Change the role sets
    -- and old rows silently mean something different.
    code_fingerprint      TEXT NOT NULL DEFAULT '',
    snapshot_id           TEXT,
    unusable_reason       TEXT,
    -- The chooser's view only: [{"id": ..., "description": ...}]. Arguments
    -- never enter the corpus; see TableRecord.note_candidates.
    candidates_json       TEXT NOT NULL,
    -- Its own column, not derivable from the blob, because the harvest spec
    -- reports on table size and a scoring harness filters on it.
    candidate_count       INTEGER NOT NULL,
    chooser_answer_id     TEXT,
    chooser_source        TEXT NOT NULL DEFAULT '',
    chooser_confidence    REAL,
    executed_candidate_id TEXT,
    status                TEXT NOT NULL,
    reason                TEXT NOT NULL DEFAULT '',
    verified              INTEGER,
    extent                TEXT NOT NULL DEFAULT '',
    exception             TEXT
);
CREATE INDEX IF NOT EXISTS idx_tables_app ON tables (app);
CREATE INDEX IF NOT EXISTS idx_tables_count ON tables (candidate_count);
CREATE INDEX IF NOT EXISTS idx_tables_driver ON tables (driver_kind);

-- Human-written. Also append-only: a revision is a NEW row with a later
-- annotated_at, and the latest wins on read. Destructive edits would erase the
-- fact that someone changed their mind.
CREATE TABLE IF NOT EXISTS annotations (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id             TEXT NOT NULL,
    correct_candidate_id  TEXT NOT NULL,
    annotated_by          TEXT NOT NULL,
    annotated_at          REAL NOT NULL,
    note                  TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_annotations_record ON annotations (record_id);
"""


def code_fingerprint() -> str:
    """A sha over the constants that DEFINE what a candidate table is.

    ``_CLICKABLE_ROLES``, ``_EDITABLE_ROLES`` and ``max_candidates`` decide
    which elements become candidates at all. Change any of them and a row
    harvested before the change describes a different universe of options —
    scoring old and new rows together would blend two experiments. Imported
    lazily so importing the corpus never drags the candidate builder in.
    """
    from prometheus.computer import candidates as _c

    parts = [
        ",".join(sorted(_c._CLICKABLE_ROLES)),
        ",".join(sorted(_c._EDITABLE_ROLES)),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _table_fingerprint(rows: list[dict[str, str]]) -> str:
    """A sha over the table itself — spots the same window captured twice."""
    joined = "|".join(f"{r['id']}\x1f{r['description']}" for r in sorted(
        rows, key=lambda r: r["id"]))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def _scrub(text: str, limit: int) -> str:
    """Redact, then bound. The only way desktop-scraped text enters a row.

    Uses ``AuditLogger.redact`` rather than a local pattern list because that
    method's own docstring names itself THE redactor for this system and warns
    that a parallel one would drift with nothing to notice the drift.
    """
    if not text:
        return ""
    scrubbed = AuditLogger.redact(text)
    if len(scrubbed) > limit:
        scrubbed = scrubbed[: limit - 1] + "…"
    return scrubbed


@dataclass
class TableRecord:
    """One captured candidate table and what became of it.

    Mutable ON PURPOSE, and only within a single ``step()``. The loop builds one
    at the top of the call and fills it in as each stage produces its fact, so a
    record still exists on the paths that return early or raise.
    """

    goal: str
    target: str
    app: str
    window_id: int
    pid: int = 0
    #: See the schema comment: fixture rows must never be mistaken for real ones.
    driver_kind: str = "unknown"
    #: "human" if a person wrote this goal without looking at the table,
    #: "derived" if it came from a candidate's own description. A corpus that
    #: cannot tell them apart cannot enforce its own harvest spec.
    goal_source: str = "unknown"
    harvest_session: str = ""
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    snapshot_id: str | None = None
    unusable_reason: str | None = None
    candidates: list[dict[str, str]] = field(default_factory=list)

    #: What the chooser ANSWERED — including a reserved id, or an id that failed
    #: validation. Distinct from what was executed, and recorded even when
    #: nothing ran.
    chooser_answer_id: str | None = None
    chooser_source: str = ""
    chooser_confidence: float | None = None

    #: What was actually EXECUTED. None on every path that ran nothing, which is
    #: most of them.
    executed_candidate_id: str | None = None

    status: str = "incomplete"
    reason: str = ""
    verified: bool | None = None
    extent: str = ""
    #: Set when step() raised rather than returned. A crash mid-step is exactly
    #: the row a corpus wants and the one a return-only recorder would lose.
    exception: str | None = None

    def note_observation(self, observation: Any) -> None:
        self.snapshot_id = getattr(observation, "snapshot_id", None)
        self.unusable_reason = getattr(observation, "unusable_reason", None)

    def note_candidates(self, candidates: list[Any]) -> None:
        """Store the chooser's view of the table — ids and descriptions only.

        Deliberately NOT the full ``Candidate``: arguments never enter the
        corpus. They carry element tokens (meaningless once the snapshot dies)
        and, for a type action, the caller's payload text. The chooser never saw
        them, and neither does anything scoring the chooser.
        """
        self.candidates = [
            {
                "id": c.candidate_id,
                "description": _scrub(c.description, MAX_DESCRIPTION_CHARS),
            }
            for c in candidates
        ]

    def note_choice(self, choice: Any) -> None:
        self.chooser_answer_id = getattr(choice, "candidate_id", None)
        self.chooser_source = getattr(choice, "source", "") or ""
        self.chooser_confidence = getattr(choice, "confidence", None)

    def note_result(self, result: Any) -> None:
        self.status = getattr(result, "status", "unknown")
        self.reason = _scrub(getattr(result, "reason", "") or "", MAX_REASON_CHARS)
        self.verified = getattr(result, "verified", None)
        self.extent = getattr(result, "extent", "") or ""
        candidate = getattr(result, "candidate", None)
        if candidate is not None and self.status == "executed":
            self.executed_candidate_id = candidate.candidate_id

    def note_exception(self, exc: BaseException) -> None:
        self.status = "raised"
        self.exception = _scrub(f"{type(exc).__name__}: {exc}", MAX_REASON_CHARS)


class CorpusStore:
    """Reads and writes ``computer_corpus.db``. Writes never raise at the caller.

    A corpus is an observation of the system, and an observer that can break the
    thing it observes is not worth having. ``capture`` logs and swallows every
    failure: a lost row is a gap in a dataset, whereas a raised exception would
    be a computer-use step that failed because of its own bookkeeping.

    Reads do NOT swallow — a scoring harness that silently sees fewer rows than
    exist computes a confidently wrong number.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._ensure_schema()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO schema_meta (key, value) VALUES (?, ?)",
                    ("schema_version", str(CORPUS_SCHEMA_VERSION)),
                )
            elif int(row["value"]) > CORPUS_SCHEMA_VERSION:
                raise RuntimeError(
                    f"{self._db_path} was written by schema version "
                    f"{row['value']}, newer than this code's "
                    f"{CORPUS_SCHEMA_VERSION}. Refusing to open it — a newer "
                    f"writer may store columns this reader would silently drop."
                )

    # -- machine side ------------------------------------------------------

    def capture(self, record: TableRecord) -> bool:
        """Append one captured table. Returns whether it landed; never raises."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO tables (
                        record_id, captured_at, schema_version, goal, target,
                        app, window_id, pid, driver_kind, goal_source,
                        harvest_session, table_fingerprint, code_fingerprint,
                        snapshot_id, unusable_reason,
                        candidates_json, candidate_count, chooser_answer_id,
                        chooser_source, chooser_confidence,
                        executed_candidate_id, status, reason, verified,
                        extent, exception
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        record.record_id, time.time(), CORPUS_SCHEMA_VERSION,
                        _scrub(record.goal, MAX_GOAL_CHARS),
                        record.target, record.app, record.window_id,
                        record.pid, record.driver_kind, record.goal_source,
                        record.harvest_session,
                        _table_fingerprint(record.candidates),
                        code_fingerprint(),
                        record.snapshot_id, record.unusable_reason,
                        json.dumps(record.candidates, ensure_ascii=False),
                        len(record.candidates), record.chooser_answer_id,
                        record.chooser_source, record.chooser_confidence,
                        record.executed_candidate_id, record.status,
                        record.reason,
                        None if record.verified is None else int(record.verified),
                        record.extent, record.exception,
                    ),
                )
            return True
        except Exception:
            log.warning(
                "computer-use corpus: could not capture record %s into %s",
                record.record_id, self._db_path, exc_info=True,
            )
            return False

    # -- human side --------------------------------------------------------

    def record_annotation(
        self,
        record_id: str,
        correct_candidate_id: str,
        *,
        annotated_by: str,
        note: str = "",
    ) -> None:
        """Append one judgment. Raises on a bad one — this is a human's input.

        ``correct_candidate_id`` is a candidate id from that record's table,
        ``ANNOTATION_NONE_CORRECT``, or ``ANNOTATION_TABLE_UNUSABLE``.

        There is deliberately NO way to write "unannotated": absence of a row
        already means that, and a second representation of the same state is how
        the two get confused. A null is refused rather than stored.

        An id that is not in the record's own table is refused too. A typo that
        lands as a valid-looking answer is unrecoverable later — nothing
        downstream can tell it from a real judgment.
        """
        if not correct_candidate_id:
            raise ValueError(
                "correct_candidate_id cannot be empty — an unannotated record "
                "is represented by having NO row, not by a blank one."
            )
        row = self.get_table(record_id)
        if row is None:
            raise ValueError(f"no captured table with record_id {record_id!r}")
        if correct_candidate_id not in SENTINEL_ANNOTATIONS:
            known = {c["id"] for c in json.loads(row["candidates_json"])}
            if correct_candidate_id not in known:
                raise ValueError(
                    f"{correct_candidate_id!r} is not a candidate in record "
                    f"{record_id!r}. Known ids: {sorted(known)}. Use "
                    f"{ANNOTATION_NONE_CORRECT!r} if no candidate was right."
                )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO annotations (
                    record_id, correct_candidate_id, annotated_by,
                    annotated_at, note
                ) VALUES (?,?,?,?,?)
                """,
                (record_id, correct_candidate_id, annotated_by, time.time(), note),
            )

    # -- reads -------------------------------------------------------------

    def get_table(self, record_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM tables WHERE record_id = ?", (record_id,)
            ).fetchone()

    def all_tables(self) -> list[dict[str, Any]]:
        """Every captured table, REFUSING any row from an unknown version."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tables ORDER BY captured_at"
            ).fetchall()
        out = []
        for row in rows:
            if row["schema_version"] != CORPUS_SCHEMA_VERSION:
                raise ValueError(
                    f"record {row['record_id']} has schema_version "
                    f"{row['schema_version']}, this reader understands only "
                    f"{CORPUS_SCHEMA_VERSION}. Migrate it — do not skip it, or "
                    f"every score computed here is quietly off."
                )
            d = dict(row)
            d["candidates"] = json.loads(d.pop("candidates_json"))
            out.append(d)
        return out

    def latest_annotations(self) -> dict[str, dict[str, Any]]:
        """Newest judgment per record. A revision supersedes, never overwrites."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM annotations ORDER BY annotated_at, id"
            ).fetchall()
        return {r["record_id"]: dict(r) for r in rows}


@dataclass
class ScorableCorpus:
    """Tables joined to judgments, with the unscorable populations named.

    These are fields rather than something a caller derives, because the number
    that matters most — how much of the corpus nobody has reviewed — is the one
    a scoring script is most likely to forget to ask about.
    """

    answered: list[dict[str, Any]] = field(default_factory=list)
    none_correct: list[dict[str, Any]] = field(default_factory=list)
    unusable: list[dict[str, Any]] = field(default_factory=list)
    unannotated: list[dict[str, Any]] = field(default_factory=list)

    @property
    def scorable(self) -> list[dict[str, Any]]:
        """Rows a chooser can be scored on. ``unusable`` and ``unannotated``
        are excluded, and excluded differently from being counted wrong."""
        return [*self.answered, *self.none_correct]

    def summary(self) -> str:
        total = (
            len(self.answered) + len(self.none_correct)
            + len(self.unusable) + len(self.unannotated)
        )
        pct = (100 * len(self.unannotated) // total) if total else 0
        return (
            f"{total} tables: {len(self.scorable)} scorable "
            f"({len(self.answered)} answered, {len(self.none_correct)} "
            f"none-correct), {len(self.unusable)} unusable-excluded, "
            f"{len(self.unannotated)} UNANNOTATED ({pct}%)"
        )


def load_corpus(store: CorpusStore) -> ScorableCorpus:
    """Join captured tables to judgments and partition them by scorability."""
    annotations = store.latest_annotations()
    out = ScorableCorpus()
    for row in store.all_tables():
        if row["candidate_count"] == 0:
            # STRUCTURALLY unscorable, not a judgment. A step blocked at the
            # precondition check never observed anything, so there is no table
            # and no chooser could have picked from one. Routing these to
            # `unannotated` would inflate the count of rows a human still owes
            # a decision on with rows nobody should ever be asked to look at —
            # and that count is the one a scoring script is meant to trust.
            out.unusable.append({**row, "correct_candidate_id": None,
                                 "annotated_by": ""})
            continue
        ann = annotations.get(row["record_id"])
        if ann is None:
            out.unannotated.append(row)
            continue
        answer = ann["correct_candidate_id"]
        merged = {
            **row,
            "correct_candidate_id": answer,
            "annotated_by": ann["annotated_by"],
        }
        if answer == ANNOTATION_TABLE_UNUSABLE:
            out.unusable.append(merged)
        elif answer == ANNOTATION_NONE_CORRECT:
            out.none_correct.append(merged)
        else:
            out.answered.append(merged)
    return out
