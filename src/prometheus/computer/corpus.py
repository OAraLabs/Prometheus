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

from prometheus.computer.types import CANDIDATE_ABSTAIN, ChoiceRequest
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

# --------------------------------------------------------------------------- #
# LABEL PROVENANCE — who said this was the right answer
#
# A number computed over model-proposed labels is AGREEMENT WITH A MODEL. It is
# not accuracy, and the two are not interchangeable: a chooser scored against
# labels another model produced can only be measured on how alike they are,
# which is exactly the "calibrating against its own author" failure
# tests/fixtures/divergence_traces.py forbids in its opening paragraph.
#
# ⚠ Like every other provenance field here, it CANNOT BE RETROFITTED. Once a
# corpus is labelled and nobody wrote down who labelled it, no later reader can
# separate the human rows from the model rows — and the calibration set is
# precisely the difference between them.
# --------------------------------------------------------------------------- #

#: A person decided, having looked at the table. The only labels against which
#: a score may be called ACCURACY.
LABEL_HUMAN = "human"

#: A model proposed it and no person checked. A score over these rows measures
#: AGREEMENT WITH A MODEL.
LABEL_MODEL = "model"

#: A model proposed it and a person then confirmed or corrected it. Weaker than
#: `human` — the person saw a suggestion first, and anchoring is real — so it
#: is tracked separately rather than folded into either neighbour.
LABEL_MODEL_CONFIRMED = "model_confirmed"

LABEL_SOURCES = frozenset({LABEL_HUMAN, LABEL_MODEL, LABEL_MODEL_CONFIRMED})

#: Sources a score may be reported as ACCURACY against. Deliberately a set of
#: one: `model_confirmed` is excluded because the confirming human saw the
#: model's answer first.
ACCURACY_GRADE_SOURCES = frozenset({LABEL_HUMAN})

#: WHY no candidate was the correct NEXT ACTION. Mandatory whenever the answer
#: is
#: ``none_correct``, because the four causes want OPPOSITE responses and a
#: corpus that cannot tell them apart measures the TABLE'S limits and reports
#: them as the CHOOSER'S.
#:
#: Established by reading ``build_candidates`` against ``ACTION_MODELS`` at
#: 424edc1: seven verbs are declared, three tool names are ever emitted, 12 of
#: 15 ``ALLOWED_KEYS`` are unreachable, and there are no modifier combinations
#: at all. ``scroll`` and ``invoke_menu`` carry full models, schemas, gate
#: wiring and operator-facing consent phrases, and cannot be selected because
#: nothing puts them in a table.
#:
#: ⚠ THE QUESTION IS ABOUT ONE STEP, NOT THE WHOLE GOAL. ``loop.step`` takes a
#: single bounded action and then RE-OBSERVES with history, so the label is the
#: correct NEXT action — not a candidate that completes the goal by itself. A
#: table offering "Main menu" for the goal "undo what I just did" contains the
#: right next action even though nothing in it undoes anything; that is a
#: CHOOSER gap, not a table defect. Judging these rows against goal-completion
#: overstates table defects and would send the fix to the wrong place.
#:
#: A row is only a table defect when NO step from this table advances the goal
#: — not when no single step finishes it.
#:
#: ⚠ CANNOT BE RETROFITTED. A corpus harvested without this cannot be split
#: afterwards — nobody recorded why. That is why it lands with step 1.

#: The verb exists in ``ACTION_MODELS`` but ``build_candidates`` never emits it
#: (``scroll``, ``invoke_menu``). Fixable deterministically; no classifier
#: needed and none would help.
REASON_VERB_NOT_OFFERED = "verb_not_offered"

#: Needs a key in ``ALLOWED_KEYS`` outside return/tab/escape, or a modifier
#: combination — which has no representation anywhere today.
REASON_KEY_NOT_OFFERED = "key_not_offered"

#: The target was not in the observation at all. Blames the OBSERVATION — the
#: element never reached the tree, so no table could have contained it.
REASON_ELEMENT_NOT_IN_TREE = "element_not_in_tree"

#: The element WAS observed, and its role is outside ``_CLICKABLE_ROLES`` so
#: ``build_candidates`` never offered it. Blames the CANDIDATE BUILDER, and the
#: fix is different from ``element_not_in_tree``: widening a role set, not
#: fixing an observation. Kept distinct for exactly that reason.
#:
#: ⚠ Widening ``_CLICKABLE_ROLES`` is a new CONSENT SURFACE, not a config
#: tweak — every added role is a class of thing the system may now be asked to
#: click. Each wants its own decision.
REASON_ROLE_NOT_CLICKABLE = "role_not_clickable"

#: The goal genuinely cannot be done in this window. THE ONLY REASON FOR WHICH
#: ABSTAINING IS THE CORRECT ANSWER — and therefore the only one against which
#: a classifier that picks something scores a REGRESSION rather than a gain.
REASON_NOT_ACHIEVABLE_HERE = "not_achievable_here"

NONE_CORRECT_REASONS = frozenset({
    REASON_VERB_NOT_OFFERED,
    REASON_KEY_NOT_OFFERED,
    REASON_ELEMENT_NOT_IN_TREE,
    REASON_ROLE_NOT_CLICKABLE,
    REASON_NOT_ACHIEVABLE_HERE,
})

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

-- One row per STEP. Machine-written except correct_id.
CREATE TABLE IF NOT EXISTS tables (
    record_id             TEXT PRIMARY KEY,
    captured_at           REAL NOT NULL,
    schema_version        INTEGER NOT NULL,

    -- ── THE REPLAY SURFACE ────────────────────────────────────────────────
    -- These four ARE a ChoiceRequest. `replay_request` rebuilds one from them
    -- and it must equal the object the chooser was handed, so nothing here is
    -- redacted, truncated or normalised. See the module docstring.
    goal                  TEXT NOT NULL,
    snapshot_id           TEXT,
    candidates_json       TEXT NOT NULL,   -- chooser_view() output, VERBATIM
    history_json          TEXT NOT NULL DEFAULT '[]',
    -- Every OBSERVED element, including the ones build_candidates rejected.
    -- Without this a `role_not_clickable` answer is unverifiable from a stored
    -- row: the corpus would record what WAS offered and never what was seen
    -- and dropped. Established the hard way — row 5's cause could only be
    -- settled by re-observing the live app, which worked by luck because the
    -- window had not changed. 120 harvest rows will not be that lucky.
    elements_json         TEXT NOT NULL DEFAULT '[]',

    target                TEXT NOT NULL,
    app                   TEXT NOT NULL,
    window_id             INTEGER NOT NULL,
    pid                   INTEGER NOT NULL DEFAULT 0,
    candidate_count       INTEGER NOT NULL,

    -- ── DETERMINISTIC: always runs, authoritative wherever it ANSWERS ─────
    deterministic_chooser     TEXT NOT NULL DEFAULT '',
    deterministic_id          TEXT,
    deterministic_confidence  REAL,
    -- Explicit, not inferred from `deterministic_id == "abstain"`. This flag
    -- defines the region where a classifier is allowed to be authoritative,
    -- so a scoring harness must not have to re-derive it from a string
    -- comparison that a later reserved-id rename would silently break.
    deterministic_abstained   INTEGER NOT NULL DEFAULT 0,

    -- ── SHADOW: recorded, NEVER acted on. All nullable — usually absent. ──
    shadow_chooser        TEXT,
    shadow_id             TEXT,
    shadow_confidence     REAL,
    shadow_latency_ms     REAL,

    -- NULL when there was no shadow answer to agree with. Not false: "they
    -- disagreed" and "there was nothing to compare" score differently.
    agreed                INTEGER,

    -- ── WHAT ACTUALLY HAPPENED ────────────────────────────────────────────
    executed_candidate_id TEXT,            -- may match NEITHER chooser
    verified              INTEGER,         -- the bool|None from _verify
    gate_decision         TEXT,
    gate_approval_required INTEGER,
    gate_approval_granted  INTEGER,

    -- ── THE LABEL ─────────────────────────────────────────────────────────
    -- NULL means nobody has judged this yet. Materialised from the latest
    -- annotations row in the same transaction that writes it.
    correct_id            TEXT,
    -- Populated iff correct_id == 'none_correct'. See NONE_CORRECT_REASONS:
    -- three of the four are table defects the deterministic path can fix, and
    -- only 'not_achievable_here' is a row where abstaining is RIGHT.
    none_correct_reason   TEXT,
    -- WHO said so: human | model | model_confirmed. Required with correct_id.
    -- A score over `model` rows is AGREEMENT WITH A MODEL, never accuracy.
    label_source          TEXT,

    -- provenance + diagnostics (NOT part of the replay surface)
    driver_kind           TEXT NOT NULL DEFAULT 'unknown',
    goal_source           TEXT NOT NULL DEFAULT 'unknown',
    harvest_session       TEXT NOT NULL DEFAULT '',
    table_fingerprint     TEXT NOT NULL DEFAULT '',
    code_fingerprint      TEXT NOT NULL DEFAULT '',
    unusable_reason       TEXT,
    status                TEXT NOT NULL,
    reason                TEXT NOT NULL DEFAULT '',
    extent                TEXT NOT NULL DEFAULT '',
    exception             TEXT
);
CREATE INDEX IF NOT EXISTS idx_tables_app ON tables (app);
CREATE INDEX IF NOT EXISTS idx_tables_count ON tables (candidate_count);
CREATE INDEX IF NOT EXISTS idx_tables_driver ON tables (driver_kind);
CREATE INDEX IF NOT EXISTS idx_tables_abstained
    ON tables (deterministic_abstained);

-- Session status. `pilot` sessions are EXCLUDED FROM ALL SCORING.
--
-- A separate table rather than a column on `tables`, because pilot-ness is a
-- property of the harvest session and not of any observation — and because
-- `tables` is machine-written and append-only, so retro-editing captured rows
-- to carry a later judgment about the run is exactly the mixing this schema
-- keeps apart everywhere else.
CREATE TABLE IF NOT EXISTS sessions (
    harvest_session TEXT PRIMARY KEY,
    status          TEXT NOT NULL,   -- 'pilot' | 'real'
    note            TEXT NOT NULL DEFAULT '',
    recorded_at     REAL NOT NULL
);

-- Diagnostic answers about ABSTAIN rows. NOT ground truth, NOT correct_id.
--
-- Answers one question per row: was a correct candidate present at all, and if
-- not, why not. That measures THE TABLE, not the chooser, and it is kept in
-- its own table precisely so it can never be mistaken for a label or consume
-- the calibration set.
CREATE TABLE IF NOT EXISTS abstain_diagnostics (
    record_id          TEXT PRIMARY KEY,
    correct_present    INTEGER NOT NULL,   -- 0/1: was ANY candidate right?
    reason             TEXT,               -- a NONE_CORRECT_REASON when not
    answered_by        TEXT NOT NULL,
    answered_at        REAL NOT NULL,
    note               TEXT NOT NULL DEFAULT ''
);

-- Append-only provenance for correct_id: who judged what, when, and what they
-- said before changing their mind. `tables.correct_id` is the materialised
-- latest value; this is the history behind it. Both are written in ONE
-- transaction so they cannot drift.
CREATE TABLE IF NOT EXISTS annotations (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id             TEXT NOT NULL,
    correct_candidate_id  TEXT NOT NULL,
    none_correct_reason   TEXT,
    label_source          TEXT NOT NULL,
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
    """One captured STEP — an eval-set row, not an audit line.

    Mutable ON PURPOSE, and only within a single ``step()``. The loop builds one
    at the top of the call and fills it in as each stage produces its fact, so a
    row still exists on the paths that return early or raise.
    """

    goal: str
    target: str
    app: str
    window_id: int
    pid: int = 0
    #: Part of the REPLAY SURFACE — the chooser saw this, so it is stored.
    history: list[str] = field(default_factory=list)
    driver_kind: str = "unknown"
    goal_source: str = "unknown"
    harvest_session: str = ""
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    snapshot_id: str | None = None
    unusable_reason: str | None = None
    candidates: list[dict[str, str]] = field(default_factory=list)
    #: Every observed element, offered or not. See the schema comment.
    elements: list[dict[str, Any]] = field(default_factory=list)

    #: DETERMINISTIC — always runs, authoritative wherever it answers.
    deterministic_chooser: str = ""
    deterministic_id: str | None = None
    deterministic_confidence: float | None = None
    deterministic_abstained: bool = False

    #: SHADOW — recorded, never acted on. Absent unless a shadow chooser ran.
    shadow_chooser: str | None = None
    shadow_id: str | None = None
    shadow_confidence: float | None = None
    shadow_latency_ms: float | None = None

    #: What actually happened.
    executed_candidate_id: str | None = None
    verified: bool | None = None
    gate_decision: str | None = None
    gate_approval_required: bool | None = None
    gate_approval_granted: bool | None = None

    status: str = "incomplete"
    reason: str = ""
    extent: str = ""
    exception: str | None = None

    @property
    def agreed(self) -> bool | None:
        """Did the two choosers pick the same candidate?

        NULL — not False — when there was no shadow answer. "They disagreed"
        and "there was nothing to compare" are different facts and a harness
        that counts the second as the first understates agreement by exactly
        the number of steps the shadow did not run on.
        """
        if self.shadow_id is None or self.deterministic_id is None:
            return None
        return self.shadow_id == self.deterministic_id

    def note_observation(self, observation: Any) -> None:
        self.snapshot_id = getattr(observation, "snapshot_id", None)
        self.unusable_reason = getattr(observation, "unusable_reason", None)
        self.note_elements(observation)

    def note_elements(self, observation: Any) -> None:
        """Record every OBSERVED element, including the rejected ones.

        The candidate table says what was offered. This says what was seen —
        and the difference between them is the only evidence that a
        ``role_not_clickable`` answer is right.
        """
        self.elements = [
            {
                "index": e.element_index,
                "role": e.role,
                "label": e.label,
                "editable": bool(e.editable),
            }
            for e in getattr(observation, "elements", ())
        ]

    def note_candidates(self, candidates: list[Any]) -> None:
        """Store the chooser's view of the table, VERBATIM.

        ``chooser_view()`` output and nothing else: arguments never enter the
        corpus — they carry element tokens that die with the snapshot and, for
        a type action, the caller's payload.

        Deliberately NOT redacted or truncated, unlike ``reason`` below. This
        is the replay surface: ``replay_request`` has to hand a future chooser
        the object the original chooser was handed, and a description shortened
        to 300 characters is a DIFFERENT input that scores a different
        question. See the module docstring for what that costs and what
        controls it instead.
        """
        self.candidates = [
            {"id": c.candidate_id, "description": c.description}
            for c in candidates
        ]

    def note_choice(self, choice: Any, *, shadow: bool = False,
                    latency_ms: float | None = None) -> None:
        cid = getattr(choice, "candidate_id", None)
        if shadow:
            self.shadow_chooser = getattr(choice, "source", "") or "unknown"
            self.shadow_id = cid
            self.shadow_confidence = getattr(choice, "confidence", None)
            self.shadow_latency_ms = latency_ms
            return
        self.deterministic_chooser = getattr(choice, "source", "") or ""
        self.deterministic_id = cid
        self.deterministic_confidence = getattr(choice, "confidence", None)
        # Set from the reserved id HERE, once, so every reader downstream sees
        # a boolean rather than re-deriving it from a string.
        self.deterministic_abstained = cid == CANDIDATE_ABSTAIN

    def note_gate(self, decision: Any) -> None:
        self.gate_decision = (
            "allowed" if getattr(decision, "allowed", False) else "denied"
        )
        self.gate_approval_required = bool(
            getattr(decision, "requires_confirmation", False)
        )

    def note_approval(self, granted: bool) -> None:
        self.gate_approval_granted = granted

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


def replay_request(row: dict[str, Any]) -> ChoiceRequest:
    """Rebuild the ChoiceRequest a chooser was handed, from a stored row.

    THE POINT OF THE WHOLE CORPUS. The driver leg cannot be exercised without a
    display and so is uncoverable in CI; the CHOOSER leg against a stored row is
    fully coverable, on any machine, with no display, no AT-SPI and no Cua.

    This is why the replay surface is stored verbatim. A reconstruction from a
    stored *observation* would be a different thing: element tokens die at the
    next observation, and rebuilding a table through ``build_candidates`` runs
    today's role sets over yesterday's tree and can yield a different table
    than the one the chooser actually saw.
    """
    return ChoiceRequest(
        goal=row["goal"],
        snapshot_id=row["snapshot_id"] or "",
        candidates=row["candidates"],
        history=row["history"],
    )


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

    #: Columns added after a corpus may already exist on disk. ``CREATE TABLE
    #: IF NOT EXISTS`` does not alter an existing table, so a file written by
    #: an earlier build keeps the old shape and every reader that assumes the
    #: new one raises KeyError. Additive only, with a default — a column that
    #: needs backfilling is not migratable this way and needs a real migration
    #: plus a schema_version bump.
    _ADDITIVE_COLUMNS: tuple[tuple[str, str, str], ...] = (
        ("tables", "elements_json", "TEXT NOT NULL DEFAULT '[]'"),
        ("tables", "none_correct_reason", "TEXT"),
        ("tables", "label_source", "TEXT"),
        ("annotations", "none_correct_reason", "TEXT"),
        ("annotations", "label_source", "TEXT NOT NULL DEFAULT 'human'"),
    )

    def _migrate_additive(self, conn: sqlite3.Connection) -> None:
        for table, column, decl in self._ADDITIVE_COLUMNS:
            have = {
                r["name"]
                for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if not have:          # table not created yet; _SCHEMA will do it
                continue
            if column not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
                log.info(
                    "computer-use corpus: added missing column %s.%s to %s",
                    table, column, self._db_path,
                )

    def _ensure_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            self._migrate_additive(conn)
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
                        record_id, captured_at, schema_version,
                        goal, snapshot_id, candidates_json, history_json,
                        elements_json,
                        target, app, window_id, pid, candidate_count,
                        deterministic_chooser, deterministic_id,
                        deterministic_confidence, deterministic_abstained,
                        shadow_chooser, shadow_id, shadow_confidence,
                        shadow_latency_ms, agreed,
                        executed_candidate_id, verified,
                        gate_decision, gate_approval_required,
                        gate_approval_granted,
                        driver_kind, goal_source, harvest_session,
                        table_fingerprint, code_fingerprint,
                        unusable_reason, status, reason, extent, exception
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        record.record_id, time.time(), CORPUS_SCHEMA_VERSION,
                        # THE REPLAY SURFACE — verbatim, see note_candidates.
                        record.goal, record.snapshot_id,
                        json.dumps(record.candidates, ensure_ascii=False),
                        json.dumps(record.history, ensure_ascii=False),
                        json.dumps(record.elements, ensure_ascii=False),
                        record.target, record.app, record.window_id,
                        record.pid, len(record.candidates),
                        record.deterministic_chooser, record.deterministic_id,
                        record.deterministic_confidence,
                        int(record.deterministic_abstained),
                        record.shadow_chooser, record.shadow_id,
                        record.shadow_confidence, record.shadow_latency_ms,
                        None if record.agreed is None else int(record.agreed),
                        record.executed_candidate_id,
                        None if record.verified is None else int(record.verified),
                        record.gate_decision,
                        None if record.gate_approval_required is None
                        else int(record.gate_approval_required),
                        None if record.gate_approval_granted is None
                        else int(record.gate_approval_granted),
                        record.driver_kind, record.goal_source,
                        record.harvest_session,
                        _table_fingerprint(record.candidates),
                        code_fingerprint(),
                        record.unusable_reason, record.status,
                        _scrub(record.reason, MAX_REASON_CHARS),
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
        label_source: str,
        none_correct_reason: str | None = None,
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

        ⚠⚠ THE LABEL IS RELATIVE TO THE OBSERVATION, NOT TO THE APP.

        The correct answer is the best action given WHAT WAS OBSERVED — not
        given ground truth about what the application can do. A GTK popover's
        contents do not exist in the accessibility tree until it is opened, so
        "the menu contains Undo" is unknowable to the chooser at the moment it
        decides. Labelling against facts the chooser could not have had would
        score it on information it never received.

        So "open the container because the target plausibly lives inside it" is
        a CORRECT label, and it stays correct even if the container turns out
        not to contain it. Exploratory is not wrong; the next observation is
        where that gets resolved, which is exactly what one-bounded-step-then-
        re-observe is for.

        This governs the real harvest more than it governs the pilot: most
        desktop state is behind closed containers — menus, popovers, expanders,
        unselected tabs — and a corpus labelled against app knowledge rather
        than tree contents would systematically mark correct exploration wrong.

        ⚠ ``correct_candidate_id`` is the correct NEXT ACTION for this goal,
        not a candidate that completes the goal on its own. The loop takes one
        bounded step and re-observes with history, so "open the menu that
        contains the thing" is a correct label when the thing itself is one
        level down. ``none_correct`` means no candidate here advances the goal
        AT ALL.

        ``none_correct_reason`` is REQUIRED with ``ANNOTATION_NONE_CORRECT`` and
        REFUSED with anything else, both enforced here rather than documented.
        The four reasons want opposite responses — three are table defects the
        deterministic path can fix, and only ``not_achievable_here`` is a row
        where abstaining is the right answer — so a ``none_correct`` row with no
        reason is a row that can never be placed on either side of that split.
        It cannot be backfilled: nobody recorded why.

        ``label_source`` is REQUIRED — there is no default. A label whose author
        is unrecorded cannot be separated from the calibration set later, and a
        score computed over model-proposed labels is AGREEMENT WITH A MODEL
        rather than accuracy. Defaulting it to ``human`` would be the
        comfortable choice and the wrong one.
        """
        if label_source not in LABEL_SOURCES:
            raise ValueError(
                f"label_source must be one of {sorted(LABEL_SOURCES)}, got "
                f"{label_source!r}. A label whose author is unrecorded cannot "
                f"be separated from the calibration set later, and a score over "
                f"model-proposed labels is AGREEMENT WITH A MODEL, not accuracy. "
                f"It cannot be added afterwards."
            )
        if correct_candidate_id == ANNOTATION_NONE_CORRECT:
            if none_correct_reason is None:
                raise ValueError(
                    f"{ANNOTATION_NONE_CORRECT!r} requires none_correct_reason. "
                    f"One of {sorted(NONE_CORRECT_REASONS)}. Without it this row "
                    f"cannot be split into 'the table could not express the goal' "
                    f"versus 'abstaining was correct', and those are the two "
                    f"halves a classifier is judged on. It cannot be added later."
                )
            if none_correct_reason not in NONE_CORRECT_REASONS:
                raise ValueError(
                    f"{none_correct_reason!r} is not a known reason. "
                    f"Known: {sorted(NONE_CORRECT_REASONS)}. A free-text reason "
                    f"cannot be aggregated, and aggregating them is the point."
                )
        elif none_correct_reason is not None:
            raise ValueError(
                f"none_correct_reason is only meaningful with "
                f"{ANNOTATION_NONE_CORRECT!r}, not with "
                f"{correct_candidate_id!r} — a reason stored beside a real "
                f"candidate id would be read as explaining a choice it does not."
            )
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
                    record_id, correct_candidate_id, none_correct_reason,
                    label_source, annotated_by, annotated_at, note
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (record_id, correct_candidate_id, none_correct_reason,
                 label_source, annotated_by, time.time(), note),
            )
            # Materialise onto the row in the SAME transaction, so the column
            # and its provenance cannot drift.
            conn.execute(
                "UPDATE tables SET correct_id = ?, none_correct_reason = ?, "
                "label_source = ? WHERE record_id = ?",
                (correct_candidate_id, none_correct_reason, label_source,
                 record_id),
            )

    def mark_session(self, session: str, status: str, note: str = "") -> None:
        """Record a harvest session as ``pilot`` or ``real``.

        A pilot session is excluded from every scoring path. Marking it is a
        decision about the RUN, so it lives beside the run rather than being
        written back onto the observations it produced.
        """
        if status not in ("pilot", "real"):
            raise ValueError(f"status must be 'pilot' or 'real', got {status!r}")
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions "
                "(harvest_session, status, note, recorded_at) VALUES (?,?,?,?)",
                (session, status, note, time.time()),
            )

    def session_status(self) -> dict[str, str]:
        with self._connect() as conn:
            return {
                r["harvest_session"]: r["status"]
                for r in conn.execute("SELECT * FROM sessions").fetchall()
            }

    def record_abstain_diagnostic(
        self,
        record_id: str,
        *,
        correct_present: bool,
        answered_by: str,
        reason: str | None = None,
        note: str = "",
    ) -> None:
        """Answer the table question for ONE abstain row. NOT a label.

        ⚠ THE QUESTION IS ONE-STEP. "Was a correct NEXT ACTION present" — not
        "did any candidate complete the goal". ``loop.step`` takes one bounded
        action and re-observes with history, so opening the menu that contains
        the target IS a correct next action. Asking the goal-completion
        question instead classifies chooser gaps as table defects and sends the
        fix to the wrong place.

        ``correct_present=False`` requires a reason from
        :data:`NONE_CORRECT_REASONS` — that is the whole measurement. A
        ``True`` answer means the table DID contain a correct next action and
        the deterministic chooser missed it: a chooser gap, not a table gap,
        and it takes no reason.

        Deliberately not ``record_annotation``: this does not set ``correct_id``
        and must never be counted as ground truth. Diagnosing which rows the
        table could not express is a different question from which candidate
        was right, and conflating them would spend the calibration set on it.
        """
        if not correct_present:
            if reason not in NONE_CORRECT_REASONS:
                raise ValueError(
                    f"correct_present=False needs a reason from "
                    f"{sorted(NONE_CORRECT_REASONS)}, got {reason!r}. The reason "
                    f"IS the measurement — it is what separates a table that "
                    f"could not express the goal from a chooser that missed it."
                )
        elif reason is not None:
            raise ValueError(
                "a reason is only meaningful when no candidate was correct; "
                "correct_present=True means the table was fine."
            )
        if self.get_table(record_id) is None:
            raise ValueError(f"no captured table with record_id {record_id!r}")
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO abstain_diagnostics "
                "(record_id, correct_present, reason, answered_by, "
                " answered_at, note) VALUES (?,?,?,?,?,?)",
                (record_id, int(correct_present), reason, answered_by,
                 time.time(), note),
            )

    def abstain_diagnostics(self) -> dict[str, dict[str, Any]]:
        with self._connect() as conn:
            return {
                r["record_id"]: dict(r)
                for r in conn.execute("SELECT * FROM abstain_diagnostics")
            }

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
            d["history"] = json.loads(d.pop("history_json") or "[]")
            d["elements"] = json.loads(d.pop("elements_json") or "[]")
            out.append(d)
        return out

    def latest_annotations(self) -> dict[str, dict[str, Any]]:
        """Newest judgment per record. A revision supersedes, never overwrites."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM annotations ORDER BY annotated_at, id"
            ).fetchall()
        return {r["record_id"]: dict(r) for r in rows}


# --------------------------------------------------------------------------- #
# CORPUS COMPLETENESS — a declared column empty in EVERY row is a defect
#
# `history` was declared on TableRecord, given a column, and never populated:
# every row stored `[]`. The corpus looked complete and would have scored
# choosers on an input the original never saw. It was found by reading the
# loop, which is not a control.
#
# The generalised form: a field that is universally NULL or empty is either
# unwired or meaningless, and BOTH are worth failing on. What it must never be
# is invisible.
#
# ⚠ WHY DECLARATION AND NOT SILENCE. A column that is legitimately absent has
# to say so, by name, with a reason. "Not wired yet" and "deliberately absent"
# are indistinguishable from the data, and the whole point of this guard is
# that the difference stops depending on somebody remembering which was which.
# Adding a name here is a decision with a sentence attached; leaving one out is
# the defect.
# --------------------------------------------------------------------------- #

#: Columns allowed to be universally empty, each with the reason. A column here
#: is EXCLUDED BY DECLARATION. Remove the entry when the field is wired.
INTENTIONALLY_ABSENT: dict[str, str] = {
    "shadow_chooser": (
        "no shadow chooser is built yet — milestone 3 ships the corpus first, "
        "by instruction. Remove when AdditiveChooser lands."
    ),
    "shadow_id": "see shadow_chooser",
    "shadow_confidence": "see shadow_chooser",
    "shadow_latency_ms": "see shadow_chooser",
    "agreed": (
        "NULL by definition while there is no shadow answer to agree with — "
        "and null here means 'nothing to compare', never 'disagreed'."
    ),
    "exception": (
        "populated only when step() RAISES. A harvest with no crashes leaves "
        "it empty, and that is the healthy outcome rather than a gap."
    ),
    "unusable_reason": (
        "populated only when the observation was unusable. Empty across a "
        "clean harvest is correct."
    ),
    "none_correct_reason": (
        "populated only on none_correct annotations. Empty before annotation "
        "and on a corpus with no none_correct rows."
    ),
    "correct_id": (
        "filled by a human AFTER harvest. Empty on a freshly harvested corpus "
        "by design — `unannotated` is the state this measures."
    ),
    "label_source": (
        "written with correct_id at annotation time, so it is empty on a "
        "freshly harvested corpus for the same reason. REQUIRED once a label "
        "exists — record_annotation refuses without it."
    ),
}


#: Columns that exist ONLY because an action was executed. A CAPTURE-ONLY
#: harvest cannot populate them, and that is not a defect — a corpus row is a
#: table plus a judgment, and execution is no part of the label.
#:
#: ⚠ SCOPED TO capture MODE, deliberately not added to INTENTIONALLY_ABSENT.
#: For a corpus harvested from real runs these columns MUST be populated, and a
#: blanket exclusion would stop the guard watching them forever to make one
#: harvest mode pass. Weakening an instrument to obtain a pass is the inversion
#: this project has refused twice already.
CAPTURE_ONLY_ABSENT: dict[str, str] = {
    "executed_candidate_id": "capture-only harvest: nothing was executed",
    "verified": "capture-only harvest: no action to verify",
    "gate_decision": "capture-only harvest: the gate is not reached",
    "gate_approval_required": "capture-only harvest: the gate is not reached",
    "gate_approval_granted": "capture-only harvest: the gate is not reached",
    "extent": (
        "derived at the gate by computer_extent_for, which a capture-only "
        "harvest never reaches"
    ),
    # ⚠ The uncomfortable one, and it gets evidence rather than an assertion.
    # `history` being empty is EXACTLY what the original defect looked like, and
    # this guard cannot tell "wired but the input was empty" from "never
    # wired". Here it is the former: each harvest row is an independent first
    # step with no prior action, and the wiring is pinned by
    # tests/test_computer_corpus_reasons.py::test_history_reaches_the_corpus,
    # which fails if the loop stops populating it. Remove this entry the moment
    # a harvest produces multi-step sequences.
    "history": (
        "each harvest row is an independent first step, so there is no prior "
        "action to record. Wiring proven by test_history_reaches_the_corpus — "
        "NOT by this exclusion."
    ),
}


class IncompleteCorpus(RuntimeError):
    """A declared column is empty in every row. Names the column."""


def _is_empty(value: Any) -> bool:
    """Empty for this purpose: NULL, "", [] or {} — not 0 and not False.

    0 and False are REAL VALUES. `deterministic_abstained` is 0 on every row of
    a corpus where the chooser always answered, and `verified` is legitimately
    False. Treating falsy as empty would fail on a healthy corpus, and a guard
    that fires on healthy data gets switched off.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in ("", "[]", "{}")
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def assert_corpus_complete(
    store: "CorpusStore", *, mode: str = "full"
) -> None:
    """Fail loud if any declared column is empty across EVERY row.

    Harvest exit criteria. Run it after harvesting and before annotating: a
    corpus with a universally-empty column cannot be fixed afterwards, because
    the value was never captured.

    ``mode="capture"`` additionally excludes the five action-outcome columns,
    for a harvest that observed windows without executing anything. It is a
    narrower guard and it says so; ``mode="full"`` remains the default so a
    corpus of real runs is still held to every column.

    Raises :class:`IncompleteCorpus` naming every offending column, rather than
    returning a bool — a caller that forgets to check a return value is the
    same silence this exists to remove.
    """
    rows = store.all_tables()
    if not rows:
        raise IncompleteCorpus(
            "the corpus is empty — nothing was harvested, so completeness is "
            "not measurable. This is a failure, not a pass."
        )

    if mode not in ("full", "capture"):
        raise ValueError(f"unknown mode {mode!r}; expected 'full' or 'capture'")
    excluded = dict(INTENTIONALLY_ABSENT)
    if mode == "capture":
        excluded.update(CAPTURE_ONLY_ABSENT)

    columns = [c for c in rows[0] if c != "candidates_json"]
    offenders = []
    for col in columns:
        if col in excluded:
            continue
        if all(_is_empty(row.get(col)) for row in rows):
            offenders.append(col)

    if offenders:
        raise IncompleteCorpus(
            f"{len(offenders)} declared column(s) are empty in ALL "
            f"{len(rows)} row(s): {sorted(offenders)}.\n"
            f"Each is either unwired or meaningless. This cannot be repaired "
            f"after the fact — the value was never captured — so the harvest "
            f"must be fixed and re-run.\n"
            f"If a column is legitimately always empty, add it to "
            f"INTENTIONALLY_ABSENT with the reason. Declaring it is a decision; "
            f"leaving it out is the defect this guard exists for."
        )


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
    #: Rows from a session marked `pilot`. Never scored, and counted separately
    #: so an empty scorable set is never mistaken for an empty corpus.
    pilot_excluded: list[dict[str, Any]] = field(default_factory=list)

    @property
    def scorable(self) -> list[dict[str, Any]]:
        """Rows a chooser can be scored on. ``unusable`` and ``unannotated``
        are excluded, and excluded differently from being counted wrong."""
        return [*self.answered, *self.none_correct]

    def none_correct_by_reason(self) -> dict[str, int]:
        """The split that decides the sequencing.

        ``verb_not_offered`` + ``key_not_offered`` are table defects: fixable
        deterministically, no dependency, no licence, no latency. If they are a
        large share, ``build_candidates`` is fixed and the corpus RE-MEASURED
        before any classifier is judged — otherwise the measurement is of the
        table's limits, reported as the chooser's.

        ``not_achievable_here`` is the only reason for which abstaining is
        CORRECT, and therefore the only population against which a classifier
        that picks something scores a regression rather than a gain.
        """
        out: dict[str, int] = {}
        for row in self.none_correct:
            out[row.get("none_correct_reason") or "(unrecorded)"] = (
                out.get(row.get("none_correct_reason") or "(unrecorded)", 0) + 1
            )
        return out

    def label_mix(self) -> dict[str, int]:
        """How many labels came from where. Reported on every score."""
        out: dict[str, int] = {}
        for row in self.scorable:
            src = row.get("label_source") or "(unrecorded)"
            out[src] = out.get(src, 0) + 1
        return out

    #: Descriptions whose target is a container-opening control rather than the
    #: thing itself. Substring match on the DESCRIPTION, which is all a label
    #: carries — deliberately crude, because the number is a smell test.
    _MENU_ISH = ("menu", "main menu", "hamburger", "more options", "view")

    def menu_opening_share(self) -> tuple[int, int]:
        """How many correct answers are just "open a container".

        Returns (menu-ish, total-answered). A HIGH share is a warning about the
        GOALS, not about the chooser: if most correct next actions are "open
        the menu", the goals are too coarse for one bounded step and the corpus
        is measuring container-opening rather than task selection.
        """
        answered = [
            r for r in self.answered
            if r.get("correct_candidate_id")
        ]
        if not answered:
            return (0, 0)
        by_id = 0
        for r in answered:
            cid = r["correct_candidate_id"]
            desc = next(
                (c["description"] for c in r.get("candidates", [])
                 if c["id"] == cid),
                "",
            ).lower()
            if any(m in desc for m in self._MENU_ISH):
                by_id += 1
        return (by_id, len(answered))

    def score_noun(self) -> str:
        """What a number over these rows MAY be called.

        Not a footnote and not a caller's choice. A score over model-proposed
        labels measures how alike two models are; calling that "accuracy" is
        the claim the corpus cannot support, and a reader who skims must not be
        able to pick up the wrong word.
        """
        mix = self.label_mix()
        if not mix:
            return "nothing scorable"
        if set(mix) <= ACCURACY_GRADE_SOURCES:
            return "accuracy (labels are human)"
        if LABEL_MODEL in mix and set(mix) == {LABEL_MODEL}:
            return "AGREEMENT WITH A MODEL — not accuracy"
        return "MIXED LABEL SOURCES — not accuracy; split before reporting"

    def summary(self) -> str:
        total = (
            len(self.answered) + len(self.none_correct)
            + len(self.unusable) + len(self.unannotated)
            + len(self.pilot_excluded)
        )
        pct = (100 * len(self.unannotated) // total) if total else 0
        base = (
            f"{total} tables: {len(self.scorable)} scorable "
            f"({len(self.answered)} answered, {len(self.none_correct)} "
            f"none-correct), {len(self.unusable)} unusable-excluded, "
            f"{len(self.unannotated)} UNANNOTATED ({pct}%)"
        )
        if self.pilot_excluded:
            base += (
                f"\n  {len(self.pilot_excluded)} row(s) EXCLUDED — pilot "
                f"session, never scored"
            )
        mix = self.label_mix()
        if mix:
            base += "\n  label sources: " + ", ".join(
                f"{k}={v}" for k, v in sorted(mix.items())
            )
            base += f"\n  a score over these rows is: {self.score_noun()}"
            if set(mix) - ACCURACY_GRADE_SOURCES:
                base += (
                    "\n  ⚠ NOT ACCURACY. Some labels were proposed by a model, "
                    "so a number over them measures how alike two models are. "
                    "Only rows labelled by a human grade as accuracy."
                )
        menu_n, menu_total = self.menu_opening_share()
        if menu_total:
            pct = 100 * menu_n // menu_total
            base += (
                f"\n  correct answers that just OPEN A CONTAINER: "
                f"{menu_n}/{menu_total} ({pct}%)"
            )
            if pct > 50:
                base += (
                    "\n  ⚠ OVER HALF. The goals are too coarse for one bounded "
                    "step — this corpus is measuring container-opening rather "
                    "than task selection. Split the goals, do not tune the "
                    "chooser against this."
                )
        by_reason = self.none_correct_by_reason()
        if by_reason:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(by_reason.items()))
            table_defects = sum(
                v for k, v in by_reason.items()
                if k in (REASON_VERB_NOT_OFFERED, REASON_KEY_NOT_OFFERED,
                         REASON_ROLE_NOT_CLICKABLE)
            )
            base += f"\n  none-correct by reason: {parts}"
            if table_defects:
                base += (
                    f"\n  ⚠ {table_defects} row(s) are TABLE DEFECTS, not "
                    f"chooser failures — fix build_candidates and re-measure "
                    f"before judging any classifier on this corpus."
                )
        return base


def load_corpus(
    store: CorpusStore, *, include_pilot: bool = False
) -> ScorableCorpus:
    """Join captured tables to judgments and partition them by scorability.

    PILOT SESSIONS ARE EXCLUDED unless asked for. A pilot corpus exists to
    shake out the harvest, and scoring anything against it would import
    whatever was wrong with the run it was built to find.
    """
    annotations = store.latest_annotations()
    pilots = {
        k for k, v in store.session_status().items() if v == "pilot"
    }
    out = ScorableCorpus()
    for row in store.all_tables():
        if not include_pilot and row.get("harvest_session") in pilots:
            out.pilot_excluded.append(row)
            continue
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
            "none_correct_reason": ann["none_correct_reason"],
            "label_source": ann["label_source"],
            "annotated_by": ann["annotated_by"],
        }
        if answer == ANNOTATION_TABLE_UNUSABLE:
            out.unusable.append(merged)
        elif answer == ANNOTATION_NONE_CORRECT:
            out.none_correct.append(merged)
        else:
            out.answered.append(merged)
    return out
