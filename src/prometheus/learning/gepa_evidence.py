"""GEPA's evidence: the runs in which the model actually loaded a skill.

A GEPA candidate is an auto skill the model LOADED. Every successful ``skill``
call writes one ``subsystem_runs`` row (``skills``/``load``, #591) carrying the
session and the file that served it, so the load counter says which skills
are in use and when. The evidence for one load is the run it happened in:

* **the request** — the newest user row the session persisted before the
  load. A user row is persisted the moment it arrives
  (``ChatSession.add_user_message``); the rest of a turn is persisted when the
  turn ends, so the run's own request is in the store before its calls.
* **the calls after the load** — the session's ``tool_calls`` rows from the
  load until the next run starts (the session's next round-0 ``loop_round``
  row), each with what it was given and whether it worked.
* **the outcome** — how those calls went, and whether the run ended with a
  reply.

Golden-trace exports are NOT read. They hold only successful calls made by a
cloud provider, so they cannot show a run that went wrong, and they miss every
run the local model served. (GEPA used to read them, through a finder that
matched a ``Skill`` tool and an export shape that no longer exist; it found
nothing.)

Everything is read through read-only connections, and every text taken from a
row passes the redactor before it is kept: rows written before X.37 were
stored unredacted.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prometheus.learning.trace_format import format_input
from prometheus.security.log_redaction import redact_json_text, redact_secrets
from prometheus.telemetry.tracker import (
    EXECUTED_ERROR_TYPES,
    POLICY_ERROR_TYPES,
    SKILL_LOAD_OPERATION,
    SKILL_LOAD_SUBSYSTEM,
    SYNTHETIC_TOOL_NAME,
)

log = logging.getLogger(__name__)

# The latest loads a candidate is judged on. Every judged document is scored
# once per run, so this bounds a candidate at (1 + variants) x 5 judge calls.
MAX_EVIDENCE_RUNS = 5

# Calls shown per run. A run can make hundreds; what followed the load is
# what the skill was steering, and the head of it is the part it steered most.
MAX_CALLS_PER_RUN = 12

# When a session has no later run to bound this one (its newest run, or a
# daemon too old to write loop_round rows), the run is read to this far past
# the load and marked unbounded.
RUN_CAP_SECONDS = 30 * 60

REQUEST_CHARS = 600
REPLY_CHARS = 300
ERROR_CHARS = 160


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def open_readonly(path: Path) -> sqlite3.Connection:
    """A read-only connection to a store the daemon may be writing.

    ``mode=ro``: nothing GEPA or its dry run does can change the file. Raises
    ``sqlite3.OperationalError`` when the file does not exist, rather than
    creating an empty database where the caller expected data.
    """
    conn = sqlite3.connect(
        f"{Path(path).expanduser().resolve().as_uri()}?mode=ro",
        uri=True,
        check_same_thread=False,
    )
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


# ── skills ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AutoSkill:
    """One file in ``skills/auto/``, with the name the registry serves it under."""

    path: Path
    served_name: str
    description: str
    text: str

    @property
    def stem(self) -> str:
        return self.path.stem


def auto_skills(auto_dir: Path) -> list[AutoSkill]:
    """The auto skills a candidate can be, sorted by file name.

    SkillRefiner's backups (``<name>.bak-<ts>.md``) sit in the same directory
    and are skipped: the registry serves the live file over them, and a backup
    is never something to improve.
    """
    from prometheus.skills.loader import _parse_skill_markdown

    if not auto_dir.is_dir():
        return []
    out: list[AutoSkill] = []
    for path in sorted(auto_dir.glob("*.md")):
        if ".bak-" in path.name:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            log.warning("GEPA: unreadable auto skill %s — skipping", path.name)
            continue
        name, description = _parse_skill_markdown(path.stem, text)
        out.append(AutoSkill(path=path, served_name=name, description=description, text=text))
    return out


# ── the load counter ─────────────────────────────────────────────────


@dataclass(frozen=True)
class LoadEvent:
    """One successful ``skill`` call, as the load counter recorded it."""

    timestamp: float
    session_id: str | None
    skill: str
    source: str | None
    file: str | None


def load_events(conn: sqlite3.Connection) -> list[LoadEvent]:
    """Every successful load, oldest first (the rows ``skill_load_stats`` counts).

    Raises ``sqlite3.Error`` when the counter cannot be read: an unreadable
    counter is not "nothing was loaded", and the caller says which it was.
    """
    rows = conn.execute(
        "SELECT timestamp, session_id, summary_json FROM subsystem_runs "
        "WHERE subsystem = ? AND operation = ? AND outcome = 'success' "
        "ORDER BY timestamp, rowid",
        (SKILL_LOAD_SUBSYSTEM, SKILL_LOAD_OPERATION),
    ).fetchall()
    out: list[LoadEvent] = []
    for ts, session_id, summary_json in rows:
        try:
            summary = json.loads(summary_json or "{}")
        except (TypeError, ValueError):
            continue
        if not isinstance(summary, dict):
            continue
        name = summary.get("skill")
        if not isinstance(name, str) or not name:
            continue
        source = summary.get("source")
        file = summary.get("file")
        out.append(LoadEvent(
            timestamp=float(ts),
            session_id=session_id or None,
            skill=name,
            source=source if isinstance(source, str) else None,
            file=file if isinstance(file, str) and file else None,
        ))
    return out


def loads_of(skill: AutoSkill, events: list[LoadEvent]) -> list[LoadEvent]:
    """The loads this file served.

    Matched on the file the load row names — the file that actually answered
    the call — and on the served name only for a row that names no file. Only
    loads served from ``skills/auto/`` count: a user skill of the same name is
    a different document.
    """
    wanted = skill.served_name.lower()
    return [
        e for e in events
        if e.source == "auto"
        and (e.file == skill.stem if e.file else e.skill.lower() == wanted)
    ]


# ── one run ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CallRecord:
    """One call made after the load, redacted and cut."""

    tool: str
    ok: bool
    error_type: str | None
    input_text: str
    error_text: str = ""

    @property
    def kind(self) -> str:
        """``ok`` | ``failed`` | ``denied`` (policy) | ``nonzero_exit`` (ran, exited non-zero)."""
        if self.ok:
            return "ok"
        if self.error_type in POLICY_ERROR_TYPES:
            return "denied"
        if self.error_type in EXECUTED_ERROR_TYPES:
            return "nonzero_exit"
        return "failed"


@dataclass(frozen=True)
class LoadRun:
    """The run around one load: its request, the calls after the load, how it ended."""

    load_ts: float
    session_id: str
    request: str | None
    request_withheld: bool
    calls: tuple[CallRecord, ...]
    calls_not_shown: int
    replied: bool | None
    reply: str
    bounded: bool

    def render(self) -> str:
        """The run as prompt text, for the variant generator and the judge."""
        lines: list[str] = []
        if self.request_withheld:
            lines.append("Request: (the run was started by an injected, untrusted message; "
                         "its text is not shown)")
        elif self.request:
            lines.append(f"Request: {self.request}")
        else:
            lines.append("Request: (not recoverable)")
        if self.calls:
            lines.append(f"Calls after loading the skill ({len(self.calls) + self.calls_not_shown}):")
            for i, call in enumerate(self.calls, 1):
                if call.ok:
                    result = "ok"
                else:
                    result = call.kind
                    if call.error_type and call.error_type != call.kind:
                        result += f" ({call.error_type})"
                    if call.error_text:
                        result += f": {call.error_text}"
                lines.append(f"{i}. {call.tool}({call.input_text}) → {result}")
            if self.calls_not_shown:
                lines.append(f"… and {self.calls_not_shown} more")
        else:
            lines.append("Calls after loading the skill: none")
        if self.replied is None:
            lines.append("Outcome: the end of the run was not recorded")
        elif self.replied:
            lines.append(f"Outcome: the run ended with a reply: {self.reply}"
                         if self.reply else "Outcome: the run ended with a reply")
        else:
            lines.append("Outcome: the run ended without a reply")
        return "\n".join(lines)


def _run_end(conn: sqlite3.Connection, session_id: str, load_ts: float) -> tuple[float, bool]:
    """When the run holding the load ended: the session's next run start, else the cap."""
    row = conn.execute(
        "SELECT MIN(timestamp) FROM subsystem_runs "
        "WHERE subsystem = 'agent_loop' AND operation = 'loop_round' "
        "AND round_index = 0 AND session_id = ? AND timestamp > ?",
        (session_id, load_ts),
    ).fetchone()
    if row and row[0] is not None:
        return float(row[0]), True
    return load_ts + RUN_CAP_SECONDS, False


def _call_input(parsed_tool_call: str | None) -> tuple[str | None, str]:
    """``(requested skill name if this is a skill call, rendered input)`` from a stored call."""
    if not parsed_tool_call:
        return None, "{}"
    try:
        parsed = json.loads(redact_json_text(parsed_tool_call) or "{}")
    except (TypeError, ValueError):
        return None, format_input({"tool_input": redact_secrets(parsed_tool_call)})
    tool_input = parsed.get("input") if isinstance(parsed, dict) else None
    if not isinstance(tool_input, dict):
        tool_input = {}
    name = tool_input.get("name")
    return (name if isinstance(name, str) else None), format_input({"tool_input": tool_input})


def _calls_after(
    conn: sqlite3.Connection,
    event: LoadEvent,
    session_id: str,
    end: float,
    max_calls: int,
) -> tuple[list[CallRecord], int]:
    """The run's calls after the load, the load's own ``skill`` call excepted."""
    rows = conn.execute(
        "SELECT tool_name, success, error_type, error_detail, parsed_tool_call "
        "FROM tool_calls WHERE session_id = ? AND timestamp > ? AND timestamp < ? "
        "AND tool_name != ? ORDER BY timestamp, rowid",
        (session_id, event.timestamp, end, SYNTHETIC_TOOL_NAME),
    ).fetchall()
    wanted = {event.skill.lower()} | ({event.file.lower()} if event.file else set())
    calls: list[CallRecord] = []
    own_call_skipped = False
    shown_limit = max(0, max_calls)
    total = 0
    for tool, success, error_type, error_detail, parsed in rows:
        requested, input_text = _call_input(parsed)
        if (
            not own_call_skipped and tool == "skill"
            and requested is not None and requested.lower() in wanted
        ):
            # The load is written inside the skill tool; its tool_calls row
            # lands a moment later. It is the load itself, not a call after it.
            own_call_skipped = True
            continue
        total += 1
        if len(calls) >= shown_limit:
            continue
        error_text = ""
        if not success and error_detail:
            error_text = _cut(redact_secrets(" ".join(str(error_detail).split())), ERROR_CHARS)
        calls.append(CallRecord(
            tool=str(tool),
            ok=bool(success),
            error_type=error_type or None,
            input_text=input_text,
            error_text=error_text,
        ))
    return calls, total - len(calls)


def _request(conn: sqlite3.Connection, session_id: str, load_ts: float) -> tuple[str | None, bool]:
    """``(request text, withheld)`` — the newest user row persisted before the load.

    An injected turn (a task result, provenance other than the user, stored
    untrusted) that started the run is withheld rather than shown: its text
    came from outside, and it is about to be put in front of a model that
    writes skills.
    """
    row = conn.execute(
        "SELECT content, is_trusted FROM lcm_messages "
        "WHERE session_id = ? AND role = 'user' AND timestamp < ? "
        "ORDER BY timestamp DESC, rowid DESC LIMIT 1",
        (session_id, load_ts),
    ).fetchone()
    if row is None:
        return None, False
    content, is_trusted = row
    if not is_trusted:
        return None, True
    text = " ".join(redact_secrets(str(content or "")).split())
    return (_cut(text, REQUEST_CHARS) if text else None), False


def _reply(conn: sqlite3.Connection, session_id: str, load_ts: float, end: float) -> tuple[bool, str]:
    """``(found, reply text)`` — the last assistant row the run persisted.

    A turn's rows are persisted when it ends, which is before the session's
    next run starts, so the window up to *end* holds them.
    """
    row = conn.execute(
        "SELECT content FROM lcm_messages "
        "WHERE session_id = ? AND role = 'assistant' AND timestamp > ? AND timestamp < ? "
        "ORDER BY timestamp DESC, rowid DESC LIMIT 1",
        (session_id, load_ts, end),
    ).fetchone()
    if row is None:
        return False, ""
    text = " ".join(redact_secrets(str(row[0] or "")).split())
    return True, _cut(text, REPLY_CHARS)


def run_for_load(
    telemetry: sqlite3.Connection,
    lcm: sqlite3.Connection | None,
    event: LoadEvent,
    *,
    max_calls: int = MAX_CALLS_PER_RUN,
) -> LoadRun | None:
    """The run around *event*, or None when the load names no session.

    A load without a session (an ephemeral turn, or a row from before loads
    carried one) cannot be tied to a run, so it is not evidence. Without an
    LCM connection the request and the reply are unknown, not empty.
    """
    session_id = event.session_id
    if not session_id:
        return None
    end, bounded = _run_end(telemetry, session_id, event.timestamp)
    calls, not_shown = _calls_after(telemetry, event, session_id, end, max_calls)
    request: str | None = None
    withheld = False
    replied: bool | None = None
    reply = ""
    if lcm is not None:
        try:
            request, withheld = _request(lcm, session_id, event.timestamp)
            found, reply = _reply(lcm, session_id, event.timestamp, end)
            # No reply inside an unbounded window is not "no reply": the run
            # may have outlasted the cap, or not have ended yet.
            replied = True if found else (False if bounded else None)
        except sqlite3.Error:
            log.warning("GEPA: conversation store unreadable for one run", exc_info=True)
    return LoadRun(
        load_ts=event.timestamp,
        session_id=session_id,
        request=request,
        request_withheld=withheld,
        calls=tuple(calls),
        calls_not_shown=not_shown,
        replied=replied,
        reply=reply,
        bounded=bounded,
    )


# ── the summary a proposal carries ───────────────────────────────────


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


@dataclass
class EvidenceSummary:
    """What a variant was judged on, as counts: no request, call input or reply text."""

    runs: int = 0
    sessions: int = 0
    loads_counted: int = 0
    calls_after_load: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)
    runs_with_reply: int = 0
    runs_without_reply: int = 0
    runs_end_unknown: int = 0
    runs_request_withheld: int = 0
    tools: dict[str, int] = field(default_factory=dict)
    first_load: str = ""
    last_load: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "sessions": self.sessions,
            "loads_counted": self.loads_counted,
            "calls_after_load": self.calls_after_load,
            "outcomes": dict(self.outcomes),
            "runs_with_reply": self.runs_with_reply,
            "runs_without_reply": self.runs_without_reply,
            "runs_end_unknown": self.runs_end_unknown,
            "runs_request_withheld": self.runs_request_withheld,
            "tools": dict(self.tools),
            "first_load": self.first_load,
            "last_load": self.last_load,
        }


def summarize(runs: list[LoadRun], *, loads_counted: int) -> EvidenceSummary:
    """Aggregate *runs* into counts. Shown calls only: the tail a run cut is not counted by kind."""
    outcomes: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    for run in runs:
        for call in run.calls:
            outcomes[call.kind] += 1
            tools[call.tool] += 1
    return EvidenceSummary(
        runs=len(runs),
        sessions=len({r.session_id for r in runs}),
        loads_counted=loads_counted,
        calls_after_load=sum(len(r.calls) + r.calls_not_shown for r in runs),
        outcomes=dict(sorted(outcomes.items())),
        runs_with_reply=sum(1 for r in runs if r.replied is True),
        runs_without_reply=sum(1 for r in runs if r.replied is False),
        runs_end_unknown=sum(1 for r in runs if r.replied is None),
        runs_request_withheld=sum(1 for r in runs if r.request_withheld),
        tools=dict(tools.most_common(10)),
        first_load=_day(min(r.load_ts for r in runs)) if runs else "",
        last_load=_day(max(r.load_ts for r in runs)) if runs else "",
    )
