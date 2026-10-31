"""Golden trace v2: the request a golden call answered, stored as it was sent.

A v1 golden trace kept the call and a ``session_id``; the export rebuilt the
input half from the LCM store, and that rebuild is lossy by construction (the
distillation audit, 2026-10-30): no system prompt, flat text only (earlier tool
calls and tool results dropped), the called tool's schema instead of the list
offered, at most 12 messages (79% of the corpus hit that cap), and no record of
which route served the call. The distillation had to rebuild every request
from ``lcm.db`` and today's tool registry, whose schemas may have drifted.

v2 records the ROUND'S REQUEST at the moment the call is recorded, for golden
calls only:

* ``messages`` — every message the provider was given, in the OpenAI chat wire
  shape the OpenAI-compatible providers send (``providers/stub.py``
  ``_build_openai_messages``): earlier assistant tool calls, tool results and
  all, after compaction, no cap. Images are replaced by a digest; their bytes
  are not training text.
* ``system`` — the system prompt, as sent that round.
* ``tools`` — the full tool list offered (OpenAI function shape), plus
  ``tool_set_hash`` (the names, joining telemetry's ``tool_sets``) and the
  build version that produced the schemas.
* ``route`` — requested model, the route key and provider the session chose,
  and whether a fallback served the round; with the serving provider and model.

STORAGE IS CONTENT-ADDRESSED. Consecutive rounds of one turn re-send the same
prefix, so each message, system prompt and tool list is a ``golden_blobs`` row
keyed by the hash of its (redacted) JSON, written once; a ``golden_requests``
row is the list of hashes. A 200-round session costs its messages once, not
200 times.

REDACTED BEFORE IT IS HASHED OR STORED, with the capture redactor every other
capture store uses (``security/log_redaction``): string values by value, and
each tool call's JSON ``arguments`` parsed and redacted value by value
(``redact_json_text``), so a token after an escaped newline is not missed.
``oara scrub`` covers ``golden_blobs.body``.

Readers: :func:`load_request` here, and ``ToolCallTelemetry``'s golden export,
which writes ``_meta.trace_format = 2`` lines (a superset of v1: same
``messages`` / ``tools`` / ``_meta`` keys, same meanings).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from prometheus.security.log_redaction import redact_capture, redact_json_text

log = logging.getLogger(__name__)

#: The trace format a request row records; v1 rows have no request.
TRACE_FORMAT = 2

#: The wire shape ``messages`` / ``tools`` are stored in.
WIRE = "openai_chat"

BLOB_SYSTEM = "system"
BLOB_TOOLS = "tools"
BLOB_MESSAGE = "message"

TABLES_SQL = """
-- GOLDEN TRACE v2 (telemetry/golden_request.py). Content-addressed parts of
-- the requests golden calls answered: one row per distinct message, system
-- prompt or tool list, keyed by the hash of its redacted JSON.
CREATE TABLE IF NOT EXISTS golden_blobs (
    hash        TEXT PRIMARY KEY,       -- sha256 of body, 32 hex
    kind        TEXT NOT NULL,          -- system / tools / message
    body        TEXT NOT NULL,          -- redacted JSON
    created_at  REAL NOT NULL
);

-- One row per distinct request; tool_calls.request_id points here.
CREATE TABLE IF NOT EXISTS golden_requests (
    id              TEXT PRIMARY KEY,   -- sha256 of the row's content, 32 hex
    created_at      REAL NOT NULL,
    trace_format    INTEGER NOT NULL,   -- 2
    wire            TEXT NOT NULL,      -- openai_chat
    system_hash     TEXT,               -- golden_blobs.hash; NULL = no system prompt
    tools_hash      TEXT,               -- golden_blobs.hash; NULL = no tools offered
    tool_set_hash   TEXT,               -- telemetry tool_sets key (the names)
    message_hashes  TEXT NOT NULL,      -- JSON array of golden_blobs.hash, in order
    message_count   INTEGER NOT NULL,
    requested_model TEXT,               -- the model the request named
    provider        TEXT,               -- the provider that SERVED the round
    route           TEXT,               -- JSON: key, provider, degraded, backend
    tool_choice     TEXT,               -- JSON
    max_tokens      INTEGER,
    build_version   TEXT                -- prometheus.__version__ that sent it
);
"""


@dataclass
class GoldenRequestCapture:
    """What the loop hands telemetry for one round: the request and its route.

    Serialized only if the call turns out golden, so a non-golden call costs
    nothing but building this object.
    """

    request: Any                      # providers.base.ApiMessageRequest
    provider: str = ""                # the provider that served the round
    route: dict[str, Any] = field(default_factory=dict)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:32]


def _canon(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _strip_images(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, list):
        return message
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            url = str((part.get("image_url") or {}).get("url", ""))
            parts.append({"type": "image_url",
                          "image_url": {"url": f"omitted:sha256:{_digest(url)[:16]}"}})
        else:
            parts.append(part)
    return {**message, "content": parts}


def _redact_message(message: dict[str, Any]) -> dict[str, Any]:
    message = _strip_images(message)
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        fixed = []
        for call in calls:
            fn = dict((call or {}).get("function") or {})
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = redact_json_text(fn["arguments"])
            fixed.append({**call, "function": fn})
        message = {**message, "tool_calls": fixed}
    return redact_capture(message)


def wire_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """The tool list in OpenAI function shape, as openai_compat sends it."""
    out = []
    for t in tools or []:
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        out.append({
            "type": "function",
            "function": {
                "name": fn.get("name", t.get("name", "")),
                "description": fn.get("description", t.get("description", "")),
                "parameters": fn.get(
                    "parameters", t.get("input_schema", t.get("parameters", {}))),
            },
        })
    return out


def serialize(capture: GoldenRequestCapture) -> dict[str, Any]:
    """The capture as redacted wire JSON: ``system``, ``messages``, ``tools`` and metadata."""
    from prometheus.providers.base import ApiMessageRequest
    from prometheus.providers.stub import _build_openai_messages
    from prometheus.telemetry.writer import tool_set_row

    request = capture.request
    # The system prompt goes on its own; the messages are built without it.
    bare = ApiMessageRequest(model=request.model, messages=list(request.messages),
                             system_prompt=None, tools=[])
    messages = [_redact_message(m) for m in _build_openai_messages(bare, allow_images=True)]
    tools = [] if getattr(request, "suppress_tools", False) else wire_tools(request.tools)
    names = [t["function"]["name"] for t in tools]
    build_version: str | None
    try:
        from prometheus import __version__ as build_version
    except Exception:  # pragma: no cover - a source tree always has one
        build_version = None
    return {
        "system": redact_capture(request.system_prompt) if request.system_prompt else None,
        "messages": messages,
        "tools": redact_capture(tools) if tools else None,
        "tool_set_hash": tool_set_row(names)["tool_set_hash"] if names else None,
        "requested_model": request.model,
        "provider": capture.provider or None,
        "route": capture.route or None,
        "tool_choice": getattr(request, "tool_choice", None),
        "max_tokens": getattr(request, "max_tokens", None),
        "build_version": build_version,
    }


class GoldenRequestStore:
    """Writes and reads ``golden_blobs`` / ``golden_requests`` on a telemetry connection.

    Keeps the hashes it has written, so the prefix a turn re-sends every round
    is not re-inserted every round. The caller owns the transaction.
    """

    MAX_KNOWN = 50_000

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._known: set[str] = set()

    def _blob(self, kind: str, value: Any, now: float) -> str:
        body = _canon(value)
        h = _digest(body)
        if h not in self._known:
            self._conn.execute(
                "INSERT OR IGNORE INTO golden_blobs (hash, kind, body, created_at) VALUES (?, ?, ?, ?)",
                (h, kind, body, now))
            if len(self._known) >= self.MAX_KNOWN:
                self._known.clear()
            self._known.add(h)
        return h

    def store(self, capture: GoldenRequestCapture) -> str:
        """Store one request (inside the caller's transaction). Returns its id."""
        data = serialize(capture)
        now = time.time()
        system_hash = self._blob(BLOB_SYSTEM, data["system"], now) if data["system"] else None
        tools_hash = self._blob(BLOB_TOOLS, data["tools"], now) if data["tools"] else None
        message_hashes = [self._blob(BLOB_MESSAGE, m, now) for m in data["messages"]]
        row = {
            "trace_format": TRACE_FORMAT,
            "wire": WIRE,
            "system_hash": system_hash,
            "tools_hash": tools_hash,
            "tool_set_hash": data["tool_set_hash"],
            "message_hashes": json.dumps(message_hashes),
            "message_count": len(message_hashes),
            "requested_model": data["requested_model"],
            "provider": data["provider"],
            "route": _canon(data["route"]) if data["route"] else None,
            "tool_choice": _canon(data["tool_choice"]) if data["tool_choice"] is not None else None,
            "max_tokens": data["max_tokens"],
            "build_version": data["build_version"],
        }
        request_id = _digest(_canon(row))
        if request_id not in self._known:
            cols = ["id", "created_at", *row]
            self._conn.execute(
                f"INSERT OR IGNORE INTO golden_requests ({', '.join(cols)}) "
                f"VALUES ({', '.join('?' for _ in cols)})",
                (request_id, now, *row.values()))
            self._known.add(request_id)
        return request_id


def load_request(conn: sqlite3.Connection, request_id: str) -> dict[str, Any] | None:
    """A stored request: ``system``, ``messages``, ``tools`` and its metadata, or None.

    None when the row is missing or any part of it is (a part can never be
    partly right: an example built from a request with a hole in it would be
    a request nobody sent).
    """
    try:
        row = conn.execute(
            "SELECT trace_format, wire, system_hash, tools_hash, tool_set_hash, message_hashes,"
            " requested_model, provider, route, tool_choice, max_tokens, build_version"
            " FROM golden_requests WHERE id = ?", (request_id,)).fetchone()
    except sqlite3.DatabaseError:
        return None
    if row is None:
        return None
    (fmt, wire, system_hash, tools_hash, tool_set_hash, message_hashes, requested_model,
     provider, route, tool_choice, max_tokens, build_version) = row
    hashes = json.loads(message_hashes or "[]")
    wanted = [h for h in (system_hash, tools_hash, *hashes) if h]
    bodies: dict[str, Any] = {}
    for i in range(0, len(wanted), 500):
        chunk = wanted[i:i + 500]
        for h, body in conn.execute(
            f"SELECT hash, body FROM golden_blobs WHERE hash IN ({', '.join('?' for _ in chunk)})",
            tuple(chunk)):
            bodies[h] = json.loads(body)
    if any(h not in bodies for h in wanted):
        log.warning("golden request %s: %d part(s) missing", request_id,
                    sum(1 for h in wanted if h not in bodies))
        return None
    return {
        "request_id": request_id,
        "trace_format": fmt,
        "wire": wire,
        "system": bodies[system_hash] if system_hash else None,
        "system_hash": system_hash,
        "messages": [bodies[h] for h in hashes],
        "tools": bodies[tools_hash] if tools_hash else [],
        "tools_hash": tools_hash,
        "tool_set_hash": tool_set_hash,
        "requested_model": requested_model,
        "provider": provider,
        "route": json.loads(route) if route else None,
        "tool_choice": json.loads(tool_choice) if tool_choice else None,
        "max_tokens": max_tokens,
        "build_version": build_version,
    }


def read_example(line: dict[str, Any]) -> dict[str, Any]:
    """One golden-trace JSONL line, v1 or v2, in one shape. Raises ValueError on a malformed one.

    Returns ``trace_format`` (1 when the line names none), ``system`` (v2;
    None on v1), ``context`` (the messages before the target, system
    excluded), ``target`` (the assistant turn carrying the call), ``tools``,
    ``meta`` and ``labels`` (``{}`` on a line exported before the labels).
    The way to read the corpus that does not care which build exported it.
    """
    messages = line.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("a golden trace line has no messages")
    target = messages[-1]
    if not isinstance(target, dict) or target.get("role") != "assistant" \
            or not target.get("tool_calls"):
        raise ValueError("a golden trace line must end in an assistant tool call")
    raw_meta = line.get("_meta")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    context = list(messages[:-1])
    system = None
    if context and isinstance(context[0], dict) and context[0].get("role") == "system":
        system = context.pop(0).get("content")
    return {
        "trace_format": int(meta.get("trace_format") or 1),
        "system": system,
        "context": context,
        "target": target,
        "tools": list(line.get("tools") or []),
        "meta": meta,
        "labels": dict(meta.get("labels") or {}),
    }
