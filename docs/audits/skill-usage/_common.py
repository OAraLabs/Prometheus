"""Shared helpers for the skill-usage audit scans.

Every scan runs ON THE MINI against a nightly snapshot opened
``mode=ro&immutable=1`` and prints aggregates only: counts, shares, sizes and
token totals. No skill names, skill bodies, message text or session ids are
printed unless a scan is started with its explicit ``--chat-only`` flag, whose
output is for the operator's terminal and is never committed.

Snapshot layout (see docs/audits/LCM-TURN-INDEX-DUPLICATES.md): telemetry is
the TOP-LEVEL ``telemetry.db``; LCM is ``data/lcm.db``.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def open_ro(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise SystemExit(f"missing: {path}")
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    return con


def telemetry(snap: Path) -> sqlite3.Connection:
    return open_ro(snap / "telemetry.db")


def lcm(snap: Path) -> sqlite3.Connection:
    return open_ro(snap / "data" / "lcm.db")


def week(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%G-W%V")


def month(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m")


def day(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d")


# Session-id prefixes that are people talking to the daemon. "web" alone is the
# routing namespace the web path filed rows under before #458 (a Beacon turn).
USER_SURFACES = frozenset({
    "beacon", "telegram", "desktop", "ios", "web", "voice", "slack", "discord",
})
TEST_SURFACES = frozenset({"smoke", "probe", "verify", "beacon-verify"})


def surface(session_id: str | None) -> str:
    """A coarse class for a session id: the surface, 'evals', 'coding', 'test' or 'none'."""
    if session_id is None:
        return "none (pre-2026-08-15, no session column)"
    if session_id == "system":
        return "evals (nightly, session 'system')"
    m = re.match(r"^([A-Za-z][A-Za-z_\-]*?)[:_]", session_id)
    prefix = m.group(1).lower() if m else session_id.lower()
    if prefix == "coding":
        return "coding runs"
    if prefix in TEST_SURFACES:
        return "test harness"
    if prefix in USER_SURFACES:
        return f"user: {prefix}"
    return "user: no prefix"


def is_user_surface(session_id: str | None) -> bool:
    return surface(session_id).startswith("user:")


def model_label(model: str | None, served: str | None = None) -> str:
    """Model identifier without host paths (a GGUF path keeps its file name)."""
    m = (model or "").strip()
    if not m:
        s = (served or "").strip()
        return f"<blank> (served {s.rsplit('/', 1)[-1]})" if s and s != "-" else "<blank>"
    return m.rsplit("/", 1)[-1]


def provider_of(label: str) -> str:
    low = label.lower()
    if low.startswith("<blank>") or low.endswith(".gguf") or low in {"gemma4-26b", "qwen3.8-27b"}:
        return "llama.cpp on the 4090 (local)"
    if low.startswith("qwen3.8-max") or low.startswith("qwen3.8-flash") or low.startswith("qwen3.7"):
        return "qwen (Alibaba cloud)"
    if low.startswith("grok"):
        return "xai (cloud)"
    if low.startswith("claude"):
        return "anthropic (cloud)"
    if ":" in low:
        return "ollama on the mini (local)"
    return "other"


def parsed_input(parsed_tool_call: str | None) -> dict[str, Any] | None:
    if not parsed_tool_call:
        return None
    try:
        data = json.loads(parsed_tool_call)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and isinstance(data.get("$json"), (dict, str)):
        data = data["$json"] if isinstance(data["$json"], dict) else json.loads(data["$json"])
    inp = data.get("input") if isinstance(data, dict) else None
    return inp if isinstance(inp, dict) else None


def blocks(content_json: str | None) -> list[dict[str, Any]]:
    if not content_json:
        return []
    try:
        data = json.loads(content_json)
    except json.JSONDecodeError:
        return []
    return [b for b in data if isinstance(b, dict)] if isinstance(data, list) else []


def human_text(role: str, provenance: str, content: str, content_json: str | None) -> str | None:
    """The text a PERSON sent, mirroring ``engine.agent_loop._human_message_from``:
    a user-role row with provenance 'user' whose text blocks are non-empty.
    Tool-result rows (user role, no text) return None."""
    if role != "user" or (provenance or "user") != "user":
        return None
    bl = blocks(content_json)
    if bl:
        if any(b.get("type") == "tool_result" for b in bl):
            return None
        text = "".join(str(b.get("text", "")) for b in bl if b.get("type") == "text")
    else:
        text = content or ""
    return text if text.strip() else None


def lcm_rows(con: sqlite3.Connection) -> Iterator[sqlite3.Row]:
    """Rows in persist order within each session (rowid), sessions interleaved by rowid."""
    yield from con.execute(
        "SELECT rowid AS rid, session_id, turn_index, role, provenance, content, content_json,"
        " timestamp FROM lcm_messages ORDER BY rowid"
    )
