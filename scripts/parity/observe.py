"""Read back everything the daemon persisted, as plain JSON.

Generic on purpose: every SQLite database and every small text file under the
isolated root is dumped, not a hand-picked list of tables. A hand-picked list
is blind to exactly the regression a seam refactor is most likely to cause —
a write that moved to a new table, or a new write nobody asked for. The
categorised view the report prints (tool calls, gate decisions, checkpoints,
memory, telemetry, reply) is a LENS over this dump, never a filter on it.

Read after the daemon has exited, so every writer has flushed and nothing is
mid-transaction. SQLite files are opened read-only through the URI form.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

SQLITE_MAGIC = b"SQLite format 3\x00"

# Files that are not side effects of a turn. Each is here for a stated reason.
SKIP_SUFFIXES = (
    "-wal", "-shm", "-journal",   # SQLite sidecars; the main file is dumped after a clean exit
    ".log",                       # process logs: interleaving is scheduling, not behavior
    ".pid", ".lock",              # process bookkeeping
    ".pyc",
)
SKIP_NAMES = {".parity-root"}      # the harness's own marker, not a daemon write
SKIP_PARTS = {
    "__pycache__",
    "logs",                       # daemon/log output directory (see .log)
}
TEXT_LIMIT = 256 * 1024           # larger text files are compared by digest


def _is_sqlite(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(16) == SQLITE_MAGIC
    except OSError:
        return False


def _cell(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"$bytes_sha256": hashlib.sha256(value).hexdigest()[:16], "len": len(value)}
    return value


def dump_sqlite(path: Path) -> dict[str, Any]:
    uri = f"file:{path}?mode=ro"
    out: dict[str, Any] = {}
    con = sqlite3.connect(uri, uri=True)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        for table in tables:
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]
            # FTS shadow tables and virtual tables have no stable rowid order
            # worth trusting; order by every column instead, which is total.
            try:
                rows = con.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
            except sqlite3.OperationalError:
                order = ", ".join(f'"{c}"' for c in cols) or "1"
                rows = con.execute(f'SELECT * FROM "{table}" ORDER BY {order}').fetchall()
            out[table] = {
                "columns": cols,
                "rows": [[_cell(v) for v in row] for row in rows],
            }
    finally:
        con.close()
    return out


def snapshot(root: Path) -> dict[str, Any]:
    """{relative path: dump} for every persisted artefact under ``root``."""
    result: dict[str, Any] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if any(part in SKIP_PARTS for part in path.relative_to(root).parts):
            continue
        if rel.endswith(SKIP_SUFFIXES) or path.name in SKIP_NAMES:
            continue
        if _is_sqlite(path):
            try:
                result[rel] = {"sqlite": dump_sqlite(path)}
            except sqlite3.DatabaseError as exc:
                result[rel] = {"sqlite_error": str(exc)}
            continue
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            result[rel] = {"binary_sha256": hashlib.sha256(data).hexdigest()[:16],
                           "len": len(data)}
            continue
        if len(data) > TEXT_LIMIT:
            result[rel] = {"text_sha256": hashlib.sha256(data).hexdigest()[:16],
                           "len": len(data)}
        elif rel.endswith(".json"):
            try:
                result[rel] = {"json": json.loads(text)}
            except json.JSONDecodeError:
                result[rel] = {"text": text}
        elif rel.endswith(".jsonl"):
            lines = []
            for line in text.splitlines():
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    lines.append(line)
            result[rel] = {"jsonl": lines}
        else:
            result[rel] = {"text": text}
    return result


def tree(root: Path) -> dict[str, str]:
    """A workspace's files → content digest. For checkpoint/undo assertions."""
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts:
            continue
        out[path.relative_to(root).as_posix()] = hashlib.sha256(
            path.read_bytes()).hexdigest()[:16]
    return out
