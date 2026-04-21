"""Permission Audit Logger — persistent trail of security gate decisions.

Every ALLOW/DENY/CONFIRM decision is logged to:
1. SQLite table (queryable history)
2. JSONL file (append-only, grep-able)
3. Standard logger (immediate visibility)

Donor patterns:
- OpenClaw bash-tools.exec-approval-request.ts: structured approval tracking

Source: Prometheus (OAra AI Lab)
License: MIT
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AuditDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    CONFIRM_PENDING = "confirm_pending"
    CONFIRM_APPROVED = "confirm_approved"
    CONFIRM_REJECTED = "confirm_rejected"


@dataclass
class AuditEntry:
    """A single security gate decision."""

    timestamp: float
    tool_name: str
    decision: AuditDecision
    trust_level: int
    reason: str
    tool_input_summary: str = ""
    user_id: str | None = None
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        d["timestamp_iso"] = datetime.fromtimestamp(self.timestamp).isoformat()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class AuditLogger:
    """Persistent audit log for security decisions."""

    # Patterns to redact from logged content
    _REDACT_PATTERNS: list[tuple[re.Pattern, str]] = [
        (re.compile(r'(api[_-]?key|token|secret|password|auth)["\']?\s*[:=]\s*["\']?[\w\-]+', re.I), r'\1=***'),
        (re.compile(r'Bearer\s+[\w\-\.]+', re.I), 'Bearer ***'),
        (re.compile(r'(sk-|pk-|xox[bpas]-|ghp_|gho_)[\w\-]+'), '***'),
    ]

    def __init__(self, data_dir: Path, max_input_chars: int = 200) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.data_dir / "permission_audit.jsonl"
        self.db_path = self.data_dir / "audit.db"
        self.max_input_chars = max_input_chars

        self._init_db()

    def _init_db(self) -> None:
        """Create audit table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS permission_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    tool_name TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    trust_level INTEGER NOT NULL,
                    reason TEXT,
                    tool_input_summary TEXT,
                    user_id TEXT,
                    session_id TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_ts ON permission_audit(timestamp DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_decision ON permission_audit(decision)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_tool ON permission_audit(tool_name)"
            )

    def _redact(self, text: str) -> str:
        """Redact potential secrets from text before logging."""
        for pattern, replacement in self._REDACT_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def _summarize_input(self, tool_input: dict | str | None) -> str:
        """Create truncated, redacted summary of tool input."""
        if tool_input is None:
            return ""

        if isinstance(tool_input, dict):
            text = json.dumps(tool_input)
        else:
            text = str(tool_input)

        text = self._redact(text)

        if len(text) > self.max_input_chars:
            text = text[: self.max_input_chars] + "..."

        return text

    def log(
        self,
        tool_name: str,
        decision: AuditDecision,
        trust_level: int,
        reason: str,
        tool_input: dict | str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> AuditEntry:
        """Log a security decision to JSONL + SQLite + standard logger."""
        entry = AuditEntry(
            timestamp=time.time(),
            tool_name=tool_name,
            decision=decision,
            trust_level=trust_level,
            reason=reason,
            tool_input_summary=self._summarize_input(tool_input),
            user_id=user_id,
            session_id=session_id,
        )

        # Append to JSONL (append-only, crash-safe)
        try:
            with open(self.log_file, "a") as f:
                f.write(entry.to_json() + "\n")
        except OSError as exc:
            logger.warning("Failed to write audit JSONL: %s", exc)

        # Insert into SQLite
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO permission_audit
                       (timestamp, tool_name, decision, trust_level, reason,
                        tool_input_summary, user_id, session_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entry.timestamp, entry.tool_name, entry.decision.value,
                        entry.trust_level, entry.reason, entry.tool_input_summary,
                        entry.user_id, entry.session_id,
                    ),
                )
        except sqlite3.Error as exc:
            logger.warning("Failed to write audit SQLite: %s", exc)

        # Standard logger
        level = logging.WARNING if decision == AuditDecision.DENY else logging.INFO
        logger.log(level, "[AUDIT] %s: %s - %s", decision.value.upper(), tool_name, reason)

        return entry

    def query_recent(
        self,
        limit: int = 50,
        decision: AuditDecision | None = None,
        tool_name: str | None = None,
    ) -> list[AuditEntry]:
        """Query recent audit entries from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM permission_audit WHERE 1=1"
            params: list[Any] = []

            if decision:
                query += " AND decision = ?"
                params.append(decision.value)

            if tool_name:
                query += " AND tool_name = ?"
                params.append(tool_name)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()

        return [
            AuditEntry(
                timestamp=row["timestamp"],
                tool_name=row["tool_name"],
                decision=AuditDecision(row["decision"]),
                trust_level=row["trust_level"],
                reason=row["reason"],
                tool_input_summary=row["tool_input_summary"] or "",
                user_id=row["user_id"],
                session_id=row["session_id"],
            )
            for row in rows
        ]

    def stats(self, hours: int = 24) -> dict[str, int]:
        """Get decision counts for recent period."""
        cutoff = time.time() - (hours * 3600)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT decision, COUNT(*) as count
                   FROM permission_audit
                   WHERE timestamp > ?
                   GROUP BY decision""",
                (cutoff,),
            ).fetchall()

        return {row[0]: row[1] for row in rows}
