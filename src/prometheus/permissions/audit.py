"""Permission Audit Logger — persistent trail of security gate decisions.

Every ALLOW/DENY/CONFIRM decision is logged to:
1. SQLite table (queryable history)
2. JSONL file (append-only, grep-able)
3. Standard logger (immediate visibility)

Donor patterns:
- OpenClaw bash-tools.exec-approval-request.ts: structured approval tracking

Source: Prometheus (OAra Labs)
License: MIT
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
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
    # SPRINT-CONSENT Phase 3/4: an expired request is an OUTCOME, not an
    # absence. Without this row a timeout is indistinguishable from a
    # request nobody ever answered — which is the ambiguity that cost a
    # live probe to resolve.
    CONFIRM_TIMEOUT = "confirm_timeout"


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

    # ── Redaction, in two layers ───────────────────────────────────────────
    #
    # LAYER 1 — BY NAME. Precise, cheap, and INCOMPLETE BY CONSTRUCTION. It
    # masks a value because the identifier next to it contains one of five
    # words. Measured against the names this repo actually uses:
    #
    #     DOCS_TOKEN           masked      ("token")
    #     API_KEY              masked      ("api_key")
    #     MY_SECRET            masked      ("secret")
    #     DOCS_ENDPOINT        NOT masked
    #     DOCS_SEED            NOT masked
    #     UPSTREAM_PASSPHRASE  NOT masked  ("passphrase" is not "password")
    #
    # A longer word list moves the boundary; it does not remove it. The next
    # credential will be called something else, and this file already carries
    # the lesson from five name enumerations elsewhere in permissions/.
    _NAME_REDACT_PATTERNS: list[tuple[re.Pattern, str]] = [
        (re.compile(r'(api[_-]?key|token|secret|password|auth)["\']?\s*[:=]\s*["\']?[\w\-]+', re.I), r'\1=***'),
        (re.compile(r'Bearer\s+[\w\-\.]+', re.I), 'Bearer ***'),
        (re.compile(r'(sk-|pk-|xox[bpas]-|ghp_|gho_)[\w\-]+'), '***'),
    ]

    # LAYER 2 — BY SHAPE. Masks a value because of what the VALUE looks like
    # and where it sits, never because of what it is called. This is the layer
    # that catches DOCS_SEED and UPSTREAM_PASSPHRASE, and it catches the next
    # name nobody has thought of yet for the same reason.
    _SHAPE_REDACT_PATTERNS: list[tuple[re.Pattern, str]] = [
        # URL userinfo: scheme://user:password@host
        (re.compile(r'([A-Za-z][A-Za-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@'), r'\1***:***@'),
        # Any URL query parameter value, whatever the parameter is called.
        # `?access_token=`, `?sig=`, `?X-Amz-Signature=` — a tokenised URL is
        # a credential regardless of the vendor's choice of key name.
        (re.compile(r'([?&][A-Za-z0-9_.\-\[\]]+=)[^&\s"\'>]{8,}'), r'\1***'),
        # An environment-style assignment carrying an opaque value.
        # Shape on BOTH sides: an UPPER_SNAKE identifier (how env vars and
        # secrets are spelled) assigned something long and opaque. Lowercase
        # `--flag=value` and short values are left alone, so ordinary command
        # lines stay readable in the log.
        (re.compile(r'\b([A-Z][A-Z0-9_]{2,})=(["\']?)[A-Za-z0-9+/=_.:\-]{12,}\2'), r'\1=***'),
    ]

    #: Applied in order. Name layer first so its nicer `\1=***` replacement
    #: wins on the values it does recognise.
    _REDACT_PATTERNS: list[tuple[re.Pattern, str]] = (
        _NAME_REDACT_PATTERNS + _SHAPE_REDACT_PATTERNS
    )

    def __init__(
        self,
        data_dir: Path,
        max_input_chars: int = 200,
        max_reason_chars: int = 500,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.data_dir / "permission_audit.jsonl"
        self.db_path = self.data_dir / "audit.db"
        self.max_input_chars = max_input_chars
        # Looser than max_input_chars on purpose. A reason is prose written to
        # be read by a human deciding whether to approve, and clipping it at
        # 200 would cut real diagnostics; a tool input is machine payload where
        # the first 200 characters are enough to recognise the call. Both are
        # bounded, which is the property that matters — the number differs
        # because what the two fields are for differs.
        self.max_reason_chars = max_reason_chars

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

    def _scrub(self, text: str, limit: int) -> str:
        """Redact, then bound. The only way text enters a stored field.

        Truncation is not cosmetic and not a substitute for redaction — it is
        the second half of the same control. Redaction removes what it
        recognises; truncation bounds what it does not, so an unrecognised
        credential in a long command line is at least not stored in full.
        """
        text = self._redact(text)
        if len(text) > limit:
            text = text[:limit] + "..."
        return text

    def _summarize_input(self, tool_input: dict | str | None) -> str:
        """Create truncated, redacted summary of tool input."""
        if tool_input is None:
            return ""

        if isinstance(tool_input, dict):
            text = json.dumps(tool_input)
        else:
            text = str(tool_input)

        return self._scrub(text, self.max_input_chars)

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
            # ⚠ REASON IS SCRUBBED, AND THAT USED TO BE `reason=reason`.
            #
            # The gate builds its reason FROM the command:
            #     reason = f"Command requires approval: {command!r}"
            # so the same string reached this method twice — once as
            # `tool_input`, which was redacted and truncated, and once as
            # `reason`, which was neither. A curl with a bearer header and a
            # tokenised URL was stored masked in one column and verbatim in
            # the next, in the same row.
            #
            # System-origin curl / wget / ssh are exactly the commands that
            # reach the approve tier, and exactly the ones carrying
            # credentials. And this is not a write-only log: `audit_query`
            # returns these rows INTO MODEL CONTEXT, so an unredacted reason
            # re-feeds the secret to the model on every later query.
            reason=self._scrub(reason, self.max_reason_chars),
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
