"""Enrolled API devices (GRAFT-MOBILE-BRIDGE 1).

One row per enrolled client (a phone, a laptop). The token itself is NEVER
stored — only its SHA-256 — so this file leaking costs an attacker nothing
they can present. Minting returns the plaintext token exactly once.

Revocation is a tombstone (``revoked_at``), not a delete: the row remains
listable so "what was enrolled and when did it die" stays answerable, and a
revoked token can never be re-minted into validity by accident.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from prometheus.config.paths import get_devices_db_path

# last_seen_at is stamped at most this often per device — a hot client must
# not turn every request into a write.
LAST_SEEN_THROTTLE_SECONDS = 60.0


def token_digest(token: str) -> str:
    """The stored form of a device token: SHA-256 hex."""
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass(frozen=True)
class DeviceRow:
    id: str
    name: str
    platform: str
    created_at: float
    last_seen_at: float | None
    revoked_at: float | None


@dataclass(frozen=True)
class PushTarget:
    """What the APNs sender needs about one device — separate from DeviceRow
    on purpose: the REST device listing must never carry the APNs token."""

    id: str
    apns_token: str
    environment: str
    bundle_id: str
    push_failures: int


class DeviceStore:
    """SQLite-backed device registry. Safe for cross-thread use the same way
    the LCM stores are (``check_same_thread=False``; callers serialize)."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path if db_path is not None else get_devices_db_path()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS api_devices (
              id           TEXT PRIMARY KEY,
              name         TEXT NOT NULL,
              platform     TEXT NOT NULL,
              token_sha256 TEXT NOT NULL UNIQUE,
              created_at   REAL NOT NULL,
              last_seen_at REAL,
              revoked_at   REAL
            );
        """)
        self._conn.commit()
        self._migrate_push_columns()
        # Throttle memory: device_id -> monotonic-ish wall time of last stamp.
        self._last_touch: dict[str, float] = {}

    def _migrate_push_columns(self) -> None:
        """GRAFT Piece 2: push registration lives on the device row. ALTER is
        additive and idempotent-by-check — a Piece-1 devices.db gains the
        columns on first open, a fresh db already has them from here."""
        have = {r["name"] for r in self._conn.execute("PRAGMA table_info(api_devices)")}
        for column, decl in (
            ("apns_token", "TEXT"),
            ("apns_environment", "TEXT"),
            ("apns_bundle_id", "TEXT"),
            ("push_failures", "INTEGER DEFAULT 0"),
        ):
            if column not in have:
                self._conn.execute(f"ALTER TABLE api_devices ADD COLUMN {column} {decl}")
        # Live Activity per-activity push tokens (device × session). Its own
        # table: a device runs at most a handful of live activities, each with
        # a token ActivityKit rotates, and a stale row must be droppable
        # without touching the device row.
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS activity_tokens (
              device_id      TEXT NOT NULL,
              session_id     TEXT NOT NULL,
              activity_token TEXT NOT NULL,
              updated_at     REAL NOT NULL,
              PRIMARY KEY (device_id, session_id)
            );
        """)
        self._conn.commit()

    # ------------------------------------------------------------------

    def mint(self, name: str, platform: str) -> dict:
        """Enrol a device. Returns the ONLY copy of the plaintext token that
        will ever exist — the store keeps the digest."""
        import secrets

        device_id = uuid.uuid4().hex
        token = secrets.token_urlsafe(32)
        now = time.time()
        self._conn.execute(
            "INSERT INTO api_devices (id, name, platform, token_sha256, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (device_id, name, platform, token_digest(token), now),
        )
        self._conn.commit()
        return {"id": device_id, "name": name, "platform": platform,
                "token": token, "created_at": now}

    def lookup(self, digest: str) -> DeviceRow | None:
        """The live (non-revoked) device for a token digest, or None."""
        row = self._conn.execute(
            "SELECT * FROM api_devices WHERE token_sha256 = ? AND revoked_at IS NULL",
            (digest,),
        ).fetchone()
        return self._row(row) if row else None

    def touch(self, device_id: str) -> None:
        """Stamp last_seen_at, at most once per LAST_SEEN_THROTTLE_SECONDS."""
        now = time.time()
        if now - self._last_touch.get(device_id, 0.0) < LAST_SEEN_THROTTLE_SECONDS:
            return
        self._last_touch[device_id] = now
        self._conn.execute(
            "UPDATE api_devices SET last_seen_at = ? WHERE id = ?", (now, device_id)
        )
        self._conn.commit()

    def list_devices(self) -> list[DeviceRow]:
        rows = self._conn.execute(
            "SELECT * FROM api_devices ORDER BY created_at"
        ).fetchall()
        return [self._row(r) for r in rows]

    def revoke(self, device_id: str) -> bool:
        """Tombstone a device. True if a live row was revoked; False for an
        unknown id. Revoking an already-revoked device is True (idempotent)."""
        exists = self._conn.execute(
            "SELECT revoked_at FROM api_devices WHERE id = ?", (device_id,)
        ).fetchone()
        if exists is None:
            return False
        if exists["revoked_at"] is None:
            self._conn.execute(
                "UPDATE api_devices SET revoked_at = ? WHERE id = ?",
                (time.time(), device_id),
            )
            self._conn.commit()
        return True

    # ------------------------------------------------------------------
    # Push registration (GRAFT Piece 2)
    # ------------------------------------------------------------------

    def set_push(self, device_id: str, apns_token: str, environment: str,
                 bundle_id: str) -> bool:
        """Register (or replace) a device's APNs token. False for an unknown
        or revoked device — a tombstone must not be re-armable for push."""
        cur = self._conn.execute(
            "UPDATE api_devices SET apns_token = ?, apns_environment = ?,"
            " apns_bundle_id = ?, push_failures = 0"
            " WHERE id = ? AND revoked_at IS NULL",
            (apns_token, environment, bundle_id, device_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def clear_push(self, device_id: str) -> bool:
        """Drop a device's push registration (user disabled notifications, or
        Apple said 410 Unregistered). True if the device exists at all."""
        cur = self._conn.execute(
            "UPDATE api_devices SET apns_token = NULL, apns_environment = NULL,"
            " apns_bundle_id = NULL, push_failures = 0 WHERE id = ?",
            (device_id,),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def record_push_failure(self, device_id: str) -> int:
        """Increment and return the device's consecutive push failure count."""
        self._conn.execute(
            "UPDATE api_devices SET push_failures = COALESCE(push_failures, 0) + 1"
            " WHERE id = ?", (device_id,),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT push_failures FROM api_devices WHERE id = ?", (device_id,)
        ).fetchone()
        return int(row["push_failures"]) if row else 0

    def reset_push_failures(self, device_id: str) -> None:
        self._conn.execute(
            "UPDATE api_devices SET push_failures = 0 WHERE id = ?", (device_id,)
        )
        self._conn.commit()

    def push_targets(self) -> list[PushTarget]:
        """Live (non-revoked) devices with a push registration. The APNs token
        is deliberately NOT on DeviceRow: GET /api/devices must never leak it."""
        rows = self._conn.execute(
            "SELECT id, apns_token, apns_environment, apns_bundle_id, push_failures"
            " FROM api_devices"
            " WHERE revoked_at IS NULL AND apns_token IS NOT NULL"
        ).fetchall()
        return [PushTarget(id=r["id"], apns_token=r["apns_token"],
                           environment=r["apns_environment"] or "production",
                           bundle_id=r["apns_bundle_id"] or "",
                           push_failures=int(r["push_failures"] or 0))
                for r in rows]

    # ------------------------------------------------------------------
    # Live Activity tokens (GRAFT Piece 2)
    # ------------------------------------------------------------------

    def set_activity_token(self, device_id: str, session_id: str, token: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO activity_tokens"
            " (device_id, session_id, activity_token, updated_at) VALUES (?, ?, ?, ?)",
            (device_id, session_id, token, time.time()),
        )
        self._conn.commit()

    def clear_activity_token(self, device_id: str, session_id: str) -> None:
        self._conn.execute(
            "DELETE FROM activity_tokens WHERE device_id = ? AND session_id = ?",
            (device_id, session_id),
        )
        self._conn.commit()

    def activity_targets(self, session_id: str) -> list[tuple[PushTarget, str]]:
        """(push target, activity_token) pairs for a session's live activities.
        Joined on live push-registered devices: a revoked or push-cleared
        device's activity token is unreachable and simply drops out."""
        rows = self._conn.execute(
            "SELECT d.id, d.apns_token, d.apns_environment, d.apns_bundle_id,"
            "       d.push_failures, a.activity_token"
            "  FROM activity_tokens a JOIN api_devices d ON d.id = a.device_id"
            " WHERE a.session_id = ? AND d.revoked_at IS NULL AND d.apns_token IS NOT NULL",
            (session_id,),
        ).fetchall()
        return [(PushTarget(id=r["id"], apns_token=r["apns_token"],
                            environment=r["apns_environment"] or "production",
                            bundle_id=r["apns_bundle_id"] or "",
                            push_failures=int(r["push_failures"] or 0)),
                 r["activity_token"]) for r in rows]

    # ------------------------------------------------------------------

    @staticmethod
    def _row(row: sqlite3.Row) -> DeviceRow:
        return DeviceRow(
            id=row["id"], name=row["name"], platform=row["platform"],
            created_at=row["created_at"], last_seen_at=row["last_seen_at"],
            revoked_at=row["revoked_at"],
        )

    def close(self) -> None:
        self._conn.close()
