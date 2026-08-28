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
        # Throttle memory: device_id -> monotonic-ish wall time of last stamp.
        self._last_touch: dict[str, float] = {}

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

    @staticmethod
    def _row(row: sqlite3.Row) -> DeviceRow:
        return DeviceRow(
            id=row["id"], name=row["name"], platform=row["platform"],
            created_at=row["created_at"], last_seen_at=row["last_seen_at"],
            revoked_at=row["revoked_at"],
        )

    def close(self) -> None:
        self._conn.close()
