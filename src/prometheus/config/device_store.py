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
import logging
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from prometheus.config.paths import get_devices_db_path

logger = logging.getLogger(__name__)

# last_seen_at is stamped at most this often per device — a hot client must
# not turn every request into a write.
LAST_SEEN_THROTTLE_SECONDS = 60.0


def registry_platform(value: object) -> str:
    """The platform the device registry keeps for a device: ``ios``, ``macos`` or ``other``.

    One narrowing for every enrolment path (``POST /api/devices`` and a pairing approval), so a device
    called "windows" is ``other`` in the registry however it arrived.
    """
    platform = str(value or "").strip().lower()
    return platform if platform in ("ios", "macos") else "other"


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
    # #348. last_seen_at is the DEVICE talking to us; these are us reaching the device — a
    # phone can be seen every minute and have received nothing for a week.
    last_push_at: float | None = None
    last_push_status: str | None = None


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
        # Called with the ids of devices that were just revoked (see add_revoke_listener).
        self._revoke_listeners: list[Callable[[list[str]], None]] = []

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
            # #348: a FAILED push was recorded (push_failures) and a SUCCESSFUL one was not,
            # so push_failures == 0 meant "delivered" and "never attempted" equally. These are
            # the positive half; they do not replace the counter.
            ("last_push_at", "REAL"),
            ("last_push_status", "TEXT"),
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
        minted = self._new_device(name, platform)
        self._conn.commit()
        return minted

    def _new_device(self, name: str, platform: str) -> dict:
        """INSERT one device row and return its mint record. Does NOT commit: the caller owns the
        transaction (mint commits at once; mint_owner commits the device and its owner mark together)."""
        import secrets

        device_id = uuid.uuid4().hex
        token = secrets.token_urlsafe(32)
        now = time.time()
        self._conn.execute(
            "INSERT INTO api_devices (id, name, platform, token_sha256, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (device_id, name, platform, token_digest(token), now),
        )
        return {"id": device_id, "name": name, "platform": platform,
                "token": token, "created_at": now}

    @property
    def connection(self) -> sqlite3.Connection:
        """The registry's SQLite connection, for a store that must commit TOGETHER with a device row.

        Pairing requests (``config/pair_requests.py``) live in this file and approve in one transaction
        with the device they mint. Not for general use: anything that writes through it owns the
        transaction and the lazily created table it writes to.
        """
        return self._conn

    def mint_in_transaction(self, name: str, platform: str) -> dict:
        """Enrol an ORDINARY (scoped) device inside a transaction the CALLER owns and commits.

        The same row ``mint`` writes, without the commit, so an approval can record itself and mint its
        device atomically: a failure after this rolls the device back, and there is never a live token
        nobody holds. There is no tier parameter, as with ``mint``: an owner device is only ever
        ``mint_owner`` (same-Mac pairing), never the product of an approval.
        """
        return self._new_device(name, platform)

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
            self._notify_revoked([device_id])
        return True

    def add_revoke_listener(self, listener: Callable[[list[str]], None]) -> None:
        """Be told, AFTER the commit, which devices were just revoked — by :meth:`revoke` or by an owner
        mint replacing earlier ones. The WebSocket bridge uses it to close a revoked token's open
        sockets: revocation is a tombstone and authentication only happens at connect, so without
        this a revoked device kept its live socket (and its frames) until it chose to leave.

        Listeners run synchronously on the revoking thread, in registration order. One that raises is
        logged and skipped: the revocation is already durable and nothing here may undo it, or the
        mint that triggered it."""
        self._revoke_listeners.append(listener)

    def _notify_revoked(self, device_ids: list[str]) -> None:
        if not device_ids:
            return
        for listener in tuple(self._revoke_listeners):
            try:
                listener(list(device_ids))
            except Exception:
                logger.warning("a device-revocation listener failed; the revocation stands",
                               exc_info=True)

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

    def record_push_success(self, device_id: str, status: int | None = None) -> None:
        """Stamp a DELIVERED push. The whole point of #348: without this, silence after a send
        is ambiguous between 'fine' and 'never happened', and a regression looks like health."""
        self._conn.execute(
            "UPDATE api_devices SET last_push_at = ?, last_push_status = ? WHERE id = ?",
            (time.time(), f"ok:{status}" if status is not None else "ok", device_id),
        )
        self._conn.commit()

    def record_push_attempt(self, device_id: str, outcome: str, status: int | None = None) -> None:
        """Stamp a NON-delivered attempt (failed / unregistered), so the row distinguishes
        'we tried and it did not land' from 'we never tried'."""
        self._conn.execute(
            "UPDATE api_devices SET last_push_at = ?, last_push_status = ? WHERE id = ?",
            (time.time(), f"{outcome}:{status}" if status is not None else outcome, device_id),
        )
        self._conn.commit()

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
    # Computer use: which devices a PERSON marked (computer-use v1.1, W3)
    # ------------------------------------------------------------------
    #
    # Its own table, created on the FIRST mark rather than at open: enrolment
    # needs only the global token (POST /api/devices), which a model can
    # read, so a device token alone is not a person — a person has to mark
    # it. A box that never uses computer use never grows the table (and the
    # parity fixtures, which record every table in this file, stay as they
    # are).

    def _has_computer_table(self) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='computer_devices'").fetchone() is not None

    def set_computer(self, device_id: str, on: bool, *, by: str) -> bool:
        """Mark (or unmark) a LIVE device for computer use. ``by`` names who
        did it (an approver label). False for an unknown or revoked id."""
        live = self._conn.execute(
            "SELECT 1 FROM api_devices WHERE id = ? AND revoked_at IS NULL",
            (device_id,)).fetchone()
        if live is None:
            return False
        if not on:
            if self._has_computer_table():
                self._conn.execute(
                    "DELETE FROM computer_devices WHERE device_id = ?",
                    (device_id,))
                self._conn.commit()
            return True
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS computer_devices (
              device_id  TEXT PRIMARY KEY,
              marked_at  REAL NOT NULL,
              marked_by  TEXT NOT NULL
            );
        """)
        self._conn.execute(
            "INSERT OR REPLACE INTO computer_devices (device_id, marked_at, "
            "marked_by) VALUES (?, ?, ?)", (device_id, time.time(), by))
        self._conn.commit()
        return True

    def computer_allowed(self, device_id: str) -> bool:
        """Is this LIVE device marked for computer use? A revoked device
        never is, whatever its mark says."""
        if not self._has_computer_table():
            return False
        row = self._conn.execute(
            "SELECT 1 FROM computer_devices c JOIN api_devices d "
            "ON d.id = c.device_id WHERE c.device_id = ? "
            "AND d.revoked_at IS NULL", (device_id,)).fetchone()
        return row is not None

    def computer_device_ids(self) -> set[str]:
        if not self._has_computer_table():
            return set()
        rows = self._conn.execute(
            "SELECT c.device_id FROM computer_devices c JOIN api_devices d "
            "ON d.id = c.device_id WHERE d.revoked_at IS NULL").fetchall()
        return {r["device_id"] for r in rows}

    # ------------------------------------------------------------------
    # Session ownership: which device brought a session into existence
    # ------------------------------------------------------------------
    #
    # A device token sees and manages only the sessions it owns; the operator's
    # global token sees all (web/session_scope.py is the policy, this is the
    # record). A session with no row here belongs to the operator — a Telegram
    # chat, or anything that predates device scoping.
    #
    # One row per session, first writer wins, never reassigned and never
    # deleted: ownership must not change under a session that is mid-turn, and
    # a purged or revoked-device session must not become claimable by someone
    # else. Revoking a device leaves its rows; the operator still reads them.
    # Anything that adds a way to REASSIGN a session (approve-to-pair handing
    # one over) must do it here, in one UPDATE, and nowhere else.
    #
    # Its own table, created on the FIRST claim rather than at open, like
    # computer_devices above: a daemon with no device activity never grows it,
    # and the parity fixtures, which record every table in this file, stay as
    # they are.

    def claim_session(self, session_id: str, device_id: str) -> bool:
        """Record *device_id* as the owner of *session_id*, if nobody owns it.

        True when the device owns the session afterwards (it already did, or this
        call took it); False when another device owns it or *device_id* is not a
        live device. Check that the session does not exist elsewhere BEFORE
        calling — this only arbitrates between devices; it cannot know that a
        Telegram chat already holds the id.
        """
        live = self._conn.execute(
            "SELECT 1 FROM api_devices WHERE id = ? AND revoked_at IS NULL",
            (device_id,)).fetchone()
        if live is None or not session_id:
            return False
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS device_sessions (
              session_id TEXT PRIMARY KEY,
              device_id  TEXT NOT NULL,
              claimed_at REAL NOT NULL
            );
        """)
        self._conn.execute(
            "INSERT OR IGNORE INTO device_sessions (session_id, device_id, claimed_at)"
            " VALUES (?, ?, ?)", (session_id, device_id, time.time()))
        self._conn.commit()
        return self.session_owner(session_id) == device_id

    def session_owner(self, session_id: str) -> str | None:
        """The device id that owns *session_id*, or None (the operator's)."""
        try:
            row = self._conn.execute(
                "SELECT device_id FROM device_sessions WHERE session_id = ?",
                (session_id,)).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc):  # a lock or I/O error is not "unowned"
                raise
            return None  # nothing was ever claimed
        return row["device_id"] if row else None

    def owned_session_ids(self, device_id: str) -> set[str]:
        """Every session id *device_id* owns."""
        try:
            rows = self._conn.execute(
                "SELECT session_id FROM device_sessions WHERE device_id = ?",
                (device_id,)).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc):
                raise
            return set()
        return {r["session_id"] for r in rows}

    # ------------------------------------------------------------------
    # The OWNER tier: the person's own device (P2)
    # ------------------------------------------------------------------
    #
    # Two ways to mint a device, on purpose, and the tier is not a parameter on either:
    #
    #   mint()        an ordinary device. Scoped to its own sessions (web/session_scope.py), cannot
    #                 approve anything. This is what a device approved from Telegram or by another
    #                 device gets.
    #   mint_owner()  the person's OWN device (same-Mac pairing, or the global token minting one
    #                 for its owner's cockpit). Operator-equivalent for session scoping and device
    #                 management; ``DeviceIdentity.is_operator`` is true. NOT root: it still cannot
    #                 enrol devices or define MCP servers (those stay global-token-only).
    #
    # The marker is its own table, created on the FIRST owner mint, like computer_devices and
    # device_sessions: api_devices keeps its columns and a daemon that never has an owner device
    # keeps its schema, and the parity fixtures, which record both, stay as they are. The device
    # row and its mark are written in ONE transaction, so there is never a token that
    # authenticates without its mark or the reverse. ``marked_by`` names the route that issued
    # it (OWNER_SOURCE_SAME_MAC, or the label of the approver who minted it).

    def mint_owner(self, name: str, platform: str, *, by: str,
                   replaces: Iterable[str] = ()) -> dict:
        """Enrol the person's own device. Returns what :meth:`mint` returns, plus ``owner: True``.

        ``replaces`` names SOURCES (``marked_by`` values): every LIVE owner device marked with one of
        them is revoked in the SAME transaction as this mint, so a crash leaves neither zero owners nor
        two — the earlier credentials stand until this one exists. A re-pairing replaces the install it
        came from instead of stacking a second standing operator credential nobody holds. Owner devices
        from other sources (another computer's), and ordinary devices, are never touched, and neither
        is the device being minted. The ids revoked come back as ``revoked_previous`` and are announced
        to the revoke listeners once the commit is durable.
        """
        if not by:
            raise ValueError("an owner device must say who marked it (by=)")
        sources = tuple(replaces)
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS owner_devices (
              device_id TEXT PRIMARY KEY,
              marked_at REAL NOT NULL,
              marked_by TEXT NOT NULL
            );
        """)
        try:
            revoked: list[str] = []
            if sources:
                marks = ",".join("?" for _ in sources)
                rows = self._conn.execute(
                    "SELECT o.device_id FROM owner_devices o JOIN api_devices d"
                    f" ON d.id = o.device_id WHERE o.marked_by IN ({marks})"
                    " AND d.revoked_at IS NULL", sources).fetchall()
                revoked = [r["device_id"] for r in rows]
                now = time.time()
                for device_id in revoked:
                    self._conn.execute(
                        "UPDATE api_devices SET revoked_at = ? WHERE id = ?", (now, device_id))
            minted = self._new_device(name, platform)
            self._conn.execute(
                "INSERT INTO owner_devices (device_id, marked_at, marked_by) VALUES (?, ?, ?)",
                (minted["id"], minted["created_at"], by))
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        minted["owner"] = True
        minted["revoked_previous"] = revoked
        self._notify_revoked(revoked)
        return minted

    def is_owner(self, device_id: str) -> bool:
        """Is this LIVE device an owner device? A revoked device never is."""
        try:
            row = self._conn.execute(
                "SELECT 1 FROM owner_devices o JOIN api_devices d ON d.id = o.device_id"
                " WHERE o.device_id = ? AND d.revoked_at IS NULL", (device_id,)).fetchone()
        except sqlite3.OperationalError as exc:  # runs on EVERY authenticated request
            if "no such table" not in str(exc):
                raise
            return False
        return row is not None

    def owner_device_ids(self) -> set[str]:
        return set(self.owner_sources())

    def owner_sources(self, *, include_revoked: bool = False) -> dict[str, str]:
        """{device id: who marked it} for every LIVE owner device — or, with ``include_revoked``,
        for every device ever marked (the device list shows a replaced credential as what it was)."""
        live = "" if include_revoked else " WHERE d.revoked_at IS NULL"
        try:
            rows = self._conn.execute(
                "SELECT o.device_id, o.marked_by FROM owner_devices o JOIN api_devices d"
                f" ON d.id = o.device_id{live}").fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc):  # a lock is not 'no owner devices'
                raise
            return {}
        return {r["device_id"]: r["marked_by"] for r in rows}

    # ------------------------------------------------------------------

    @staticmethod
    def _row(row: sqlite3.Row) -> DeviceRow:
        keys = row.keys()
        return DeviceRow(
            id=row["id"], name=row["name"], platform=row["platform"],
            created_at=row["created_at"], last_seen_at=row["last_seen_at"],
            revoked_at=row["revoked_at"],
            # Tolerate a SELECT that predates the #348 columns rather than requiring every
            # query to be updated in lockstep.
            last_push_at=row["last_push_at"] if "last_push_at" in keys else None,
            last_push_status=row["last_push_status"] if "last_push_status" in keys else None,
        )

    def close(self) -> None:
        self._conn.close()
