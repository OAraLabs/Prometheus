"""Pairing requests: a stranger's ask to join, held while the owner decides.

A new device on the home network asks (``POST /api/pair/requests``) and waits; the owner approves or denies
on a device they already use; on approval the daemon mints an ordinary, scoped device token and seals it to
a public key the requester sent. This module is the table and the state machine. Contract:
docs/PAIRING-APPROVAL-API.md, section 4.

    pending --approve--> approved --acknowledged--> delivered
       |                     \\--5 min uncollected--> uncollected   (the device is revoked)
       |--deny--> denied
       |--ttl---> expired
       \\--cancel> canceled

What makes it safe to hold a request from a stranger:

* **One guarded statement per transition**, never read-then-write. Two decisions at once have exactly one
  winner (the CLI and the daemon are two processes on one SQLite file), and a decision after the TTL is
  ``RequestExpired``, not a quiet success.
* **Approve is atomic with the mint.** The state change, the device row and the sealed blob commit together;
  a failure while sealing rolls the device back, so there is never a live token nobody holds.
* **The plaintext token exists only inside ``approve``.** The store keeps the SHA-256 of the poll secret and
  a blob sealed to the requester's key; the registry keeps the SHA-256 of the token. Nothing here returns,
  logs or stores the token itself.
* **Approval grants an ordinary scoped device**, through ``mint_paired_device`` and nothing else. Never
  ``mint_owner`` or ``issue_owner_credential``: the owner tier is only ever same-Mac pairing.
* **The table appears on first use**, like ``computer_devices`` and ``device_sessions``, so a daemon that
  never pairs a device keeps the exact ``devices.db`` schema the parity fixtures record. Reads before that
  find nothing and create nothing.
* **A device nobody collects is revoked** five minutes after approval, through ``DeviceStore.revoke``, so
  the registry's listener closes its sockets too. Revoke first, mark second: a crash between the two leaves
  the row ``approved`` past its deadline, and the next sweep finishes the job.

Every create, approval, denial, cancellation, expiry, delivery and auto-revoke leaves one ``pairing:`` audit
line. Never in a log line: the poll secret, the token, the sealed blob.

Source: novel code for Prometheus, 2026-10-08.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import secrets
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from prometheus.config import pair_seal
from prometheus.config.device_store import DeviceStore, registry_platform

logger = logging.getLogger("prometheus.pairing")

PENDING = "pending"
APPROVED = "approved"
DENIED = "denied"
CANCELED = "canceled"
EXPIRED = "expired"
DELIVERED = "delivered"
UNCOLLECTED = "uncollected"

#: How long an approved token may sit uncollected before the device is revoked.
DELIVERY_WINDOW_SECONDS = 300
#: Finished rows are kept this long (the audit line is the record; the row is for debugging a bad day).
PURGE_AFTER_SECONDS = 7 * 86400
_HOUR = 3600

TTL_DEFAULT = 300
TTL_MIN = 60
TTL_MAX = 900

_SCHEMA = """
    CREATE TABLE IF NOT EXISTS pair_requests (
      id                TEXT PRIMARY KEY,
      secret_sha256     TEXT NOT NULL,
      device_name       TEXT NOT NULL,
      platform          TEXT NOT NULL,
      public_key        TEXT NOT NULL,
      source            TEXT NOT NULL,
      match_code        TEXT NOT NULL,
      state             TEXT NOT NULL,
      created_at        REAL NOT NULL,
      expires_at        REAL NOT NULL,
      decided_at        REAL,
      decided_by        TEXT,
      approved_name     TEXT,
      device_id         TEXT,
      sealed            TEXT,
      delivery_deadline REAL,
      finished_at       REAL
    );
    CREATE INDEX IF NOT EXISTS pair_requests_source ON pair_requests (source, created_at);
    CREATE INDEX IF NOT EXISTS pair_requests_state ON pair_requests (state);
"""

# What an unknown id is compared against, so it costs the same as a wrong secret.
_DUMMY_DIGEST = hashlib.sha256(b"no such pairing request").hexdigest()


# ── settings ─────────────────────────────────────────────────────────────────

def _whole(section: Mapping[str, Any], key: str, default: int) -> int:
    value = section.get(key)
    if value is None:                                  # a YAML key left empty is "unset", not an error
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        logger.warning("pairing.%s must be a whole number of at least 1, got %r — using %d", key, value, default)
        return default
    return value


def _ttl(section: Mapping[str, Any]) -> int:
    value = section.get("request_ttl_seconds")
    if value is None:
        return TTL_DEFAULT
    if isinstance(value, bool) or not isinstance(value, int):
        logger.warning("pairing.request_ttl_seconds must be a whole number of seconds, got %r — using %d",
                       value, TTL_DEFAULT)
        return TTL_DEFAULT
    clamped = min(TTL_MAX, max(TTL_MIN, value))
    if clamped != value:
        logger.warning("pairing.request_ttl_seconds %d is outside %d-%d — using %d", value, TTL_MIN, TTL_MAX, clamped)
    return clamped


@dataclass(frozen=True)
class PairingSettings:
    """The ``pairing:`` section of the config, read here and nowhere else."""

    requests_enabled: bool = True
    request_ttl_seconds: int = TTL_DEFAULT
    max_pending: int = 3
    max_pending_per_source: int = 1
    max_requests_per_source_per_hour: int = 10

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> PairingSettings:
        section = config.get("pairing") if isinstance(config, Mapping) else None
        if not isinstance(section, Mapping):
            return cls()
        enabled = section.get("requests_enabled")
        if enabled is not None and not isinstance(enabled, bool):
            logger.warning("pairing.requests_enabled must be true or false, got %r — using true", enabled)
            enabled = None
        return cls(
            requests_enabled=True if enabled is None else enabled,
            request_ttl_seconds=_ttl(section),
            max_pending=_whole(section, "max_pending", 3),
            max_pending_per_source=_whole(section, "max_pending_per_source", 1),
            max_requests_per_source_per_hour=_whole(section, "max_requests_per_source_per_hour", 10),
        )


# ── errors and results ───────────────────────────────────────────────────────

class PairError(Exception):
    """Base for a refusal the caller turns into a status code."""


class UnknownRequest(PairError):
    """No such request."""


class RequestExpired(PairError):
    """The decision arrived after the TTL."""


class NotPending(PairError):
    """Someone decided first. ``status`` is what the request became."""

    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


class LimitExceeded(PairError):
    """A creation limit. ``reason`` is ``per_source_pending``, ``pending_full`` or ``hourly``."""

    def __init__(self, reason: str, retry_after: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after = max(1, int(retry_after))


@dataclass(frozen=True)
class PairRequest:
    """One request as the caller may see it. Never holds the poll secret or the sealed blob."""

    id: str
    device_name: str
    platform: str
    public_key: str
    source: str
    match_code: str
    state: str
    created_at: float
    expires_at: float
    decided_at: float | None = None
    decided_by: str | None = None
    approved_name: str | None = None
    device_id: str | None = None
    delivery_deadline: float | None = None


@dataclass(frozen=True)
class Created:
    request: PairRequest
    #: The only copy of the poll secret that will ever exist outside the requester.
    poll_secret: str


@dataclass(frozen=True)
class Approved:
    request: PairRequest
    device_id: str
    name: str


@dataclass(frozen=True)
class SweepResult:
    expired: list[PairRequest] = field(default_factory=list)
    uncollected: list[tuple[PairRequest, str]] = field(default_factory=list)
    purged: int = 0


def mint_paired_device(devices: DeviceStore, name: str, platform: str) -> dict:
    """The ONE place an approval creates a device: an ordinary, scoped one.

    ``DeviceStore.mint_in_transaction`` has no tier parameter and writes no ``owner_devices`` row, and this
    passes no scope, no approver flag and no permission, because the request has no field to ask for any.
    Not ``mint_owner`` and not ``api_token.issue_owner_credential``: those are the same-Mac tier only
    (Will, 2026-10-08), so Jennifer's Mac approved from Will's sees only its own conversations.
    """
    return devices.mint_in_transaction(name, registry_platform(platform))


# ── the store ────────────────────────────────────────────────────────────────

def _effective(row: sqlite3.Row, now: float) -> str:
    state = row["state"]
    if state == PENDING and now >= row["expires_at"]:
        return EXPIRED
    if state == APPROVED and row["delivery_deadline"] is not None and now >= row["delivery_deadline"]:
        return UNCOLLECTED
    return state


def _request(row: sqlite3.Row, now: float) -> PairRequest:
    return PairRequest(
        id=row["id"], device_name=row["device_name"], platform=row["platform"], public_key=row["public_key"],
        source=row["source"], match_code=row["match_code"], state=_effective(row, now),
        created_at=row["created_at"], expires_at=row["expires_at"], decided_at=row["decided_at"],
        decided_by=row["decided_by"], approved_name=row["approved_name"], device_id=row["device_id"],
        delivery_deadline=row["delivery_deadline"],
    )


class PairRequestStore:
    """Pairing requests in ``devices.db``, on the registry's own connection."""

    def __init__(self, devices: DeviceStore, *, settings: PairingSettings,
                 clock: Callable[[], float] = time.time) -> None:
        self._devices = devices
        self._conn = devices.connection
        self._settings = settings
        self._clock = clock
        self._ready = False

    # -- plumbing ---------------------------------------------------------

    def _ensure(self) -> None:
        if not self._ready:
            self._conn.executescript(_SCHEMA)
            self._ready = True

    def _exists(self) -> bool:
        if self._ready:
            return True
        found = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pair_requests'").fetchone()
        self._ready = found is not None
        return self._ready

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """One transaction holding the write lock from its first statement, so a check and its write
        cannot be separated by another connection's write."""
        self._ensure()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()

    def _row(self, conn: sqlite3.Connection, request_id: str) -> sqlite3.Row | None:
        return conn.execute("SELECT * FROM pair_requests WHERE id = ?", (str(request_id)[:64],)).fetchone()

    def _need(self, conn: sqlite3.Connection, request_id: str) -> sqlite3.Row:
        """A row this transaction just wrote: it exists, or something is badly wrong."""
        row = self._row(conn, request_id)
        if row is None:
            raise RuntimeError(f"pairing request {request_id} vanished inside its own transaction")
        return row

    @staticmethod
    def _audit(event: str, req: PairRequest, **extra: object) -> None:
        # name is validated (no control characters) at the route; repr anyway, it is attacker-typed.
        tail = "".join(f" {key}={value}" for key, value in extra.items() if value is not None)
        logger.info("pairing: %s id=%s name=%r platform=%s source=%s%s",
                    event, req.id, req.device_name, req.platform, req.source, tail)

    def _why_not(self, row: sqlite3.Row | None, now: float) -> PairError:
        """The refusal for a guarded UPDATE that matched no row."""
        if row is None:
            return UnknownRequest(None)
        state = _effective(row, now)
        if state == PENDING:
            return NotPending(PENDING)           # unreachable in practice: the guard would have matched
        if state == EXPIRED:
            # Whether the sweep has recorded it yet or not (it usually has: it runs at the start of every
            # request), a decision that arrives after the TTL is "too late", never "someone decided first".
            return RequestExpired(row["id"])
        # An approval nobody collected (its device revoked) was still approved: that is who won.
        return NotPending(APPROVED if state == UNCOLLECTED else state)

    # -- create -----------------------------------------------------------

    def create(self, *, device_name: str, platform: str, public_key: str, source: str,
               instance_public_key_der: bytes) -> Created:
        """Record a request, or raise :class:`LimitExceeded`. *public_key* is base64url, already validated."""
        now = self._clock()
        request_id = secrets.token_hex(16)
        poll_secret = secrets.token_urlsafe(32)
        code = pair_seal.match_code(pair_seal.b64url_decode(public_key), instance_public_key_der, request_id)
        with self._write() as conn:
            self._check_limits(conn, source, now)
            conn.execute(
                "INSERT INTO pair_requests (id, secret_sha256, device_name, platform, public_key, source,"
                " match_code, state, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (request_id, hashlib.sha256(poll_secret.encode()).hexdigest(), device_name, platform,
                 public_key, source, code, PENDING, now, now + self._settings.request_ttl_seconds))
            row = self._need(conn, request_id)
        req = _request(row, now)
        self._audit("created", req, ttl=self._settings.request_ttl_seconds)
        return Created(req, poll_secret)

    def _check_limits(self, conn: sqlite3.Connection, source: str, now: float) -> None:
        s = self._settings
        live = "state = 'pending' AND expires_at > ?"
        count, soonest = conn.execute(
            f"SELECT COUNT(*), MIN(expires_at) FROM pair_requests WHERE {live} AND source = ?", (now, source)).fetchone()
        if count >= s.max_pending_per_source:
            raise LimitExceeded("per_source_pending", math.ceil(soonest - now))
        count, soonest = conn.execute(
            f"SELECT COUNT(*), MIN(expires_at) FROM pair_requests WHERE {live}", (now,)).fetchone()
        if count >= s.max_pending:
            raise LimitExceeded("pending_full", math.ceil(soonest - now))
        count, oldest = conn.execute(
            "SELECT COUNT(*), MIN(created_at) FROM pair_requests WHERE source = ? AND created_at > ?",
            (source, now - _HOUR)).fetchone()
        if count >= s.max_requests_per_source_per_hour:
            raise LimitExceeded("hourly", math.ceil(oldest + _HOUR - now))

    # -- reading ----------------------------------------------------------

    def verify(self, request_id: str, secret: str) -> PairRequest | None:
        """The request, if *secret* is its poll secret; ``None`` for an unknown id and a wrong secret alike.

        Exactly one constant-time comparison either way, so an unknown id cannot be told from a wrong
        secret by how long the answer takes.
        """
        now = self._clock()
        row = self._row(self._conn, request_id) if self._exists() else None
        stored = row["secret_sha256"] if row is not None else _DUMMY_DIGEST
        presented = hashlib.sha256((secret if isinstance(secret, str) else "").encode()).hexdigest()
        matches = hmac.compare_digest(stored, presented)
        return _request(row, now) if (row is not None and matches) else None

    def get(self, request_id: str) -> PairRequest | None:
        if not self._exists():
            return None
        row = self._row(self._conn, request_id)
        return _request(row, self._clock()) if row is not None else None

    def pending(self) -> list[PairRequest]:
        """Live pending requests, newest first."""
        if not self._exists():
            return []
        now = self._clock()
        rows = self._conn.execute(
            "SELECT * FROM pair_requests WHERE state = 'pending' AND expires_at > ?"
            " ORDER BY created_at DESC, rowid DESC", (now,)).fetchall()
        return [_request(r, now) for r in rows]

    def sealed_blob(self, request_id: str) -> dict[str, str] | None:
        """The sealed token while it may still be collected, else ``None``."""
        if not self._exists():
            return None
        row = self._conn.execute(
            "SELECT sealed FROM pair_requests WHERE id = ? AND state = 'approved' AND delivery_deadline > ?"
            " AND sealed IS NOT NULL", (str(request_id)[:64], self._clock())).fetchone()
        return json.loads(row["sealed"]) if row is not None else None

    # -- deciding ---------------------------------------------------------

    def approve(self, request_id: str, *, name: str | None, decided_by: str) -> Approved:
        """Mint an ordinary scoped device for a pending request and seal its token to the requester.

        One transaction: the state change, the device row and the sealed blob commit together or not at all.
        """
        now = self._clock()
        with self._write() as conn:
            changed = conn.execute(
                "UPDATE pair_requests SET state = 'approved', decided_at = ?, decided_by = ?"
                " WHERE id = ? AND state = 'pending' AND expires_at > ?",
                (now, decided_by, str(request_id)[:64], now)).rowcount
            if changed != 1:
                raise self._why_not(self._row(conn, request_id), now)
            row = self._need(conn, request_id)
            device_name = name if name is not None else row["device_name"]
            minted = mint_paired_device(self._devices, device_name, row["platform"])
            sealed = pair_seal.seal_token(
                requester_public_key=pair_seal.b64url_decode(row["public_key"]),
                request_id=row["id"],
                payload={"token": minted["token"], "device_id": minted["id"], "name": device_name},
            )
            conn.execute(
                "UPDATE pair_requests SET device_id = ?, approved_name = ?, sealed = ?, delivery_deadline = ?"
                " WHERE id = ?",
                (minted["id"], device_name, json.dumps(sealed, separators=(",", ":")),
                 now + DELIVERY_WINDOW_SECONDS, row["id"]))
            row = self._need(conn, request_id)
        req = _request(row, now)
        self._audit("approved", req, by=decided_by, device=minted["id"], enrolled_as=repr(device_name))
        return Approved(req, minted["id"], device_name)

    def deny(self, request_id: str, *, decided_by: str) -> PairRequest:
        now = self._clock()
        with self._write() as conn:
            changed = conn.execute(
                "UPDATE pair_requests SET state = 'denied', decided_at = ?, decided_by = ?, finished_at = ?"
                " WHERE id = ? AND state = 'pending' AND expires_at > ?",
                (now, decided_by, now, str(request_id)[:64], now)).rowcount
            if changed != 1:
                raise self._why_not(self._row(conn, request_id), now)
            row = self._need(conn, request_id)
        req = _request(row, now)
        self._audit("denied", req, by=decided_by)
        return req

    def cancel(self, request_id: str) -> PairRequest | None:
        """The requester withdraws a pending request. ``None`` when it was not pending (no change)."""
        if not self._exists():
            return None
        now = self._clock()
        with self._write() as conn:
            changed = conn.execute(
                "UPDATE pair_requests SET state = 'canceled', decided_at = ?, decided_by = 'requester',"
                " finished_at = ? WHERE id = ? AND state = 'pending' AND expires_at > ?",
                (now, now, str(request_id)[:64], now)).rowcount
            row = self._need(conn, request_id) if changed == 1 else None
        if row is None:
            return None
        req = _request(row, now)
        self._audit("canceled", req)
        return req

    def acknowledge(self, request_id: str) -> bool:
        """The requester confirms it holds the token: the sealed blob is wiped. Once, and only in the window."""
        if not self._exists():
            return False
        now = self._clock()
        with self._write() as conn:
            changed = conn.execute(
                "UPDATE pair_requests SET state = 'delivered', sealed = NULL, finished_at = ?"
                " WHERE id = ? AND state = 'approved' AND delivery_deadline > ?",
                (now, str(request_id)[:64], now)).rowcount
            row = self._need(conn, request_id) if changed == 1 else None
        if row is None:
            return False
        self._audit("delivered", _request(row, now), device=row["device_id"])
        return True

    # -- housekeeping -----------------------------------------------------

    def sweep(self) -> SweepResult:
        """Expire what ran out, revoke what nobody collected, and delete what is old. Creates nothing."""
        if not self._exists():
            return SweepResult()
        now = self._clock()
        expired: list[PairRequest] = []
        with self._write() as conn:
            for row in conn.execute(
                    "SELECT * FROM pair_requests WHERE state = 'pending' AND expires_at <= ?", (now,)).fetchall():
                conn.execute(
                    "UPDATE pair_requests SET state = 'expired', decided_at = ?, decided_by = 'system',"
                    " finished_at = ? WHERE id = ? AND state = 'pending'", (now, now, row["id"]))
                expired.append(_request(self._need(conn, row["id"]), now))
            candidates = conn.execute(
                "SELECT * FROM pair_requests WHERE state = 'approved' AND delivery_deadline <= ?", (now,)).fetchall()
        uncollected: list[tuple[PairRequest, str]] = []
        for row in candidates:
            # Revoke FIRST (its own commit, and the registry's listener closes the device's sockets), then
            # mark: a crash in between leaves the row 'approved' past its deadline and the next sweep
            # finishes it. The other order could leave a live token marked uncollected.
            self._devices.revoke(row["device_id"])
            with self._write() as conn:
                conn.execute(
                    "UPDATE pair_requests SET state = 'uncollected', sealed = NULL, finished_at = ?"
                    " WHERE id = ? AND state = 'approved'", (now, row["id"]))
                updated = self._need(conn, row["id"])
            uncollected.append((_request(updated, now), row["device_id"]))
        with self._write() as conn:
            purged = conn.execute(
                "DELETE FROM pair_requests WHERE finished_at IS NOT NULL AND finished_at < ?",
                (now - PURGE_AFTER_SECONDS,)).rowcount
        for req in expired:
            self._audit("expired", req)
        for req, device_id in uncollected:
            self._audit("uncollected", req, device=device_id, action="device revoked")
        return SweepResult(expired, uncollected, purged)
