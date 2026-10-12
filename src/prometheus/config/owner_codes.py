"""One-time owner codes: how a Beacon on ANOTHER computer becomes an owner device.

Owner device tokens used to reach only a client on the daemon's own machine (``POST /api/pair/local``, or the API
token minting one from loopback). Since #744 a worker's request is approved only by a device token, because a
model can read the API token, so a Beacon on another computer that held the API token could approve no worker at
all. This is the way up for it:

    oara pair --owner                 on the daemon's machine: prints a code like  K7QM-2XW9-RTHP
    POST /api/pair/owner              from the other computer: the code + a public key -> an owner device, sealed

Where the authority comes from
------------------------------
**A code is made only by writing this file.** ``issue_owner_code`` is called by the terminal command and by
nothing else (``tests/test_owner_code_promotion.py`` scans the source for it), and there is no route that makes
one, so nothing that only reaches the daemon over the network can mint a code, whatever bearer it holds. Being
able to write ``devices.db`` is already being able to mint any device outright, so issuing through it widens
nothing. The API token alone mints no owner device anywhere: the route also needs a code, and the loopback-only
``POST /api/devices`` refuses ``owner`` from another address.

The code
--------
12 characters from :data:`ALPHABET` (no 0, 1, I, L, O or U, so it can be read off one screen and typed into
another), printed as three groups of four: ``30**12`` is about ``2**59``. It is good for
:data:`CODE_TTL_SECONDS` and for one use, and there is at most one live code: issuing replaces the one before.
Every wrong guess, from anyone, counts against the live code and the :data:`MAX_FAILURES`-th burns it, so the
whole network gets five guesses at a 59-bit value; the route also slows each source to
:data:`WRONG_CODES_PER_MINUTE`. Only the SHA-256 of the code is stored. That is enough against someone who later
reads a copy of the file: there are ten minutes, and ``2**59`` hashes take a GPU about a year.

A malformed code (wrong length, a character outside the alphabet) is not a guess at the live code: it is refused
before the table is touched and spends nothing, so a typo of an ``O`` for a ``0`` costs the person nothing.

Consistency
-----------
The terminal and the daemon are two processes on one SQLite file. A redemption is ONE ``BEGIN IMMEDIATE``
transaction: find the live code, compare in constant time, then either count the failure (and burn at the limit)
or delete the code AND mint the owner device, together. Two redemptions of one code have exactly one winner, and a
crash leaves neither a spent code with no device nor a device whose code still works.

The tables appear on first use, like ``owner_devices`` and ``pair_requests``: a daemon nobody ran
``oara pair --owner`` for keeps the exact ``devices.db`` schema the parity fixtures record. Redeeming when no
code was ever issued reads nothing and creates nothing.

Never in a log line: the code (right or wrong) or the token.

Source: novel code for Prometheus, 2026-10-11.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass

from prometheus.config.api_token import OWNER_SOURCE_OWNER_CODE
from prometheus.config.device_store import DeviceStore, registry_platform

logger = logging.getLogger("prometheus.pairing")

#: Read off one screen and typed into another: no 0/O, 1/I/L, and no U.
ALPHABET = "23456789ABCDEFGHJKMNPQRSTVWXYZ"
CODE_CHARS = 12
GROUP = 4
CODE_TTL_SECONDS = 600
#: Wrong guesses, from anyone, that burn the live code.
MAX_FAILURES = 5
#: Wrong guesses one source may make in a minute before it is told to wait (the route's limiter).
WRONG_CODES_PER_MINUTE = 3
#: ``owner_devices.marked_by`` for an owner device bought with a code.
OWNER_SOURCE = OWNER_SOURCE_OWNER_CODE

_SCHEMA = """
    CREATE TABLE IF NOT EXISTS owner_codes (
      code_sha256 TEXT PRIMARY KEY,
      created_at  REAL NOT NULL,
      expires_at  REAL NOT NULL,
      failures    INTEGER NOT NULL DEFAULT 0,
      issued_by   TEXT NOT NULL
    );
"""
_SEPARATORS = str.maketrans("", "", " -\t")


@dataclass(frozen=True)
class IssuedCode:
    #: As it is printed: ``XXXX-XXXX-XXXX``.
    code: str
    expires_at: float


@dataclass(frozen=True)
class Redeemed:
    device_id: str
    token: str
    name: str
    platform: str


def format_code(raw: str) -> str:
    """``K7QM2XW9RTHP`` -> ``K7QM-2XW9-RTHP``."""
    return "-".join(raw[i:i + GROUP] for i in range(0, len(raw), GROUP))


def normalize_code(value: object) -> str | None:
    """The 12 characters a person meant, or None for anything that cannot be a code.

    Upper-cased, with spaces and hyphens dropped (it is printed in groups and may be typed in groups). Nothing
    else is forgiven: a character outside the alphabet makes it malformed rather than silently mapped to another
    code.
    """
    if not isinstance(value, str):
        return None
    text = value.translate(_SEPARATORS).upper()
    if len(text) != CODE_CHARS or any(ch not in ALPHABET for ch in text):
        return None
    return text


def _digest(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def _has_table(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='owner_codes'").fetchone() is not None


def issue_owner_code(devices: DeviceStore, *, by: str, clock: Callable[[], float] = time.time) -> IssuedCode:
    """Make THE live owner code, replacing any earlier one. Returns the only copy of it that will exist.

    Called by ``oara pair --owner`` on the daemon's machine, and by nothing reachable over the network (see the
    module docstring). *by* names who issued it, for the audit line and the row.
    """
    if not by:
        raise ValueError("an owner code must say who issued it (by=)")
    raw = "".join(secrets.choice(ALPHABET) for _ in range(CODE_CHARS))
    now = clock()
    conn = devices.connection
    conn.executescript(_SCHEMA)
    devices.ensure_owner_table()          # so a redemption never has to create a table inside its transaction
    conn.execute("BEGIN IMMEDIATE")
    try:
        replaced = conn.execute("DELETE FROM owner_codes").rowcount
        conn.execute(
            "INSERT INTO owner_codes (code_sha256, created_at, expires_at, failures, issued_by) VALUES (?, ?, ?, 0, ?)",
            (_digest(raw), now, now + CODE_TTL_SECONDS, by))
    except BaseException:
        conn.rollback()
        raise
    conn.commit()
    logger.info("pairing: owner code issued by=%s ttl=%ds replaced=%d", by, CODE_TTL_SECONDS, max(replaced, 0))
    return IssuedCode(format_code(raw), now + CODE_TTL_SECONDS)


class OwnerCodeStore:
    """Redeeming the live owner code, on the registry's own connection (the daemon's side)."""

    def __init__(self, devices: DeviceStore, *, clock: Callable[[], float] = time.time) -> None:
        self._devices = devices
        self._conn = devices.connection
        self._clock = clock

    def redeem(self, typed: str, *, name: str, platform: str, source: str = "", by: str = "") -> Redeemed | None:
        """Spend the live code on an owner device, or count a wrong guess. None for every failure alike.

        *typed* is the code as the person typed it (see :func:`normalize_code`). One that cannot be a code is a
        None that touches nothing (the route refuses it with a 400 before it gets here). *source* and *by* are
        for the audit line only.
        """
        normalized = normalize_code(typed)
        if normalized is None or not _has_table(self._conn):
            return None
        presented = _digest(normalized)
        now = self._clock()
        self._devices.ensure_owner_table()
        conn = self._conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("DELETE FROM owner_codes WHERE expires_at <= ? OR failures >= ?", (now, MAX_FAILURES))
            row = conn.execute(
                "SELECT code_sha256, failures FROM owner_codes ORDER BY created_at DESC LIMIT 1").fetchone()
            if row is None:
                conn.commit()
                logger.warning("pairing: owner code refused source=%s (no live code)", source or "?")
                return None
            if not hmac.compare_digest(row["code_sha256"], presented):
                failures = row["failures"] + 1
                if failures >= MAX_FAILURES:
                    conn.execute("DELETE FROM owner_codes WHERE code_sha256 = ?", (row["code_sha256"],))
                else:
                    conn.execute("UPDATE owner_codes SET failures = ? WHERE code_sha256 = ?",
                                 (failures, row["code_sha256"]))
                conn.commit()
                logger.warning("pairing: owner code refused source=%s failures=%d/%d%s", source or "?", failures,
                               MAX_FAILURES, " (burned: run `oara pair --owner` again)" if failures >= MAX_FAILURES
                               else "")
                return None
            spent = conn.execute("DELETE FROM owner_codes WHERE code_sha256 = ?", (presented,)).rowcount
            if spent != 1:
                raise RuntimeError("the live owner code vanished inside its own transaction")
            minted = self._devices.mint_owner_in_transaction(name, registry_platform(platform), by=OWNER_SOURCE)
        except BaseException:
            conn.rollback()
            raise
        conn.commit()
        logger.info("pairing: owner code redeemed device=%s name=%r platform=%s source=%s by=%s",
                    minted["id"], minted["name"], minted["platform"], source or "?", by or "?")
        return Redeemed(str(minted["id"]), str(minted["token"]), str(minted["name"]), str(minted["platform"]))
