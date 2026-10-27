"""One-time OWNER codes: how a Beacon on another computer becomes the owner's own device.

An owner device is operator-equivalent (``DeviceIdentity.is_operator``) and, unlike the API token, "a person"
for a worker's approval (#744). Same-Mac pairing issues one to a Beacon on the daemon's own machine
(``api_token.issue_owner_credential``). A Beacon on another computer (a MacBook reaching a Mac mini over
Tailscale) had no way to get one: an approved device is always scoped, and the API token is refused where a
person is required.

So the person at the machine runs ``oara pair --owner``. The daemon mints a code (``POST /api/pair/owner-codes``:
the global token, from this host only, web/owner_code_routes.py); the terminal prints it; the remote Beacon types
it (``POST /api/pair/owner``, public) and receives an owner device token of its own. Nothing remote can mint a
code, so nothing remote can promote itself.

The code
--------
* **8 ASCII digits**, drawn uniformly (``secrets.randbelow(10**8)``), shown as ``1234-5678``. Spaces and hyphens
  are ignored when it is typed back; anything else that is not exactly eight ASCII digits is not a code (``isdigit``
  alone would accept full-width digits).
* **Five minutes, once.** A redemption mints at most one device: the check, the mint and the mark happen under one
  lock. A mint that raises leaves the code pending (nothing was granted), so the person can simply try again.
* **One live code.** Issuing another replaces it (``replaced``), so there is never more than one thing to guess.
* **No wrong guess burns the code.** Redeeming is a public route, so a code that strangers could burn would be a
  stranger's off switch for the owner. Guessing is bounded instead, by refusing tries UNREAD (not compared, not
  counted, the code untouched):

  - **Five wrong tries per source per code.** A source is the TCP peer's IPv4 address, or its IPv6 /64 (one host
    commonly holds a whole /64, so rotating inside it is not a new source; an IPv4-mapped address is its IPv4).
  - **Ten compared wrong tries a minute in all** (:data:`MAX_WRONG_PER_MINUTE`, a sliding window). While it is
    full every try is refused unread, the right code included, so a guesser with any number of addresses gets about
    ``10 x 5 = 50`` chances in 10^8 per code. Once the guessing stops the code works again, for the rest of its five
    minutes: an attacker can delay a redemption, never cancel one.

  While a code is live, ANY compared presentation that is not the code counts, junk included (a body whose ``code``
  is not a string is the route's 400 and never gets here). With no code live there is nothing to guess, and nothing
  is counted. The terminal that asked is told how many wrong tries there were (``wrong_attempts``), and a full
  window logs a ``guessing`` warning: the alarm, without the off switch.
* **Five minutes by the MONOTONIC clock.** A wall-clock step (NTP, a sleep, a person) neither stretches nor cuts a
  code's life; the wall clock is read once, to tell a person until when (``expires_wall``).
* **Kept as a SHA-256 digest**, never the code, and in memory only: a restart ends every code, which is the right
  answer for something that lives five minutes. A digest of eight digits is no secret from anyone who can read this
  process's memory, so it is not meant as one; it keeps the code out of reprs, dumps and logs.
* **Nothing here logs the code or a token.** Each issue, redemption, lock, expiry, replacement and cancel leaves one
  ``pairing: owner-code`` line with the code's id, and a redemption says who (the typed name) and from where.

Finished codes are remembered (``KEEP_FINISHED``) so the terminal that asked can learn what became of its code.

This module knows nothing about HTTP. Who may mint, and the response shapes, belong to the routes.

Source: novel code for Prometheus, 2026-10-11.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import logging
import secrets
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, replace

logger = logging.getLogger("prometheus.pairing")

CODE_DIGITS = 8
TTL_SECONDS = 300
#: Wrong tries one source (an IPv4 address, or an IPv6 /64) may make at a code before it is refused unread.
MAX_WRONG_PER_SOURCE = 5
#: Wrong tries compared in any sixty seconds, from all sources together; past it every try is refused unread.
MAX_WRONG_PER_MINUTE = 10
_WINDOW_SECONDS = 60.0
#: Finished codes the terminal can still ask about. Older ones answer "unknown".
KEEP_FINISHED = 16

PENDING = "pending"
REDEEMED = "redeemed"
EXPIRED = "expired"
REPLACED = "replaced"
CANCELED = "canceled"

_SEPARATORS = str.maketrans("", "", " -")


def normalize(presented: object) -> str | None:
    """The eight digits of a typed code, or None for anything that is not one. Spaces and hyphens are ignored."""
    if not isinstance(presented, str):
        return None
    digits = presented.strip().translate(_SEPARATORS)
    if len(digits) != CODE_DIGITS or not (digits.isascii() and digits.isdigit()):
        return None
    return digits


def source_key(peer: str) -> str:
    """What one guesser is, for the per-source budget: an IPv4 address, or the /64 of an IPv6 one (a mapped IPv4
    address is that IPv4 address). Anything that is not an address is itself."""
    try:
        address = ipaddress.ip_address(peer)
    except ValueError:
        return peer
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        return str(ipaddress.IPv6Network((address, 64), strict=False))
    return str(address)


def format_code(code: str) -> str:
    """How a person reads it: two groups of four."""
    return f"{code[:4]}-{code[4:]}"


def _digest(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


@dataclass(frozen=True)
class OwnerCode:
    """What the store says about a code. It carries no code: the plaintext exists only in ``issue``'s return.

    ``created_at`` / ``expires_at`` are on the store's monotonic clock and decide the code's life; ``expires_wall`` is
    the wall-clock time to show a person.
    """

    id: str
    created_at: float
    expires_at: float
    expires_wall: float = 0.0
    status: str = PENDING
    wrong_attempts: int = 0
    device_id: str | None = None
    device_name: str | None = None
    platform: str | None = None
    source: str | None = None
    #: Whether the redeemer was on the daemon's own machine (``loopback.is_this_host`` at redeem time), so the terminal
    #: can say "this Mac" or "another computer" truthfully. An address alone cannot: a relay on this machine
    #: (Tailscale serve, cloudflared) brings another computer in from 127.0.0.1.
    this_host: bool = False
    finished_at: float | None = None


class OwnerCodes:
    """The live owner code, if any, and the recent finished ones. Thread-safe; in memory only."""

    def __init__(self, clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time) -> None:
        #: Both read at every call, so a test can swap them (as ``PairingRuntime.clock``). ``clock`` decides a code's
        #: life and must not jump; ``wall`` only says until when, for a person.
        self.clock = clock
        self.wall = wall
        self._lock = threading.Lock()
        self._codes: OrderedDict[str, OwnerCode] = OrderedDict()
        self._live_id: str | None = None
        self._live_digest: str | None = None
        #: Wrong tries per source (``source_key``) at the live code; cleared whenever the live code changes.
        self._wrong_by_source: dict[str, int] = {}
        #: When each compared wrong try of the last minute happened (any code): the global bound.
        self._recent_wrong: deque[float] = deque()

    # -- the one live code -------------------------------------------------

    def _finish(self, code_id: str, status: str, now: float, **fields: object) -> OwnerCode:
        record = replace(self._codes[code_id], status=status, finished_at=now, **fields)  # type: ignore[arg-type]
        self._codes[code_id] = record
        if self._live_id == code_id:
            self._live_id = self._live_digest = None
            self._wrong_by_source = {}
        return record

    def _settle(self, now: float) -> None:
        """Expire the live code once its five minutes are up (time passing is the only event that is not a call)."""
        live_id = self._live_id
        if live_id is not None and now >= self._codes[live_id].expires_at:
            self._finish(live_id, EXPIRED, now)
            logger.info("pairing: owner-code expired id=%s (unused)", live_id)

    def issue(self) -> tuple[OwnerCode, str]:
        """Mint a new code, replacing any live one. Returns its record and the code, which nothing else keeps."""
        code = f"{secrets.randbelow(10 ** CODE_DIGITS):0{CODE_DIGITS}d}"
        with self._lock:
            now = self.clock()
            self._settle(now)
            if self._live_id is not None:
                replaced_id = self._live_id
                self._finish(replaced_id, REPLACED, now)
                logger.info("pairing: owner-code replaced id=%s (a newer code was issued)", replaced_id)
            record = OwnerCode(id=secrets.token_hex(16), created_at=now, expires_at=now + TTL_SECONDS,
                               expires_wall=self.wall() + TTL_SECONDS)
            self._codes[record.id] = record
            self._live_id, self._live_digest = record.id, _digest(code)
            self._wrong_by_source = {}
            while len(self._codes) > KEEP_FINISHED + 1:
                oldest = next(iter(self._codes))
                if oldest == self._live_id:          # never evict the live one (it is the newest anyway)
                    break
                del self._codes[oldest]
        logger.info("pairing: owner-code issued id=%s ttl=%ds", record.id, TTL_SECONDS)
        return record, code

    def redeem(self, presented: object, mint: Callable[[], dict], *, source: str, this_host: bool = False) -> dict | None:
        """Use the live code once. For the right code, call ``mint`` and return what it minted; otherwise None.

        ``mint`` runs under the lock, so two right answers at once mint one device. If it raises, the exception
        propagates and the code stays pending with no wrong try counted: nothing was granted.
        """
        typed = normalize(presented)
        key = source_key(source)
        with self._lock:
            now = self.clock()
            self._settle(now)
            if self._live_id is None or self._live_digest is None:
                return None
            live = self._codes[self._live_id]
            spent = self._wrong_by_source.get(key, 0)
            if spent >= MAX_WRONG_PER_SOURCE:
                return None                     # refused unread: this try is not compared, so it is not counted
            while self._recent_wrong and now - self._recent_wrong[0] >= _WINDOW_SECONDS:
                self._recent_wrong.popleft()
            if len(self._recent_wrong) >= MAX_WRONG_PER_MINUTE:
                return None                     # refused unread, the right code included; the code is untouched
            right = typed is not None and hmac.compare_digest(_digest(typed), self._live_digest)
            if not right:
                wrong = live.wrong_attempts + 1
                self._wrong_by_source[key] = spent + 1
                self._recent_wrong.append(now)
                self._codes[live.id] = replace(live, wrong_attempts=wrong)
                logger.warning("pairing: owner-code wrong try id=%s source=%s (%d so far; %d of %d from it)",
                               live.id, source, wrong, spent + 1, MAX_WRONG_PER_SOURCE)
                if spent + 1 >= MAX_WRONG_PER_SOURCE:
                    logger.warning("pairing: owner-code source %s shut out of id=%s after %d wrong tries",
                                   key, live.id, spent + 1)
                if len(self._recent_wrong) == MAX_WRONG_PER_MINUTE:   # logged once per filling, not per refusal
                    logger.warning("pairing: owner-code guessing: %d wrong tries in a minute; every try is refused "
                                   "unread until it passes (id=%s stays live)", MAX_WRONG_PER_MINUTE, live.id)
                return None
            minted = mint()
            self._finish(live.id, REDEEMED, now, device_id=str(minted["id"]), device_name=str(minted["name"]),
                         platform=str(minted["platform"]), source=source, this_host=bool(this_host))
        logger.info("pairing: owner-code redeemed id=%s device=%s name=%r platform=%s source=%s",
                    live.id, minted["id"], minted["name"], minted["platform"], source)
        return minted

    def live(self) -> OwnerCode | None:
        """The pending code's record, if one is live (it carries no code)."""
        with self._lock:
            self._settle(self.clock())
            return self._codes[self._live_id] if self._live_id is not None else None

    def status(self, code_id: object) -> OwnerCode | None:
        """What became of a code, or None for one this process never issued (or has forgotten)."""
        with self._lock:
            self._settle(self.clock())
            return self._codes.get(code_id) if isinstance(code_id, str) else None

    def cancel(self, code_id: object) -> bool:
        """End a pending code. True if it was the live one; False for anything unknown or already finished."""
        with self._lock:
            now = self.clock()
            self._settle(now)
            if not isinstance(code_id, str) or code_id != self._live_id:
                return False
            self._finish(code_id, CANCELED, now)
        logger.info("pairing: owner-code canceled id=%s", code_id)
        return True
