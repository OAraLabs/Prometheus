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
* **Five wrong tries burn it** (``locked``). With one live code, an online guesser has 5 chances in 10^8 per code
  the person asks for. While a code is live, ANY presentation that is not it counts, junk included. With no code
  live there is nothing to guess, and nothing is counted. The price is that a neighbour can burn a code by guessing
  badly; the terminal says "locked after 5 wrong tries", which is also the alarm.
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
import logging
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace

logger = logging.getLogger("prometheus.pairing")

CODE_DIGITS = 8
TTL_SECONDS = 300
MAX_WRONG_ATTEMPTS = 5
#: Finished codes the terminal can still ask about. Older ones answer "unknown".
KEEP_FINISHED = 16

PENDING = "pending"
REDEEMED = "redeemed"
EXPIRED = "expired"
LOCKED = "locked"
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


def format_code(code: str) -> str:
    """How a person reads it: two groups of four."""
    return f"{code[:4]}-{code[4:]}"


def _digest(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


@dataclass(frozen=True)
class OwnerCode:
    """What the store says about a code. It carries no code: the plaintext exists only in ``issue``'s return."""

    id: str
    created_at: float
    expires_at: float
    status: str = PENDING
    wrong_attempts: int = 0
    device_id: str | None = None
    device_name: str | None = None
    platform: str | None = None
    source: str | None = None
    finished_at: float | None = None


class OwnerCodes:
    """The live owner code, if any, and the recent finished ones. Thread-safe; in memory only."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        #: Read at every call, so a test can swap it (as ``PairingRuntime.clock``).
        self.clock = clock
        self._lock = threading.Lock()
        self._codes: OrderedDict[str, OwnerCode] = OrderedDict()
        self._live_id: str | None = None
        self._live_digest: str | None = None

    # -- the one live code -------------------------------------------------

    def _finish(self, code_id: str, status: str, now: float, **fields: object) -> OwnerCode:
        record = replace(self._codes[code_id], status=status, finished_at=now, **fields)  # type: ignore[arg-type]
        self._codes[code_id] = record
        if self._live_id == code_id:
            self._live_id = self._live_digest = None
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
            record = OwnerCode(id=secrets.token_hex(16), created_at=now, expires_at=now + TTL_SECONDS)
            self._codes[record.id] = record
            self._live_id, self._live_digest = record.id, _digest(code)
            while len(self._codes) > KEEP_FINISHED + 1:
                oldest = next(iter(self._codes))
                if oldest == self._live_id:          # never evict the live one (it is the newest anyway)
                    break
                del self._codes[oldest]
        logger.info("pairing: owner-code issued id=%s ttl=%ds", record.id, TTL_SECONDS)
        return record, code

    def redeem(self, presented: object, mint: Callable[[], dict], *, source: str) -> dict | None:
        """Use the live code once. For the right code, call ``mint`` and return what it minted; otherwise None.

        ``mint`` runs under the lock, so two right answers at once mint one device. If it raises, the exception
        propagates and the code stays pending with no wrong try counted: nothing was granted.
        """
        typed = normalize(presented)
        with self._lock:
            now = self.clock()
            self._settle(now)
            if self._live_id is None or self._live_digest is None:
                return None
            live = self._codes[self._live_id]
            right = typed is not None and hmac.compare_digest(_digest(typed), self._live_digest)
            if not right:
                wrong = live.wrong_attempts + 1
                if wrong >= MAX_WRONG_ATTEMPTS:
                    self._finish(live.id, LOCKED, now, wrong_attempts=wrong)
                    logger.warning("pairing: owner-code locked id=%s after %d wrong tries (last from %s)",
                                   live.id, wrong, source)
                else:
                    self._codes[live.id] = replace(live, wrong_attempts=wrong)
                    logger.warning("pairing: owner-code wrong try id=%s source=%s (%d of %d)",
                                   live.id, source, wrong, MAX_WRONG_ATTEMPTS)
                return None
            minted = mint()
            self._finish(live.id, REDEEMED, now, device_id=str(minted["id"]), device_name=str(minted["name"]),
                         platform=str(minted["platform"]), source=source)
        logger.info("pairing: owner-code redeemed id=%s device=%s name=%r platform=%s source=%s",
                    live.id, minted["id"], minted["name"], minted["platform"], source)
        return minted

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
