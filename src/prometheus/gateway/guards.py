"""Guard classification — every check declares what it is, in code.

THE RULE THIS FILE MAKES UNAVOIDABLE
------------------------------------
A check either protects the system (a **CONTROL**) or improves it (a
**CONVENIENCE**), and the two must fail in opposite directions:

* A CONTROL that degrades to "allow" under error is not a control. An
  allowlist that lets everything through when its sniffer raises is worse
  than no allowlist, because a reader greps the config, finds the allowlist,
  and stops looking.
* A CONVENIENCE that degrades to "reject" turns an optimisation into an
  outage. If the media cache is unwritable, the message should still be
  processed.

That distinction was previously *inferred* per check. Here it is declared:
``Guard`` raises at construction if ``enforcement`` is missing, so the next
person adding a check cannot avoid answering the question. A test asserts
every registered guard carries one.

Source: novel code for Prometheus, 2026-08-03 (Telegram surface hardening).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Enforcement(str, Enum):
    """Which way a check fails when it cannot run."""

    CONTROL = "control"
    """Fails CLOSED. Cannot run -> reject. Protects the system."""

    CONVENIENCE = "convenience"
    """Fails OPEN. Cannot run -> proceed. Improves the system."""


class GuardDeclarationError(TypeError):
    """A guard was defined without saying whether it is a control."""


@dataclass(frozen=True)
class Guard:
    """A named check with an explicit failure direction.

    ``enforcement`` is deliberately NOT defaulted. A default would let the
    next check inherit a classification nobody chose, which is the exact
    failure this type exists to prevent.
    """

    name: str
    enforcement: Enforcement
    why: str

    def __post_init__(self) -> None:
        if not isinstance(self.enforcement, Enforcement):
            raise GuardDeclarationError(
                f"guard {self.name!r} must declare Enforcement.CONTROL or "
                f"Enforcement.CONVENIENCE — a check whose failure direction "
                f"nobody chose is a control by accident or an outage by accident"
            )
        if not self.why.strip():
            raise GuardDeclarationError(
                f"guard {self.name!r} must say WHY it has that enforcement — "
                f"an unexplained classification is the next hiding place"
            )

    @property
    def fails_closed(self) -> bool:
        return self.enforcement is Enforcement.CONTROL

    def on_error(self, exc: Exception) -> bool:
        """Return True if the subject should be ALLOWED after *exc*.

        The single place the classification turns into behaviour, so a guard
        cannot be classified one way and implemented the other.
        """
        del exc
        return not self.fails_closed


# ── The registry. Every check on the Telegram surface appears here. ──────────

RATE_LIMIT_PER_CHAT = Guard(
    name="rate_limit.per_chat",
    enforcement=Enforcement.CONTROL,
    why="a single peer must not be able to exhaust the daemon",
)

RATE_LIMIT_GLOBAL = Guard(
    name="rate_limit.global",
    enforcement=Enforcement.CONTROL,
    why="aggregate load protection above the per-chat budget",
)

MEDIA_SIZE_PRECHECK = Guard(
    name="media.size_precheck",
    enforcement=Enforcement.CONTROL,
    why="a 2GB file must be refused BEFORE it is transferred, not after",
)

MEDIA_BYTE_CEILING = Guard(
    name="media.byte_ceiling",
    enforcement=Enforcement.CONTROL,
    why="file_size is attacker-supplied; the pre-check believes it, this does not",
)

MEDIA_MIME_DECLARED = Guard(
    name="media.mime_declared",
    enforcement=Enforcement.CONTROL,
    why="declared type is free and pre-download, so it is checked first",
)

MEDIA_MIME_SNIFFED = Guard(
    name="media.mime_sniffed",
    enforcement=Enforcement.CONTROL,
    why="declared type is attacker-controlled; magic bytes are not",
)

MEDIA_ALLOWLIST = Guard(
    name="media.allowlist",
    enforcement=Enforcement.CONTROL,
    why="the config claims this allowlist; it must actually deny",
)

CACHE_WRITE = Guard(
    name="cache.write",
    enforcement=Enforcement.CONVENIENCE,
    why="caching is an optimisation — an unwritable cache must not drop a message",
)

CACHE_EVICTION = Guard(
    name="cache.eviction",
    enforcement=Enforcement.CONVENIENCE,
    why="failing to evict must not block the write that triggered it",
)

CACHE_FREE_DISK_FLOOR = Guard(
    name="cache.free_disk_floor",
    enforcement=Enforcement.CONVENIENCE,
    why="refuse to CACHE below the floor, never refuse to SERVE",
)

ALL_GUARDS: tuple[Guard, ...] = (
    RATE_LIMIT_PER_CHAT,
    RATE_LIMIT_GLOBAL,
    MEDIA_SIZE_PRECHECK,
    MEDIA_BYTE_CEILING,
    MEDIA_MIME_DECLARED,
    MEDIA_MIME_SNIFFED,
    MEDIA_ALLOWLIST,
    CACHE_WRITE,
    CACHE_EVICTION,
    CACHE_FREE_DISK_FLOOR,
)
