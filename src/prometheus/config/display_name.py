"""The name a Prometheus shows to people who are not yet paired.

``GET /api/hello``, the mDNS record and the operator's Approve prompt all show it, so there is ONE
function. Order: the owner's ``pairing.display_name``, then the computer's own name (macOS
``scutil --get ComputerName``), then the host name, then ``Prometheus``.

It is put on other people's screens, and by an unauthenticated route, so:

* control, format and line-separator characters are removed (a right-to-left override makes one name
  render as another, which is a spoofing tool), and it is at most 64 characters;
* a missing or failing ``scutil`` is a fallback, never an error;
* the computer name is cached for a minute, because a subprocess per unauthenticated request would let
  an outsider spawn processes on this machine.

Source: novel code for Prometheus, 2026-10-08.
"""

from __future__ import annotations

import platform
import subprocess
import time
import unicodedata
from typing import Any

DEFAULT_NAME = "Prometheus"
MAX_CHARS = 64
_CACHE_SECONDS = 60.0
_SCUTIL_TIMEOUT_SECONDS = 2

_cache: tuple[float, str] | None = None


def reset_cache() -> None:
    """Forget the cached computer name (tests; a daemon never needs it)."""
    global _cache
    _cache = None


def clean_label(value: Any) -> str:
    """*value* made safe to show: no control/format/separator characters, trimmed, at most 64 characters.

    Anything that is not text is the empty string.
    """
    if not isinstance(value, str):
        return ""
    kept = "".join(
        ch for ch in value
        if unicodedata.category(ch)[0] != "C" and unicodedata.category(ch) not in ("Zl", "Zp")
    )
    return kept.strip()[:MAX_CHARS].rstrip()


def _lookup_computer_name() -> str:
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["scutil", "--get", "ComputerName"],
                capture_output=True, text=True, timeout=_SCUTIL_TIMEOUT_SECONDS, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0:
            name = clean_label(result.stdout)
            if name:
                return name
    return clean_label(platform.node())


def _computer_name() -> str:
    global _cache
    now = time.monotonic()
    if _cache is not None and now - _cache[0] < _CACHE_SECONDS:
        return _cache[1]
    name = _lookup_computer_name()
    _cache = (now, name)
    return name


def display_name(config: dict[str, Any] | None) -> str:
    """The name to show for this Prometheus. Never empty."""
    pairing = (config or {}).get("pairing") if isinstance(config, dict) else None
    configured = clean_label(pairing.get("display_name")) if isinstance(pairing, dict) else ""
    return configured or _computer_name() or DEFAULT_NAME
