"""The worker → daemon approval channel's rules, in one place (#720 review).

A spawned worker asks its parent conversation for approval over
``POST /api/workers/approvals``. What keeps that channel narrow:

* **The token is never in the worker's environment.** On Linux
  ``/proc/<pid>/environ`` (and ``ps eww``) shows the block a process STARTED
  with, so popping a variable later does not hide it from the model's shell,
  which can read ``/proc/$PPID/environ``. The manager writes the token as the
  first stdin line, ahead of the brief (:func:`token_line`); the worker reads
  and strips it before anything else runs.
* **Who is asking travels in headers** (:data:`HEADER_TASK`,
  :data:`HEADER_TOKEN`), so the daemon checks it before it reads a body.
* **Bounded:** :data:`MAX_PENDING_PER_WORKER` open requests, a per-worker
  :data:`RATE_LIMIT`, a :data:`MAX_BODY_BYTES` cap that holds for a chunked
  body too.
* **Card text is the model's words** (tool name, reason, the task's
  description): :func:`clean_card_text` caps it and strips control
  characters, newlines and bidi controls before Telegram or Beacon shows it.

Source: novel code for Prometheus, 2026-10-10.
"""

from __future__ import annotations

import ipaddress
import threading
import time
import unicodedata
from collections import deque

TOKEN_LINE_PREFIX = "PROMETHEUS_WORKER_TOKEN "
HEADER_TASK = "X-Prometheus-Worker-Task"
HEADER_TOKEN = "X-Prometheus-Worker-Token"

MAX_PENDING_PER_WORKER = 3
#: (requests, per seconds), per worker. Every request waits for a person, so
#: a worker that needs more than this is not asking a person anything.
RATE_LIMIT: tuple[int, float] = (10, 60.0)
MAX_BODY_BYTES = 64 * 1024

TOOL_NAME_CHARS = 64
REASON_CHARS = 500
DESCRIPTION_CHARS = 200


def token_line(token: str) -> str:
    """The first stdin line a worker reads: its approval token."""
    return f"{TOKEN_LINE_PREFIX}{token}\n"


def clean_card_text(text: object, limit: int) -> str:
    """*text* on one line, without control or format characters, at most *limit*."""
    out = []
    for ch in str(text or ""):
        cat = unicodedata.category(ch)
        # Cc: controls (newlines, ESC, BEL); Cf: format (bidi overrides,
        # zero-width); Zl/Zp: line and paragraph separators.
        out.append(" " if cat in ("Cc", "Cf", "Zl", "Zp") else ch)
    clean = " ".join("".join(out).split())
    if len(clean) > limit:
        clean = clean[: max(0, limit - 1)].rstrip() + "…"
    return clean


def worker_approval_url(host: str | None, port: int) -> str:
    """The endpoint a worker on THIS machine reaches: loopback for a wildcard
    bind, the bound address otherwise, an IPv6 address in brackets."""
    h = (host or "").strip()
    if h in ("", "0.0.0.0", "::", "[::]"):
        h = "127.0.0.1"
    h = h.strip("[]")
    try:
        if ipaddress.ip_address(h).version == 6:
            h = f"[{h}]"
    except ValueError:
        pass  # a host name
    return f"http://{h}:{port}/api/workers/approvals"


class RateLimiter:
    """A sliding-window count per key, read against :data:`RATE_LIMIT` per call."""

    def __init__(self) -> None:
        self._seen: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        limit, window = RATE_LIMIT
        now = time.monotonic()
        with self._lock:
            q = self._seen.setdefault(key, deque())
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True
