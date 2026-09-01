"""Which sessions are MACHINE traffic rather than a conversation with the user.

This started life private to :mod:`prometheus.memory.extractor`, where it kept eval chatter out
of the wiki. It is promoted here because the same question has three more callers coming:

* retention — machine sessions are disposable and should not need a human to delete them
* ``/api/status`` — "what fraction of the store is disposable" is a health figure
* the clients — a probe that names itself ``smoke:`` is asking to be treated as disposable

The audit that motivated the move: 92% of the durable store was tombstoned probe traffic, and
213 of those sessions were purged by hand. Of them only 68 (``bakeoff:``) were nameable by this
list — the rest arrived as ``desktop:``, ``ios:`` and ``beacon:`` because the harnesses that
created them chose ordinary-looking ids. The daemon CANNOT tell those from a real conversation,
which is the whole reason the naming contract has to be honoured by the client.
"""

from __future__ import annotations

#: Reserved namespaces. A session id starting with one of these is fixture/eval material.
MACHINE_SESSION_PREFIXES: tuple[str, ...] = (
    "bakeoff:",
    "coding:",
    "eval:",
    "gym:",
    "smoke:",
)

#: Reserved bare ids (not a namespace, an exact name).
MACHINE_SESSION_IDS = frozenset({"system"})


def is_machine_session(session_id: str | None) -> bool:
    """True iff *session_id* belongs to a machine harness, not a user chat."""
    sid = (session_id or "").strip()
    return sid in MACHINE_SESSION_IDS or sid.startswith(MACHINE_SESSION_PREFIXES)
