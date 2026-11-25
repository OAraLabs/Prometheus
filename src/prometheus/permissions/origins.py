"""Origin-scoped capability sets — where a request came from decides what it may touch.

The OAra Voice rule ("no terminal from the kitchen"), written as POLICY in
the gate, not as per-device config. A turn's origin is derived from the
AUTHENTICATED DEVICE CREDENTIAL — the identity the bearer middleware already
resolved (``config/api_token.verify_token``) — never from a session id, a
request body, or anything in a transcript. A node cannot claim an origin;
the daemon reads it off the token the node holds.

Origins:

* ``terminal``   — the global token or an owner device: the full kit.
* ``voice_box``  — a scoped device marked as a voice box: read-only, timers,
  reminders, home status, memory read. Anyone in the room can talk to it;
  misheard audio is a real risk, so it gets the smallest surface.
* ``android_ptt`` / ``glasses`` / ``ios_ptt`` — a deliberate press on a
  device a person carries: the voice-box set plus messaging and memory
  writes. Memory writes still pass the owner-approval card the memory tool
  already requires — the origin widens the ADVERTISEMENT, never the consent.

A scoped device with no origin mark keeps today's behaviour on the TYPED
chat routes (the gate's system origin still applies in full): an ordinary
approved phone is not a voice box. On the VOICE event route an unmarked
scoped credential fails closed to ``voice_box`` — the smallest surface,
never the full kit. The mark itself is the operator's explicit grant (its
own table on the device registry, set only through an operator route), the
same shape as the computer-use mark. A marked device's turns are voice
turns on every route: the mark declares what the node IS.

Enforcement is the profile mechanism the loop already has: the origin's
capability set is an ``AgentProfile`` with ``held=True``, resolved per run
from the credential, applied to the advertised schema AND the call path
(``profile_permits``), with no fallback that widens it. Risky actions a
voice origin may not run directly go to the Beacon approval queue through
the gate's existing APPROVE tier — the origin never bypasses the floor.

Source: novel code for Prometheus, 2026-11-24.
"""

from __future__ import annotations

from dataclasses import replace

from prometheus.config.profiles import AgentProfile

#: The origins a credential may resolve to. ``terminal`` is the operator's.
ORIGINS = ("terminal", "voice_box", "android_ptt", "glasses", "ios_ptt")

#: What a voice box may do: read, remember, schedule — never write files,
#: run a shell, or touch the desktop. Timers/reminders ride cron_create with
#: the gate's own approval tier still in force.
_VOICE_BOX_TOOLS = [
    "read_file", "grep", "glob",
    "wiki_query", "lcm_grep", "lcm_expand", "lcm_describe", "lcm_expand_query",
    "vault_search", "vault_read",
    "memory",                      # list only in effect: writes need the owner's card
    "cron_list", "cron_create",    # timers and reminders; the gate still rules on each
    "sentinel_status", "anatomy", "dashboard",
    "web_search", "web_fetch",
    "todo_write",
]

#: A deliberate press on a carried device adds messaging. Memory writes are
#: advertised but still require the owner-approval card (the memory tool's
#: own rule, unchanged).
_PTT_EXTRA = ["message"]

_CAPABILITY_SETS: dict[str, AgentProfile] = {
    "voice_box": AgentProfile(
        name="origin:voice_box",
        description="Always-on voice box: read-only, timers, reminders, home status, memory read.",
        bootstrap_files=[],
        tools=list(_VOICE_BOX_TOOLS),
        held=True,
        bootstrap_declared=True,   # names no file: a voice turn is not sent the owner's prompt
    ),
    "android_ptt": AgentProfile(
        name="origin:android_ptt",
        description="Push-to-talk on a carried Android device: voice-box set plus messaging.",
        bootstrap_files=[],
        tools=_VOICE_BOX_TOOLS + _PTT_EXTRA,
        held=True,
        bootstrap_declared=True,
    ),
    "glasses": AgentProfile(
        name="origin:glasses",
        description="Glasses + ring PTT: the carried-device set.",
        bootstrap_files=[],
        tools=_VOICE_BOX_TOOLS + _PTT_EXTRA,
        held=True,
        bootstrap_declared=True,
    ),
    "ios_ptt": AgentProfile(
        name="origin:ios_ptt",
        description="iPhone push-to-talk: the carried-device set.",
        bootstrap_files=[],
        tools=_VOICE_BOX_TOOLS + _PTT_EXTRA,
        held=True,
        bootstrap_declared=True,
    ),
}


def origin_for_identity(identity: object | None, *, marked_origin: str | None = None) -> str | None:
    """The origin of an authenticated identity on the TYPED routes, or None.

    *identity* is the ``DeviceIdentity`` the bearer middleware resolved (or
    None when auth is off / nobody resolved). The global token and an owner
    device are ``terminal``. A scoped device is whatever the operator marked
    it as; an UNMARKED scoped device is None — today's typed-chat behaviour,
    unchanged (the gate's system origin still governs it). An unknown mark
    is None as well: a mark the policy does not know grants nothing.

    ``marked_origin`` comes ONLY from the device registry (the operator's
    mark), never from the request.
    """
    if identity is None:
        # Auth off: the daemon is deliberately open, like every other route;
        # the caller is the operator at the machine.
        return "terminal"
    if getattr(identity, "is_operator", False):
        return "terminal"
    if marked_origin in _CAPABILITY_SETS:
        return str(marked_origin)
    return None


def voice_origin_for(identity: object | None, *, marked_origin: str | None = None) -> str:
    """The origin of a VOICE event. Fail closed: an unmarked scoped credential
    is ``voice_box``, the smallest surface — a node that speaks to the daemon
    without the operator having said what it is gets the kitchen-safe set."""
    origin = origin_for_identity(identity, marked_origin=marked_origin)
    return origin if origin is not None else "voice_box"


def capability_profile(origin: str) -> AgentProfile | None:
    """The held profile for *origin*, or None for ``terminal`` (no limit).

    The returned profile is a copy: a caller cannot mutate the shared set.
    """
    profile = _CAPABILITY_SETS.get(origin)
    return replace(profile) if profile is not None else None
