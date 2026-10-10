"""Which routes a SCOPED device may use: one allowlist, default-deny, and the table that keeps it honest.

WHY THIS EXISTS
---------------
A device that was approved through ``POST /api/pair/requests`` holds a *scoped* token (``identity.is_operator``
is false). #692 scoped what that token can see in the session routes and left every other route open to any
valid token, so an approved phone could list every session's pending tool calls and approve them with a
persistent grant, create and run cron jobs (outside the shell floor, with the daemon's environment), and
overwrite provider keys. An independent review found it; ``tests/test_scoped_devices_default_deny.py`` has each
exploit.

THE RULE
--------
The bearer middleware (``web/server.py``) answers ``403 operator_only`` for a scoped token on any route that is
not in :data:`SCOPED_ALLOWED`. It is DEFAULT-DENY: a route added tomorrow, a method nobody listed, an unknown
path, ``/v1/*``, are all refused to a scoped device without anyone remembering to refuse them. The status is 403,
not 401, because the token is valid and a client that read 401 as "my token died" would throw away a healthy
credential.

What a scoped device is: a chat client for its own conversations. It may use hello; its own device (list itself,
sign itself out, its push and live-activity registration); chat; its own sessions (``SessionAccess`` already
answers 404 for anyone else's); and approve or deny the tool calls of ITS OWN sessions, with the scope
``once`` or ``until-restart`` only (:data:`SCOPED_APPROVE_SCOPES`). Everything else is the operator's: the global
token or an owner device.

Two things are deliberately wider, because the computer-use door already decided them (docs/design, W3):

* the desktop-task STOP is open to every valid token, because a stop can only end a task;
* a device an operator has MARKED for computer use is "a person", and :data:`MARKED_DEVICE_ALLOWED` is what that
  mark buys on REST (start and read a task, bind a session's computer, mark another device, and answer a desktop
  prompt). An unmarked device is denied those like everything else. The mark is the operator's explicit grant;
  an approved phone does not get it by being approved.

The handlers behind four of these routes add their own narrowing, because a route-level allow cannot say "own":
``/api/approvals*`` (own sessions' requests, no persistent or widened grant, no ``all``), the device routes
(own device only), and the activity route (own sessions only).

THE TABLE
---------
Default-deny makes a forgotten route SAFE; it does not make it VISIBLE. :data:`OPERATOR_ONLY` therefore lists,
explicitly, every other route the app registers, and ``tests/test_route_access.py`` fails when a registered route
is in no class, when a class names a route that does not exist, and when a route is in two. A new route has to be
classified on purpose. ``SCOPED_ALLOWED`` is also written out in that test, so widening what an approved phone can
do shows up as a changed line there.

Matching is the one ``web/public_routes.py`` uses: an exact method and path, where ``{name}`` is exactly one
non-empty path segment. ``HEAD`` is judged as the ``GET`` Starlette would run, and one trailing slash is the route
it redirects to.

Source: novel code for Prometheus, 2026-10-09.
"""

from __future__ import annotations

from collections.abc import Callable

from prometheus.web.public_routes import _matches

#: The approval scopes a scoped device may answer with. ``until-restart`` is process-wide (there is one gate per
#: process and its grants are never cleared), so the grant a device makes for its own call applies to every
#: matching call until the daemon restarts: this is the one place that reaches beyond the device's own sessions,
#: and the reason it is a named constant to narrow to ``("once",)`` rather than an inline literal.
SCOPED_APPROVE_SCOPES: tuple[str, ...] = ("once", "until-restart")

#: Open by design and not under the bearer prefixes at all (the dashboard page and the health probe).
OUTSIDE_THE_GATE: frozenset[tuple[str, str]] = frozenset({
    ("GET", "/"),
    ("GET", "/health"),
})

#: What a scoped device may use. Exact, and written out in full in tests/test_route_access.py.
SCOPED_ALLOWED: frozenset[tuple[str, str]] = frozenset({
    # its own device
    ("GET", "/api/devices"),                              # lists itself only
    ("DELETE", "/api/devices/{device_id}"),               # signs itself out (another id is 403)
    ("PUT", "/api/devices/{device_id}/push"),             # its own registration (another id is refused)
    ("DELETE", "/api/devices/{device_id}/push"),
    ("POST", "/api/devices/{device_id}/activity"),        # for a session it owns
    ("DELETE", "/api/devices/{device_id}/activity"),
    # chat
    ("POST", "/api/chat"),
    ("POST", "/api/chat/send"),
    ("POST", "/api/chat/interrupt"),
    # its own sessions
    ("GET", "/api/sessions"),
    ("POST", "/api/sessions"),
    ("DELETE", "/api/sessions/{session_id}"),
    ("GET", "/api/sessions/{session_id}/messages"),
    ("PUT", "/api/sessions/{session_id}/title"),
    ("PUT", "/api/sessions/{session_id}/pin"),
    ("GET", "/api/sessions/{session_id}/fork"),
    ("POST", "/api/sessions/{session_id}/fork"),
    ("GET", "/api/events/recent"),
    ("GET", "/api/activity/recent"),                      # session-filtered for a scoped caller, like events
    ("POST", "/api/search"),
    # the one desktop route open to every valid token: a stop can only end a task
    ("POST", "/api/computer/tasks/{task_id}/stop"),
    # the tool calls of its own sessions (the handlers narrow to own sessions and to SCOPED_APPROVE_SCOPES)
    ("GET", "/api/approvals"),
    ("POST", "/api/approvals/{request_id}/approve"),
    ("POST", "/api/approvals/{request_id}/deny"),
})

#: What a device an operator MARKED for computer use may additionally do over REST (the door's W3 ruling: a marked
#: device is a person). The handlers still check the mark and session ownership themselves; this only lets a marked
#: device past the gate. An unmarked scoped device is denied these.
MARKED_DEVICE_ALLOWED: frozenset[tuple[str, str]] = frozenset({
    ("GET", "/api/computer/apps"),
    ("POST", "/api/computer/tasks"),
    ("GET", "/api/computer/tasks/{task_id}"),
    ("GET", "/api/sessions/{session_id}/computer"),
    ("PUT", "/api/sessions/{session_id}/computer"),
    ("DELETE", "/api/sessions/{session_id}/computer"),
    ("PUT", "/api/devices/{device_id}/computer"),          # only a person marks another
})

#: The slash commands a scoped device may type. A message that is a command is not a conversation turn and runs
#: with the OPERATOR's authority (``/gate off`` turns the permission gate off for the whole process, ``/revoke``
#: removes a grant, and a dozen read daemon-wide state), so a scoped device's chat is default-deny here too: this
#: is ``/help`` and the commands that act on the device's own session. Any other known command is answered with a
#: refusal in the chat and does nothing. An unknown ``/word`` is still just text for the agent.
SCOPED_SLASH_COMMANDS: frozenset[str] = frozenset({
    "help",
    "reset", "clear",                        # empty its own session's history
    "steer", "queue", "unqueue", "clearsteers",
    "ephemeral",                             # its own session's flag
})

#: Everything else the app registers: the operator's (the global token or an owner device). Listed so that a
#: new route cannot arrive unclassified; the middleware denies a scoped device these whether or not they are
#: listed. Includes the operator halves of the pairing routes, which also check ``is_operator`` themselves.
OPERATOR_ONLY: frozenset[tuple[str, str]] = frozenset({
    # /api/activity
    # /api/approvals
    ("GET", "/api/approvals/grants"),
    ("DELETE", "/api/approvals/grants/{grant_id}"),
    # /api/artifacts
    ("GET", "/api/artifacts"),
    ("GET", "/api/artifacts/{artifact_id}"),
    # /api/backends
    ("GET", "/api/backends"),
    ("POST", "/api/backends/{name}/probe"),
    # /api/benchmarks
    ("POST", "/api/benchmarks/run"),
    # /api/chat
    ("POST", "/v1/chat/completions"),
    # /api/code
    ("POST", "/api/code"),
    ("GET", "/api/code/{task_id}"),
    ("GET", "/api/code/{task_id}/diff"),
    ("POST", "/api/code/{task_id}/inject"),
    ("POST", "/api/code/{task_id}/pause"),
    ("POST", "/api/code/{task_id}/resume"),
    ("POST", "/api/code/{task_id}/stop"),
    # /api/computer
    # /api/config
    ("GET", "/api/config"),
    # /api/cron
    ("GET", "/api/cron"),
    ("POST", "/api/cron"),
    ("DELETE", "/api/cron/{name}"),
    ("PUT", "/api/cron/{name}"),
    ("POST", "/api/cron/{name}/run"),
    # /api/devices
    ("POST", "/api/devices"),
    # /api/documents
    ("GET", "/api/documents"),
    ("GET", "/api/documents/content"),
    ("PUT", "/api/documents/content"),
    ("POST", "/api/documents/edit"),
    ("POST", "/api/documents/suggest"),
    # /api/files
    ("GET", "/api/files"),
    ("GET", "/api/files/read"),
    # /api/integrations
    ("GET", "/api/integrations/computer"),
    ("POST", "/api/integrations/computer/probe"),
    # /api/lcm
    ("GET", "/api/lcm/{session_id}"),
    # /api/learning
    ("POST", "/api/learning/live-upload"),
    ("GET", "/api/learning/skill-drafts"),
    ("GET", "/api/learning/skill-drafts/{draft_id}"),
    ("POST", "/api/learning/skill-drafts/{draft_id}/accept"),
    ("POST", "/api/learning/skill-drafts/{draft_id}/reject"),
    ("POST", "/api/learning/video-ingest"),
    # /api/mcp
    ("GET", "/api/mcp/servers"),
    ("POST", "/api/mcp/servers"),
    ("DELETE", "/api/mcp/servers/{name}"),
    ("PATCH", "/api/mcp/servers/{name}"),
    # /api/media
    ("GET", "/api/media"),
    # /api/memory
    ("GET", "/api/memory/current"),
    ("PUT", "/api/memory/current"),
    # /api/models
    ("GET", "/api/models"),
    ("GET", "/v1/models"),
    # /api/network
    ("GET", "/api/network"),
    ("PUT", "/api/network"),
    # /api/packs
    ("GET", "/api/packs"),
    # /api/pair
    ("GET", "/api/pair/requests"),
    ("POST", "/api/pair/requests/{request_id}/approve"),
    ("POST", "/api/pair/requests/{request_id}/deny"),
    # /api/pairs
    ("GET", "/api/pairs"),
    # /api/paperclip
    ("POST", "/api/paperclip/wake"),
    # /api/profiles
    ("GET", "/api/profiles"),
    ("PUT", "/api/profiles/active"),
    # /api/project-file
    ("GET", "/api/project-file"),
    ("PUT", "/api/project-file"),
    # /api/projects
    ("GET", "/api/projects"),
    ("POST", "/api/projects"),
    ("DELETE", "/api/projects/{project_id}"),
    ("PUT", "/api/projects/{project_id}"),
    # /api/providers
    ("GET", "/api/providers/keys"),
    ("PUT", "/api/providers/keys/{service_id}"),
    ("DELETE", "/api/providers/xai/oauth"),
    ("GET", "/api/providers/xai/oauth"),
    ("POST", "/api/providers/xai/oauth/login"),
    # /api/sentinel
    ("GET", "/api/sentinel"),
    # /api/sessions
    ("GET", "/api/sessions/{session_id}/checkpoints"),
    ("GET", "/api/sessions/{session_id}/checkpoints/{checkpoint_id}"),
    ("POST", "/api/sessions/{session_id}/checkpoints/{checkpoint_id}/restore"),
    ("DELETE", "/api/sessions/{session_id}/model"),
    ("GET", "/api/sessions/{session_id}/model"),
    ("POST", "/api/sessions/{session_id}/model"),
    ("GET", "/api/sessions/{session_id}/profile"),
    ("PUT", "/api/sessions/{session_id}/profile"),
    ("POST", "/api/sessions/{session_id}/purge"),
    ("DELETE", "/api/sessions/{session_id}/workspace"),
    ("GET", "/api/sessions/{session_id}/workspace"),
    ("PUT", "/api/sessions/{session_id}/workspace"),
    # /api/skills
    ("GET", "/api/skills"),
    ("GET", "/api/skills/list"),
    ("GET", "/api/skills/{name}"),
    ("DELETE", "/api/skills/{name}/pin"),
    ("POST", "/api/skills/{name}/pin"),
    # /api/status
    ("GET", "/api/status"),
    # /api/stories
    ("GET", "/api/stories"),
    ("POST", "/api/stories"),
    ("POST", "/api/stories/reorder"),
    ("DELETE", "/api/stories/{story_pk}"),
    ("PUT", "/api/stories/{story_pk}"),
    ("POST", "/api/stories/{story_pk}/dispatch"),
    ("POST", "/api/stories/{story_pk}/undispatch"),
    # /api/tasks
    ("GET", "/api/tasks"),
    ("GET", "/api/tasks/{task_id}"),
    ("POST", "/api/tasks/{task_id}/stop"),
    # /api/telemetry
    ("GET", "/api/telemetry"),
    # /api/tools
    ("GET", "/api/tools/deferred"),
    ("PUT", "/api/tools/deferred"),
    ("GET", "/api/tools/recent"),
    # /api/usage
    ("GET", "/api/usage"),
    # /api/wiki
    ("GET", "/api/wiki/page"),
    ("GET", "/api/wiki/pages"),
    ("GET", "/api/wiki/search"),
    ("GET", "/api/wiki/stats"),
})


def _listed(table: frozenset[tuple[str, str]], method: str, path: str) -> bool:
    verb = (method or "").upper()
    if verb == "HEAD":
        verb = "GET"
    if len(path) > 1 and path.endswith("/") and not path.endswith("//"):
        path = path[:-1]
    return any(m == verb and _matches(pattern, path) for m, pattern in table)


def is_scoped_allowed(method: str, path: str, *, marked: Callable[[], bool] | None = None) -> bool:
    """May a scoped device call *method* *path*? *path* is the ROUTED path (``scope["path"]``), never the URL.

    *marked* answers "has an operator marked THIS device for computer use?"; it is called only for a route in
    :data:`MARKED_DEVICE_ALLOWED`, so the common request never touches the registry for it. A failing answer is no.
    """
    if _listed(SCOPED_ALLOWED, method, path):
        return True
    if marked is not None and _listed(MARKED_DEVICE_ALLOWED, method, path):
        try:
            return bool(marked())
        except Exception:                                  # noqa: BLE001 - fail closed on any doubt
            return False
    return False
