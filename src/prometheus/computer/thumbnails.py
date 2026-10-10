"""Cockpit level 2 — a thumbnail of the APPROVED WINDOW after each step.

computer-use v1.1 PR 6b (design §5.2.5). This module is the pure half: the
skip taxonomy, the frame's exact key set, the size floor, and the opt-in
persist path. It imports nothing from the driver, the bus or the socket, so
every rule below is CI-coverable — the driver leg that actually produces the
pixels is not (see ``computer/cua.py``'s module docstring).

THE THREE THINGS THAT MAKE THIS SAFE
------------------------------------
1. **It is a picture for a PERSON and never reaches a model.** Not the
   chooser, not step history, not a prompt, not an audit row. ``observe``
   still asks for no screenshot.
2. **The frame is direct-only.** ``computer_step_thumbnail`` is deliberately
   OUTSIDE ``COMPUTER_FRAME_KINDS``, so ``web.ws_server`` never promotes it,
   it is never in ``signal_events``, never in ``GET /api/events/recent``, and
   never in a backfill. SignalBus persists everything it is given; a
   screenshot must not be given to it.
3. **Redaction is skip, never blur.** A masked image still shows the layout
   around a secret, and a blur only promises the pixels underneath are gone.
   So any doubt — a password field anywhere in the walk, a walk that cannot
   vouch for its own completeness — means NO picture, and the step frame says
   so with a reason.

THE CAPTURE SCOPE
-----------------
The bound app's frontmost on-screen window, and nothing else. Never the
desktop, a display or another app's window; nothing here calls
``get_desktop_state`` or a screen-capture verb.

⚠ NOT ESTABLISHED ON 0.28.2, and deferred to the on-box check: whether an X11
window capture can contain another window's pixels (an overlap, a
notification). The guard we DO have is the driver's own
``screenshot_frame_valid``; if it reports false the capture is skipped
(``frame_invalid``). If the on-box check shows overlap is possible while that
flag is true, the flag is not sufficient and this needs a stronger guard.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

#: The ONE spelling rule for an app term — the binding's names arrive folded,
#: so the driver's raw app_name must be folded the same way before they are
#: compared (see decide_capture).
from prometheus.permissions.computer_extent import normalise_term

# ── floors and defaults ────────────────────────────────────────────────────
# The 256 KB cap and the password skip are FLOORS, not config keys: a
# misconfiguration must not be able to lift them. ``max_dimension`` is a key
# because it is a bandwidth/legibility tradeoff, not a safety one.
MAX_THUMBNAIL_BYTES = 256 * 1024
DEFAULT_MAX_DIMENSION = 480

#: Every reason a step can carry ``thumbnail: "skipped"``. Closed set: a new
#: skip reason is a new thing a person can be told, so it is added here
#: deliberately and the test below pins the set.
SKIP_REASONS = frozenset({
    # A node anywhere in the walk — tokenless ones included — has a role
    # containing "password" (AT-SPI spells it "password text").
    "password_field",
    # The walk cannot vouch for its own completeness, so a password field
    # cannot be RULED OUT. Positive evidence is required; absence of evidence
    # is not evidence of absence.
    "incomplete_walk",
    # The window vanished, or the driver reports it now belongs to another
    # app. Never capture a window the person did not pick.
    "window_changed",
    # The driver returned no usable inline image (call failed, or — see the
    # 0.28.2 caveat above — only a file path we do not read).
    "capture_failed",
    # The driver says the frame is not trustworthy (an overlap it detected).
    "frame_invalid",
    # Over the 256 KB floor. Skipped whole — never sent in pieces.
    "too_large",
    # No eligible device is connected, so nothing is captured at all. The
    # picture exists only to be watched.
    "no_viewer",
})

#: The frame's EXACT payload keys. Everything else is excluded on purpose:
#: no pid, no window id, no window title (page-authored text in a browser),
#: no snapshot id, no element data. A test asserts the built frame matches
#: this set exactly, in both directions, so a future field cannot ride along.
FRAME_KEYS = frozenset({
    "session_id", "task_id", "seq", "app", "mime_type",
    "width", "height", "data_base64", "captured_at",
})

_TASK_ID_RE = re.compile(r"^[0-9a-fA-F-]{1,64}$")
_EXT_FOR_MIME = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}


#: The three things a step's thumbnail can be, before any capture happens.
#: The step frame renders these as ``thumbnail: null`` (NONE — the feature is
#: off or the step did not execute, so no decision was ever made),
#: ``thumbnail: "skipped"`` + a reason (SKIP — decided against), or a capture
#: attempt that becomes ``"sent"`` or a post-capture SKIP. ``None`` could not
#: carry all three without ambiguity, which is why this is a named tri-state
#: rather than an Optional return.
CAPTURE = "capture"
SKIP = "skip"
NONE = "none"


def capture_plan(*, status: str, enabled: bool, viewers: int) -> tuple[str, str | None]:
    """Decide, BEFORE capturing, what this step's thumbnail is.

    Returns ``(action, reason)`` where action is CAPTURE, SKIP or NONE.

    Only an ``executed`` step gets one. Not a refused, abstained or blocked
    step (nothing changed on screen), not the call that was in flight at a
    stop (its outcome is unknown and the person asked us to stop), and not
    while an approval is pending (the screen is not ours to show yet). Those
    are NONE, not SKIP: no decision was made, so the step frame says
    ``thumbnail: null`` rather than claiming a skip.

    ``enabled`` False is NONE for the same reason — the feature is off, not
    decided against.

    ``no_viewer`` IS a skip: the feature is on and the step executed, we just
    did not capture because nobody eligible is watching. The picture exists
    only to be watched, so not capturing is the right call and the step frame
    says why.
    """
    if not enabled:
        return NONE, None
    if status != "executed":
        return NONE, None
    if viewers <= 0:
        return SKIP, "no_viewer"
    return CAPTURE, None


def walk_hides_a_password(roles: Any) -> bool:
    """Does ANY node in the walk have a password role?

    The WHOLE walk, tokenless nodes included — a password field is exactly
    the kind of node a filter might drop from the clickable set, and the skip
    has to see what the chooser never sees.

    ⚠ Deliberately does NOT consult ``elements_complete``: 0.28.2 hard-codes
    it False on Linux (see the pinned fixture in ``test_cua_adapter.py``), so
    gating on it would skip every thumbnail on the only platform this runs on.
    That field is carried, never relied on.
    """
    for role in roles or ():
        if isinstance(role, str) and "password" in role.lower():
            return True
    return False


def walk_is_positive_evidence(capture: Any) -> bool:
    """Can this walk vouch for its own completeness?

    ``degraded`` or ``truncated`` means a password field could be hiding in
    the part we did not get, so the absence of one is not evidence. Skip —
    never guess.
    """
    return not (getattr(capture, "degraded", False)
                or getattr(capture, "truncated", False))


def decoded_size(data_base64: str) -> int | None:
    """The frame's byte length, or None if it is not valid base64.

    Measured on the DECODED bytes: the cap is about the picture, and base64
    inflates it by ~4/3, so capping the encoded string would let a 340 KB
    image through.
    """
    try:
        return len(base64.b64decode(data_base64, validate=True))
    except Exception:  # noqa: BLE001 - malformed is "no size", not a crash
        return None


def frame_valid(capture: Any) -> bool:
    """Does the driver vouch for this frame?

    ``frame_valid`` is False when the driver knows the capture is not
    trustworthy (on the SDK side, ``screenshot_frame_valid``). None — the
    field absent or unset — is NOT a refusal: it means the driver did not
    report either way, and refusing on None would skip every thumbnail on a
    driver that does not populate it. Only an explicit False skips.
    """
    return getattr(capture, "frame_valid", None) is not False


def decide_capture(
    capture: Any, *, app_names: Any, max_bytes: int = MAX_THUMBNAIL_BYTES,
) -> tuple[str | None, dict[str, Any] | None]:
    """Apply every post-capture rule to one ``WindowCapture``.

    Returns ``(skip_reason, image)``: a reason and no image, or no reason and
    the image to send. All the checks live here, in this order, and the order
    is deliberate:

    1. ``window_changed`` — the driver's own ``app_name`` at capture time is
       not one of the names the door resolved this app by (D19). An app_name
       of None is also a change: a window the driver cannot attribute is a
       window we did not prove is ours.
    2. ``incomplete_walk`` — degraded or truncated, so a password field cannot
       be ruled out.
    3. ``password_field`` — anywhere in the walk, tokenless nodes included.
    4. ``frame_invalid`` — the driver says the frame is not trustworthy.
    5. ``capture_failed`` — no usable inline image.
    6. ``too_large`` — over the byte floor.

    The privacy gates (1-4) run BEFORE the mechanical ones (5-6) because the
    mechanical answers are about transport, not about whether these pixels may
    leave the machine. A frame that fails a privacy gate must be dropped
    without ever being examined for size, and the reason a person is told
    should be the one that actually decided it — "too large" would be a lie
    about a picture we refused to show for a different reason.
    """
    names = {normalise_term(n) for n in (app_names or ())}
    reported = getattr(capture, "app_name", None)
    # Fold BOTH sides the same way the binding's own covers() does: the names
    # arrive normalised (Binding._app_names) while the driver reports a raw
    # display name ("GNOME Text Editor" vs the picked "gedit"). Comparing raw
    # would skip every real capture as window_changed.
    reported = normalise_term(reported) if isinstance(reported, str) else None
    # Fail CLOSED: with no names to compare against we cannot prove this
    # window is the one the person picked, and an app_name the driver cannot
    # attribute is the same position. Both skip rather than guess — positive
    # evidence, as everywhere else in this file.
    if reported is None or reported not in names:
        return "window_changed", None
    if not walk_is_positive_evidence(capture):
        return "incomplete_walk", None
    if walk_hides_a_password(getattr(capture, "roles", None)):
        return "password_field", None
    if not frame_valid(capture):
        return "frame_invalid", None
    data = getattr(capture, "image_base64", None)
    mime = getattr(capture, "image_mime", None)
    if not isinstance(data, str) or not data or mime not in _EXT_FOR_MIME:
        # No usable inline image, or one we cannot label honestly. We do NOT
        # fall back to a screenshot_file_path: a file the driver wrote to disk
        # is a second copy of the desktop whose lifetime we do not control.
        return "capture_failed", None
    size = decoded_size(data)
    if size is None or size > max_bytes:
        return "too_large", None
    return None, {
        "mime_type": mime, "data_base64": data,
        "width": getattr(capture, "image_width", None),
        "height": getattr(capture, "image_height", None),
    }


def build_frame(
    *, session_id: str, task_id: str, seq: int, app: str,
    image: dict[str, Any], captured_at: str,
) -> dict[str, Any]:
    """The ``computer_step_thumbnail`` payload, with exactly FRAME_KEYS.

    ``image`` is the dict ``decide_capture`` returns on success. ``app`` is the
    bound app as the extent names it — not a window title, which in a browser
    is page-authored text and would cross a content boundary the rest of the
    log is careful about.
    """
    frame: dict[str, Any] = {
        "session_id": session_id,
        "task_id": task_id,
        "seq": seq,
        "app": app,
        "mime_type": image.get("mime_type"),
        "width": image.get("width"),
        "height": image.get("height"),
        "data_base64": image.get("data_base64"),
        "captured_at": captured_at,
    }
    # Asserted, not documented: a frame that grows a pid or a title is a
    # content leak, and this is the one place that can catch it structurally.
    assert set(frame) == FRAME_KEYS, (
        f"thumbnail frame keys drifted: extra={set(frame) - FRAME_KEYS} "
        f"missing={FRAME_KEYS - set(frame)}")
    return frame


def persist_path(data_dir: str, task_id: str, seq: int,
                 mime_type: str) -> str | None:
    """Where an opted-in thumbnail is written, or None if it must not be.

    ``<data>/computer/thumbnails/<task_id>/<seq>.<ext>``, written 0600 by the
    caller. Off by default (``thumbnails.persist: false``); even when on, no
    route serves these in v1.1 and they never enter ``signal_events``.

    ``task_id`` is our own hex, but it becomes a path component, so it is
    validated rather than trusted — a traversal here would write a desktop
    screenshot somewhere outside the thumbnails tree. The extension comes
    from the closed mime map, never from the driver's string.
    """
    ext = _EXT_FOR_MIME.get(mime_type)
    if ext is None or not _TASK_ID_RE.match(task_id) or not isinstance(seq, int):
        return None
    return f"{data_dir.rstrip('/')}/computer/thumbnails/{task_id}/{seq}.{ext}"


# ── configuration ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ThumbnailConfig:
    """``computer_use.thumbnails.*``.

    ``enabled`` and ``persist`` are booleans; ``max_dimension`` is the one
    tuning knob (a bandwidth/legibility tradeoff, not a safety one). The 256 KB
    cap and the password skip are FLOORS and deliberately have no key — a
    misconfiguration must not be able to lift them.
    """

    enabled: bool = True
    max_dimension: int = DEFAULT_MAX_DIMENSION
    persist: bool = False

    @classmethod
    def from_config(cls, config: Any,
                    errors: list[str] | None = None) -> "ThumbnailConfig":
        block = (((config or {}).get("computer_use") or {})
                 if isinstance(config, dict) else {})
        raw = ((block.get("thumbnails") or {})
               if isinstance(block, dict) else {})
        if not isinstance(raw, dict):
            raw = {}

        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            if errors is not None:
                errors.append("computer_use.thumbnails.enabled must be a "
                              "boolean; using true")
            enabled = True

        persist = raw.get("persist", False)
        if not isinstance(persist, bool):
            if errors is not None:
                errors.append("computer_use.thumbnails.persist must be a "
                              "boolean; using false")
            persist = False
            # An opt-in STORAGE path that failed to parse must not silently
            # become the default-and-keep-going: it is the one key here whose
            # wrong answer writes desktop pixels to disk. Fail to OFF.

        dim = raw.get("max_dimension", DEFAULT_MAX_DIMENSION)
        if isinstance(dim, bool) or not isinstance(dim, int) or not (16 <= dim <= 2048):
            if errors is not None:
                errors.append("computer_use.thumbnails.max_dimension must be "
                              "an integer between 16 and 2048; using "
                              f"{DEFAULT_MAX_DIMENSION}")
            dim = DEFAULT_MAX_DIMENSION

        return cls(enabled=enabled, max_dimension=dim, persist=persist)


# ── the transport seam ─────────────────────────────────────────────────────

@runtime_checkable
class ThumbnailSink(Protocol):
    """Where the runner sends a thumbnail. ``web.ws_server`` implements it.

    Two methods, and the split is the point:

    * ``viewer_count(session_id)`` is asked BEFORE any capture, so a task
      nobody is watching never calls the driver at all (``no_viewer``). The
      picture exists only to be watched.
    * ``send_thumbnail(session_id, frame)`` is a DIRECT send. It does not go
      through ``broadcast`` (every socket), the SignalBus (which persists
      everything it is given) or the bridge's frame path. A thumbnail reaches
      only the sockets that (a) authenticated with a device token, (b) that
      device is marked ``computer: true``, (c) declared the
      ``computer-thumbnails`` capability, and (d) are attached to this
      session. A global-token connection NEVER receives one — a model holding
      that token must not be able to watch the screen through it (D15).

    The method names are the bridge's own, not an abstraction of them: naming
    it ``send`` here while the bridge implements ``send_thumbnail`` would have
    type-checked and then failed at the first desktop task. A test pins that
    ``WebSocketBridge`` really does satisfy this protocol.

    Keeping this a protocol means ``livestream`` and ``task`` never import
    ws_server, and the eligibility rules below stay testable with a fake.
    """

    def viewer_count(self, session_id: str) -> int:
        raise NotImplementedError

    async def send_thumbnail(self, session_id: str,
                             frame: dict[str, Any]) -> None:
        raise NotImplementedError


class _NoSink(ThumbnailSink):
    """The sink when no bridge is wired: nobody is watching, ever."""

    def viewer_count(self, session_id: str) -> int:
        return 0

    async def send_thumbnail(self, session_id: str,
                             frame: dict[str, Any]) -> None:
        return None


NO_SINK = _NoSink()
