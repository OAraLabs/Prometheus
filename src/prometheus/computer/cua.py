"""The real desktop driver: a ``Driver`` adapter over the Cua SDK.

⚠ THIS MODULE IS STRUCTURALLY UNCOVERABLE BY CI, AND THAT IS NOT A GAP TO
   BE PAPERED OVER — IT IS A FACT TO STATE.

Every line below that matters needs a display, an accessibility bus and a
live desktop. CI has none of those and cannot acquire them. So **the on-box
outcome check is the only evidence this works** — the same shape as the MCP
stdio transport, whose only proof is boot logs, and which is exactly how
``mcp>=1.0`` resolved to 2.x with a green suite.

What the tests below this module CAN cover is the translation: that a
Prometheus action dict becomes the right SDK call with the right arguments,
and that an SDK result becomes the right verdict. That is worth having and it
is not the same as "the driver works". Do not let a green suite read as the
second thing.

WHAT IT DELIBERATELY DOES NOT OFFER
------------------------------------
The SDK exposes ~50 methods on ``CuaDriver``. This adapter implements the
seven-verb ``Driver`` Protocol and nothing else, so the loop has nowhere to
call ``clipboard_read``, ``drag``, ``get_desktop_state`` or the browser
surface even if something wanted to. The narrowness IS the boundary; a
passthrough would have quietly restored the whole tool set.

⚠ THE SDK IS ASYNC AND THE ``Driver`` PROTOCOL IS SYNC
--------------------------------------------------------
Every SDK method that does anything — ``get_window_state``, ``click``,
``press_key`` — is a coroutine; only ``create``/``is_available``/
``execution_mode``/``socket_path`` are synchronous. Found by calling one and
getting ``'coroutine' object has no attribute 'windows'``, which is to say:
by running it, not by reading the signatures.

The merged ``Driver`` Protocol is synchronous, so this adapter owns a
dedicated event loop on its own thread and blocks the caller on it. That
keeps the contract the loop was built against, unchanged.

⚠ AND IT MEANS A DAEMON INTEGRATION NEEDS ONE MORE THING THIS MILESTONE DOES
NOT BUILD. ``ComputerUseLoop.step`` calls ``driver.observe()`` directly, so a
blocking driver blocks whatever event loop the step runs on. For a one-shot
probe that is fine — nothing else is on that loop. Inside the daemon it would
be the #416 shape exactly (synchronous work on the loop, seen as watchdog
stalls), and the fix is for the loop to call the driver through
``asyncio.to_thread``. Named here rather than discovered later.

⚠ ``SUSPECTED_NOOP`` IS NOT SUCCESS
------------------------------------
Cua's ``ActionEffect`` is ``CONFIRMED | PARTIAL | UNVERIFIABLE |
SUSPECTED_NOOP | REFUSED``. The driver already distinguishes "I dispatched
this and nothing appears to have happened" from "confirmed" — which is the
exact failure this entire subsystem is built around, handed to us for free by
upstream.

Mapping it to success because the call returned without raising would be the
trust-the-success-message shape at the one place we were warned about it. So:
only ``CONFIRMED`` and ``PARTIAL`` are effects; ``SUSPECTED_NOOP`` and
``REFUSED`` raise, and ``UNVERIFIABLE`` is reported as such rather than as
either.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from prometheus.computer.driver import DriverUnavailable, StaleSnapshot
from prometheus.computer.types import Element, Observation

logger = logging.getLogger(__name__)

#: Effects that mean the action landed. Everything else is not success.
_EFFECT_OK = frozenset({"CONFIRMED", "PARTIAL"})

#: Effects that mean it demonstrably did not land. These RAISE.
_EFFECT_BAD = frozenset({"SUSPECTED_NOOP", "REFUSED"})

#: Bound on one SDK call. Longer than a slow accessibility walk, short
#: enough that a hung driver surfaces as a failure rather than a wedge.
OPERATION_TIMEOUT_SECONDS = 60

#: How many elements to pull per snapshot. Bounded because the candidate
#: table is reviewed by a human in the approval prompt, and a thousand-row
#: tree makes a table nobody can check.
MAX_ELEMENTS = 200

#: Substrings in a driver error that mean "your snapshot is stale". Cua
#: returns an explicit stale error; this maps it onto our own exception so
#: the loop's existing refusal path fires rather than a generic failure.
_STALE_MARKERS = ("stale", "snapshot_id_required", "superseded")


def _require_sdk() -> Any:
    """Import the SDK, or fail with something an operator can act on."""
    try:
        import cua_driver
    except ImportError as exc:
        raise DriverUnavailable(
            "the Cua driver SDK is not installed — "
            "`uv sync --extra computer` (or pip install 'oara-prometheus"
            "[computer]'). It is deliberately absent from CI's extras and "
            "from `full`, because the driver cannot be exercised without a "
            "display."
        ) from exc
    return cua_driver


class CuaDriverAdapter:
    """A ``Driver`` over the in-process Cua SDK. Local target only.

    ``CuaDriver.create()`` loads the runtime in THIS process — no daemon, no
    socket, no network. That is the only hosting mode used here: ``connect()``
    and ``create_private_worker()`` exist, and neither is a remote transport
    (Cua has none), so neither buys anything a local target needs.
    """

    def __init__(self, target: str, session: str | None = None) -> None:
        self._target = target
        self._session = session
        self._sdk = _require_sdk()
        self._driver: Any = None
        # A private event loop on its own thread: the SDK is async, the
        # Driver Protocol is not, and this is the seam between them.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        #: The snapshot this driver last handed out, per (pid, window_id).
        #: Cua binds element tokens to a snapshot; we refuse a stale one
        #: BEFORE the call as well, so the refusal does not depend on the
        #: driver's error text staying the same. A MISSING entry is refused
        #: too — see `act` — because no record means the snapshot cannot be
        #: vouched for, not that it is fresh.
        #:
        #: ⚠ TWO KNOWN GAPS, BOTH DEFERRED 2026-09-20, recorded together
        #: because anyone fixing either will be working on this key.
        #:
        #: 1. PID REUSE. This dict is keyed on (pid, window_id) and is never
        #:    pruned — nothing here observes window close or process death.
        #:    A long-lived adapter that observed pid P, where P then dies and
        #:    the OS reissues the number to an unrelated process with a
        #:    coincident window id, would compare a new snapshot against a
        #:    dead window's record. Low likelihood; the consequence is a
        #:    wrong staleness verdict in EITHER direction, so it is not
        #:    fail-safe. A fix prunes on DriverUnavailable from a window
        #:    target, or stops trusting the pid as identity.
        #:
        #: 2. NO DISCOVERY PATH. Nothing in `src/` ever calls the SDK's
        #:    `list_windows`, and `observe` does not read pid/window_id back
        #:    off the response — it echoes the caller's arguments into the
        #:    Observation (note `app` two lines below does the opposite, and
        #:    takes `out.app_name`). So every caller must already know both
        #:    numbers, and the only way to obtain them today is out of band.
        #:    Nothing validates them either: both are plain required ints
        #:    with no bound, and a shipped script defaults both to 0.
        #:    The SDK does refuse `pid=0` (`invalid_action_target`), so the
        #:    failure is loud rather than silent — which is why this is a gap
        #:    and not a defect.
        self._snapshots: dict[tuple[int, int], str] = {}

    # ── lifecycle ──────────────────────────────────────────────────────

    def _await(self, coro: Any) -> Any:
        """Run one SDK coroutine on the adapter's own loop and block."""
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(
            timeout=OPERATION_TIMEOUT_SECONDS)

    def start(self) -> None:
        if self._driver is not None:
            return
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._loop.run_forever, daemon=True,
                name="cua-driver-loop")
            self._thread.start()
        try:
            self._driver = self._sdk.CuaDriver.create()
        except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
            raise DriverUnavailable(
                f"the Cua runtime could not start: "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc
        if not self._driver.is_available():
            raise DriverUnavailable(
                "the Cua runtime started but reports itself unavailable — "
                "on Linux this usually means no reachable display or no "
                "accessibility bus. Check /api/status's computer.substrate."
            )

    def shutdown(self) -> None:
        driver, self._driver = self._driver, None
        if driver is not None:
            try:
                self._await(driver.shutdown())
            except Exception:  # noqa: BLE001
                logger.warning("Cua driver shutdown failed", exc_info=True)
        loop, self._loop = self._loop, None
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> CuaDriverAdapter:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    # ── the Driver Protocol ────────────────────────────────────────────

    def observe(
        self, target: str, app: str, pid: int, window_id: int
    ) -> Observation:
        """One fresh snapshot, translated into our Observation."""
        self._assert_target(target)
        self.start()
        sdk = self._sdk
        try:
            out = self._await(self._driver.get_window_state(
                sdk.GetWindowStateInput(
                pid=pid,
                window_id=window_id,
                session=self._session,
                query=None,
                include_accessibility_tree=True,
                # ⚠ FALSE, deliberately. A screenshot is not needed to build
                # a candidate table, it is the expensive half of the call,
                # and a frame we do not need is a frame that could end up
                # somewhere it should not be (see the capture ruling).
                include_screenshot=False,
                screenshot_out_file=None,
                max_elements=MAX_ELEMENTS,
                max_depth=None,
                max_dimension=None,
            )))
        except Exception as exc:  # noqa: BLE001
            raise DriverUnavailable(
                f"observation failed on {target}: "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc

        snapshot_id = getattr(out, "snapshot_id", None)
        elements = tuple(
            _element(e) for e in (getattr(out, "elements", None) or [])
            if getattr(e, "element_token", None)
        )
        if not snapshot_id:
            # No snapshot id means no element can be safely addressed later.
            # UNUSABLE, not empty — the loop must refuse rather than build a
            # table whose tokens it cannot bind.
            return Observation(
                target=target, app=app, pid=pid, window_id=window_id,
                snapshot_id="", elements=(),
                unusable_reason=(
                    "the driver returned no snapshot id, so element tokens "
                    "could not be bound to an observation"
                ),
            )
        self._snapshots[(pid, window_id)] = snapshot_id
        return Observation(
            target=target,
            app=getattr(out, "app_name", None) or app,
            pid=pid,
            window_id=window_id,
            snapshot_id=snapshot_id,
            elements=elements,
            unusable_reason=None,
        )

    def act(self, verb: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one bounded action, and rule on what came back."""
        self._assert_target(arguments.get("target", self._target))
        pid = int(arguments["pid"])
        window_id = int(arguments["window_id"])

        # ⚠ OUR OWN STALENESS CHECK, BEFORE THE DRIVER'S. The loop already
        # validates the candidate against the live observation; this is the
        # third refusal, and it exists so the guarantee does not depend on
        # the driver's error text staying the same across versions.
        #
        # ⚠ AND BEFORE start(), so a call that was always going to be refused
        # does not spin up the Cua runtime on its way to being refused.
        snapshot = arguments.get("snapshot_id")
        current = self._snapshots.get((pid, window_id))
        if snapshot and current is None:
            # NO RECORD IS "CANNOT DETERMINE", NOT "FRESH". This read
            # `if snapshot and current and ...`, so a missing record made the
            # whole check evaporate and the action went to the driver
            # unvouched. Reachable whenever the acting adapter is not the one
            # that observed — a fresh adapter, a restart, a second adapter on
            # the same target — and in exactly those cases the only thing left
            # was the driver's error text, which is what this check exists NOT
            # to depend on.
            #
            # Safe to be strict: one adapter serves both halves everywhere.
            # `ComputerUseLoop._driver` is assigned once and used for observe,
            # act and the post-action verify; `build_computer_tools` binds one
            # driver to every tool. So "this adapter never observed that
            # window" really does mean the snapshot cannot be vouched for.
            raise StaleSnapshot(
                f"no observation on record for pid {pid} window {window_id} "
                f"on this adapter, so snapshot {snapshot!r} cannot be vouched "
                f"for — observe before acting"
            )
        if snapshot and snapshot != current:
            raise StaleSnapshot(
                f"snapshot {snapshot!r} has been superseded by {current!r} "
                f"— re-observe before acting"
            )
        self.start()

        builder = _BUILDERS.get(verb)
        if builder is None:
            # Not a generic dispatch: a verb with no builder is one this
            # adapter deliberately does not implement.
            raise DriverUnavailable(
                f"{verb!r} is not implemented by this adapter (implemented: "
                f"{', '.join(sorted(_BUILDERS))})"
            )
        method_name, make_input = builder
        try:
            result = self._await(getattr(self._driver, method_name)(
                make_input(self._sdk, arguments, self._session)))
        except Exception as exc:  # noqa: BLE001
            text = f"{exc}".lower()
            if any(m in text for m in _STALE_MARKERS):
                raise StaleSnapshot(str(exc)) from exc
            raise DriverUnavailable(
                f"{verb} failed: {exc.__class__.__name__}: {exc}") from exc

        return _verdict(verb, result)

    # ── internals ──────────────────────────────────────────────────────

    def _assert_target(self, target: str) -> None:
        """Refuse an action labelled for another machine.

        A registry mis-binding would otherwise execute machine B's action on
        machine A, with the gate having ruled on B.
        """
        if target and target != self._target:
            raise DriverUnavailable(
                f"this driver is bound to target {self._target!r}, not "
                f"{target!r} — refusing rather than acting on the wrong "
                f"machine"
            )


def _element(e: Any) -> Element:
    return Element(
        element_index=int(getattr(e, "element_index", 0)),
        element_token=str(getattr(e, "element_token", "")),
        role=str(getattr(e, "role", "") or ""),
        label=str(getattr(e, "label", "") or ""),
        value=getattr(e, "value", None),
        actions=tuple(getattr(e, "actions", None) or ()),
        editable=bool(getattr(e, "editable", False)),
    )


def _verdict(verb: str, result: Any) -> dict[str, Any]:
    """Turn an SDK result into an outcome — or raise when it did not land.

    ⚠ THE POINT OF THIS FUNCTION. `SUSPECTED_NOOP` is upstream telling us
    the action appears to have done nothing. Returning success on it would be
    trusting the success message at the one place this whole subsystem was
    built to distrust.
    """
    effect = _effect_name(result)
    if effect in _EFFECT_BAD:
        raise DriverUnavailable(
            f"{verb} did not land: the driver reported {effect} — the action "
            f"was dispatched and there is no evidence it took effect"
        )
    # ⚠ NO KEY CALLED "ok" HERE, DELIBERATELY. `StepResult.ok` already means
    # "the step executed", and a second `ok` in the driver's own result
    # meaning "the driver confirmed the effect" is two fields with one name
    # and different answers — observed live: a click that demonstrably landed
    # (the target app printed CLICKED 1) came back UNVERIFIABLE, so
    # `status: executed` sat next to `ok: False` in one payload.
    #
    # They are both true and they are answering different questions. Naming
    # them differently is the whole fix.
    return {
        "verb": verb,
        "effect": effect,
        # Did the DRIVER confirm it, as opposed to merely dispatch it?
        # UNVERIFIABLE is reported as itself — neither success nor failure —
        # and a caller wanting certainty verifies against fresh state, which
        # is what ComputerUseLoop._verify does.
        "confirmed_by_driver": effect == "CONFIRMED",
        "landed": effect in _EFFECT_OK,
    }


def _effect_name(result: Any) -> str:
    effect = getattr(result, "effect", None)
    if effect is None:
        return "UNVERIFIABLE"
    return str(getattr(effect, "name", effect)).upper()


# ── per-verb input builders ────────────────────────────────────────────
#
# One entry per implemented verb. A table rather than a chain of ifs so the
# implemented set is READABLE — and so `act` can refuse an absent verb by
# looking it up rather than by falling through to a generic call.


def _delivery(sdk: Any, arguments: dict[str, Any]) -> Any:
    mode = str(arguments.get("delivery_mode", "background")).upper()
    return getattr(sdk.InputDeliveryMode, mode)


def _window(sdk: Any, arguments: dict[str, Any]) -> Any:
    return sdk.ActionTarget.WINDOW(
        pid=int(arguments["pid"]), window_id=int(arguments["window_id"]))


def _click_input(sdk: Any, a: dict[str, Any], session: str | None) -> Any:
    return sdk.ClickInput(
        target=_window(sdk, a),
        position=sdk.ClickPosition.ELEMENT(element_token=a["element_token"]),
        delivery_mode=_delivery(sdk, a),
        session=session,
        button=sdk.ClickButton.LEFT,
        count=1,
    )


def _press_key_input(sdk: Any, a: dict[str, Any], session: str | None) -> Any:
    return sdk.PressKeyInput(
        key=str(a["key"]), target=_window(sdk, a), scope=None,
        session=session, modifiers=None,
    )


def _scroll_input(sdk: Any, a: dict[str, Any], session: str | None) -> Any:
    return sdk.ScrollInput(
        x=0.0, y=0.0,
        direction=getattr(sdk.ScrollDirection,
                          str(a.get("direction", "down")).upper()),
        target=_window(sdk, a), scope=None, session=session,
        by=None, amount=int(a.get("amount", 1)),
    )


def _type_text_input(sdk: Any, a: dict[str, Any], session: str | None) -> Any:
    return sdk.TypeTextInput(
        text=str(a["text"]), target=_window(sdk, a), scope=None,
        session=session,
    )


def _invoke_menu_input(sdk: Any, a: dict[str, Any], session: str | None) -> Any:
    return sdk.InvokeMenuInput(
        pid=int(a["pid"]), window_id=int(a["window_id"]),
        path=list(a["path"]), session=session,
    )


#: verb -> (SDK method, input builder). The IMPLEMENTED SET, and `act`
#: refuses anything absent from it rather than dispatching generically.
_BUILDERS: dict[str, tuple[str, Any]] = {
    "click": ("click", _click_input),
    "press_key": ("press_key", _press_key_input),
    "scroll": ("scroll", _scroll_input),
    "type_text": ("type_text", _type_text_input),
    "invoke_menu": ("invoke_menu", _invoke_menu_input),
}
