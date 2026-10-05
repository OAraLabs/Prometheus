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

⚠ AND THE EFFECT IS NOT IN THE SAME PLACE FOR EVERY VERB. ``click`` returns an
``ActionResult`` (effect at the top). ``press_key``, ``scroll``, ``type_text``
and ``invoke_menu`` return a ``ToolResult``: the effect sits at
``.action.effect``, beside ``is_error`` and ``error_code``. Reading only the
top level made every non-click no-op UNVERIFIABLE, so on four verbs of five the
rule above could not fire (computer-use v1.1, D3). ``_verdict`` reads both.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import TYPE_CHECKING, Any

from prometheus.computer.driver import (
    ActionOutcomeUnknown,
    DriverBusy,
    DriverSessionEnded,
    DriverUnavailable,
    StaleSnapshot,
)
from prometheus.computer.types import Element, Observation
from prometheus.permissions.computer_schema import DELIVERY_BACKGROUND

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from prometheus.computer.discovery import AppRecord, WindowRecord

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

#: The SDK's ``error_code`` for a driver whose session Cua has ended (idle or
#: absolute expiry). Read off the exception's attribute, never its text.
_SESSION_ENDED = "session_ended"

#: Verbs whose 0.28.2 input type can CARRY a delivery mode. Only
#: ``ClickInput`` takes one; for the others the driver decides, so the
#: extent's ``:background`` term names a request the driver was never sent.
#: The result says which it was rather than letting the extent assert it
#: (computer-use v1.1, D5).
_DELIVERY_IN_INPUT: frozenset[str] = frozenset({"click"})


def _require_sdk() -> Any:
    """Import the SDK, or fail with something an operator can act on.

    ⚠ THE TELEMETRY FLOOR GOES FIRST. cua-driver reports usage by default and
    reads its opt-outs when it loads, so they are forced off BEFORE the import
    — after it would be too late (computer/integration.py, design Q1).
    """
    from prometheus.computer.integration import apply_telemetry_floor

    apply_telemetry_floor()
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
    socket, no network. That is the only hosting mode used here. Cua does
    offer others — a supervised private worker process (its mode for crash
    containment), a daemon, MCP, and a remote transport — and choosing among
    them belongs to the driver's Integration (computer-use v1.1 §5.3), not to
    this adapter. A remote transport is out under the local-only ruling.
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
        #: Windows whose CURRENT snapshot was unusable (empty or degraded),
        #: and why. ``act`` refuses them: the wrapped-tool path reaches
        #: ``act`` without a candidate table, so the refusal cannot live only
        #: in ``build_candidates``.
        self._unusable: dict[tuple[int, int], str] = {}
        #: ONE SDK CALL AT A TIME — see ``_await``.
        self._call_lock = threading.Lock()

    # ── lifecycle ──────────────────────────────────────────────────────

    def _await(self, coro: Any) -> Any:
        """Run one SDK coroutine on the adapter's own loop and block.

        ⚠ ONE CALL AT A TIME, AND THE LOCK OUTLIVES THE WAIT (D13). Two steps
        sharing an adapter would otherwise interleave on its loop. The lock is
        released when the coroutine FINISHES, not when this caller stops
        waiting: ``.result(timeout=…)`` does not cancel the SDK call, so a
        timed-out action is still in flight, and the next call must not
        overtake it. A caller that cannot get in within the timeout is
        refused with ``DriverBusy`` — nothing is sent.
        """
        assert self._loop is not None
        lock = self._call_lock
        if not lock.acquire(timeout=OPERATION_TIMEOUT_SECONDS):
            coro.close()
            raise DriverBusy(
                "an earlier driver call has not finished, so this one was "
                "not sent — the driver handles one call at a time"
            )
        try:
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        except BaseException:
            lock.release()
            coro.close()
            raise
        future.add_done_callback(lambda _f: lock.release())
        return future.result(timeout=OPERATION_TIMEOUT_SECONDS)

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
            driver = self._sdk.CuaDriver.create()
        except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
            raise DriverUnavailable(
                f"the Cua runtime could not start: "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc
        if not driver.is_available():
            # ⚠ NOT KEPT (D11). This assigned the driver first and checked it
            # second, so a runtime that reported itself unavailable stayed
            # assigned — and the NEXT start() returned early on
            # `self._driver is not None`, reporting nothing. A failed health
            # check must fail every time it is asked.
            try:
                self._await(driver.shutdown())
            except Exception:  # noqa: BLE001
                logger.warning("Cua driver shutdown after a failed health "
                               "check failed", exc_info=True)
            raise DriverUnavailable(
                "the Cua runtime started but reports itself unavailable — "
                "on Linux this usually means no reachable display or no "
                "accessibility bus. Check /api/status's computer.substrate."
            )
        self._driver = driver

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
        # A call still in flight on the stopped loop will never finish, so its
        # lock would never be released. The next start() gets a fresh one.
        self._call_lock = threading.Lock()

    def __enter__(self) -> CuaDriverAdapter:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    # ── discovery (D8) ─────────────────────────────────────────────────
    #
    # Read-only, and the ONLY two discovery calls: there is no launch path
    # here, so "my editor" can resolve to a running window or to a question,
    # never to a process this adapter started.

    def list_apps(self) -> list[AppRecord]:
        self.start()
        try:
            out = self._await(self._driver.list_apps(
                _list_apps_input(self._sdk)))
        except Exception as exc:  # noqa: BLE001
            # ⚠ AN ENDED SESSION IS NAMED, BY THE SDK'S OWN CODE. ``start()``
            # keeps this driver, and once Cua has ended its session every call
            # answers ``session_ended``. The probe replaces the driver once on
            # exactly this (integration._run_checks); the text is not read.
            if getattr(exc, "error_code", None) == _SESSION_ENDED:
                raise DriverSessionEnded(
                    f"list_apps failed: the driver session ended "
                    f"({exc.__class__.__name__}: {exc})") from exc
            raise DriverUnavailable(
                f"list_apps failed: {exc.__class__.__name__}: {exc}") from exc
        return [_app_record(a) for a in (getattr(out, "apps", None) or [])]

    def list_windows(
        self, pid: int | None = None, on_screen_only: bool = True
    ) -> list[WindowRecord]:
        self.start()
        try:
            out = self._await(self._driver.list_windows(_list_windows_input(
                self._sdk, pid=pid, on_screen_only=on_screen_only)))
        except Exception as exc:  # noqa: BLE001
            raise DriverUnavailable(
                f"list_windows failed: {exc.__class__.__name__}: {exc}"
            ) from exc
        return [_window_record(w)
                for w in (getattr(out, "windows", None) or [])]

    # ── the Driver Protocol ────────────────────────────────────────────

    def observe(
        self, target: str, app: str, pid: int, window_id: int
    ) -> Observation:
        """One fresh snapshot, translated into our Observation."""
        self._assert_target(target)
        self.start()
        key = (pid, window_id)
        try:
            out = self._await(self._driver.get_window_state(
                _window_state_input(
                    self._sdk, pid=pid, window_id=window_id,
                    session=self._session)))
        except DriverUnavailable:
            raise  # DriverBusy: nothing was sent, nothing to forget
        except Exception as exc:  # noqa: BLE001
            # A call that failed or timed out may still have minted a new
            # snapshot on the driver side, invalidating the tokens we hold
            # for this window. The record can no longer be vouched for.
            self._forget(key)
            raise DriverUnavailable(
                f"observation failed on {target}: "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc

        snapshot_id = getattr(out, "snapshot_id", None)
        if not snapshot_id:
            # No snapshot id means no element can be safely addressed later.
            # UNUSABLE, not empty — the loop must refuse rather than build a
            # table whose tokens it cannot bind.
            self._forget(key)
            return Observation(
                target=target, app=app, pid=pid, window_id=window_id,
                snapshot_id="", elements=(),
                unusable_reason=(
                    "the driver returned no snapshot id, so element tokens "
                    "could not be bound to an observation"
                ),
            )

        raw = list(getattr(out, "elements", None) or [])
        elements = tuple(
            _element(e) for e in raw if getattr(e, "element_token", None)
        )
        degraded = bool(getattr(out, "degraded", False))
        degraded_reason = _opt_str(getattr(out, "degraded_reason", None))
        unusable: str | None = None
        if degraded:
            unusable = (
                "the driver reported this tree as degraded"
                + (f" ({degraded_reason})" if degraded_reason else "")
                + " — it may be missing elements that are on screen, so it "
                "is refused rather than built into actions"
            )
        elif not elements:
            # D10. An empty tree and an idle window are the same shape; acting
            # on one is the "reports success and does nothing" failure.
            unusable = (
                "the driver returned no elements with a token for this window "
                "— an empty tree is refused, never read as an idle window"
            )

        # The NEW snapshot is current whether or not it is usable: the driver
        # has invalidated every earlier token for this window either way.
        self._snapshots[key] = snapshot_id
        if unusable:
            self._unusable[key] = unusable
        else:
            self._unusable.pop(key, None)
        return Observation(
            target=target,
            # ⚠ THE DRIVER'S ANSWER OR NONE (D19). This was
            # `out.app_name or app`: with no app name from the driver, the
            # consent term became whatever the CALLER claimed. Empty here
            # makes the extent unknown and the loop refuse the window.
            app=str(getattr(out, "app_name", None) or ""),
            pid=pid,
            window_id=window_id,
            snapshot_id=snapshot_id,
            elements=elements,
            unusable_reason=unusable,
            degraded=degraded,
            degraded_reason=degraded_reason,
            truncated=bool(getattr(out, "truncated", False)),
            truncation_reason=_opt_str(getattr(out, "truncation_reason", None)),
            elements_complete=_opt_bool(getattr(out, "elements_complete", None)),
            total_element_count=_opt_int(
                getattr(out, "total_element_count", None)),
            returned_element_count=_opt_int(
                getattr(out, "returned_element_count", None)),
            # Over the WHOLE walk, before the token filter above.
            web_content_seen=(
                any(_is_web_or_document(e) for e in raw) if raw else None),
        )

    def act(self, verb: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one bounded action, and rule on what came back."""
        self._assert_target(arguments.get("target", self._target))
        pid = int(arguments["pid"])
        window_id = int(arguments["window_id"])
        key = (pid, window_id)

        # ⚠ OUR OWN STALENESS CHECK, BEFORE THE DRIVER'S. The loop already
        # validates the candidate against the live observation; this is the
        # third refusal, and it exists so the guarantee does not depend on
        # the driver's error text staying the same across versions.
        #
        # ⚠ AND BEFORE start(), so a call that was always going to be refused
        # does not spin up the Cua runtime on its way to being refused.
        snapshot = arguments.get("snapshot_id")
        current = self._snapshots.get(key)
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
        unusable = self._unusable.get(key)
        if unusable:
            raise DriverUnavailable(
                f"refusing to {verb} in pid {pid} window {window_id}: its "
                f"latest observation is unusable — {unusable}"
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
            result = self._await(_dispatch(
                self._driver, method_name,
                make_input(self._sdk, arguments, self._session)))
        except TimeoutError as exc:
            # ⚠ NOT "FAILED" (D13). The wait gave up; the action did not.
            # Forget the snapshot so nothing can act on this window again
            # until a fresh observation says what actually happened.
            self._forget(key)
            raise ActionOutcomeUnknown(
                f"{verb} in pid {pid} window {window_id} did not answer "
                f"within {OPERATION_TIMEOUT_SECONDS}s. It was dispatched and "
                f"may still land — observe again before deciding what "
                f"happened"
            ) from exc
        except DriverUnavailable:
            raise  # DriverBusy: refused before anything was sent
        except Exception as exc:  # noqa: BLE001
            text = f"{exc}".lower()
            if any(m in text for m in _STALE_MARKERS):
                raise StaleSnapshot(str(exc)) from exc
            raise DriverUnavailable(
                f"{verb} failed: {exc.__class__.__name__}: {exc}") from exc

        return _verdict(verb, result, arguments)

    # ── internals ──────────────────────────────────────────────────────

    def _forget(self, key: tuple[int, int]) -> None:
        """Drop what we know about a window's snapshot: it cannot be vouched
        for, so ``act`` refuses until the next ``observe``."""
        self._snapshots.pop(key, None)
        self._unusable.pop(key, None)

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


def _window_state_input(
    sdk: Any, *, pid: int, window_id: int, session: str | None
) -> Any:
    """The observe call's input. A function so the real-SDK tests can build it
    with the PINNED types — from 0.28.3 it gains a required keyword this
    call does not pass, which is what the exact pin exists to keep out."""
    return sdk.GetWindowStateInput(
        pid=pid,
        window_id=window_id,
        session=session,
        query=None,
        include_accessibility_tree=True,
        # ⚠ FALSE, deliberately. A screenshot is not needed to build a
        # candidate table, it is the expensive half of the call, and a frame
        # we do not need is a frame that could end up somewhere it should not
        # be (see the capture ruling).
        include_screenshot=False,
        screenshot_out_file=None,
        max_elements=MAX_ELEMENTS,
        max_depth=None,
        max_dimension=None,
    )


def _is_web_or_document(e: Any) -> bool:
    """Web content, or a node that hosts documents (a browser page, an
    Electron view, an embedded frame). Over-matching only ever means more
    prompts; under-matching would let a page pass as a plain app."""
    if getattr(e, "in_web_content", None) is True:
        return True
    role = str(getattr(e, "role", "") or "").lower()
    return "document" in role or role == "embedded"


def _opt_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _element(e: Any) -> Element:
    # ``editable`` is NOT read: 0.28.2's ``WindowElement`` has no such field,
    # so reading it could only ever return the default (D4). It stays a
    # fixture-only field on ``Element``.
    return Element(
        element_index=int(getattr(e, "element_index", 0)),
        element_token=str(getattr(e, "element_token", "")),
        role=str(getattr(e, "role", "") or ""),
        label=str(getattr(e, "label", "") or ""),
        value=getattr(e, "value", None),
        actions=tuple(getattr(e, "actions", None) or ()),
        enabled=_opt_bool(getattr(e, "enabled", None)),
        selected=_opt_bool(getattr(e, "selected", None)),
        in_web_content=_opt_bool(getattr(e, "in_web_content", None)),
        parent_index=_opt_int(getattr(e, "parent_index", None)),
    )


def _verdict(verb: str, result: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    """Turn an SDK result into an outcome — or raise when it did not land.

    ⚠ THE POINT OF THIS FUNCTION. `SUSPECTED_NOOP` is upstream telling us
    the action appears to have done nothing. Returning success on it would be
    trusting the success message at the one place this whole subsystem was
    built to distrust.

    Both result shapes are read (see the module docstring): an
    ``ActionResult`` carries the effect itself; a ``ToolResult`` carries it at
    ``.action`` and may instead state an error outright with ``is_error``.
    """
    if getattr(result, "is_error", False):
        code = getattr(result, "error_code", None) or "unspecified"
        text = str(getattr(result, "text", "") or "").strip()
        detail = f"{code}: {text}" if text else str(code)
        if any(m in detail.lower() for m in _STALE_MARKERS):
            raise StaleSnapshot(f"{verb}: {detail}")
        raise DriverUnavailable(
            f"{verb} did not land: the driver reported an error ({detail})"
        )
    action = _action_of(result)
    effect = _effect_name(action)
    if effect in _EFFECT_BAD:
        raise DriverUnavailable(
            f"{verb} did not land: the driver reported {effect} — the action "
            f"was dispatched and there is no evidence it took effect"
        )
    requested = str(arguments.get("delivery_mode", DELIVERY_BACKGROUND)).lower()
    reported = _delivery_reported(action)
    matches = (
        None if reported in (None, "unknown", "not_applicable")
        else reported == requested
    )
    escalation = _escalation_text(action)
    if matches is False:
        # The extent the operator consented to names the REQUESTED delivery.
        # When the driver did something else, that must be visible.
        logger.warning(
            "%s: asked for %s delivery, the driver reports %s%s",
            verb, requested, reported,
            f" (escalation: {escalation})" if escalation else "",
        )
    elif escalation:
        logger.info("%s: the driver advises escalation: %s", verb, escalation)
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
        # What was asked, what the driver says it did, and whether the input
        # could even carry the request (D5).
        "delivery_requested": requested,
        "delivery_reported": reported,
        "delivery_matches": matches,
        "delivery_enforced": verb in _DELIVERY_IN_INPUT,
        "escalation": escalation,
    }


def _action_of(result: Any) -> Any:
    """The ``ActionResult`` inside *result*, wherever this verb puts it."""
    if result is None or hasattr(result, "effect"):
        return result
    return getattr(result, "action", None)


def _effect_name(action: Any) -> str:
    effect = getattr(action, "effect", None)
    if effect is None:
        return "UNVERIFIABLE"
    return str(getattr(effect, "name", effect)).upper()


def _delivery_reported(action: Any) -> str | None:
    mode = getattr(getattr(action, "delivery", None), "mode", None)
    if mode is None:
        return None
    return str(getattr(mode, "name", mode)).lower()


def _escalation_text(action: Any) -> str | None:
    escalation = getattr(action, "escalation", None)
    if escalation is None:
        return None
    parts = [
        str(getattr(term, "name", term)).lower()
        for term in (getattr(escalation, "target", None),
                     getattr(escalation, "reason", None))
        if term is not None
    ]
    return ": ".join(parts) or None


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


def _set_value_call(sdk: Any, a: dict[str, Any],
                    session: str | None) -> tuple[str, str]:
    """``set_value`` by ELEMENT TOKEN (D2), through ``call_tool``.

    0.28.2 has no typed method for it; the generic surface reaches the same
    driver tool, whose schema (platform-linux/src/tools/impl_.rs:5480-5490 at
    the 0.28.2 tag) takes the token, the snapshot and the value, with
    ``additionalProperties: false``. A missing token is a KeyError HERE,
    before anything is sent — an untargeted set would be the defect again.
    """
    del sdk
    payload: dict[str, Any] = {
        "pid": int(a["pid"]),
        "window_id": int(a["window_id"]),
        "element_token": str(a["element_token"]),
        "snapshot_id": a.get("snapshot_id"),
        "value": str(a["text"]),
    }
    if session:
        payload["session"] = session
    return "set_value", json.dumps(
        {k: v for k, v in payload.items() if v is not None})


def _dispatch(driver: Any, method_name: str, built: Any) -> Any:
    """The SDK coroutine for one built input. ``call_tool`` is the one method
    that takes ``(name, arguments_json)`` rather than a typed input."""
    if method_name == "call_tool":
        name, arguments_json = built
        return driver.call_tool(name, arguments_json)
    return getattr(driver, method_name)(built)


#: verb -> (SDK method, input builder). The IMPLEMENTED SET, and `act`
#: refuses anything absent from it rather than dispatching generically.
#: ``set_value`` goes through ``call_tool`` by NAME, and only by this name —
#: the generic surface is not a passthrough for anything else.
_BUILDERS: dict[str, tuple[str, Any]] = {
    "click": ("click", _click_input),
    "press_key": ("press_key", _press_key_input),
    "scroll": ("scroll", _scroll_input),
    "type_text": ("type_text", _type_text_input),
    "set_value": ("call_tool", _set_value_call),
    "invoke_menu": ("invoke_menu", _invoke_menu_input),
}


# ── discovery inputs and records (D8) ──────────────────────────────────────


def _list_apps_input(sdk: Any) -> Any:
    return sdk.ListAppsInput()


def _list_windows_input(
    sdk: Any, *, pid: int | None, on_screen_only: bool
) -> Any:
    return sdk.ListWindowsInput(pid=pid, on_screen_only=on_screen_only)


def _app_record(a: Any) -> AppRecord:
    from prometheus.computer.discovery import AppRecord

    return AppRecord(
        pid=int(getattr(a, "pid", 0)),
        name=str(getattr(a, "name", "") or ""),
        running=bool(getattr(a, "running", False)),
        active=bool(getattr(a, "active", False)),
        bundle_id=getattr(a, "bundle_id", None),
        launch_path=getattr(a, "launch_path", None),
    )


def _window_record(w: Any) -> WindowRecord:
    from prometheus.computer.discovery import WindowRecord

    pid = getattr(w, "pid", None)
    z_index = getattr(w, "z_index", None)
    minimized = getattr(w, "minimized", None)
    return WindowRecord(
        window_id=int(getattr(w, "window_id", 0)),
        pid=None if pid is None else int(pid),
        app_name=str(getattr(w, "app_name", "") or ""),
        title=str(getattr(w, "title", "") or ""),
        is_on_screen=bool(getattr(w, "is_on_screen", False)),
        z_index=None if z_index is None else int(z_index),
        minimized=None if minimized is None else bool(minimized),
    )
