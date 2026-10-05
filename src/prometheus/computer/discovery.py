"""Turning "my editor" into ONE window, or into a question.

WHY THIS EXISTS (computer-use v1.1, D8)
----------------------------------------
``list_apps`` and ``list_windows`` were in the SDK and nothing called them, so
every caller had to already know a pid and a window id — and the only way to
know them was out of band. The door has to turn a person's word into a
window, and it must never guess while doing it.

THE RULES (design §5.1.3)
-------------------------
* Match in this order: an operator ALIAS (``computer_use.apps.aliases``),
  then a case-insensitive match on the app's name, its bundle id, or the
  basename of its launch path.
* Only RUNNING apps with an ON-SCREEN, unminimised window are candidates.
  Exactly one → propose it. Zero or several → ASK, listing app NAMES ONLY:
  no pid and no window id ever reaches the question — they are not
  meaningful to a person, and they are not consent terms.
* NOTHING IS EVER LAUNCHED. An app that is not running is not proposed; the
  adapter has no launch path at all.
* The window is the app's FRONTMOST on-screen window (highest ``z_index``),
  RE-RESOLVED before every step (:func:`resolve_window`); a window that has
  vanished resolves to None and ends the task rather than being replaced by
  a guess.

Window titles are deliberately not used and not shown: a browser's title is
page-authored text.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from prometheus.permissions.computer_extent import normalise_term

RESOLVED = "match"
ASK = "ask"


@dataclass(frozen=True)
class AppRecord:
    """One application the driver reports (``AppInfo``)."""

    pid: int
    name: str
    running: bool = True
    active: bool = False
    bundle_id: str | None = None
    launch_path: str | None = None


@dataclass(frozen=True)
class WindowRecord:
    """One window the driver reports (``WindowInfo``). No bounds: nothing here
    needs them, and coordinates are not something the door acts on."""

    window_id: int
    pid: int | None
    app_name: str
    title: str = ""
    is_on_screen: bool = True
    z_index: int | None = None
    minimized: bool | None = None


@dataclass(frozen=True)
class Resolution:
    """What a phrase resolved to: one app and window, or a question."""

    status: str  # RESOLVED | ASK
    app: AppRecord | None = None
    window: WindowRecord | None = None
    #: App NAMES the person can pick from. Never a pid, never a window id.
    options: list[str] = field(default_factory=list)
    question: str = ""


class _ListsWindows(Protocol):
    def list_windows(
        self, pid: int | None = None, on_screen_only: bool = True
    ) -> list[WindowRecord]: ...


def _fold(text: str | None) -> str:
    return str(text or "").strip().lower()


def _identifiers(app: AppRecord) -> tuple[str, ...]:
    """The app's own names, as the driver spells them: its display name, its
    bundle/desktop id and its launch-path basename. Each once, in that
    order, and never an empty one."""
    raw = [app.name, app.bundle_id or "",
           os.path.basename(app.launch_path) if app.launch_path else ""]
    out: list[str] = []
    seen: set[str] = set()
    for name in (str(n).strip() for n in raw):
        if name and _fold(name) not in seen:
            seen.add(_fold(name))
            out.append(name)
    return tuple(out)


def _keys_of(app: AppRecord) -> set[str]:
    return {_fold(name) for name in _identifiers(app)}


@dataclass(frozen=True)
class AppIdentity:
    """The app the door resolved: every identifier it is known by, and its
    pid where known. What check 1b compares the driver's report against.

    ⚠ WHY MORE THAN ONE NAME. The door resolves a phrase through ANY of an
    app's identifiers, and the driver names a window's app its own way. On
    GNOME, ``list_apps`` calls it "Text Editor" (the ``.desktop`` name) while
    the window's state calls it ``gnome-text-editor``; keeping only the
    display name refused the very app the person picked (task 36ed5742).

    ⚠ ONLY THE APP'S OWN NAMES. An operator alias, or the phrase a person
    typed, is how the app was FOUND, not what it is called; neither is
    recorded. And none of this reaches consent: the extent's app term is
    still the driver's report, and a binding still names one app.
    """

    names: tuple[str, ...]
    pid: int | None = None

    @classmethod
    def of(cls, app: AppRecord) -> AppIdentity:
        return cls(names=_identifiers(app),
                   pid=app.pid if app.pid and app.pid > 0 else None)

    def matches(self, reported: str | None, reported_pid: int | None) -> bool:
        """The driver's report is this app: it names it by one of its
        identifiers, and — when both pids are known — in the same process."""
        if not any(same_app(reported, name) for name in self.names):
            return False
        return not self.pid_differs(reported_pid)

    def pid_differs(self, reported_pid: int | None) -> bool:
        """Both pids known, and not the same process."""
        known = reported_pid is not None and reported_pid > 0
        return self.pid is not None and known and reported_pid != self.pid


def same_app(reported: str | None, wanted: str) -> bool:
    """Is the app the DRIVER reported the app we were asked to act in?

    Folded with the extent's own spelling rule, so "GEdit" and "gedit" are
    one app here exactly as they are one grant. An empty report is never a
    match: the consent term cannot come from the caller (D19).
    """
    reported = str(reported or "").strip()
    return bool(reported) and normalise_term(reported) == normalise_term(wanted)


def frontmost_window(
    windows: Sequence[WindowRecord], pid: int
) -> WindowRecord | None:
    """The app's frontmost on-screen, unminimised window, or None."""
    mine = [w for w in windows
            if w.pid == pid and w.is_on_screen and not w.minimized]
    if not mine:
        return None
    return max(mine, key=lambda w: -1 if w.z_index is None else w.z_index)


def resolve_window(driver: _ListsWindows, pid: int) -> WindowRecord | None:
    """Re-resolve before EVERY step. None means the window is gone — the task
    ends; it is never swapped for another app's window."""
    return frontmost_window(
        driver.list_windows(pid=pid, on_screen_only=True), pid)


def resolve_app(
    phrase: str,
    apps: Sequence[AppRecord],
    windows: Sequence[WindowRecord],
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> Resolution:
    """One running app with an on-screen window, or a question. Never a guess."""
    wanted = _fold(phrase)
    folded_aliases = {_fold(k): v for k, v in (aliases or {}).items()}
    keys = (
        {_fold(name) for name in folded_aliases[wanted]}
        if wanted in folded_aliases else {wanted}
    )
    running = [a for a in apps if a.running]
    visible = [(a, frontmost_window(windows, a.pid)) for a in running]
    visible_names = [a.name for a, w in visible if w is not None]

    matches = [(a, w) for a, w in visible if _keys_of(a) & keys]
    shown = [(a, w) for a, w in matches if w is not None]
    if len(shown) == 1:
        app, window = shown[0]
        return Resolution(status=RESOLVED, app=app, window=window)
    if len(shown) > 1:
        return Resolution(
            status=ASK, options=[a.name for a, _ in shown],
            question=(f"{len(shown)} running apps match {phrase.strip()!r} — "
                      f"which one?"))
    if matches:
        return Resolution(
            status=ASK, options=visible_names,
            question=(f"{phrase.strip()!r} is running but has no on-screen "
                      f"window. Bring it up, or pick one that is on screen."))
    return Resolution(
        status=ASK, options=visible_names,
        question=(f"No running app with an on-screen window matches "
                  f"{phrase.strip()!r}. Nothing is launched — pick one that "
                  f"is running."))
