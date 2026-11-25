"""The arguments an operator is shown when asked to approve a tool call.

WHY THIS EXISTS
---------------
The approval prompt named the tool and the reason and **nothing else**
(``agent_loop`` passed only ``(tool_name, decision.reason)``). For a file
write that was survivable, because the reason carries the path. For a desktop
action it is not: ``computer_type_text requires confirmation`` tells the
operator neither what is being typed nor where, and an approval given on that
basis is not consent to anything in particular.

Ruled 2026-09-18 (MCP consent survey) and promoted to a prerequisite on
2026-09-19: **the approval must show the arguments being approved.**

THREE DECISIONS THIS FILE MAKES, EACH FOR A STATED REASON
---------------------------------------------------------
1. **Redaction reuses the audit redactor, deliberately.** ``AuditLogger``
   already carries two layers — by NAME (``token``/``secret``/``api_key``…)
   and by SHAPE (URL userinfo, query values, opaque UPPER_SNAKE assignments) —
   and the second layer exists precisely because the first is incomplete by
   construction. A second, parallel redactor here would drift from it, and
   the drift would be invisible until something leaked. One redactor.

   ⚠ It is NOT sufficient on its own and must not be described as if it were.
   An argument can carry a secret that neither layer matches — a bare
   passphrase in a ``text`` field has no name beside it and no distinctive
   shape. That is a real residual risk of showing arguments at all, and the
   answer is that the operator seeing their own machine's prompt is the
   intended reader. It is recorded here rather than left for someone to
   discover.

2. **Truncation is per-value AND overall.** A single huge argument must not
   push the actual decision off a phone screen, and a call with forty
   arguments must not either. Both limits are stated as constants rather than
   buried in a format string.

3. **The ``…`` is never silent.** A truncated value says how much was cut.
   "Approve this text" where the text is quietly the first 80 of 4000
   characters is the same defect as a narrow prompt for a wide grant.

4. **The prompt never hides the action, and says what it hid.** The agent
   chooses the arguments, so the redactor is a way to hide a command if a mask
   can swallow syntax (``ls ?a=;rm${IFS}-rf${IFS}~`` was shown ``ls ?a=***``;
   ``audit.py`` now masks only the characters a credential is made of). Nothing
   is dropped silently either: a whitespace-only value (``computer_type_text``
   with ``"\n"`` presses Enter) is drawn as ``⏎``/``␣``/``⇥``, ``None`` and empty
   values are drawn as what they are, every name and value is one visible line
   (invisible and direction-override characters are written ``⟨U+XXXX⟩``), and
   one leading ``⚠`` line says what was MASKED, CUT or NOT SHOWN (internal
   handles). A client that does not know about the line draws it as an argument.
"""

from __future__ import annotations

import json
import math
import os
import unicodedata
from dataclasses import dataclass
from typing import Any

from prometheus.permissions.audit import AuditLogger
from prometheus.permissions.tool_paths import TOOL_PATH_PARAM

#: Longest single rendered value before it is cut. Chosen against the audit
#: summary (200) and reason (500) limits already in the system: an argument is
#: shown alongside a reason, so it gets less room than the reason does.
MAX_VALUE_CHARS = 160

#: Longest whole rendered block. A Telegram message and a Beacon card both
#: have to stay glanceable; past this the operator is scrolling rather than
#: deciding.
MAX_TOTAL_CHARS = 700

#: Arguments that hold a filesystem path: the parameter each tool's gate reads as its path
#: (``tool_paths.TOOL_PATH_PARAM``, one source of truth) and the names tools use most. They are redacted like every other
#: argument but CUT differently: from the middle, so the start and the filename both show. A value cut from the end lost
#: its filename; see :func:`_truncate_path`.
PATH_ARGUMENT_NAMES: frozenset[str] = frozenset(TOOL_PATH_PARAM.values()) | frozenset({
    "path", "file_path", "cwd", "directory", "dest", "destination",
})

#: The key of the one line that says what the view masked, cut or left out. A real argument with this name is renamed.
NOTE_KEY = "⚠"

#: The key of the line that says how many arguments did not fit.
OVERFLOW_KEY = "…"

#: The most arguments one prompt lists. A call with more says how many were left out, in the leading line and at the end.
MAX_ARGS_SHOWN = 40

#: A call with more arguments than this is not approved on a partial view: the queue refuses it (permissions/approval_queue.py).
MAX_ARGUMENTS = MAX_ARGS_SHOWN

#: The least room an argument's value gets, however many arguments there are: a call with thirty arguments shows each
#: of them rather than the first few and nothing of the rest.
MIN_VALUE_SHARE = 20

#: The most characters of a name that are looked at at all.
_NAME_CAP = 256

#: Longest argument name shown. A name is the model's to choose, so it is cleaned and capped like a value.
MAX_NAME_CHARS = 40

#: Past this a value is cut BEFORE redaction, keeping both ends, so a model-chosen megabyte cannot stall the daemon in a
#: regex. Both ends are kept because the display keeps both ends.
_REDACT_CAP = 8192


def view_was_cut(view: dict[str, str] | None) -> bool:
    """Does a rendered view (:func:`redact_arguments`'s answer) say that anything was masked, cut or left out?

    It does when it opens with the :data:`NOTE_KEY` line (a value masked or cut, an argument named ``_x`` left out, a
    worker's arguments not all sent) or ends with the :data:`OVERFLOW_KEY` line (more arguments than fit). Read from
    the view itself, so it describes exactly what every surface shows, and cannot disagree with it. A real argument
    named like either marker is renamed by the view, so the keys are the view's own. Never raises.
    """
    try:
        return bool(view) and (NOTE_KEY in view or OVERFLOW_KEY in view)  # type: ignore[operator]
    except Exception:                                      # noqa: BLE001 - fail toward critical
        return True


def _bounded(text: str) -> tuple[str, int]:
    """The text, or its first and last halves of :data:`_REDACT_CAP` with a marker (and how many characters that left out),
    so a regex never sees a huge value."""
    if len(text) <= 2 * _REDACT_CAP:
        return text, 0
    return f"{text[:_REDACT_CAP]} … {text[-_REDACT_CAP:]}", len(text) - 2 * _REDACT_CAP - 3  # 3 = the marker


#: Arguments never worth a line in a consent prompt: machine plumbing the
#: operator cannot act on. Kept SHORT and justified — an exclusion list is how
#: the thing you needed to see goes missing, so nothing is added here without
#: a reason that survives "what if that was the dangerous part?". These three
#: are handles the daemon validates against live state (every desktop call
#: carries them, so a warning for them would be wallpaper). Any other argument
#: left out (a name starting ``_``) is COUNTED in the leading ``⚠`` line.
_NOISE_KEYS: frozenset[str] = frozenset({
    "session",        # driver session handle, opaque
    "snapshot_id",    # snapshot binding — validated in code, not by a human
    "element_token",  # opaque per-snapshot handle, unreadable by design
})

#: How many names the leading line spells out before it says "and N more".
_NOTE_NAMES = 4


@dataclass(frozen=True)
class RenderedArguments:
    """What :func:`render_arguments` made of a call: the view every surface shows, and whether it left anything out."""

    view: dict[str, str] | None
    #: Something a person would want to see was NOT sent: an argument named ``_x`` left out, a name or a value cut (even
    #: when masking then left nothing of the cut to see), more arguments than fit (the overflow row), or a worker's
    #: arguments not all sent. NOT: one of the three validated desktop handles left out of a DESKTOP prompt, an empty
    #: value (shown), or a value only masked (it is there, and says so). The ``arguments_cut`` of a
    #: served approval request is this (``ApprovalQueue.serialize_pending``); ``critical`` reads the broader
    #: :func:`view_was_cut`, which counts a mask too.
    left_out: bool


def redact_arguments(
    arguments: dict[str, Any] | None,
    *,
    max_value: int = MAX_VALUE_CHARS,
    max_total: int = MAX_TOTAL_CHARS,
    wire: bool = False,
    desktop: bool = False,
) -> dict[str, str] | None:
    """The view :func:`render_arguments` makes of *arguments* (see there)."""
    return render_arguments(arguments, max_value=max_value, max_total=max_total, wire=wire, desktop=desktop).view


def render_arguments(
    arguments: dict[str, Any] | None,
    *,
    max_value: int = MAX_VALUE_CHARS,
    max_total: int = MAX_TOTAL_CHARS,
    wire: bool = False,
    desktop: bool = False,
) -> RenderedArguments:
    """Render tool arguments for human consent: redacted, truncated, ordered, and honest about what it did.

    The view is ``None`` when the call has no arguments at all. Otherwise its first entry is the ``⚠`` line when anything was
    masked, cut or left out (see :data:`NOTE_KEY`); every other entry is one argument as one visible line. Every argument
    is shown (up to :data:`MAX_ARGS_SHOWN`): *max_total* is shared out so that a short value leaves its room to a long one
    and no argument gets less than :data:`MIN_VALUE_SHARE` characters, so padding a call with junk arguments cannot push
    the one that matters out of sight.

    *wire*: the arguments came from :func:`bound_for_wire` (a worker's request): its count of arguments not sent becomes
    a line of the ``⚠`` note, and is not an argument.

    *desktop*: the call is a desktop action (``ComputerApprovalChannel``), which carries the three validated handles in
    :data:`_NOISE_KEYS` on every call: they are left out of its view without a word, because a warning on every desktop
    action would teach people to ignore it. Any other call SHOWS them like any argument: a tool that takes a ``session``
    is not carrying plumbing, and a person answering for it would want to see which one.
    """
    if not arguments:
        return RenderedArguments(None, False)

    not_sent = 0
    shown_keys: list[tuple[str, Any]] = []
    hidden_names: list[str] = []
    for key in sorted(arguments, key=str):
        name = str(key)[:_NAME_CAP]
        value = arguments[key]
        if wire and name == WIRE_DROPPED_KEY and isinstance(value, int) and not isinstance(value, bool) and value > 0:
            not_sent = value                               # a count of zero or less says nothing: it is an ordinary argument
            continue
        if desktop and name in _NOISE_KEYS and _is_plain_handle(value):
            continue                                       # a desktop call's own handle, left out without a word
        if name.startswith("_") and _is_plain_handle(value):
            hidden_names.append(_name_visible(name))       # counted and named, because a person cannot see what it was
            continue
        shown_keys.append((name, value))

    over = max(0, len(shown_keys) - MAX_ARGS_SHOWN)
    shown_keys = shown_keys[:MAX_ARGS_SHOWN]
    rendered = []
    for name, value in shown_keys:
        is_path = name in PATH_ARGUMENT_NAMES and isinstance(value, (str, os.PathLike))
        rendered.append((name, value, is_path, *_show(name, value, max_value, path=is_path, wire=wire)))
    shares = _shares([len(r[3]) for r in rendered], max_total, max_value)
    out: dict[str, str] = {}
    masked_names: list[str] = []
    cut_names: list[str] = []
    short_names: list[str] = []
    for (name, value, is_path, text, was_masked, was_cut), share in zip(rendered, shares):
        if len(text) > share:
            text, was_masked, was_cut = _show(name, value, share, path=is_path, wire=wire)
        label = _unique(_name_visible(name), out)
        out[label] = text
        if was_masked:
            masked_names.append(label)
        if was_cut:
            cut_names.append(label)
        if len(_name_text(name)) > MAX_NAME_CHARS:
            short_names.append(label)                      # shortened with its own ``…``: the wire flag counts it, the note does not
    if over:
        out[OVERFLOW_KEY] = f"({over} more argument(s) not shown)"
    note = _note(masked_names, cut_names, hidden_names, over, not_sent)
    if note:
        out = {NOTE_KEY: note, **out}
    return RenderedArguments(out or None, bool(hidden_names or cut_names or short_names or over or not_sent))


def _shares(lengths: list[int], total: int, cap: int) -> list[int]:
    """How many characters each value may take: what it needs, up to *cap*, with *total* shared out so that a short value
    leaves its room to the long ones (water-filling) and none gets less than :data:`MIN_VALUE_SHARE`."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    shares = [0] * len(lengths)
    left, remaining = total, len(lengths)
    for i in order:
        fair = max(MIN_VALUE_SHARE, left // remaining)
        shares[i] = min(cap, max(lengths[i], MIN_VALUE_SHARE) if lengths[i] <= fair else fair)
        left -= min(shares[i], lengths[i])
        remaining -= 1
    return shares


#: A worker's POST body may be 64 KiB (``tasks/worker_auth.MAX_BODY_BYTES``) and JSON can spell a non-ASCII character in
#: six bytes, so the whole copy stays within this many characters.
_WIRE_CHARS = 10_000
_WIRE_ARGS = MAX_ARGS_SHOWN
_WIRE_BYTES = 40_000
_WIRE_NAME = 64

#: The wire entry that says how many arguments were not sent. Only a worker request is read for it (``wire=True``).
WIRE_DROPPED_KEY = "_arguments_not_sent"


def bound_for_wire(arguments: dict[str, Any] | None) -> dict[str, Any] | None:
    """A size-bounded RAW copy of a call's arguments, for a worker to send to the daemon, which renders them once.

    A worker used to render its own view and the daemon rendered it again, and the second pass re-cut, re-noted and renamed
    the first one's lines. Raw and bounded keeps one renderer. What the daemon would not show anyway (a validated handle, a
    name starting ``_``) is sent too, but does not count against the limit; at most :data:`_WIRE_ARGS` of the rest, sorted as the daemon
    sorts them, and what did not fit is SAID under :data:`WIRE_DROPPED_KEY`, which the daemon turns into a line of its
    ``⚠`` note (a real argument with that name is renamed). A name is at most :data:`_WIRE_NAME` characters (two that cut
    alike are told apart); a text value is cut to its two ends with the count of what was left out; a finite number, a
    boolean and ``None`` are kept as they are. The whole copy stays under the worker body cap.
    """
    if not arguments:
        return None
    ordered = sorted(arguments, key=str)
    quiet = [k for k in ordered if (str(k) in _NOISE_KEYS or str(k).startswith("_")) and _is_plain_handle(arguments[k])]
    quiet_set = set(quiet)
    keys = [k for k in ordered if k not in quiet_set]
    kept = quiet[:_WIRE_ARGS // 2] + keys[:_WIRE_ARGS]
    dropped = max(0, len(keys) - _WIRE_ARGS) + max(0, len(quiet) - _WIRE_ARGS // 2)
    per = max(80, (_WIRE_CHARS - _WIRE_NAME * len(kept)) // max(1, len(kept)))
    while True:
        out = _wire_copy(arguments, kept, dropped, per)
        # JSON writes a control character in six bytes: measure, and halve the values until the body fits the cap.
        if per <= 16 or len(json.dumps(out).encode()) <= _WIRE_BYTES:
            return out
        per //= 2


def _encodable(text: str) -> str:
    """*text* with a lone surrogate (a model's JSON can hold one) written out, so it can travel as UTF-8."""
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def _wire_copy(arguments: dict[str, Any], kept: list[Any], dropped: int, per: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if dropped > 0:
        out[WIRE_DROPPED_KEY] = dropped
    for key in kept:
        value = arguments[key]
        if value is None or isinstance(value, bool) or (isinstance(value, int) and abs(value) < 10**15) or (
                isinstance(value, float) and math.isfinite(value)):
            wired: Any = value
        else:
            text = _raw(value)
            if len(text) > per:
                half = per // 2
                text = f"{text[:half]}{_WIRE_CUT_HEAD}{len(text) - 2 * half}{_WIRE_CUT_TAIL}{text[-half:]}"
            wired = _encodable(text)
        label, n = _encodable(str(key)[:_WIRE_NAME]), 1
        while label in out or label == WIRE_DROPPED_KEY:
            n += 1
            label = _encodable(f"{str(key)[:_WIRE_NAME - 5]} ({n})")
        out[label] = wired
    return out


#: A short identifier: nothing in it can be a command, a path or an option.
_PLAIN_HANDLE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")


def _is_plain_handle(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 64 and all(ch in _PLAIN_HANDLE_CHARS for ch in value)


#: What a worker puts between the two ends of a value it cut (``_wire_copy``: head, the count, tail), and how to find it
#: again. Plain text matching: this module has no patterns of its own (the redactor is the audit logger's).
_WIRE_CUT_HEAD = " … [+"
_WIRE_CUT_TAIL = " more chars] … "


def _worker_cut(text: str) -> bool:
    """Does *text* hold the marker a worker leaves where it cut a value (:func:`bound_for_wire`)?"""
    at = text.find(_WIRE_CUT_HEAD)
    while at != -1:
        digits = at + len(_WIRE_CUT_HEAD)
        end = digits
        while end < len(text) and text[end] in "0123456789":
            end += 1
        if end > digits and text.startswith(_WIRE_CUT_TAIL, end):
            return True
        at = text.find(_WIRE_CUT_HEAD, at + 1)
    return False


def _show(name: str, value: Any, max_value: int, *, path: bool, wire: bool = False) -> tuple[str, bool, bool]:
    """One argument value as one visible, redacted, cut line, and whether it was masked and whether it was cut.

    "Cut" is whatever left part of the value out: the display cut, the bound before redaction, or (*wire*) the worker's own
    cut. Masking can shrink what is left until the display cut has nothing to do, so it is not enough to compare the
    displayed text with what redaction left."""
    raw = _raw(value)
    if raw == "":
        return '""', False, False
    if not raw.strip():
        symbols = _whitespace_only(raw)
        cut = _truncate(symbols, max_value)
        return cut, False, cut != symbols
    spaced, left_out = _bounded(_newlines_visible(raw))
    # Redact with the CONSENT redactor (audit.py), which lives beside the storage one and shares its key rule: a prompt is
    # read to decide, so a mask here covers only a span proven to be a credential. The BY-NAME rule is told the argument's name.
    # Every argument gets the SAME redaction (a path is not exempt). It runs BEFORE the invisible characters are written out (an ANSI sequence in front of a key is
    # matched as the raw escape) and BEFORE the cut (a key cannot straddle a cut it has already been masked across).
    masked = AuditLogger.redact_for_consent(spaced, name=name)
    was_masked = masked != spaced
    visible = _visible(masked)
    cut = (_truncate_path if path else _truncate)(visible, max_value, left_out)
    return cut, was_masked, cut != visible or left_out > 0 or (wire and _worker_cut(raw))


def _raw(value: Any) -> str:
    """The value as text, with nothing left out: ``None`` is ``null``, an empty list ``[]``, an empty dict ``{}``."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(str(v) for v in value) + "]"
        return str(value)
    except Exception:  # noqa: BLE001 - an object that cannot be printed is said to be, never skipped
        return f"<{type(value).__name__} that cannot be printed>"


def _newlines_visible(text: str) -> str:
    """Every line break as `` ⏎ `` (a trailing one too: a typed Enter must show), so one value is one line."""
    parts = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return " ⏎ ".join(parts).strip(" ") if len(parts) > 1 else text


#: Letters that render as nothing (Hangul fillers, the braille blank, the combining grapheme joiner).
_INVISIBLE_LETTERS = frozenset("\u115f\u1160\u3164\uffa0\u2800\u034f")


def _visible(text: str) -> str:
    """Write out what a person cannot see: control, format (zero-width, direction override), separator, non-ASCII space,
    combining and invisible-letter characters become ``⟨U+XXXX⟩``, and a tab becomes ``⇥``. The text is composed first
    (NFC), so an ordinary accent is one letter and is left alone. Nothing else changes."""
    return "".join(
        "⇥" if ch == "\t"
        else f"⟨U+{ord(ch):04X}⟩" if ch != " " and (unicodedata.category(ch)[0] in "CZM" or ch in _INVISIBLE_LETTERS)
        else ch
        for ch in unicodedata.normalize("NFC", text))


def _whitespace_only(text: str) -> str:
    """A value of nothing but white space, drawn so it is not nothing: Enter ``⏎``, space ``␣``, tab ``⇥``."""
    out = []
    for ch in text.replace("\r\n", "\n").replace("\r", "\n"):
        out.append("⏎" if ch == "\n" else "␣" if ch == " " else "⇥" if ch == "\t" else f"⟨U+{ord(ch):04X}⟩")
    return "".join(out)


def _name_text(name: str) -> str:
    """An argument name written out in full as one line of identifier characters: anything else (a space, a colon, a
    newline, a symbol, an invisible character) is written ``⟨U+XXXX⟩``."""
    return "".join(ch if ch.isascii() and (ch.isalnum() or ch in "_.-[]")
                   else f"⟨U+{ord(ch):04X}⟩" for ch in unicodedata.normalize("NFC", name)) or "(unnamed)"


def _name_visible(name: str) -> str:
    """An argument name as one short line of identifier characters. The model chooses names, so a name cannot look like a
    second argument (``x: rm -rf ~; y``) or like the ``⚠`` line, and two names differing by a trailing space differ on
    screen. One over :data:`MAX_NAME_CHARS` is cut with a ``…`` (which :class:`RenderedArguments` counts as left out)."""
    text = _name_text(name)
    return text if len(text) <= MAX_NAME_CHARS else text[:MAX_NAME_CHARS - 1] + "…"


def _unique(label: str, taken: dict[str, str]) -> str:
    """*label*, made different from the ones already shown and from the two lines this view adds itself."""
    candidate, n = label, 1
    while candidate in taken or candidate in (NOTE_KEY, OVERFLOW_KEY):
        n += 1
        candidate = f"{label} ({n})"
    return candidate


def _note(masked: list[str], cut: list[str], hidden: list[str], over: int, not_sent: int = 0) -> str:
    """The one line that says what the view did to the call, or '' when it did nothing."""
    def names(items: list[str]) -> str:
        shown = ", ".join(items[:_NOTE_NAMES])
        return shown + (f" and {len(items) - _NOTE_NAMES} more" if len(items) > _NOTE_NAMES else "")

    parts = []
    if masked:
        parts.append(f"masked: {names(masked)}")
    if cut:
        parts.append(f"cut: {names(cut)}")
    if hidden:
        parts.append(f"not shown: {names(hidden)}")
    if over:
        parts.append(f"{over} more argument(s) not shown")
    if not_sent:
        parts.append(f"{not_sent} more argument(s) were not sent by the worker")
    return "; ".join(parts)


def _truncate(text: str, limit: int, left_out: int = 0) -> str:
    """Cut to *limit*, and SAY how much was cut. Never a silent ellipsis. *left_out*: what was cut before, to bound a regex."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (+{len(text) - limit + left_out} more chars)"


def _truncate_path(text: str, limit: int, left_out: int = 0) -> str:
    """Cut a PATH to about *limit* from the MIDDLE, and SAY how much was cut.

    The start says where it begins and the end is the filename the owner is asked about. Cutting from the end (as
    :func:`_truncate` does for prose) lost the filename, and a path padded to the limit could hide where it pointed.
    """
    if len(text) <= limit:
        return text
    if limit < 3:
        return _truncate(text, limit, left_out)  # no room for a head and a tail (text[-0:] would be the whole path)
    head = limit // 3
    tail = limit - head
    return f"{text[:head]}… (+{len(text) - limit + left_out} more chars) …{text[-tail:]}"


def format_arguments(rendered: dict[str, str] | None) -> list[str]:
    """Prompt lines for a rendered argument map. Empty list when there is none.

    One line per argument rather than a single joined blob: the operator is
    scanning for the one value that decides the answer, and a wall of
    ``k=v, k=v, k=v`` is the shape that gets skimmed.
    """
    if not rendered:
        return []
    return [f"  {key}: {value}" for key, value in rendered.items()]
