"""Turn outcomes for telemetry v2 (WP-X.54 T-4).

T-3 writes one ``turns`` row per turn and leaves ``outcome`` NULL. This module
fills ``outcome`` / ``outcome_source`` / ``outcome_at``, under Will's rulings
of 2026-10-03:

=====================  ======================  =================================
``outcome``            ``outcome_source``      set by
=====================  ======================  =================================
``forced_stop``        ``daemon``              turn end: the loop stopped it
``error_terminal``     ``daemon``              turn end: it ended in an error
``accepted_user``      ``user_signal``         the next message: a thanks/ack,
                                               or a clearly different request
``user_corrected``     ``user_signal``         the next message: a clear
                                               correction, or the same request
``abandoned``          ``daemon``              the heartbeat sweep: no message
                                               within ``WINDOW_SECONDS``
``accepted_verified``  ``acceptance_command``  coding mode: acceptance passed
``rejected_verified``  ``acceptance_command``  coding mode: acceptance failed
NULL                   NULL                    not known, or not clear
=====================  ======================  =================================

PLAIN RULES, NO GUESSING. Every label comes from a fixed rule over the text of
one message or the times on the rows; no model judges anything (Instinct may
relabel later). When a signal is unclear the outcome stays NULL.

FIRST WRITE WINS. Every write here is "only while ``outcome`` IS NULL", so a
turn-end ``forced_stop`` is never turned into ``user_corrected`` by the next
message, and the sweep can run any number of times.

NOTHING WAITS. Every write goes through the telemetry v2 queued writer
(``TelemetryV2Writer.call``), which runs it on the writer thread after the
turn rows queued before it. The caller only classifies a string and queues.
Nothing here raises into a turn, an ingress or the heartbeat.

``abandoned`` IS NEUTRAL (for T-5 and anyone training on these rows). It means
no message followed within the window. That usually means the user got what
they needed and left, so it must never be used as a negative label. The value
keeps its spec name.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict
from typing import Any

log = logging.getLogger(__name__)

#: How long after a turn ends its next message still counts as a reply to it.
#: Past this the heartbeat sweep calls the turn ``abandoned`` (NEUTRAL, above).
WINDOW_SECONDS = 30 * 60

#: Most turns one sweep labels; the rest wait for the next sweep.
SWEEP_BATCH = 500

#: Surfaces whose ingress calls ``note_user_message``. A turn on any other
#: surface (the CLI, whose ingress is deferred; coding mode) never gets a user
#: signal, so the sweep must not call it abandoned either.
USER_SIGNAL_SURFACES = ("beacon", "telegram", "slack", "discord", "rest")

#: Sessions that are never a conversation: ``system`` (the /benchmark
#: diagnostics) and ``""`` (an ephemeral turn, which records no session).
_NOT_CONVERSATIONS = frozenset({"", "system"})

# Turn end (ruling 2). Every _StopReason value in engine/agent_loop.py is on
# exactly one side; tests/test_telemetry_v2_outcomes.py holds the split closed.
#: The loop ended a turn the model was still going on.
FORCED_STOP_REASONS = frozenset({
    "circuit_breaker_trip", "max_iterations_hit", "unproductive_repeat", "divergence_halt",
    "max_turns_exhausted", "forced_tool_not_honored", "boundary_escape", "cancelled",
})
#: The turn ended in an error: no usable model response.
ERROR_TERMINAL_REASONS = frozenset({
    "provider_error", "no_final_message", "error", "empty_response", "context_preflight_refusal",
})


def turn_end_outcome(forced_stop_reason: str | None, terminal_kind: str | None) -> str | None:
    """The daemon's outcome for a turn as it ends, or None to leave it to the user."""
    if forced_stop_reason in ERROR_TERMINAL_REASONS:
        return "error_terminal"
    if forced_stop_reason is not None:
        return "forced_stop"
    if terminal_kind == "error":
        return "error_terminal"
    return None


# --------------------------------------------------------------------------- #
# The classifier: what the next message says about the turn before it
# --------------------------------------------------------------------------- #

#: A message that OPENS with one of these is a correction.
_CORRECTION_OPENERS = (
    "no", "nope", "wrong", "that's wrong", "that is wrong", "incorrect", "that's incorrect",
    "not what i asked", "that's not", "that is not", "i meant", "i said", "i asked",
    "you didn't", "you did not", "you forgot", "you missed", "try again", "retry", "redo",
    "still broken",
)
#: A message that CONTAINS one of these anywhere is a correction.
_CORRECTION_PHRASES = (
    "didn't work", "did not work", "doesn't work", "does not work", "not working",
    "still broken", "still failing", "same error", "not what i asked", "you misunderstood",
    "wrong answer",
)
#: Open with "no"/"nope" and correct nothing (ruling 1b).
_NOT_CORRECTIONS = ("no worries", "no problem", "no thanks", "nope that's it", "nope, that's it")
#: On their own these say nothing clear (ruling 1a). With a correction opener
#: or phrase after them, the message is a correction.
_AMBIGUOUS_OPENERS = ("actually", "wait", "still", "again")
#: Every word of an acknowledgement comes from here.
_ACK_WORDS = frozenset({
    "thanks", "thank", "you", "thx", "ty", "cheers", "great", "perfect", "awesome", "nice",
    "excellent", "cool", "ok", "okay", "got", "it", "works", "worked", "that", "thats",
    "looks", "good", "lgtm", "much", "so", "very", "appreciated", "brilliant", "wonderful",
    "fantastic", "amazing", "all", "set",
})
_ACK_EMOJI = frozenset({"\U0001f44d", "\U0001f64f", "❤️", "❤", "\U0001f44c",
                        "\U0001f389", "✅"})

#: Overlap (Jaccard, over words of 3+ letters) at or above which the message
#: repeats the previous request (ruling 1c) ...
REPEAT_OVERLAP = 0.6
#: ... and at or below which it is a different request. Between: NULL.
DIFFERENT_OVERLAP = 0.25
#: Fewer qualifying words than this and overlap measures nothing (ruling 1c).
MIN_QUALIFYING_WORDS = 4

_WORD = re.compile(r"[a-z0-9_']+")


def _normalize(text: str) -> str:
    t = text.replace("’", "'").replace("‘", "'").lower().strip()
    return re.sub(r"\s+", " ", t)


def _opens_with(text: str, phrases) -> bool:  # noqa: ANN001
    for p in phrases:
        if text == p or (text.startswith(p) and not text[len(p)].isalnum()
                         and text[len(p)] != "'"):
            return True
    return False


def _strip_please(text: str) -> str:
    for p in ("please ", "pls ", "plz "):
        if text.startswith(p):
            return text[len(p):]
    return text


def _is_correction(text: str) -> bool:
    if _opens_with(text, _NOT_CORRECTIONS):
        return False
    return _opens_with(_strip_please(text), _CORRECTION_OPENERS) or any(
        p in text for p in _CORRECTION_PHRASES)


def _qualifying(text: str) -> set[str]:
    return {w.replace("'", "") for w in _WORD.findall(text) if len(w.replace("'", "")) >= 3}


def _overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def _is_ack(text: str) -> bool:
    bare = text
    for e in _ACK_EMOJI:
        bare = bare.replace(e, " ")
    words = [w.replace("'", "") for w in _WORD.findall(bare)]
    if not words:
        return bare != text  # emoji only
    return all(w in _ACK_WORDS for w in words)


def classify_next_message(text: str, previous: str | None) -> str | None:
    """``user_corrected`` / ``accepted_user`` from the message after a turn, or None.

    ``previous`` is the request before it (the message that started the
    turn), or None when it is not known, e.g. after a restart. Clear signals
    only (rulings 1, 1a-1c):

    * a correction: the message opens with a correction ("no", "that's
      wrong", "not what I asked", "try again", "redo", ...) or contains a
      correction phrase ("didn't work", "still broken", "same error", ...);
      "no worries", "no problem", "no thanks" and "nope, that's it" are not;
    * "actually", "wait", "still", "again": a correction only together with a
      correction after them, else None;
    * the same request again: at least ``MIN_QUALIFYING_WORDS`` words of 3+
      letters and ``REPEAT_OVERLAP`` overlap with ``previous``;
    * an acknowledgement ("thanks", "perfect", a thumbs-up): accepted;
    * a different request: as many words, overlap at most ``DIFFERENT_OVERLAP``;
    * anything else: None.
    """
    t = _normalize(text)
    if not t:
        return None
    if _opens_with(t, _AMBIGUOUS_OPENERS):
        for amb in _AMBIGUOUS_OPENERS:
            if _opens_with(t, (amb,)):
                rest = t[len(amb):].lstrip(" ,.;:!-")
                if _is_correction(rest) or any(p in t for p in _CORRECTION_PHRASES):
                    return "user_corrected"
                return None
    if _is_correction(t):
        return "user_corrected"
    if _is_ack(t):
        return "accepted_user"
    if previous is None:
        return None
    words = _qualifying(t)
    if len(words) < MIN_QUALIFYING_WORDS:
        return None
    overlap = _overlap(words, _qualifying(_normalize(previous)))
    if overlap >= REPEAT_OVERLAP:
        return "user_corrected"
    if overlap <= DIFFERENT_OVERLAP:
        return "accepted_user"
    return None


# --------------------------------------------------------------------------- #
# Writes. Each is a conditional UPDATE run on the writer thread.
# --------------------------------------------------------------------------- #


def _writer(telemetry: Any):  # noqa: ANN202
    if telemetry is None:
        return None
    get = getattr(telemetry, "v2_writer", None)
    return get() if callable(get) else None


def _queue(telemetry: Any, label: str, fn) -> None:  # noqa: ANN001
    """Queue ``fn`` on the telemetry's v2 writer. Never raises."""
    try:
        writer = _writer(telemetry)
        if writer is not None:
            writer.call(label, fn)
    except Exception:
        log.warning("telemetry v2 outcomes: could not queue %s", label, exc_info=True)


def stamp_turn_end(writer: Any, turn_id: str, outcome: str, at: float) -> None:
    """Queue the daemon's turn-end outcome on ``writer`` (the turn's own writer)."""
    def write(conn) -> None:  # noqa: ANN001
        conn.execute(
            "UPDATE turns SET outcome = ?, outcome_source = 'daemon', outcome_at = ? "
            "WHERE turn_id = ? AND outcome IS NULL", (outcome, at, turn_id))

    writer.call("outcome_turn_end", write)


class _Recent:
    """The last message seen per session: its text and when it arrived.

    Memory only, never stored: the previous request for the repeat check, and
    the time that says whether a message is the FIRST one after a turn ended.
    Bounded; after a restart it is empty, and the rules that need it say None.
    """

    MAX_SESSIONS = 1024

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_session: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def swap(self, session_id: str, text: str, at: float) -> tuple[str, float] | None:
        with self._lock:
            prev = self._by_session.pop(session_id, None)
            self._by_session[session_id] = (text[:4000], at)
            while len(self._by_session) > self.MAX_SESSIONS:
                self._by_session.popitem(last=False)
            return prev

    def clear(self) -> None:
        with self._lock:
            self._by_session.clear()


_RECENT = _Recent()


def reset_memory() -> None:
    """Forget every session's last message (tests)."""
    _RECENT.clear()


def note_user_message(session_id: str, text: str, *, telemetry: Any = None,
                      at: float | None = None) -> None:
    """Ingress hook: a user message arrived in ``session_id``. Never raises.

    Called where each surface receives a message for the agent, BEFORE the
    message joins the session or waits for the turn lock, with the session id
    the turn will be recorded under. Not for slash commands or machine-built
    prompts. ``telemetry`` defaults to the daemon's tracker; with telemetry
    off there is none, and nothing happens.

    It labels at most one turn: the session's latest turn, if that turn had
    ENDED when this message arrived, ended within ``WINDOW_SECONDS``, has no
    outcome, and this is the first message since it ended. So a message sent
    while a turn is still running never labels that turn (ruling); it counts
    only for the turn before it. The label is ``classify_next_message``'s;
    when that is None nothing is written.
    """
    try:
        at = time.time() if at is None else at
        if session_id in _NOT_CONVERSATIONS:
            return
        if telemetry is None:
            from prometheus.telemetry.tracker import get_telemetry_handle

            telemetry = get_telemetry_handle()
        if telemetry is None:
            return
        prev = _RECENT.swap(session_id, text, at)
        verdict = classify_next_message(text, prev[0] if prev else None)
        if verdict is None:
            return
        prev_at = prev[1] if prev else None
    except Exception:
        log.warning("telemetry v2 outcomes: ingress hook failed", exc_info=True)
        return

    def write(conn) -> None:  # noqa: ANN001
        row = conn.execute(
            "SELECT turn_id, ended_at, outcome, coding_run_id FROM turns "
            "WHERE session_id = ? AND started_at <= ? "
            "ORDER BY started_at DESC, rowid DESC LIMIT 1", (session_id, at)).fetchone()
        if row is None:
            return
        turn_id, ended_at, outcome, coding_run_id = row
        if outcome is not None or coding_run_id is not None:
            return
        if ended_at is None or ended_at > at:
            return  # still running when this message arrived
        if at - ended_at > WINDOW_SECONDS:
            return  # the sweep's: abandoned
        if prev_at is not None and prev_at >= ended_at:
            return  # an earlier message already answered this turn
        conn.execute(
            "UPDATE turns SET outcome = ?, outcome_source = 'user_signal', outcome_at = ? "
            "WHERE turn_id = ? AND outcome IS NULL", (verdict, at, turn_id))

    _queue(telemetry, "outcome_user_signal", write)


def sweep_abandoned(telemetry: Any, *, now: float | None = None,
                    batch: int = SWEEP_BATCH) -> None:
    """Heartbeat sweep: ``abandoned`` for turns no message followed. Never raises.

    A turn qualifies when it ended more than ``WINDOW_SECONDS`` ago, has no
    outcome, ended without a daemon stop (a turn from before T-4 whose
    turn-end outcome never landed must not be called abandoned), is on a
    surface with an ingress hook, is not coding/system/ephemeral, and no later
    turn in its session started within the window (a later turn means a
    message came). ``outcome_at`` is the moment the window closed. At most
    ``batch`` turns per call; the next sweep takes the rest.
    """
    now = time.time() if now is None else now
    marks = ", ".join("?" for _ in USER_SIGNAL_SURFACES)
    excluded = ", ".join("?" for _ in _NOT_CONVERSATIONS)

    def write(conn) -> None:  # noqa: ANN001
        conn.execute(
            "UPDATE turns SET outcome = 'abandoned', outcome_source = 'daemon', "
            "outcome_at = ended_at + ? "
            "WHERE outcome IS NULL AND turn_id IN ("
            " SELECT t.turn_id FROM turns t"
            " WHERE t.outcome IS NULL AND t.ended_at IS NOT NULL AND t.ended_at < ?"
            " AND t.forced_stop_reason IS NULL AND COALESCE(t.terminal_kind, '') != 'error'"
            " AND t.coding_run_id IS NULL"
            f" AND t.surface IN ({marks}) AND t.session_id NOT IN ({excluded})"
            " AND NOT EXISTS (SELECT 1 FROM turns u WHERE u.session_id = t.session_id"
            "  AND u.started_at > t.started_at AND u.started_at <= t.ended_at + ?)"
            " ORDER BY t.ended_at LIMIT ?)",
            (WINDOW_SECONDS, now - WINDOW_SECONDS, *USER_SIGNAL_SURFACES,
             *sorted(_NOT_CONVERSATIONS), WINDOW_SECONDS, batch))

    _queue(telemetry, "outcome_sweep", write)


def coding_acceptance(telemetry: Any, coding_run_id: str, exit_code: int | None, *,
                      at: float | None = None) -> None:
    """Coding mode: the acceptance command judged the run's latest episode. Never raises.

    Exit 0 is ``accepted_verified``; anything else, a timeout (None)
    included, is ``rejected_verified``, the session's own verdict. Only the
    episode it judged, the latest one (ruling 3): an episode rejected for
    showing no evidence was never judged by the command and stays NULL.
    """
    at = time.time() if at is None else at
    outcome = "accepted_verified" if exit_code == 0 else "rejected_verified"

    def write(conn) -> None:  # noqa: ANN001
        conn.execute(
            "UPDATE turns SET outcome = ?, outcome_source = 'acceptance_command', "
            "outcome_at = ? WHERE outcome IS NULL AND ended_at IS NOT NULL AND turn_id = ("
            " SELECT turn_id FROM turns WHERE coding_run_id = ?"
            " ORDER BY started_at DESC, rowid DESC LIMIT 1)",
            (outcome, at, coding_run_id))

    _queue(telemetry, "outcome_acceptance", write)
