"""Natural-language cron schedule parser (no external deps).

Pure-Python regex + datetime. The cron-builtin tool tries standard
5-field cron syntax first; on failure it falls through here. Recognised
patterns:

  - every <N> <minutes|hours|days>          → recurring */N
  - every <weekday> at <time>               → recurring per weekday
  - every weekday at <time>                 → recurring Mon-Fri
  - every weekend at <time>                 → recurring Sat-Sun
  - every day at <time>  /  daily at <time> → recurring once a day
  - in <N> <minutes|hours|days>             → ONE-SHOT
  - today at <time>                         → ONE-SHOT
  - tomorrow at <time>                      → ONE-SHOT
  - at <time>                               → ONE-SHOT (today if future, else tomorrow)
  - hourly | daily | weekly                 → recurring at top of unit

Time strings supported: ``9am``, ``9:30am``, ``9:30 am``, ``15:30``,
``noon``, ``midnight``.

LLM fallback (``set_llm_fallback``) is a hook a host can install so that
unparseable strings can be sent to a local model with constrained
decoding. Default is no-op — the sprint design says "LLM only as last
resort" and most users should never hit it.

Source: Novel code for Prometheus Polish & Platform sprint, work
stream 3 (natural-language cron).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

_WEEKDAY_NAMES: dict[str, int] = {
    "sun": 0, "sunday": 0,
    "mon": 1, "monday": 1,
    "tue": 2, "tues": 2, "tuesday": 2,
    "wed": 3, "weds": 3, "wednesday": 3,
    "thu": 4, "thur": 4, "thurs": 4, "thursday": 4,
    "fri": 5, "friday": 5,
    "sat": 6, "saturday": 6,
}


@dataclass(frozen=True)
class ParsedSchedule:
    """Result of a successful natural-language parse.

    ``one_shot`` flags schedules whose cron expression only describes one
    specific calendar moment ("2026-05-23 14:37"). The cron daemon will
    technically re-fire it annually; the tool's output warns the user.
    """

    cron: str
    one_shot: bool = False
    target: datetime | None = None  # When one_shot=True, the calendar moment.


# ----------------------------------------------------------------------
# LLM fallback hook (sprint doc: "LLM only as last resort")
# ----------------------------------------------------------------------

_LlmFallback = Callable[[str], "ParsedSchedule | None"]
_llm_fallback: _LlmFallback | None = None


def set_llm_fallback(fn: _LlmFallback | None) -> None:
    """Install (or clear) the LLM fallback parser.

    The host wires this from the daemon when a model provider is
    available, e.g.::

        from prometheus.tools.builtin.cron_nl import set_llm_fallback
        set_llm_fallback(my_local_model_parser)

    The function must accept the raw NL text and return either a
    ``ParsedSchedule`` or ``None`` if it cannot produce a valid cron.
    """
    global _llm_fallback
    _llm_fallback = fn


# ----------------------------------------------------------------------
# Time / weekday helpers
# ----------------------------------------------------------------------


def _parse_time(s: str) -> tuple[int, int] | None:
    """Parse '9am' / '15:30' / 'noon' / 'midnight' → (hour, minute)."""
    s = s.strip().lower().replace(".", "")
    if s == "noon":
        return (12, 0)
    if s == "midnight":
        return (0, 0)
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", s)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2)) if m.group(2) else 0
    suffix = m.group(3)
    if suffix == "pm" and hour < 12:
        hour += 12
    elif suffix == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return (hour, minute)


def _one_shot_from_target(target: datetime) -> ParsedSchedule:
    """Build a single-fire cron from a concrete datetime."""
    return ParsedSchedule(
        cron=f"{target.minute} {target.hour} {target.day} {target.month} *",
        one_shot=True,
        target=target,
    )


# ----------------------------------------------------------------------
# The main entry point
# ----------------------------------------------------------------------


def parse_natural_schedule(
    text: str, *, now: datetime | None = None,
) -> ParsedSchedule | None:
    """Convert *text* to a cron-style schedule, or return None.

    None means "unparseable" — callers may then try the LLM fallback via
    :func:`try_llm_fallback`, or surface an error to the user.
    """
    if not text:
        return None
    if now is None:
        now = datetime.now().astimezone()

    s = text.strip().lower()
    # Collapse internal whitespace for easier regex matching.
    s = re.sub(r"\s+", " ", s)

    # in <N> <units> — one-shot
    m = re.fullmatch(
        r"in (\d+) (minute|minutes|min|hour|hours|hr|hrs|day|days)", s
    )
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("min"):
            target = now + timedelta(minutes=n)
        elif unit.startswith("hr") or unit.startswith("hour"):
            target = now + timedelta(hours=n)
        else:
            target = now + timedelta(days=n)
        return _one_shot_from_target(target)

    # tomorrow at <time>
    m = re.fullmatch(r"tomorrow at (.+)", s)
    if m:
        tm = _parse_time(m.group(1))
        if tm is None:
            return None
        target = (now + timedelta(days=1)).replace(
            hour=tm[0], minute=tm[1], second=0, microsecond=0,
        )
        return _one_shot_from_target(target)

    # today at <time>
    m = re.fullmatch(r"today at (.+)", s)
    if m:
        tm = _parse_time(m.group(1))
        if tm is None:
            return None
        target = now.replace(hour=tm[0], minute=tm[1], second=0, microsecond=0)
        return _one_shot_from_target(target)

    # every N units — recurring
    m = re.fullmatch(
        r"every (\d+) (minute|minutes|min|hour|hours|hr|hrs|day|days)", s
    )
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("min"):
            if not (1 <= n <= 59):
                return None
            return ParsedSchedule(cron=f"*/{n} * * * *")
        if unit.startswith("hr") or unit.startswith("hour"):
            if not (1 <= n <= 23):
                return None
            return ParsedSchedule(cron=f"0 */{n} * * *")
        if not (1 <= n <= 31):
            return None
        return ParsedSchedule(cron=f"0 0 */{n} * *")

    # every weekday at <time>  (Mon-Fri)
    m = re.fullmatch(r"every weekdays? at (.+)", s)
    if m:
        tm = _parse_time(m.group(1))
        if tm is None:
            return None
        return ParsedSchedule(cron=f"{tm[1]} {tm[0]} * * 1-5")

    # every weekend at <time>  (Sat-Sun, cron 0=Sunday, 6=Saturday)
    m = re.fullmatch(r"every weekend(?:s)? at (.+)", s)
    if m:
        tm = _parse_time(m.group(1))
        if tm is None:
            return None
        return ParsedSchedule(cron=f"{tm[1]} {tm[0]} * * 0,6")

    # every day at <time>  /  daily at <time>
    m = re.fullmatch(r"(?:every day|daily) at (.+)", s)
    if m:
        tm = _parse_time(m.group(1))
        if tm is None:
            return None
        return ParsedSchedule(cron=f"{tm[1]} {tm[0]} * * *")

    # every <weekday> at <time>
    m = re.fullmatch(r"every (\w+) at (.+)", s)
    if m:
        dow = _WEEKDAY_NAMES.get(m.group(1))
        if dow is None:
            return None
        tm = _parse_time(m.group(2))
        if tm is None:
            return None
        return ParsedSchedule(cron=f"{tm[1]} {tm[0]} * * {dow}")

    # at <time> — today if future, tomorrow if past (one-shot)
    m = re.fullmatch(r"at (.+)", s)
    if m:
        tm = _parse_time(m.group(1))
        if tm is None:
            return None
        target = now.replace(hour=tm[0], minute=tm[1], second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return _one_shot_from_target(target)

    # bare frequency words
    if s in ("hourly", "every hour"):
        return ParsedSchedule(cron="0 * * * *")
    if s in ("daily", "every day"):
        return ParsedSchedule(cron="0 0 * * *")
    if s in ("weekly", "every week"):
        return ParsedSchedule(cron="0 0 * * 0")
    if s in ("monthly", "every month"):
        return ParsedSchedule(cron="0 0 1 * *")

    return None


def try_llm_fallback(text: str) -> ParsedSchedule | None:
    """Call the installed LLM fallback, if any. Returns None on absence or error.

    Kept separate from :func:`parse_natural_schedule` so the regex path
    stays pure / dependency-free for testing.
    """
    fn = _llm_fallback
    if fn is None:
        return None
    try:
        return fn(text)
    except Exception:
        return None
