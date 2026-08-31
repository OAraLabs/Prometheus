"""Keep gateway bot tokens out of everything the logging stack emits.

The failure mode: the Telegram Bot API carries the bot token in the URL
*path*, and ``httpx`` logs every request line at INFO — so each
``getUpdates`` poll emits

    httpx INFO HTTP Request: POST https://api.telegram.org/bot<TOKEN>/getUpdates

into the journal and into ``~/.prometheus/logs/daemon.log``, at roughly
one line a second for as long as the gateway is up. #95 pinned ``httpx``
to WARNING in ``prometheus.daemon.main`` to stop it.

That gate is necessary but not sufficient, which is why this module
exists:

  * It was **per-entry-point**. ``main()`` in ``prometheus.daemon`` had it;
    ``prometheus.__main__`` (which runs the setup wizard's ``getMe``
    token check), ``jobs.daily_briefing`` (which sends via Telegram) and
    ``benchmarks.runner`` each call ``basicConfig`` of their own and did
    not. A control that has to be remembered at four call sites is a
    control that gets missed at the fifth.
  * It is **level-scoped**. WARNING silences the INFO request line, but
    an *error* path still formats the URL: ``httpx`` embeds the request
    URL in ``HTTPStatusError``, and ``telegram.py`` logs exception
    strings at ERROR (``_on_polling_error``) — above the gate, not below
    it.

So redaction is applied where every log line has to pass regardless of
logger, level or entry point: the handler. ``daemon.py`` already scrubbed
the *Slack* token out of one exception string by hand — the same failure
mode, caught once, at one site. This generalises that fix.

Redaction is deliberately unconditional and not config-gated: a secret
scrubber a config file can switch off is a secret scrubber that will be
found off during the next incident.
"""

from __future__ import annotations

import logging
import re

REDACTED = "<redacted>"

# Ordered; each is applied in turn.
#
# 1. Telegram token behind a "bot" prefix. Anchored on the prefix rather
#    than on the token's shape, so a token that does not match the
#    documented "<digits>:<35 chars>" form is still redacted, and a
#    custom API base_url (gateway/telegram_network.py allows one) is
#    covered as well as api.telegram.org.
#
#    The prefix is "bot", NOT "/bot". Real logged data carries the token
#    with no slash in front of it — a captured shell command reads
#    TOK="bot<token>"; curl "https://api.telegram.org/$TOK/..." — and an
#    URL-only pattern walks straight past it. That form also defeats
#    pattern 2 below, because there is no word boundary between the "t"
#    of "bot" and the leading digit. Between them the two patterns cover
#    the token whether it is prefixed, embedded in a URL, or bare.
# 2. A bare Telegram token. The {30,} secret length keeps timestamps
#    ("23:42:01") and "host:port" out of the blast radius.
# 3. Slack bot/app/user tokens (xoxb-/xoxp-/… and the app-level
#    xapp-), which travel in headers and exception strings rather
#    than URLs.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(bot)(\d{5,}:[^/\s\"'\\]+)"), r"\1" + REDACTED),
    (re.compile(r"\b\d{5,}:[A-Za-z0-9_-]{30,}\b"), REDACTED),
    (re.compile(r"\b(?:xox[abprs]|xapp)-[A-Za-z0-9-]{10,}"), REDACTED),
)


def redact_secrets(text: str) -> str:
    """Return ``text`` with any known gateway-token shape replaced.

    Cheap enough to run on every emitted log line: the common case is
    three failed regex scans over a short string.
    """
    if not text:
        return text
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class RedactingFilter(logging.Filter):
    """Redact a record's message in place.

    Rewrites ``msg``/``args`` only when redaction actually changed
    something, so records that carry no secret reach the formatter
    byte-identical to before.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # a broken %-format must not drop the record
            return True
        redacted = redact_secrets(message)
        if redacted != message:
            # args are already interpolated into `redacted`; clearing
            # them stops the formatter re-applying %-substitution (an
            # empty tuple is falsy, so getMessage() returns msg as-is).
            record.msg = redacted
            record.args = ()
        if getattr(record, "exc_text", None):
            record.exc_text = redact_secrets(record.exc_text)
        return True


class RedactingFormatter(logging.Formatter):
    """Wrap another formatter and redact its finished output.

    The filter above cannot see an exception traceback — ``exc_text`` is
    rendered by the formatter, after filters have run. Wrapping the
    formatter is what closes the ERROR/``exc_info`` path.
    """

    def __init__(self, inner: logging.Formatter) -> None:
        super().__init__()
        self._inner = inner

    @property
    def inner(self) -> logging.Formatter:
        return self._inner

    def format(self, record: logging.LogRecord) -> str:
        return redact_secrets(self._inner.format(record))


def install_log_redaction(logger: logging.Logger | None = None) -> None:
    """Arm redaction on every handler of ``logger`` (default: root).

    Call once, immediately after ``logging.basicConfig``. Idempotent, so
    a second call after a ``basicConfig(force=True)`` re-arm is safe and
    will not stack wrappers.

    Also pins the ``httpx`` request logger to WARNING. Redaction alone
    would make those lines safe, but they are ~1/second of pure noise
    that buries the real signal in daemon.log.
    """
    target = logger if logger is not None else logging.getLogger()

    for handler in target.handlers:
        if not any(isinstance(f, RedactingFilter) for f in handler.filters):
            handler.addFilter(RedactingFilter())
        formatter = handler.formatter
        if not isinstance(formatter, RedactingFormatter):
            handler.setFormatter(
                RedactingFormatter(formatter or logging.Formatter())
            )

    # SPRINT G3 (#95) kept verbatim: httpx logs every request URL at INFO
    # and the Telegram token rides in that URL.
    logging.getLogger("httpx").setLevel(logging.WARNING)
