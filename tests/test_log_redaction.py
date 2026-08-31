"""Gateway bot tokens must never reach a log handler.

httpx logs request URLs at INFO and the Telegram Bot API puts the bot
token in the URL path, so an unguarded daemon writes its own credential
to the journal once per getUpdates poll. #95 pinned httpx to WARNING in
the daemon; these tests pin the stronger property — redaction at the
handler, so the token is stripped regardless of logger, level, or entry
point.

The secrets below are fabricated and deliberately sit *below* the length
floors in .githooks/pre-commit ("Length floors are set above the repo's
fixture lengths so obvious fakes don't match in the first place"): six
leading digits instead of a real token's eight-plus, and a short Slack
body. They still exercise every redaction pattern here — which is the
point of the floors — while keeping the hook and the sdist artifact scan
(tests/test_sdist_contents.py) clean. Do not "fix" them to look more
real: that reintroduces a scanner hit for no test value.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import pytest

from prometheus.security import (
    REDACTED,
    RedactingFormatter,
    install_log_redaction,
    redact_secrets,
)

FAKE_TOKEN = "123456:AAF-FakeTokenForTestsOnly_0123456789x"
SRC = Path(__file__).resolve().parents[1] / "src" / "prometheus"


@pytest.fixture
def captured():
    """A private logger + handler pair with redaction armed."""
    logger = logging.getLogger("test_log_redaction")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        logging.Formatter("%(name)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    install_log_redaction(logger)
    yield logger, stream
    logger.handlers.clear()


# ── the pure function ────────────────────────────────────────────────

def test_redacts_the_exact_incident_line():
    """The verbatim shape found in the journal."""
    line = (
        f"HTTP Request: POST https://api.telegram.org/bot{FAKE_TOKEN}"
        '/getUpdates "HTTP/1.1 200 OK"'
    )
    out = redact_secrets(line)
    assert FAKE_TOKEN not in out
    assert REDACTED in out
    # the diagnostic value of the line survives
    assert "getUpdates" in out and "200 OK" in out


def test_redacts_the_slashless_bot_prefix_form():
    """The shape that actually appears in captured tool output.

    A stored shell command reads TOK="bot<token>"; curl ".../$TOK/getMe".
    The token has no slash in front of it, so a "/bot"-anchored pattern
    walks straight past it — and the bare-token pattern cannot catch it
    either, because there is no word boundary between the "t" of "bot"
    and the leading digit. Found in telemetry.db, training.db and a
    trajectories/ export, i.e. in fine-tune capture data.
    """
    line = (
        '{"name": "bash", "input": {"command": "TOK=\\"bot'
        + FAKE_TOKEN
        + '\\"; curl -s https://api.telegram.org/$TOK/getMe"}}'
    )
    assert FAKE_TOKEN in line, "fixture no longer contains the token"
    out = redact_secrets(line)
    assert FAKE_TOKEN not in out
    assert REDACTED in out


def test_redacts_bare_token_and_custom_api_base():
    assert FAKE_TOKEN not in redact_secrets(f"token={FAKE_TOKEN}")
    # gateway/telegram_network.py allows overriding the API base URL —
    # anchoring on /bot rather than the hostname keeps that covered.
    assert FAKE_TOKEN not in redact_secrets(
        f"POST https://tg-proxy.internal/bot{FAKE_TOKEN}/sendMessage"
    )


def test_redacts_slack_tokens():
    for tok in ("xoxb-1234567890-abc", "xapp-1-A0123-456-dead"):
        assert tok not in redact_secrets(f"Slack auth failed for {tok}")


@pytest.mark.parametrize(
    "benign",
    [
        "2026-08-30 23:42:01,613 prometheus.context.compactor INFO done",
        "connecting to 127.0.0.1:8080",
        "HTTP Request: POST https://api.openai.com/v1/messages 200 OK",
        "ratio 50% of 12345:678",
        "",
    ],
)
def test_benign_lines_pass_through_untouched(benign):
    """No secret, no rewrite — byte-identical."""
    assert redact_secrets(benign) == benign


# ── the handler-level install ────────────────────────────────────────

def test_info_request_line_is_redacted_at_the_handler(captured):
    logger, stream = captured
    logger.info(
        "HTTP Request: POST https://api.telegram.org/bot%s/getUpdates",
        FAKE_TOKEN,
    )
    assert FAKE_TOKEN not in stream.getvalue()
    assert REDACTED in stream.getvalue()


def test_redaction_survives_percent_args(captured):
    """A redacted record must not be re-%-formatted (IndexError bait)."""
    logger, stream = captured
    logger.warning("bot %s reported %d%% failure", FAKE_TOKEN, 50)
    out = stream.getvalue()
    assert FAKE_TOKEN not in out
    assert "50%" in out


def test_error_traceback_is_redacted(captured):
    """The case a WARNING level gate cannot reach.

    httpx embeds the request URL in HTTPStatusError, and telegram.py
    logs exception strings at ERROR (_on_polling_error) — above the
    gate, not below it.
    """
    logger, stream = captured
    try:
        raise RuntimeError(
            "Client error '401 Unauthorized' for url "
            f"'https://api.telegram.org/bot{FAKE_TOKEN}/getMe'"
        )
    except RuntimeError:
        logger.exception("Telegram polling error")
    out = stream.getvalue()
    assert FAKE_TOKEN not in out
    assert "Traceback" in out  # the traceback itself is preserved
    assert REDACTED in out


def test_install_pins_httpx_to_warning():
    install_log_redaction()
    assert logging.getLogger("httpx").level == logging.WARNING


def test_install_is_idempotent(captured):
    """basicConfig(force=True) re-arms; wrappers must not stack."""
    logger, stream = captured
    install_log_redaction(logger)
    install_log_redaction(logger)
    handler = logger.handlers[0]
    assert isinstance(handler.formatter, RedactingFormatter)
    assert not isinstance(handler.formatter.inner, RedactingFormatter)
    assert sum(1 for f in handler.filters if f.__class__.__name__ ==
               "RedactingFilter") == 1
    logger.info("POST https://api.telegram.org/bot%s/getMe", FAKE_TOKEN)
    assert FAKE_TOKEN not in stream.getvalue()


# ── the invariant that keeps a fifth entry point from regressing ─────

def test_every_logging_entry_point_arms_redaction():
    """Whoever configures logging must also arm redaction.

    This control was missed at three of four entry points for months
    precisely because it had to be remembered per-site. Fail loudly when
    a new basicConfig() call appears without it.
    """
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*logging\.basicConfig\(", text, re.MULTILINE):
            if "install_log_redaction()" not in text:
                offenders.append(str(path.relative_to(SRC)))
    assert not offenders, (
        "these modules configure logging without arming token redaction: "
        + ", ".join(sorted(offenders))
    )
