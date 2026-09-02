"""Daemon and CLI logs must rotate.

An unrotated log keeps every line it was ever given, forever. That is a
capacity problem on its own (daemon.log reached 1.4 GB on a long-lived
install), but the sharper edge is that it turns any line the process
should not have written into permanent on-disk residue instead of
something that ages out on its own.

Redaction (test_log_redaction.py) stops the writing. Rotation bounds what
survives when something slips past it anyway. The two are tested together
below: a record that rotates out must be redacted in the rotated file as
well as the live one.

The fabricated token sits below .githooks/pre-commit's length floors —
see the note in test_log_redaction.py.
"""

from __future__ import annotations

import logging
import logging.handlers
import re
from pathlib import Path

from prometheus.security import install_log_redaction

FAKE_TOKEN = "123456:AAF-FakeTokenForTestsOnly_0123456789x"
SRC = Path(__file__).resolve().parents[1] / "src" / "prometheus"


def test_redaction_holds_across_a_rotation(tmp_path):
    """Every rotated file, not just the live one, must be clean."""
    log = tmp_path / "daemon.log"
    logger = logging.getLogger("test_log_rotation")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.handlers.RotatingFileHandler(
        log, maxBytes=2048, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    install_log_redaction(logger)

    for _ in range(200):
        logger.info(
            "HTTP Request: POST https://api.telegram.org/bot%s/getUpdates",
            FAKE_TOKEN,
        )
    handler.close()
    logger.handlers.clear()

    written = sorted(tmp_path.glob("daemon.log*"))
    assert len(written) > 1, "maxBytes was never reached — rotation untested"
    for path in written:
        assert FAKE_TOKEN not in path.read_text(encoding="utf-8"), path.name


def test_the_daemon_and_cli_logs_are_rotating_handlers():
    """A plain FileHandler on these paths is the unbounded-growth bug."""
    daemon = (SRC / "daemon.py").read_text(encoding="utf-8")
    cli = (SRC / "__main__.py").read_text(encoding="utf-8")

    assert re.search(
        r"RotatingFileHandler\(\s*log_dir\s*/\s*[\"']daemon\.log[\"']", daemon
    ), "daemon.log must use a RotatingFileHandler"
    assert re.search(
        r"RotatingFileHandler\(\s*\n?\s*get_logs_dir\(\)\s*/\s*[\"']cli\.log[\"']",
        cli,
    ), "cli.log must use a RotatingFileHandler"

    # and neither may fall back to the unbounded handler on those paths
    for label, text in (("daemon.py", daemon), ("__main__.py", cli)):
        assert not re.search(
            r"(?<!Rotating)FileHandler\([^)]*(daemon|cli)\.log", text
        ), f"{label} still opens a log with a non-rotating FileHandler"


def test_rotation_is_bounded_to_a_sane_ceiling():
    """Guard the constants against an accidental unbounded edit."""
    from prometheus.daemon import _LOG_BACKUPS, _LOG_MAX_BYTES

    assert 0 < _LOG_MAX_BYTES <= 128 * 1024 * 1024
    assert 0 < _LOG_BACKUPS <= 10
    total = _LOG_MAX_BYTES * (_LOG_BACKUPS + 1)
    assert total <= 1024 * 1024 * 1024, f"log dir ceiling too high: {total}"
