"""FIRSTLIGHT GAP-3 — the CLI's first answer must not be buried in logs.

The stranger walk (minute six): a first-ever ``prometheus --once`` reply
arrived interleaved with httpx/agent_loop/audit INFO lines — operator
logging on first contact, which a stranger reads as "broken". The fix:
on the CLI surfaces (interactive chat and ``--once``) the console defaults
to WARNING, the full INFO stream lands in ``~/.prometheus/logs/cli.log``
(previously the console spam was also the ONLY record of a CLI run —
there was no file log at all), ``-v/--verbose`` restores the old console
stream verbatim, ``--debug`` is unchanged, and the daemon path is
untouched.

Guards in BOTH directions (§2c) plus the file-log pin, all through the
real CLI as a subprocess against the round-1 stub model — no mocks on
the path the complaint lived in.
"""

from __future__ import annotations

import os
import site
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
STUB = REPO / "scripts" / "firstlight_stub_model.py"
MARKER = "FIRSTLIGHT-COMPLETE"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def cli_home(tmp_path):
    """Isolated HOME with a config pointing at a live stub model."""
    port = _free_port()
    stub = subprocess.Popen(
        [sys.executable, str(STUB), "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)
    else:
        stub.kill()
        pytest.fail("stub model never came up")

    home = tmp_path / "home"
    (home / ".prometheus").mkdir(parents=True)
    (home / ".prometheus" / "prometheus.yaml").write_text(textwrap.dedent(f"""\
        model:
          provider: llama_cpp
          base_url: http://127.0.0.1:{port}
          model: firstlight-stub-model
        gateway:
          telegram_enabled: false
        tools:
          deferred_loading:
            enabled: auto
            always_loaded: [glob]
        """), encoding="utf-8")
    try:
        yield home
    finally:
        stub.terminate()
        try:
            stub.wait(timeout=5)
        except subprocess.TimeoutExpired:
            stub.kill()


def _run_once(home: Path, *extra: str) -> subprocess.CompletedProcess:
    env = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONUNBUFFERED": "1",
        "LANG": "C.UTF-8",
    }
    for passthrough in ("PYTHONPATH", "VIRTUAL_ENV"):
        if passthrough in os.environ:
            env[passthrough] = os.environ[passthrough]
    # A foreign HOME moves the user site-packages (site.getusersitepackages
    # derives from HOME), so a box whose deps live in ~/.local lost pydantic
    # in the child and every test here failed with ModuleNotFoundError.
    # Isolation is about config, not about which packages the interpreter
    # has: pin the parent's user base explicitly.
    env["PYTHONUSERBASE"] = site.getuserbase()
    # --config is EXPLICIT on purpose. load_config() searches the
    # repo-local config/prometheus.yaml FIRST and only then
    # $PROMETHEUS_CONFIG_DIR — so setting HOME is not isolation: run from
    # a checkout that has a live config (the operator's own) and these
    # tests silently drive the REAL model instead of the stub. They passed
    # in CI and in worktrees only because neither has that file
    # (Standing-Principles §3b — green for an environmental accident).
    return subprocess.run(
        [sys.executable, "-m", "prometheus",
         "--config", str(home / ".prometheus" / "prometheus.yaml"),
         *extra, "--once", "list files"],
        capture_output=True, text=True, timeout=120, env=env, cwd=home,
    )


def test_default_console_is_quiet_and_the_reply_is_not(cli_home):
    """Quiet direction: the reply arrives; INFO logging does not."""
    proc = _run_once(cli_home)
    assert proc.returncode == 0, proc.stderr[-2000:]
    combined = proc.stdout + proc.stderr
    assert MARKER in combined, "the actual answer must still be printed"
    assert " INFO " not in combined, (
        "INFO logger lines reached the default console — GAP-3 regressed; "
        "first lines seen:\n" +
        "\n".join(l for l in combined.splitlines() if " INFO " in l)[:500]
    )


def test_cli_log_file_carries_the_info_stream(cli_home):
    """The file-log pin: quiet console must not mean lost diagnostics."""
    proc = _run_once(cli_home)
    assert proc.returncode == 0
    cli_log = cli_home / ".prometheus" / "logs" / "cli.log"
    assert cli_log.exists(), "cli.log was not created on a CLI run"
    text = cli_log.read_text(encoding="utf-8", errors="replace")
    assert " INFO " in text, (
        "cli.log exists but carries no INFO lines — the console went quiet "
        "AND the record went missing, which is strictly worse than GAP-3"
    )


def test_verbose_restores_the_info_console_stream(cli_home):
    """Loud direction: -v puts the old stream back on the console."""
    proc = _run_once(cli_home, "-v")
    assert proc.returncode == 0, proc.stderr[-2000:]
    combined = proc.stdout + proc.stderr
    assert MARKER in combined
    assert " INFO " in combined, (
        "-v/--verbose did not restore INFO console logging"
    )


# ---------------------------------------------------------------------------
# `prometheus token show` — machine-readable: the token is line 1 of stdout
# and no INFO line reaches the console by default
# ---------------------------------------------------------------------------

TOKEN = "firstlight-test-token-0123456789abcdef"


@pytest.fixture
def token_home(tmp_path):
    """Isolated HOME with a CLOUD config, the API token in the env file, and
    (in the child's environment, see _run_token_show) the exported cloud key
    that made the FIRSTLIGHT cloud leg fail at S6: env_override logs
    "Applied env overrides: OPENAI_API_KEY=…" at INFO during load_config,
    before the token is printed."""
    home = tmp_path / "home"
    (home / ".prometheus").mkdir(parents=True)
    (home / ".prometheus" / "prometheus.yaml").write_text(textwrap.dedent("""\
        model:
          provider: openai
          model: gpt-4o
          api_key_env: OPENAI_API_KEY
        gateway:
          telegram_enabled: false
        web:
          enabled: true
        """), encoding="utf-8")
    env_dir = home / ".config" / "prometheus"
    env_dir.mkdir(parents=True)
    (env_dir / "env").write_text(
        f"PROMETHEUS_API_TOKEN={TOKEN}\nOPENAI_API_KEY=dummy-openai-key-for-token-show-test\n",
        encoding="utf-8",
    )
    return home


def _run_token_show(home: Path, *extra: str) -> subprocess.CompletedProcess:
    env = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONUNBUFFERED": "1",
        "LANG": "C.UTF-8",
        "PYTHONUSERBASE": site.getuserbase(),
        # The stranger's shape: the cloud key exported in the shell. This is
        # what env_override picks up and announces at INFO.
        "OPENAI_API_KEY": "dummy-openai-key-for-token-show-test",
    }
    for passthrough in ("PYTHONPATH", "VIRTUAL_ENV"):
        if passthrough in os.environ:
            env[passthrough] = os.environ[passthrough]
    return subprocess.run(
        [sys.executable, "-m", "prometheus",
         "--config", str(home / ".prometheus" / "prometheus.yaml"),
         *extra, "token", "show"],
        capture_output=True, text=True, timeout=120, env=env, cwd=home,
    )


def test_token_show_prints_the_token_first_and_no_info_by_default(token_home):
    """The FIRSTLIGHT cloud leg's S6: `prometheus token show 2>&1 | head -1`
    read an INFO line as the token. A machine-readable subcommand's
    console is quiet unless asked."""
    proc = _run_token_show(token_home)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.splitlines()[0] == TOKEN
    combined = proc.stdout + proc.stderr
    assert " INFO " not in combined, (
        "INFO reached the console of a machine-readable subcommand:\n" +
        "\n".join(l for l in combined.splitlines() if " INFO " in l)[:500]
    )
    # And the merged stream — what a `2>&1 | head -1` consumer sees — leads
    # with the token too, which is the exact contract the harness reads.
    assert combined.splitlines()[0] == TOKEN


def test_token_show_verbose_restores_the_info_stream(token_home):
    """-v is the loud direction, same as the chat surfaces: the env-override
    line is still produced, so nothing was silenced — only defaulted."""
    proc = _run_token_show(token_home, "-v")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.splitlines()[0] == TOKEN          # stdout is still the answer
    assert "Applied env overrides" in proc.stderr          # the log line exists, on stderr
    assert "dummy-openai-key-for-token-show-test" not in proc.stderr   # and stays masked


def test_daemon_subcommand_console_is_unchanged():
    """`daemon` is read by journald and keeps INFO — the quiet set is
    exactly the machine-readable subcommands, not every subcommand."""
    from prometheus import __main__ as cli
    assert cli._QUIET_CONSOLE_COMMANDS == frozenset({"token"})

