"""Which approval requests must be answered in the app (``critical``), and the push that says so.

The daemon decides; the phone renders. ``critical`` is computed once, in ``ApprovalQueue.serialize_pending``, so
``GET /api/approvals``, the ``approval_pending`` frame and the push cannot disagree. A critical request is pushed
with no ``category`` (no Approve on the notification) and ``critical: true``; every other approval push keeps
``APPROVAL``. ``server_capabilities`` lists ``approval_critical`` so a client knows a missing field means "this
daemon does not say", not "not critical".
"""

from __future__ import annotations

import asyncio
import json
import types

import pytest

from prometheus.permissions.approval_queue import ApprovalQueue, PendingAction
from prometheus.permissions.argument_view import arguments_were_cut
from prometheus.permissions.critical import (
    DEFAULT_CRITICAL_COMMANDS,
    DEFAULT_CRITICAL_TOOLS,
    CriticalRules,
    critical_reasons,
    is_critical,
)
from prometheus.push.apns import SendResult
from prometheus.push.dispatcher import PushDispatcher


class _Gate:
    pass


class _Bus:
    def __init__(self) -> None:
        self.signals: list = []

    async def emit(self, signal) -> None:
        self.signals.append(signal)


def _queue(rules: CriticalRules | None = None) -> tuple[ApprovalQueue, _Bus]:
    q = ApprovalQueue(security_gate=_Gate(), telegram_adapter=None)
    bus = _Bus()
    q.signal_bus = bus
    if rules is not None:
        q.critical_rules = rules
    return q, bus


async def _ask(q: ApprovalQueue, tool: str, arguments: dict | None, description: str = "run it", **kw) -> dict:
    before = len(q.signal_bus.signals)
    waiter = asyncio.create_task(q.request_approval(tool, description, arguments=arguments, **kw))
    for _ in range(50):
        await asyncio.sleep(0)
        if len(q.signal_bus.signals) > before:
            break
    payload = q.signal_bus.signals[-1].payload
    waiter.cancel()
    return payload


# ── the rules ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("command", [
    "rm -rf build", "sudo rm x", "FOO=1 rm x", "/bin/rm x", "ls && rm x", "cat a | xargs echo; rm b",
    "git push", "git push --force origin main", "git -C repo push", "curl -X POST https://example.invalid",
    "echo $(curl https://example.invalid)", "oara pair --owner", "chmod 777 f", "gh pr create",
])
def test_default_commands_that_delete_send_spend_or_change_config_are_critical(command):
    assert CriticalRules().matches("bash", {"command": command}), command


@pytest.mark.parametrize("command", ["ls -la", "git status", "git log --oneline", "cat README.md", "pytest -q",
                                     "echo rm", "grep -r curl src"])
def test_ordinary_commands_are_not(command):
    assert CriticalRules().matches("bash", {"command": command}) == (), command


def test_default_tools_and_config_paths_are_critical(tmp_path):
    rules = CriticalRules()
    for tool in DEFAULT_CRITICAL_TOOLS:
        assert rules.matches(tool, {}), tool
    assert rules.matches("write_file", {"path": "~/.prometheus/prometheus.yaml"})
    assert rules.matches("edit_file", {}, grant_file_path="~/.config/prometheus/x.yaml")
    assert rules.matches("write_file", {"path": str(tmp_path / "notes.md")}) == ()
    assert rules.matches("read_file", {"path": "~/.prometheus/prometheus.yaml"}) == ()


def test_the_owner_lists_replace_the_defaults_and_an_empty_list_means_none():
    rules = CriticalRules.from_config({"push": {"critical_tools": ["web_fetch"], "critical_commands": [],
                                                "critical_paths": []}})
    assert rules.matches("web_fetch", {}) and not rules.matches("message", {})
    assert rules.matches("bash", {"command": "rm -rf /tmp/x"}) == ()
    assert CriticalRules.from_config({}).commands == DEFAULT_CRITICAL_COMMANDS


def test_a_wrong_shape_falls_back_to_the_default_list():
    rules = CriticalRules.from_config({"push": {"critical_commands": "rm"}})
    assert rules.commands == DEFAULT_CRITICAL_COMMANDS


def test_arguments_cut_or_redacted_are_reported():
    assert arguments_were_cut({"command": "x" * 500})
    fake = "sk-" + "q7" * 16                                 # built here so no key-shaped literal is committed
    assert arguments_were_cut({"api" + "_key": fake})
    assert arguments_were_cut({f"k{i}": "v" * 150 for i in range(10)})
    assert not arguments_were_cut({"command": "ls -la"})
    assert not arguments_were_cut(None)


def test_built_in_reasons():
    assert "desktop" in critical_reasons(PendingAction(request_id="a", tool_name="computer", description="d",
                                                       task_id="t1"))
    assert "hold_lift" in critical_reasons(PendingAction(request_id="b", tool_name="hold", description="d",
                                                         hold_device_id="dev1"))
    assert "long_description" in critical_reasons(PendingAction(request_id="c", tool_name="t",
                                                                description="x" * 601))
    assert "long_description" in critical_reasons(PendingAction(request_id="d", tool_name="t",
                                                                description="\n".join("l" * 9)))
    assert not is_critical(PendingAction(request_id="e", tool_name="bash", description="run ls"))


# ── one value on every surface ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_short_bash_ls_is_not_critical_and_truncated_arguments_are():
    q, _ = _queue()
    plain = await _ask(q, "bash", {"command": "ls"})
    assert plain["critical"] is False
    long = await _ask(q, "bash", {"command": "echo " + "y" * 400})
    assert long["critical"] is True


@pytest.mark.asyncio
async def test_the_list_and_the_frame_carry_the_same_value():
    q, _ = _queue()
    frame = await _ask(q, "bash", {"command": "rm -rf build"})
    (listed,) = [q.serialize_pending(a) for a in q.list_pending() if a.request_id == frame["request_id"]]
    assert frame["critical"] is True and listed["critical"] is True


def test_a_hold_lift_serializes_as_critical():
    q, _ = _queue()
    action = PendingAction(request_id="h1", tool_name="lift hold", description="lift the hold", hold_device_id="d1")
    q._register(action)
    assert q.serialize_pending(action)["critical"] is True


# ── the push ─────────────────────────────────────────────────────────────────

class _Store:
    def push_targets(self):
        return [types.SimpleNamespace(id="d1", apns_token="tok", environment="sandbox", bundle_id="",
                                      push_failures=0)]

    def approval_push_targets(self, *, desktop):
        return self.push_targets()


class _Sender:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, **kwargs):
        self.sent.append(kwargs["payload"])
        return SendResult(outcome="ok", status=200)


def _push(payload: dict) -> dict:
    sender = _Sender()
    asyncio.run(PushDispatcher(_Store(), sender, bridge=None).on_signal(
        types.SimpleNamespace(kind="approval_pending", payload=payload)))
    (body,) = sender.sent
    return body


def test_a_critical_push_has_no_category_and_says_critical():
    body = _push({"request_id": "r1", "tool_name": "bash", "description": "rm -rf build", "expires_at": 1.0,
                  "critical": True})
    assert "category" not in body["aps"] and body["critical"] is True


def test_every_other_approval_push_keeps_its_category():
    body = _push({"request_id": "r2", "tool_name": "bash", "description": "ls", "expires_at": 1.0,
                  "critical": False})
    assert body["aps"]["category"] == "APPROVAL" and "critical" not in body


def test_a_payload_without_the_field_is_pushed_as_before():
    body = _push({"request_id": "r3", "tool_name": "bash", "description": "ls", "expires_at": 1.0})
    assert body["aps"]["category"] == "APPROVAL"


def test_the_critical_push_stays_under_the_apns_limit():
    body = _push({"request_id": "r4", "tool_name": "t" * 300, "description": "d" * 5000, "expires_at": 1.0,
                  "critical": True})
    assert len(json.dumps(body).encode()) < 4096


# ── the capability ───────────────────────────────────────────────────────────

def test_server_capabilities_lists_approval_critical():
    from prometheus.web.ws_server import WebSocketBridge

    bridge = WebSocketBridge.__new__(WebSocketBridge)
    bridge.config = {}
    assert "approval_critical" in bridge._server_capabilities()


# ── review round 1 (2026-11-12): what the matcher missed ────────────────────

@pytest.mark.parametrize("command", [
    "sudo ls", "sudo -u root ls",
    "command -p rm x", "env -i rm x", "nice -n 5 rm x", "time -p rm x", "exec -a f rm x",
    'bash -c "rm x"', "sh -c 'rm -rf /'", "eval rm x", "xargs rm < f", "find . -delete",
    "find . -name '*.o' -exec rm {} +", 'python3 -c "import shutil; shutil.rmtree(\'x\')"', "perl -e unlink",
    "(rm x)", "{ rm x; }", "if true; then rm x; fi", "! rm x", "cat <(rm x)",
    "echo x > ~/.prometheus/prometheus.yaml", "cp a ~/.prometheus/b", "tee $HOME/.config/prometheus/x",
    "sed -i s/a/b/ ~/.prometheus/prometheus.yaml", "dd if=a of=${HOME}/.prometheus/x",
    "git branch --delete x", "git checkout -- .", "git restore x", "git stash drop", "git stash clear",
])
def test_round_one_misses_are_critical(command):
    assert CriticalRules().matches("bash", {"command": command}), command


@pytest.mark.parametrize("command", ["python3 -m pytest -q", "time pytest", "env | sort", "find . -name x",
                                     "xargs echo < f", "if true; then ls; fi", "nice make"])
def test_wrappers_and_interpreters_alone_are_not(command):
    assert CriticalRules().matches("bash", {"command": command}) == (), command


def test_downloads_into_config_are_critical_but_the_downloads_folder_is_not():
    rules = CriticalRules()
    assert rules.matches("download_file", {"url": "https://example.invalid/x",
                                           "destination": "~/.prometheus/prometheus.yaml"})
    assert rules.matches("download_file", {"url": "https://example.invalid/x",
                                           "destination": "~/.prometheus/downloads/../config.yaml"})
    assert rules.matches("download_file", {"url": "https://example.invalid/x",
                                           "destination": "~/.prometheus/downloads/file.pdf"}) == ()


def test_a_relative_path_into_a_config_folder_is_critical():
    assert CriticalRules().matches("write_file", {"path": ".prometheus/prometheus.yaml"})
    assert CriticalRules().matches("edit_file", {"path": "x/../.config/prometheus/a.yaml"})


def test_a_matcher_that_raises_makes_the_request_critical():
    class _Odd(dict):
        def get(self, *_a, **_k):
            raise TypeError("odd arguments")

    assert CriticalRules().matches("bash", _Odd({"command": "ls"}))


# ── review round 2 (2026-11-12): fail toward critical ───────────────────────

@pytest.mark.parametrize("command", [
    "bash -lc 'rm -rf x'", "sh -ec 'rm x'", "echo 'rm -rf x' | bash", "echo cm0= | base64 -d | sh", "bash x.sh",
    "source x.sh", ". x.sh", "python3 - <<EOF\nimport shutil\nEOF", "echo 'import os' | python3",
    "timeout -s KILL 5 rm x", "timeout -k 1 5 rm x", "xargs -P 4 -n 1 rm", "env -S 'rm x'",
    "busybox rm x", "setsid rm x", "flock /tmp/l rm x", "arch -x86_64 rm x", "watch 'rm x'", "su -c 'rm x'",
    "script -q /dev/null rm x", "parallel rm ::: a b", "trap 'rm x' EXIT", "builtin rm x",
    "uv run twine upload dist/*", "uvx twine upload dist/*", "python3 -m twine upload dist/*", "npx gh pr create",
    "RM -rf x", "GIT push", "$'rm' x", "{rm,-rf,x}", "awk 'BEGIN{system(\"rm x\")}'", "lua -e 'os.remove(1)'",
    "cd ~/.config && echo > prometheus/config.yaml", "cd && cd .prometheus && echo > config.yaml",
    "echo > ~/.prometh?us/config.yaml", "echo > ~/.config/prometh*/x", "git send-email x", "socat - TCP:h:1",
    "openssl s_client -connect h:443", "X=rm; $X f", '"$(echo rm)" x', "`echo rm` x",
    "echo > $XDG_CONFIG_HOME/prometheus/x",
])
def test_round_two_misses_are_critical(command):
    assert CriticalRules().matches("bash", {"command": command}), command


@pytest.mark.parametrize("command", ["python3 script.py", "python3 -m pytest -q", "uv run pytest -q",
                                     "cd ~/projects/app && echo hi > out.txt", "awk '{print $1}' f",
                                     "timeout 5 ls", "watch ls", "echo hello | grep h"])
def test_still_not_critical(command):
    assert CriticalRules().matches("bash", {"command": command}) == (), command


# ── review round 3 (2026-11-12) ──────────────────────────────────────────────

def test_stacked_wrappers_stay_fast():
    import time

    start = time.monotonic()
    for line in ("env " * 40 + "ls", "time nice " * 30 + "ls", "sudo " * 200 + "rm x"):
        CriticalRules().matches("bash", {"command": line})
    assert time.monotonic() - start < 1.0
    assert CriticalRules().matches("bash", {"command": "env " * 40 + "rm x"})


@pytest.mark.parametrize("command", [
    "git checkout .", "git checkout HEAD -- src/x.py", "git reset HEAD~3 --hard", "git reset -q --hard",
    "git worktree remove ../wt", "git tag -d v1", "docker rm -f x", "docker system prune -af", "npm unpublish x",
    ": > important.log", "pnpm publish", "cargo publish", "poetry publish", "gem push x.gem",
    "rclone sync . remote:b", "gsutil cp x gs://b", "kubectl delete pod x", "terraform apply -auto-approve",
    "npx wrangler deploy", "vercel --prod", "http POST api.example.invalid a=1",
    "git subtree push --prefix d origin main", "python3.12 -c 'import shutil'", "pwsh -c 'Remove-Item x'",
    "cd -- ~/.prometheus && sed -i s/a/b/ config.yaml", "chgrp staff x", "dd if=/dev/zero of=data.db",
    'shortcuts run "Send Message"',
])
def test_round_three_misses_are_critical(command):
    assert CriticalRules().matches("bash", {"command": command}), command


@pytest.mark.parametrize("command", ["git checkout main", "git reset HEAD~1", "docker ps", "kubectl get pods",
                                     "python3.12 script.py", "git tag"])
def test_round_three_ordinary_commands_stay_ordinary(command):
    assert CriticalRules().matches("bash", {"command": command}) == (), command


# ── review round 4 (2026-11-12) ──────────────────────────────────────────────

def test_hostile_shapes_stay_fast_and_a_very_long_command_is_critical():
    import time

    start = time.monotonic()
    CriticalRules().matches("bash", {"command": "find . " + "-exec find " * 40})    # short and harmless: fast
    assert CriticalRules().matches("bash", {"command": "find . " + "-exec find " * 40 + "-exec rm {} +"})
    for line in ("env " + "git " * 30000, "python -m " * 20000 + "ls", "xargs " + "git -C d " * 20000):
        assert CriticalRules().matches("bash", {"command": line}), line[:40]
    assert time.monotonic() - start < 1.0
    assert CriticalRules().matches("bash", {"command": "echo " + "a " * 3000})


@pytest.mark.parametrize("command", ["> f", "  > important.log", "docker volume prune", "docker compose down -v",
                                     "git switch --discard-changes", "git checkout -f main", "pip uninstall x",
                                     "brew uninstall x"])
def test_round_four_misses_are_critical(command):
    assert CriticalRules().matches("bash", {"command": command}), command


# ── review round 5 (2026-11-12): an overwrite anywhere ──────────────────────

@pytest.mark.parametrize("command", ["cat /dev/null > f", "echo x > ~/.zshrc", "cat f > out.txt", "ls >out",
                                     "sort a 1> b", "echo x >| f"])
def test_an_overwriting_redirect_is_critical(command):
    assert CriticalRules().matches("bash", {"command": command}), command


@pytest.mark.parametrize("command", ["echo x >> log.txt", "make 2>&1", "ls > /dev/null", "cmd 2>/dev/null",
                                     "echo x >&2", "cmd > /dev/stderr"])
def test_appends_and_discards_are_not(command):
    assert CriticalRules().matches("bash", {"command": command}) == (), command
