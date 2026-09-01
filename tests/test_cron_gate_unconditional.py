"""THE INVARIANT: cron command vetting is wired whenever a SecurityGate exists,
independent of which chat surfaces are enabled.

WHAT THIS PINS
--------------
``set_cron_security_gate(security_gate)`` used to sit inside
``if not args.telegram_only:`` in daemon.py. That made cron's security posture
a function of an unrelated flag. ``--telegram-only`` skips the scheduler loop
but does NOT stop cron jobs running: the web bridge starts on its own
``web.enabled`` check at the same indentation, and ``POST /api/cron/{name}/run``
calls ``execute_job`` directly.

With the gate unwired, ``_get_security_gate()`` lazily built
``SecurityGate.from_config()`` — no argument, so DEFAULTS_PATH, which resolves
to a file that exists on no checkout, so ``sec = {}``. The hardcoded
denied_paths floor survived, but the ten CONFIG-supplied denied_commands did
not.

That fails OPEN. Measured on the live config: ``cat /etc/shadow`` went
DENY -> ALLOW.

WHY THE ASSERTIONS ARE ON CONTENTS, NOT EXISTENCE
-------------------------------------------------
A gate built from ``{}`` is still a gate. Any test asking "is a gate wired?"
passes on the broken path — which is exactly how this survived. So the
behavioural tests below assert a command that ONLY the configured denials
block, and they assert it in both directions: with the real gate it is
refused, and with the fallback gate it is not. If the negative case ever
starts passing too, the test has stopped discriminating and says so.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from prometheus.gateway import cron_scheduler
from prometheus.permissions.checker import SecurityGate

# Straight from the shipped template's security.denied_commands. Deliberately
# NOT one the hardcoded floor also covers: `rm -rf /` is refused either way and
# would make this test pass against the broken wiring.
CONFIG_ONLY_DENIAL = "cat /etc/shadow"

CONFIGURED_DENIALS = [
    "rm -rf /", "rm -rf ~", "DROP TABLE", "mkfs",
    "cat /etc/passwd", "cat /etc/shadow", "cat /etc/sudoers",
    "/etc/passwd", "/etc/shadow", "/etc/sudoers",
]


@pytest.fixture(autouse=True)
def _reset_gate():
    cron_scheduler.set_cron_security_gate(None)
    yield
    cron_scheduler.set_cron_security_gate(None)


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path))
    return tmp_path


class _FakeProc:
    returncode = 0

    async def communicate(self):
        return b"", b""


@pytest.fixture()
def spawns(monkeypatch):
    """Record what reaches the subprocess boundary; never spawn for real.

    The fake COMPLETES rather than raising: execute_job wraps the spawn in its
    own try/except, so an exception here would be swallowed into an "error"
    history entry and the test would read that as "it was stopped". Reaching
    this list is the signal that vetting let the command through.
    """
    spawned: list[tuple] = []

    async def _fake(*a, **k):
        spawned.append(a)
        return _FakeProc()

    monkeypatch.setattr("asyncio.create_subprocess_exec", _fake)
    return spawned


def _configured_gate() -> SecurityGate:
    """A gate carrying the config's denied_commands — what the daemon builds."""
    return SecurityGate(denied_commands=list(CONFIGURED_DENIALS))


# ── the structural pin: the wiring is not inside the telegram_only branch ──


def _set_gate_call_ancestors() -> list[list[ast.AST]]:
    """Ancestor chains for every ``set_cron_security_gate(...)`` call in daemon.py."""
    from prometheus import daemon

    tree = ast.parse(Path(inspect.getfile(daemon)).read_text(encoding="utf-8"))
    chains: list[list[ast.AST]] = []

    def walk(node: ast.AST, stack: list[ast.AST]) -> None:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "set_cron_security_gate"
        ):
            chains.append(list(stack))
        for child in ast.iter_child_nodes(node):
            walk(child, stack + [node])

    walk(tree, [])
    return chains


def test_daemon_wires_the_cron_gate_outside_any_surface_conditional():
    """The regression pin for the code move itself.

    A behavioural test alone would not catch someone re-nesting this call
    under a different surface flag later, because the unit tests never boot
    the daemon.
    """
    chains = _set_gate_call_ancestors()
    assert chains, "daemon.py no longer wires the cron SecurityGate at all"

    for chain in chains:
        for ancestor in chain:
            if isinstance(ancestor, ast.If):
                test_src = ast.unparse(ancestor.test)
                assert "telegram_only" not in test_src, (
                    "set_cron_security_gate is nested under "
                    f"`if {test_src}` — cron vetting must not depend on which "
                    "chat surface is enabled. --telegram-only still serves "
                    "POST /api/cron/{name}/run through the web bridge."
                )


def test_the_structural_detector_would_catch_the_original():
    """Replay: the pre-fix shape must be flagged, or the check above is blind."""
    original = (
        "def run_daemon(args):\n"
        "    if not args.telegram_only:\n"
        "        set_cron_notifier(telegram, _notify_chat)\n"
        "        set_cron_security_gate(security_gate)\n"
    )
    tree = ast.parse(original)
    found = []

    def walk(node, stack):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "set_cron_security_gate"):
            found.append([a for a in stack if isinstance(a, ast.If)])
        for child in ast.iter_child_nodes(node):
            walk(child, stack + [node])

    walk(tree, [])
    assert found and found[0], "replay produced no enclosing If"
    assert "telegram_only" in ast.unparse(found[0][0].test)


# ── the behavioural pin: CONTENTS, both directions ────────────────────────


async def test_wired_gate_refuses_a_config_only_denial(isolated, spawns):
    """With the daemon's real gate wired, the configured denial is enforced."""
    cron_scheduler.set_cron_security_gate(_configured_gate())

    entry = await cron_scheduler.execute_job(
        {"name": "leak", "command": CONFIG_ONLY_DENIAL, "cwd": str(isolated)}
    )
    assert entry["status"] == "blocked"
    assert entry["returncode"] == 126
    assert "SecurityGate" in entry["stderr"]
    assert spawns == [], "a blocked command must never reach the spawn"


async def test_unwired_gate_does_not_refuse_it(isolated, spawns):
    """The discrimination check — and the fail-open itself, stated as a test.

    With no gate wired, cron falls back to SecurityGate.from_config(), whose
    default path loads nothing, so the configured denials are absent and the
    command reaches the spawn boundary. THIS IS THE BUG, pinned: it is here so
    that the positive test above is known to be measuring the configured
    denials rather than the hardcoded floor.

    If this ever starts returning 'blocked', either the fallback learned to
    read a config (then delete this test along with the fallback) or the floor
    grew to cover it (then CONFIG_ONLY_DENIAL must move to a command the floor
    still misses) — either way the positive test has stopped discriminating
    and this failure is the notice.
    """
    assert cron_scheduler._get_security_gate() is not None, (
        "cron must never be ungated — the fallback exists so that a missing "
        "wire cannot mean 'no vetting at all'"
    )
    entry = await cron_scheduler.execute_job(
        {"name": "leak", "command": CONFIG_ONLY_DENIAL, "cwd": str(isolated)}
    )
    assert entry["status"] != "blocked"
    assert len(spawns) == 1, (
        "expected the unwired fallback to let a config-only denial through to "
        f"the spawn; got status={entry['status']!r}, spawns={spawns!r}"
    )


async def test_floor_holds_in_both_configurations(isolated, spawns):
    """Control: the hardcoded floor is refused with or without a wired gate.

    Without this, a uniform-DENY regression would make the positive test above
    look like it was measuring the configured denials when it was not.
    """
    for label, gate in (("wired", _configured_gate()), ("fallback", None)):
        cron_scheduler.set_cron_security_gate(gate)
        entry = await cron_scheduler.execute_job(
            {"name": f"floor-{label}", "command": "rm -rf /", "cwd": str(isolated)}
        )
        assert entry["status"] == "blocked", f"floor breached in the {label} gate"
    assert spawns == [], "the floor must stop these before any spawn"
