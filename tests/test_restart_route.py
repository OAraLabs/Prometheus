"""``POST /api/restart``: the operator applies a saved network mode without touching the Mac.

Beacon's v0.4 spec (§2.4): ``PUT /api/network`` only saves; ``POST /api/restart`` (operator only) answers
``202 {"restarting": true}``, every socket drops, and the daemon comes back; Beacon then waits for ``GET /api/hello``
to answer with the SAME ``fp``. How it comes back: after the answer is sent the daemon runs its ordinary graceful
shutdown and then RE-EXECS ITS OWN COMMAND LINE (``sys.orig_argv``), in the same process. That works under launchd
(the app), systemd and a terminal alike, and needs no supervisor to notice an exit. The instance key is a file, so
the ``fp`` is the same after it.

Only the operator (the global token or an owner device): a scoped device is 403 ``operator_only`` at the gate, like
every route it is not allowed. A daemon that cannot restart itself (an app built without the hook) says so,
``503 restart_unavailable``, rather than pretending.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from tests.support.device_world import World  # noqa: E402


@pytest.fixture
def w(tmp_path):
    world = World(tmp_path)
    world.restarts = []
    world.app.state.request_restart = lambda: world.restarts.append("restart")
    return world


def test_the_operator_restarts_it_and_is_told_so_before_it_goes(w):
    response = w.call("op", "POST", "/api/restart")
    assert response.status_code == 202, response.text
    assert response.json() == {"restarting": True}
    assert w.restarts == ["restart"], "the restart is requested AFTER the answer, exactly once"


def test_a_scoped_device_may_not(w):
    response = w.call("a", "POST", "/api/restart")
    assert response.status_code == 403 and response.json()["error"] == "operator_only"
    assert w.restarts == []


def test_nobody_without_a_token_may(w):
    assert w.client.post("/api/restart").status_code == 401
    assert w.restarts == []


def test_a_daemon_that_cannot_restart_itself_says_so(w):
    del w.app.state.request_restart
    response = w.call("op", "POST", "/api/restart")
    assert response.status_code == 503 and response.json()["error"] == "restart_unavailable"


def test_the_route_is_the_operators_in_the_route_list():
    from prometheus.web import route_access

    assert ("POST", "/api/restart") in route_access.OPERATOR_ONLY


# ── coming back: re-exec of the same command line ────────────────────────────

def test_the_restart_command_line_is_the_one_this_process_was_started_with(monkeypatch):
    import sys

    from prometheus import daemon

    monkeypatch.setattr(sys, "orig_argv", ["/x/python3.12", "-I", "-B", "-m", "prometheus", "daemon"])
    assert daemon.restart_argv() == [sys.executable, "-I", "-B", "-m", "prometheus", "daemon"]


def test_main_re_execs_when_the_daemon_asked_to_restart(monkeypatch):
    """main() runs the daemon; when run_daemon reports a requested restart it re-execs, after the lock is gone."""
    import ast
    from pathlib import Path

    from prometheus import daemon

    tree = ast.parse(Path(daemon.__file__).read_text(encoding="utf-8"))
    main = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")
    called = {(n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None))
              for n in ast.walk(main) if isinstance(n, ast.Call)}
    assert "reexec" in called, "main() never re-execs after a requested restart"

    seen = []
    monkeypatch.setattr(daemon.os, "execv", lambda exe, argv: seen.append((exe, argv)))
    monkeypatch.setattr(daemon, "release_daemon_lock", lambda: seen.append("lock released"))
    daemon.reexec()
    assert seen[0] == "lock released" and seen[1][1] == daemon.restart_argv()
