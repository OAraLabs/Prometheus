"""``GET /api/sessions/{id}/git``, the ``git/1`` capability and the ``workspace_git_changed`` frame.

Real git on real repositories, the real FastAPI app, and a stand-in for the bridge's sockets.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.security import git_pr, git_read  # noqa: E402
from prometheus.web import git_routes  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402
from prometheus.web.ws_server import WebSocketBridge  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("git") is None, reason="needs git and a POSIX host")

GLOBAL = "git-route-global-" + secrets.token_hex(8)
SID = "beacon:git-route"


def _g(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=Smoke", "-c", "user.email=smoke@example.invalid", "-c", "init.defaultBranch=main",
         "-c", "commit.gpgsign=false", *args], cwd=cwd, capture_output=True, text=True, timeout=30, check=True,
        env={**os.environ, "LANG": "C", "LC_ALL": "C"},
    ).stdout


class _Store:
    """The conversation store's one method the route uses."""

    def __init__(self) -> None:
        self.bound: dict[str, str] = {}

    def get_session_workspace(self, sid: str):
        return self.bound.get(sid)

    def set_session_workspace(self, sid: str, path: str, set_by: str = "") -> None:
        if path:
            self.bound[sid] = path
        else:
            self.bound.pop(sid, None)


@pytest.fixture
def world(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "config"))
    for k in list(os.environ):
        if k.startswith("GIT_") and k != "GIT_CONFIG_NOSYSTEM":
            monkeypatch.delenv(k, raising=False)
    git_read.forget_reads()
    git_read.forget_probe()
    if not asyncio.run(git_read.fork_limit_enforced()):
        pytest.skip("the process limit is not enforced for this user")
    git_pr.reset(())
    repo = tmp_path / "work" / "site"
    repo.mkdir(parents=True)
    _g(repo, "init", "-q", ".")
    (repo / "a.txt").write_text("one\n")
    _g(repo, "add", "-A")
    _g(repo, "commit", "-q", "-m", "first")
    store = _Store()
    store.bound[SID] = str(repo)
    devices = DeviceStore(tmp_path / "devices.db")
    app = create_app({"web": {"api_token": GLOBAL}}, device_store=devices,
                     lcm_engine=SimpleNamespace(conversation_store=store))
    client = TestClient(app, client=("127.0.0.1", 50123))
    scoped = devices.mint("a scoped phone", "ios")
    yield SimpleNamespace(client=client, app=app, repo=repo, store=store, devices=devices,
                          owner={"Authorization": f"Bearer {GLOBAL}"},
                          scoped={"Authorization": f"Bearer {scoped['token']}"})
    git_pr.reset(("/opt/homebrew/bin/gh", "/usr/local/bin/gh", "/usr/bin/gh", "/home/linuxbrew/.linuxbrew/bin/gh"))
    git_read.forget_reads()


def get(w, sid: str = SID, headers=None):
    git_read.forget_reads()
    return w.client.get(f"/api/sessions/{sid}/git", headers=headers or w.owner)


class TestTheRoute:
    def test_a_repository_reads_as_the_contract_shape(self, world):
        (world.repo / "new.txt").write_text("x\n")
        r = get(world)
        assert r.status_code == 200
        body = r.json()
        assert body["v"] == 1 and body["kind"] == "state" and body["top"] == str(world.repo.resolve())
        assert body["state"]["branch"] == "main" and body["state"]["untracked"] == 1 and body["state"]["changed"] == 1
        assert body["truncated"] is False and body["pr"] is None and isinstance(body["read_at"], float)

    def test_a_session_with_no_work_folder_is_none(self, world):
        body = get(world, "beacon:unbound").json()
        assert body["kind"] == "none" and "state" not in body and "top" not in body and "pr" not in body

    def test_a_folder_that_is_not_in_a_repository_is_none(self, world, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        world.store.bound[SID] = str(plain)
        assert get(world).json()["kind"] == "none"

    def test_where_the_limit_does_not_hold_it_is_unreadable_unsafe_here_with_nothing_else(self, world, monkeypatch):
        monkeypatch.setattr(git_read, "_enforced", False)
        body = get(world).json()
        assert body == {"v": 1, "kind": "unreadable", "why": "unsafe_here", "read_at": body["read_at"]}

    def test_a_scoped_device_gets_403_operator_only(self, world):
        r = world.client.get(f"/api/sessions/{SID}/git", headers=world.scoped)
        assert r.status_code == 403 and r.json()["error"] == "operator_only"

    def test_no_token_is_refused(self, world):
        assert world.client.get(f"/api/sessions/{SID}/git").status_code in (401, 403)

    def test_a_top_the_daemon_would_not_accept_as_a_workspace_is_not_read(self, world):
        world.app.state.git_watch._config = {"security": {"denied_paths": [str(world.repo.parent)]}}
        assert get(world).json()["kind"] == "none"

    def test_the_answer_holds_no_file_name_message_or_remote(self, world):
        (world.repo / "SECRET-FILE-NAME.txt").write_text("x\n")
        _g(world.repo, "commit", "-q", "--allow-empty", "-m", "SECRET-COMMIT-MESSAGE")
        _g(world.repo, "remote", "add", "origin", "https://github.com/acme/SECRET-REMOTE.git")
        text = get(world).text
        assert not any(s in text for s in ("SECRET-FILE-NAME", "SECRET-COMMIT-MESSAGE", "SECRET-REMOTE"))

    def test_etag_and_304(self, world):
        first = get(world)
        tag = first.headers["etag"]
        assert tag.startswith('W/"')
        again = world.client.get(f"/api/sessions/{SID}/git", headers={**world.owner, "If-None-Match": tag})
        assert again.status_code == 304 and again.content == b"" and again.headers["etag"] == tag
        (world.repo / "new.txt").write_text("x\n")
        git_read.forget_reads()
        changed = world.client.get(f"/api/sessions/{SID}/git", headers={**world.owner, "If-None-Match": tag})
        assert changed.status_code == 200 and changed.headers["etag"] != tag

    def test_the_validator_ignores_the_time_it_was_read(self):
        a = {"v": 1, "kind": "none", "read_at": 1.0}
        b = {"v": 1, "kind": "none", "read_at": 2.0}
        assert git_routes.etag_of(a) == git_routes.etag_of(b)
        assert git_routes.etag_of(a) != git_routes.etag_of({"v": 1, "kind": "unreadable", "why": "x", "read_at": 1.0})
        assert git_routes.not_modified("*", git_routes.etag_of(a)) and git_routes.not_modified(
            f'"zzz", {git_routes.etag_of(a)}', git_routes.etag_of(a)) and not git_routes.not_modified(None, "W/\"x\"")

    def test_a_hostile_repository_answers_and_runs_nothing(self, world, tmp_path):
        mark = tmp_path / "mark"
        prog = tmp_path / "prog.sh"
        prog.write_text(f'#!/bin/sh\necho ran >> "{mark}"\nexit 0\n')
        prog.chmod(0o755)
        _g(world.repo, "config", "core.fsmonitor", str(prog))
        (world.repo / ".gitattributes").write_text("* filter=evil\n")
        _g(world.repo, "config", "filter.evil.clean", str(prog))
        (world.repo / "a.txt").write_text("ONE\n")
        t = time.time() + 5000
        os.utime(world.repo / "a.txt", (t, t))
        t0 = time.monotonic()
        body = get(world).json()
        assert body["kind"] == "state" and not mark.exists() and time.monotonic() - t0 < 9


class TestTheCapability:
    def test_advertised_only_after_the_self_test_proved_the_floor_and_git_was_found(self, monkeypatch):
        bridge = WebSocketBridge(loop_context=object(), config={})
        monkeypatch.setattr(git_read, "_enforced", None)
        monkeypatch.setattr(git_read, "_git_checked", False)
        assert "git/1" not in bridge._server_capabilities()
        monkeypatch.setattr(git_read, "_git_checked", True)
        monkeypatch.setattr(git_read, "_git_program", "/usr/bin/git")
        monkeypatch.setattr(git_read, "_enforced", True)
        assert "git/1" in bridge._server_capabilities()
        monkeypatch.setattr(git_read, "_enforced", False)       # root, or a host that does not count processes
        assert "git/1" not in bridge._server_capabilities()
        monkeypatch.setattr(git_read, "_enforced", True)
        monkeypatch.setattr(git_read, "_git_program", None)     # no git here
        assert "git/1" not in bridge._server_capabilities()

    async def test_the_startup_self_test_turns_it_on_here_and_off_where_a_child_can_still_start(self, monkeypatch):
        git_read.forget_probe()
        monkeypatch.setattr(git_read, "_git_checked", False)
        assert await git_read.prepare() is True and git_read.capability_ready()
        real_run = git_read.run

        async def fake(argv, cwd, env, **kw):
            if argv[:2] == git_read._PROBE[:2]:
                return git_read.Ran(code=0, out=b"started:0\n")
            return await real_run(argv, cwd, env, **kw)

        monkeypatch.setattr(git_read, "run", fake)
        git_read.forget_probe()
        assert await git_read.prepare() is False and not git_read.capability_ready()


class _Bridge:
    """Stand-in for the bridge's sockets: who is watching, and what was sent."""

    def __init__(self, watching: bool = True) -> None:
        self.watching = watching
        self.frames: list[tuple[str, dict]] = []
        self.frame_observers: list = []
        self.broadcasts: list = []

    def operator_sockets_watching(self, sid: str):
        return ["ws"] if self.watching else []

    async def send_to_operators_watching(self, event, sid):
        if not self.watching:
            return 0
        self.frames.append((sid, event))
        return 1

    async def broadcast(self, event):                       # must never be used for this frame
        self.broadcasts.append(event)


@pytest.fixture
def watch(world, monkeypatch):
    monkeypatch.setattr(git_routes, "SETTLE_S", 0.01)
    monkeypatch.setattr(git_routes, "COALESCE_S", 0.4)
    w = world.app.state.git_watch
    w.bridge = _Bridge()
    return w


class TestTheFrame:
    async def test_it_goes_out_on_a_real_change_carries_no_state_and_not_through_the_broadcast(self, world, watch):
        await watch.check(SID)                               # first reading: nothing known yet → told
        watch.bridge.frames.clear()
        assert await watch.check(SID) is False and watch.bridge.frames == []     # same reading: nothing to say
        (world.repo / "SECRET-FILE-NAME.txt").write_text("x\n")
        assert await watch.check(SID) is True
        (sid, frame), = watch.bridge.frames
        assert sid == SID and frame["type"] == "workspace_git_changed"
        assert set(frame["payload"]) == {"session_id", "read_at"} and frame["payload"]["session_id"] == SID
        assert "SECRET-FILE-NAME" not in json.dumps(frame) and "state" not in json.dumps(frame["payload"])
        assert watch.bridge.broadcasts == []                 # sent to the sockets directly: never the durable feed

    async def test_what_a_client_fetched_is_what_it_knows(self, world, watch):
        body = await watch.snapshot(SID)
        watch.note_seen(SID, body)
        assert await watch.check(SID) is False               # it already has this reading
        (world.repo / "x.txt").write_text("x\n")
        assert await watch.check(SID) is True

    async def test_nothing_is_sent_when_no_owner_device_is_watching(self, world, watch):
        watch.bridge.watching = False
        watch.touch(SID)
        assert SID not in watch._pending
        (world.repo / "x.txt").write_text("x\n")
        assert await watch.check(SID) is False and watch.bridge.frames == []

    async def test_a_turn_ending_a_tool_call_and_a_binding_change_re_read_and_a_burst_is_one_frame_per_window(self, world, watch):
        watch.note_seen(SID, await watch.snapshot(SID))
        (world.repo / "x.txt").write_text("x\n")
        for kind in ("chat_done", "tool_call_end", "tool_call_end"):
            watch.observe({"type": kind, "payload": {"session_id": SID}})
        watch.touch(SID)
        await asyncio.sleep(0.2)
        assert len(watch.bridge.frames) == 1
        (world.repo / "y.txt").write_text("y\n")
        watch.observe({"type": "tool_call_end", "payload": {"session_id": SID}})
        await asyncio.sleep(0.15)
        assert len(watch.bridge.frames) == 1                 # inside the window: held back
        await asyncio.sleep(0.6)
        assert len(watch.bridge.frames) == 2                 # then said once, with the latest reading

    async def test_a_frame_that_is_not_a_trigger_or_names_no_session_does_nothing(self, world, watch):
        watch.observe({"type": "chat_delta", "payload": {"session_id": SID}})
        watch.observe({"type": "chat_done", "payload": {}})
        watch.observe({"type": "chat_done"})
        await asyncio.sleep(0.1)
        assert not watch._pending

    async def test_the_slow_timer_reads_only_sessions_that_are_active_and_watched(self, world, watch):
        watch.note_seen(SID, await watch.snapshot(SID))
        (world.repo / "x.txt").write_text("x\n")
        await watch._tick()
        assert len(watch.bridge.frames) == 1
        watch.bridge.frames.clear()
        (world.repo / "y.txt").write_text("y\n")
        watch._active[SID] = time.monotonic() - git_routes.ACTIVE_S - 1       # not active lately
        await watch._tick()
        assert watch.bridge.frames == []
        watch._active[SID] = time.monotonic()
        watch.bridge.watching = False                                       # no owner device watching
        await watch._tick()
        assert watch.bridge.frames == []

    def test_changing_a_conversation_s_workspace_over_rest_touches_the_watcher(self, world, watch):
        touched: list[str] = []
        watch.touch = touched.append  # type: ignore[method-assign]
        other = world.repo.parent / "other"
        other.mkdir()
        assert world.client.put(f"/api/sessions/{SID}/workspace", headers=world.owner, json={"path": str(other)}).status_code == 200
        assert world.client.delete(f"/api/sessions/{SID}/workspace", headers=world.owner).status_code == 200
        assert touched == [SID, SID]

    async def test_a_pull_request_that_arrives_after_the_route_stopped_waiting_is_a_change(self, world, watch, tmp_path):
        _g(world.repo, "remote", "add", "origin", "https://github.com/acme/site.git")
        _g(world.repo, "config", "branch.main.remote", "origin")
        _g(world.repo, "config", "branch.main.merge", "refs/heads/main")
        answer = tmp_path / "answer.json"
        answer.write_text(json.dumps([{"number": 12, "state": "OPEN", "isDraft": False, "mergeable": "MERGEABLE",
                                       "headRefName": "main", "isCrossRepository": False, "statusCheckRollup": []}]))
        gh = tmp_path / "gh"
        gh.write_text(f"#!/bin/sh\nsleep 1\ncat '{answer}'\n")
        gh.chmod(0o755)
        git_pr.reset((str(gh),))
        quick = await watch.snapshot(SID, fresh=True, pr_wait=0.05)          # gh is slower than the wait
        assert quick["pr"] is None
        watch.note_seen(SID, quick)
        await asyncio.sleep(2.5)                                             # the lookup lands in the background
        assert watch.bridge.frames, "a pull request that turned up was not announced"
        again = await watch.snapshot(SID, fresh=True)
        assert again["pr"]["number"] == 12 and again["pr"]["url"] == "https://github.com/acme/site/pull/12"


class TestTheBridge:
    def _bridge(self):
        b = WebSocketBridge(loop_context=object(), config={})
        sent: dict[str, list] = {}

        def sock(name: str, op: bool, subs):
            class WS:
                async def send(self, raw):
                    sent.setdefault(name, []).append(json.loads(raw))
            ws = WS()
            b._clients.add(ws)
            b._ws_identity[ws] = SimpleNamespace(is_operator=op)
            if subs is not None:
                b._ws_filters[ws] = subs
            return ws

        return b, sent, sock

    async def test_only_owner_sockets_that_watch_the_session_are_sent_the_frame(self):
        b, sent, sock = self._bridge()
        sock("firehose", True, None)
        sock("named", True, {SID})
        sock("elsewhere", True, {"beacon:other"})
        sock("scoped", False, None)
        n = await b.send_to_operators_watching({"type": "workspace_git_changed", "payload": {"session_id": SID}}, SID)
        assert n == 2 and set(sent) == {"firehose", "named"}
        assert len(b.operator_sockets_watching(SID)) == 2 and len(b.operator_sockets_watching("beacon:other")) == 2

    async def test_a_socket_that_fails_is_dropped_not_raised(self):
        b, sent, sock = self._bridge()

        class Dead:
            async def send(self, raw):
                raise ConnectionError("gone")

        dead = Dead()
        b._clients.add(dead)
        b._ws_identity[dead] = SimpleNamespace(is_operator=True)
        assert await b.send_to_operators_watching({"type": "x", "payload": {}}, SID) == 0
        assert dead not in b._clients
