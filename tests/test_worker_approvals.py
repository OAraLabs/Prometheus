"""A worker's approval request reaches the conversation that spawned it, and a
PERSON answers it. Nothing approves it on their behalf.

WHAT HAPPENED
-------------
A spawned worker is a ``python -m prometheus`` child with no approval
channel. Under ``permission_mode: default`` an action that needs confirmation
— a write whose path the gate cannot resolve, a destructive command — was
refused outright ("Permission denied for write_file"), so a worker could only
do real work if the operator ran everything ``autonomous``. (Seen in the
scratch reproduction for #704.)

THE RULE THESE TESTS PIN (Will's ruling: PR-6)
----------------------------------------------
* The worker's permission prompt asks the daemon (``POST
  /api/workers/approvals``) with a per-worker token the manager mints and
  holds in memory only — it travels in the child's environment overlay, like
  the API key: never in the command, the record or the database.
* The daemon files the request in its approval queue under the worker's
  PARENT conversation, tagged with the worker, approve-ONCE only (the
  worker's gate is not the daemon's, so a lasting grant would mean nothing),
  and waits for a person.
* Nothing but a person approves it: no queue, an unreachable daemon, an
  expired request, a wrong token — each is a denial.
* The endpoint answers only a loopback peer holding a RUNNING worker's token.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from tests.fixtures.fake_llama_server import WORKER_MARK, FakeLlamaServer

_LOOPBACK = ("127.0.0.1", 40123)
_BASE = "http://127.0.0.1:8005"


class _Gate:
    """The queue requires a gate reference; nothing here records a grant."""


async def _until(predicate, seconds: float = 10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.05)
    return predicate()


async def _standin_worker(tmp_path, monkeypatch, *, session_id="beacon:abc"):
    """A RUNNING worker task whose process is a harmless sleep."""
    from prometheus.tasks import manager as mgr_mod
    from prometheus.tasks.manager import BackgroundTaskManager

    monkeypatch.setattr(mgr_mod, "prometheus_argv",
                        lambda *a: ["/bin/sh", "-c", "sleep 30", "standin", *a])
    mgr = BackgroundTaskManager()
    mgr.worker_approval_url = _BASE + "/api/workers/approvals"
    rec = await mgr.create_agent_task(
        prompt="p", description="the worker", cwd=str(tmp_path), route="local",
        worker=True, session_id=session_id, on_complete="reengage")
    token = mgr._worker_tokens[rec.id]
    return mgr, rec, token


def _app(queue=None, mgr=None, monkeypatch=None):
    from prometheus.web.server import create_app

    if mgr is not None and monkeypatch is not None:
        monkeypatch.setattr("prometheus.tasks.manager.get_task_manager", lambda: mgr)
    app = create_app({})
    app.state.approval_queue = queue
    return app


def _client(app, peer=_LOOPBACK):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=peer),
                             base_url=_BASE)


def _ask(task_id: str, token: str, **body) -> dict:
    """The worker's request: who it is in HEADERS (checked before the body is
    read), what it wants in the body."""
    return {
        "headers": {"X-Prometheus-Worker-Task": task_id, "X-Prometheus-Worker-Token": token},
        "json": {"tool_name": "write_file", "reason": "write_file needs approval: ./notes.md",
                 "arguments": {"path": "./notes.md"}, **body},
    }


@pytest.mark.asyncio
async def test_a_workers_request_lands_in_its_conversations_queue_and_waits_for_a_person(
        tmp_path, monkeypatch):
    from prometheus.permissions.approval_queue import ApprovalQueue
    from prometheus.permissions.approver import in_process

    mgr, rec, token = await _standin_worker(tmp_path, monkeypatch)
    queue = ApprovalQueue(security_gate=_Gate())
    try:
        async with _client(_app(queue, mgr, monkeypatch)) as c:
            req = asyncio.create_task(c.post("/api/workers/approvals", **_ask(rec.id, token)))
            pending = await _until(queue.list_pending)
            assert pending, "the request never reached the approval queue"
            action = pending[0]
            assert action.session_id == "beacon:abc", action.session_id
            assert action.worker_task_id == rec.id and action.once_only
            wire = queue.serialize_pending(action)
            assert wire["worker_task_id"] == rec.id and wire["session_id"] == "beacon:abc"
            assert wire["extents"] == {}, "a worker request must offer no lasting grant"
            await asyncio.sleep(0.2)
            assert not req.done(), "the worker was answered before anyone approved"
            await queue.approve(action.request_id, by=in_process("a person"))
            resp = await req
        assert resp.status_code == 200 and resp.json()["result"] == "approved", resp.text
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_nothing_but_a_person_approves_it(tmp_path, monkeypatch):
    mgr, rec, token = await _standin_worker(tmp_path, monkeypatch)
    try:
        # No approval queue: denied, never waved through.
        async with _client(_app(None, mgr, monkeypatch)) as c:
            r = await c.post("/api/workers/approvals", **_ask(rec.id, token))
        assert r.status_code == 200 and r.json()["result"] == "denied", r.text
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_only_a_loopback_peer_with_the_running_workers_token_is_answered(
        tmp_path, monkeypatch):
    from prometheus.permissions.approval_queue import ApprovalQueue

    mgr, rec, token = await _standin_worker(tmp_path, monkeypatch)
    queue = ApprovalQueue(security_gate=_Gate())
    app = _app(queue, mgr, monkeypatch)
    try:
        async with _client(app) as c:
            wrong = await c.post("/api/workers/approvals", **_ask(rec.id, "not" + "it"))
            unknown = await c.post("/api/workers/approvals", **_ask("a00000000", token))
        async with _client(app, peer=("192.168.1.50", 40000)) as c:
            remote = await c.post("/api/workers/approvals", **_ask(rec.id, token))
        await mgr.stop_task(rec.id)
        async with _client(app) as c:
            stopped = await c.post("/api/workers/approvals", **_ask(rec.id, token))
    finally:
        await mgr.shutdown()
    assert wrong.status_code == 401 and unknown.status_code == 401, (wrong.text, unknown.text)
    assert remote.status_code == 403, remote.text
    assert stopped.status_code == 401, stopped.text
    assert not queue.list_pending()


@pytest.mark.asyncio
async def test_a_worker_request_can_only_be_approved_once(tmp_path, monkeypatch):
    from prometheus.permissions.approval_queue import ApprovalQueue

    mgr, rec, token = await _standin_worker(tmp_path, monkeypatch)
    queue = ApprovalQueue(security_gate=_Gate())
    try:
        async with _client(_app(queue, mgr, monkeypatch)) as c:
            req = asyncio.create_task(c.post("/api/workers/approvals", **_ask(rec.id, token)))
            pending = await _until(queue.list_pending)
            r = await c.post(f"/api/approvals/{pending[0].request_id}/approve",
                             json={"scope": "always"})
            assert r.status_code == 409, r.text
            await c.post(f"/api/approvals/{pending[0].request_id}/deny")
            resp = await req
        assert resp.json()["result"] == "denied"
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_the_worker_token_is_never_written_down(tmp_path, monkeypatch):
    from prometheus.tasks.store import TaskStore

    mgr, rec, token = await _standin_worker(tmp_path, monkeypatch)
    try:
        mgr.store = TaskStore(tmp_path / "tasks.db")
        mgr._persist(mgr.get_task(rec.id))
        assert token not in (rec.command or "")
        assert token not in json.dumps(rec.metadata)
        assert token not in json.dumps(mgr._task_env.get(rec.id) or {})
        dump = "\n".join(str(row) for row in sqlite3.connect(tmp_path / "tasks.db").execute(
            "select * from tasks"))
        assert token not in dump
    finally:
        await mgr.shutdown()


# ---------------------------------------------------------------------------
# The worker's side: its permission prompt asks the daemon
# ---------------------------------------------------------------------------

class _Standin:
    """A stand-in for the daemon's endpoint: records each request, answers."""

    def __init__(self, answer: str = "approved") -> None:
        self.answer = answer
        self.seen: list[dict] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                body["_task"] = self.headers.get("X-Prometheus-Worker-Task")
                body["_token"] = self.headers.get("X-Prometheus-Worker-Token")
                outer.seen.append(body)
                data = json.dumps({"result": outer.answer}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/api/workers/approvals"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.mark.asyncio
async def test_the_workers_prompt_asks_the_daemon_and_denies_on_any_failure():
    from prometheus.cli import worker_approval as wa

    tok = "fake" + "worker" + "token"
    env = {"PROMETHEUS_WORKER_APPROVAL_URL": "http://127.0.0.1:9/x", "PROMETHEUS_WORKER_TASK_ID": "a1"}
    assert wa.approval_prompt_from_env({}) is None
    yes, no = _Standin("approved"), _Standin("denied")
    try:
        wa.set_worker_token(tok)
        ask_yes = wa.approval_prompt_from_env({**env, "PROMETHEUS_WORKER_APPROVAL_URL": yes.url})
        assert await ask_yes("write_file", "needs approval", arguments={"path": "x"}) is True
        assert yes.seen[0]["_task"] == "a1" and yes.seen[0]["_token"] == tok
        assert "token" not in yes.seen[0], "the token belongs in a header, not the body"
        ask_no = wa.approval_prompt_from_env({**env, "PROMETHEUS_WORKER_APPROVAL_URL": no.url})
        assert await ask_no("write_file", "needs approval") is False
    finally:
        yes.close()
        no.close()
    assert await wa.approval_prompt_from_env(env)("write_file", "needs approval") is False
    wa.set_worker_token(None)
    # No token at all: nobody to ask as, so denied.
    assert await wa.approval_prompt_from_env({**env})("write_file", "x") is False


@pytest.fixture
def child_env(tmp_path, monkeypatch):
    site_dir = tmp_path / "child-site"
    site_dir.mkdir()
    (site_dir / "sitecustomize.py").write_text(
        "import prometheus.config.defaults as _defaults\n"
        f"_defaults.REPO_CONFIG_PATH = _defaults.Path({str(tmp_path / 'no-repo' / 'prometheus.yaml')!r})\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(p for p in (str(site_dir), os.environ.get("PYTHONPATH", "")) if p),
    )
    for var in ("ANTHROPIC_API_KEY", "QWEN_API_KEY", "QWEN_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    work = tmp_path / "work"
    work.mkdir()
    return work


@pytest.mark.asyncio
async def test_a_worker_under_the_default_mode_gets_its_write_approved_by_the_parent(child_env):
    """End to end: a worker in permission_mode default writes a path the gate
    cannot resolve; the request reaches the (stand-in) daemon, which approves;
    the file is written."""
    from prometheus.tasks.manager import get_task_manager
    from prometheus.tools.base import ToolExecutionContext
    from prometheus.tools.builtin.task_create import TaskCreateTool, TaskCreateToolInput

    work = child_env
    daemon = _Standin("approved")
    try:
        # work_dir "." makes the fake model write "./step_0.py": a relative path.
        with FakeLlamaServer(work_dir=".", writes_per_turn=1) as box:
            cfg_dir = Path(os.environ["PROMETHEUS_CONFIG_DIR"])
            cfg_dir.mkdir(parents=True, exist_ok=True)
            (cfg_dir / "prometheus.yaml").write_text(
                f"model:\n  provider: llama_cpp\n  base_url: {box.url}\n  model: fake-local-27b\n"
                f"security:\n  permission_mode: default\n  workspace_root: {work}\n",
                encoding="utf-8")
            mgr = get_task_manager()
            mgr.worker_approval_url = daemon.url
            result = await TaskCreateTool().execute(
                TaskCreateToolInput(type="local_agent", description="w",
                                    prompt=f"{WORKER_MARK} W.", timeout_seconds=60),
                ToolExecutionContext(cwd=work, metadata={
                    "session_id": "web", "effective_session_id": "beacon:abc",
                    "ephemeral": False,
                    "serving_route": {"key": "local", "provider": "llama_cpp",
                                      "model": "fake-local-27b", "backend": None}}))
            task_id = result.metadata["task_id"]
            deadline = time.monotonic() + 30
            while mgr.get_task(task_id).status == "running" and time.monotonic() < deadline:
                await asyncio.sleep(0.25)
            if mgr.get_task(task_id).status == "running":
                await mgr.stop_task(task_id)
    finally:
        daemon.close()
    assert daemon.seen, "the worker never asked anyone"
    assert daemon.seen[0]["_task"] == task_id and daemon.seen[0]["tool_name"] == "write_file"
    assert daemon.seen[0]["_token"], "the worker asked without its token"
    assert (work / "step_0.py").exists(), "the approved write did not happen"
