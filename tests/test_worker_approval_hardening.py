"""Worker approvals, hardened (independent security review of #720).

(a) The token is never in a worker's ENVIRONMENT. Popping it from
    ``os.environ`` is not enough on its own: Linux's ``/proc/<pid>/environ``
    (what ``ps eww`` shows, and what the model's shell can read through
    ``/proc/$PPID``) is the block the process STARTED with. So the manager
    hands the token over on stdin, ahead of the brief, and the worker still
    pops any copy it finds in its environment.
(b) The approval card's text is the model's: tool name, reason and task
    description are capped and stripped of control characters and newlines
    before Telegram or Beacon sees them.
(c) A worker may hold at most 3 pending requests and is rate-limited; the
    body is capped even when it arrives chunked (no Content-Length); the
    loopback and token checks run before the body is read.
(d) The chat path refuses a lasting scope on an approve-once request by name,
    and ``/approve all`` leaves worker requests alone, like desktop ones.
(e) A restarted worker gets a fresh token; one that blocks or fails at launch
    leaves none behind.
(f) An IPv6 host is bracketed in the endpoint URL.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from tests.fixtures.fake_llama_server import WORKER_MARK, FakeLlamaServer

_LOOPBACK = ("127.0.0.1", 40123)
_BASE = "http://127.0.0.1:8005"


class _Gate:
    pass


async def _until(predicate, seconds: float = 10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.05)
    return predicate()


async def _standin_worker(tmp_path, monkeypatch, *, description="the worker"):
    from prometheus.tasks import manager as mgr_mod
    from prometheus.tasks.manager import BackgroundTaskManager

    monkeypatch.setattr(mgr_mod, "prometheus_argv",
                        lambda *a: ["/bin/sh", "-c", "sleep 30", "standin", *a])
    mgr = BackgroundTaskManager()
    mgr.worker_approval_url = _BASE + "/api/workers/approvals"
    rec = await mgr.create_agent_task(
        prompt="p", description=description, cwd=str(tmp_path), route="local",
        worker=True, session_id="beacon:abc", on_complete="reengage")
    return mgr, rec, mgr._worker_tokens[rec.id]


def _app(queue, mgr, monkeypatch):
    from prometheus.web.server import create_app

    monkeypatch.setattr("prometheus.tasks.manager.get_task_manager", lambda: mgr)
    app = create_app({})
    app.state.approval_queue = queue
    return app


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=_LOOPBACK),
                             base_url=_BASE)


def _headers(task_id: str, token: str) -> dict:
    return {"X-Prometheus-Worker-Task": task_id, "X-Prometheus-Worker-Token": token}


_BODY = {"tool_name": "write_file", "reason": "needs approval", "arguments": {"path": "./x"}}


# --- (a) the token is not in any worker process's environment --------------

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


def _group_environs(pgid: int) -> list[bytes]:
    out = subprocess.run(["ps", "-o", "pid=", "-g", str(pgid)], capture_output=True, text=True)
    blocks = []
    for pid in out.stdout.split():
        try:
            blocks.append(Path(f"/proc/{pid}/environ").read_bytes())
        except OSError:
            pass
    return blocks


@pytest.mark.asyncio
@pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="needs /proc")
async def test_the_token_is_not_in_any_worker_processs_environment(child_env):
    from prometheus.tasks.manager import get_task_manager
    from prometheus.tools.base import ToolExecutionContext
    from prometheus.tools.builtin.task_create import TaskCreateTool, TaskCreateToolInput

    work = child_env
    with FakeLlamaServer(work_dir=work, writes_per_turn=30, n_ctx=1_000_000) as box:
        cfg_dir = Path(os.environ["PROMETHEUS_CONFIG_DIR"])
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "prometheus.yaml").write_text(
            f"model:\n  provider: llama_cpp\n  base_url: {box.url}\n  model: fake-local-27b\n"
            f"security:\n  permission_mode: autonomous\n  workspace_root: {work}\n",
            encoding="utf-8")
        mgr = get_task_manager()
        mgr.worker_approval_url = _BASE + "/api/workers/approvals"
        result = await TaskCreateTool().execute(
            TaskCreateToolInput(type="local_agent", description="w",
                                prompt=f"{WORKER_MARK} E.", timeout_seconds=60),
            ToolExecutionContext(cwd=work, metadata={
                "session_id": "web", "effective_session_id": "beacon:abc", "ephemeral": False,
                "serving_route": {"key": "local", "provider": "llama_cpp",
                                  "model": "fake-local-27b", "backend": None}}))
        task_id = result.metadata["task_id"]
        token = mgr._worker_tokens[task_id].encode()
        proc = mgr._processes[task_id]
        # Let the python worker start (it is a grandchild of the task shell).
        await _until(lambda: len(_group_environs(proc.pid)) >= 1 and box.chat_rows(), 20)
        environs = _group_environs(proc.pid)
        await mgr.stop_task(task_id)
    assert environs, "no worker process was found to inspect"
    leaking = [e for e in environs if token in e]
    assert not leaking, f"the worker token is readable in {len(leaking)} process environment(s)"


def test_the_worker_pops_any_token_it_finds_in_its_environment():
    from prometheus.cli import worker_approval as wa

    env = {"PROMETHEUS_WORKER_APPROVAL_URL": "http://127.0.0.1:9/x",
           "PROMETHEUS_WORKER_TASK_ID": "a1", "PROMETHEUS_WORKER_TOKEN": "fake" + "envtoken"}
    try:
        assert wa.approval_prompt_from_env(env) is not None
        assert "PROMETHEUS_WORKER_TOKEN" not in env
    finally:
        wa.set_worker_token(None)


def test_the_brief_carries_the_token_line_and_the_worker_strips_it():
    from prometheus.cli.worker_approval import split_token_line
    from prometheus.tasks.worker_auth import token_line

    token, brief = split_token_line(token_line("fake" + "stdintoken") + "Do the thing.\nStep 2.")
    assert token == "fake" + "stdintoken" and brief == "Do the thing.\nStep 2."
    assert split_token_line("Do the thing.") == (None, "Do the thing.")


# --- (b) card text is capped and clean --------------------------------------

@pytest.mark.asyncio
async def test_the_cards_text_is_capped_and_has_no_control_characters(tmp_path, monkeypatch):
    from prometheus.permissions.approval_queue import ApprovalQueue

    mgr, rec, token = await _standin_worker(
        tmp_path, monkeypatch, description="desc\r\nwith a break\x1b[31m and " + "d" * 500)
    queue = ApprovalQueue(security_gate=_Gate())
    try:
        async with _client(_app(queue, mgr, monkeypatch)) as c:
            body = {"tool_name": "write_file\n/approve always " + "t" * 300,
                    "reason": "line one\nline two‮\x07" + "r" * 5000,
                    "arguments": {"path": "./x"}}
            req = asyncio.create_task(c.post("/api/workers/approvals",
                                             headers=_headers(rec.id, token), json=body))
            pending = await _until(queue.list_pending)
            action = pending[0]
            await queue.deny(action.request_id, by=__import__(
                "prometheus.permissions.approver", fromlist=["in_process"]).in_process("t"))
            await req
    finally:
        await mgr.shutdown()
    for text, cap in ((action.tool_name, 64), (action.description, 800)):
        assert len(text) <= cap, (len(text), cap)
        assert not any(ch in text for ch in "\r\n\x1b\x07‮"), repr(text[:120])


# --- (c) caps, rate limit, chunked body, auth before body --------------------

@pytest.mark.asyncio
async def test_a_worker_holds_at_most_three_pending_requests(tmp_path, monkeypatch):
    from prometheus.permissions.approval_queue import ApprovalQueue

    mgr, rec, token = await _standin_worker(tmp_path, monkeypatch)
    queue = ApprovalQueue(security_gate=_Gate())
    try:
        async with _client(_app(queue, mgr, monkeypatch)) as c:
            held = [asyncio.create_task(c.post("/api/workers/approvals",
                                               headers=_headers(rec.id, token), json=_BODY))
                    for _ in range(3)]
            await _until(lambda: len(queue.list_pending()) == 3)
            fourth = await c.post("/api/workers/approvals",
                                  headers=_headers(rec.id, token), json=_BODY)
            assert fourth.status_code == 429 and fourth.json()["result"] == "denied", fourth.text
            assert len(queue.list_pending()) == 3
            for a in queue.list_pending():
                await queue.deny(a.request_id, by=__import__(
                    "prometheus.permissions.approver", fromlist=["in_process"]).in_process("t"))
            await asyncio.gather(*held)
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_a_worker_is_rate_limited(tmp_path, monkeypatch):
    from prometheus.permissions.approval_queue import ApprovalQueue
    from prometheus.tasks import worker_auth

    monkeypatch.setattr(worker_auth, "RATE_LIMIT", (2, 60.0))
    mgr, rec, token = await _standin_worker(tmp_path, monkeypatch)
    queue = ApprovalQueue(security_gate=_Gate(), timeout_seconds=1)
    try:
        async with _client(_app(queue, mgr, monkeypatch)) as c:
            codes = []
            for _ in range(3):
                r = await c.post("/api/workers/approvals",
                                 headers=_headers(rec.id, token), json=_BODY)
                codes.append(r.status_code)
    finally:
        await mgr.shutdown()
    assert codes[:2] == [200, 200] and codes[2] == 429, codes


@pytest.mark.asyncio
async def test_a_chunked_body_over_the_cap_is_refused(tmp_path, monkeypatch):
    from prometheus.permissions.approval_queue import ApprovalQueue

    mgr, rec, token = await _standin_worker(tmp_path, monkeypatch)
    queue = ApprovalQueue(security_gate=_Gate())

    async def chunks():
        yield b'{"tool_name": "write_file", "reason": "'
        for _ in range(200):
            yield b"x" * 4096
        yield b'"}'

    try:
        async with _client(_app(queue, mgr, monkeypatch)) as c:
            r = await c.post("/api/workers/approvals", headers=_headers(rec.id, token),
                             content=chunks())
    finally:
        await mgr.shutdown()
    assert r.status_code == 413, (r.status_code, r.text[:200])
    assert not queue.list_pending()


@pytest.mark.asyncio
async def test_the_token_is_checked_before_the_body_is_read(tmp_path, monkeypatch):
    from prometheus.permissions.approval_queue import ApprovalQueue

    mgr, rec, _token = await _standin_worker(tmp_path, monkeypatch)
    try:
        async with _client(_app(ApprovalQueue(security_gate=_Gate()), mgr, monkeypatch)) as c:
            r = await c.post("/api/workers/approvals", headers=_headers(rec.id, "wrong"),
                             content=b"{not json at all")
    finally:
        await mgr.shutdown()
    assert r.status_code == 401, (r.status_code, r.text[:200])


# --- (d) the chat path ---------------------------------------------------------

def _worker_action(request_id: str):
    from prometheus.permissions.approval_queue import PendingAction

    return PendingAction(request_id=request_id, tool_name="write_file", description="d",
                         session_id="beacon:abc", once_only=True, worker_task_id="a1")


@pytest.mark.asyncio
async def test_chat_approve_always_refuses_a_once_only_request_by_name():
    from prometheus.gateway.commands import approve_detail
    from prometheus.permissions.approval_queue import ApprovalQueue
    from prometheus.permissions.approver import in_process

    queue = ApprovalQueue(security_gate=_Gate())
    queue._register(_worker_action("abcd1234"))
    from prometheus.permissions.approval_queue import ApprovalResult

    for verb in ("always", "until-restart"):
        out = await approve_detail(queue, f"{verb} abcd1234", by=in_process("t"))
        assert not out.resolved and "once" in out.message.lower(), out.message
        assert queue.pending["abcd1234"]._result is ApprovalResult.TIMEOUT, "it was approved"


@pytest.mark.asyncio
async def test_approve_all_leaves_worker_requests_alone():
    from prometheus.gateway.commands import approve_detail
    from prometheus.permissions.approval_queue import ApprovalQueue, PendingAction
    from prometheus.permissions.approver import in_process

    queue = ApprovalQueue(security_gate=_Gate())
    queue._register(_worker_action("abcd1234"))
    queue._register(PendingAction(request_id="eeee5555", tool_name="bash", description="ls"))
    from prometheus.permissions.approval_queue import ApprovalResult

    out = await approve_detail(queue, "all", by=in_process("t"))
    assert queue.pending["abcd1234"]._result is ApprovalResult.TIMEOUT, \
        "approve all approved a worker's request"
    assert queue.pending["eeee5555"]._result is ApprovalResult.APPROVED
    assert out.skipped == 1, out


# --- (e) token lifecycle -----------------------------------------------------------

@pytest.mark.asyncio
async def test_a_restarted_worker_gets_a_fresh_token(tmp_path, monkeypatch):
    mgr, rec, first = await _standin_worker(tmp_path, monkeypatch)
    try:
        proc = mgr._processes[rec.id]
        proc.kill()
        await _until(lambda: mgr.get_task(rec.id).status != "running")
        await mgr._restart_agent_task(mgr.get_task(rec.id))
        second = mgr._worker_tokens.get(rec.id)
        assert second and second != first, "a restart kept the old token"
        assert mgr.verify_worker_token(rec.id, first) is None
        assert mgr.verify_worker_token(rec.id, second) is not None
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_a_worker_that_is_refused_at_launch_leaves_no_token(tmp_path, monkeypatch):
    from prometheus.tasks import manager as mgr_mod
    from prometheus.tasks.manager import BackgroundTaskManager

    mgr = BackgroundTaskManager()
    mgr.worker_approval_url = _BASE + "/api/workers/approvals"
    monkeypatch.setattr(mgr, "_vet_command", lambda command: "blocked: not allowed here")
    monkeypatch.setattr(mgr_mod, "prometheus_argv", lambda *a: ["/bin/true", *a])
    rec = await mgr.create_agent_task(prompt="p", description="d", cwd=str(tmp_path),
                                      route="local", worker=True, session_id="beacon:abc")
    assert rec.status == "blocked"
    assert rec.id not in mgr._worker_tokens and not mgr._worker_tokens


# --- (f) IPv6 hosts are bracketed ------------------------------------------------

def test_the_endpoint_url_brackets_an_ipv6_host():
    from prometheus.tasks.worker_auth import worker_approval_url

    assert worker_approval_url("::1", 8005) == "http://[::1]:8005/api/workers/approvals"
    assert worker_approval_url("0.0.0.0", 8005) == "http://127.0.0.1:8005/api/workers/approvals"
    assert worker_approval_url("::", 8005) == "http://127.0.0.1:8005/api/workers/approvals"
    assert worker_approval_url(None, 8005) == "http://127.0.0.1:8005/api/workers/approvals"
    assert worker_approval_url("fe80::1", 9) == "http://[fe80::1]:9/api/workers/approvals"


# --- the token line never lands anywhere durable, nor in the model's context ---

@pytest.mark.asyncio
async def test_the_token_line_lands_in_no_record_file_log_telemetry_or_model_request(
        child_env, tmp_path, caplog):
    """The token rides stdin ahead of the brief. A real worker runs a turn;
    then every durable place this run could have written it is searched for
    the literal token: the task row in tasks.db (the stored prompt is the brief
    alone), the task's output file, every file under the test's config, data
    and log directories (the worker's cli.log, telemetry.db, lcm.db, ...), the
    daemon side's log records, and the raw requests the model was sent."""
    import logging

    from prometheus.tasks.manager import get_task_manager
    from prometheus.tasks.store import TaskStore
    from prometheus.tools.base import ToolExecutionContext
    from prometheus.tools.builtin.task_create import TaskCreateTool, TaskCreateToolInput

    caplog.set_level(logging.DEBUG)
    work = child_env
    brief = f"{WORKER_MARK} L. Write the step files."
    with FakeLlamaServer(work_dir=work, writes_per_turn=1, keep_bodies=True) as box:
        cfg_dir = Path(os.environ["PROMETHEUS_CONFIG_DIR"])
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "prometheus.yaml").write_text(
            f"model:\n  provider: llama_cpp\n  base_url: {box.url}\n  model: fake-local-27b\n"
            f"security:\n  permission_mode: autonomous\n  workspace_root: {work}\n",
            encoding="utf-8")
        mgr = get_task_manager()
        mgr.store = TaskStore(tmp_path / "tasks.db")
        mgr.worker_approval_url = _BASE + "/api/workers/approvals"
        result = await TaskCreateTool().execute(
            TaskCreateToolInput(type="local_agent", description="w", prompt=brief,
                                timeout_seconds=60),
            ToolExecutionContext(cwd=work, metadata={
                "session_id": "web", "effective_session_id": "beacon:abc", "ephemeral": False,
                "serving_route": {"key": "local", "provider": "llama_cpp",
                                  "model": "fake-local-27b", "backend": None}}))
        task_id = result.metadata["task_id"]
        token = mgr._worker_tokens[task_id]
        deadline = time.monotonic() + 40
        while mgr.get_task(task_id).status == "running" and time.monotonic() < deadline:
            await asyncio.sleep(0.25)
        rec = mgr.get_task(task_id)
        bodies = list(box.bodies)
    assert rec.status == "completed", (rec.status, rec.error)
    assert bodies, "the worker never reached the model"
    needle = token.encode()

    # The stored prompt is the brief and nothing else.
    import sqlite3

    rows = sqlite3.connect(tmp_path / "tasks.db").execute(
        "select prompt, command, metadata from tasks where id = ?", (task_id,)).fetchall()
    assert rows and rows[0][0] == brief, rows
    assert all(needle not in str(col).encode() for col in rows[0])

    # Every file this run could have written.
    roots = {Path(os.environ[v]) for v in ("PROMETHEUS_CONFIG_DIR", "PROMETHEUS_DATA_DIR")
             if os.environ.get(v)} | {tmp_path}
    leaked = sorted(str(p) for root in roots for p in root.rglob("*")
                    if p.is_file() and needle in p.read_bytes())
    assert not leaked, f"the token is written in: {leaked}"
    assert needle not in rec.output_file.read_bytes()

    # The daemon side's logs, and the model's context.
    assert not [r for r in caplog.records if token in r.getMessage()]
    assert not [b for b in bodies if needle in b], "the token reached the model's context"
    assert any(brief.encode() in b for b in bodies), "the brief itself never reached the model"
