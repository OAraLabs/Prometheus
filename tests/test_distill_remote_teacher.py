"""The remote teacher ("option 3"): the 4090 serves the 27B on the tailnet, the 3090 Ti host runs the gym.

Covers teacher_gen.py's remote mode (tailnet-only URL, the key file, key enforcement, concurrency against the
server's slots, the pause while the server is gone or loading and the resume, the stop time across the DST
change, the row tags build.py reads, bwrap only, the tool host), gen_runner.py ``remote`` / ``remote-setup``,
teacher_ab.py's remote arm B, mini_ab.py's AB_ARM_B flag and tower/serve_teacher.sh. A fake OpenAI-compatible
llama-server on loopback stands in for the 4090 (its tailnet check is bypassed by setting the checked URL);
nothing touches a GPU, a systemd manager, a real host or a real key.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

REPO = Path(__file__).resolve().parents[1]
_DIR = REPO / "scripts" / "distill"
pytestmark = pytest.mark.skipif(not sys.platform.startswith(("linux", "darwin")), reason="posix only")


def _load(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tg = _load("distill_teacher_gen", _DIR / "teacher_gen.py")
ab = _load("distill_teacher_ab", _DIR / "teacher_ab.py")
H = _load("_remote_tg_helpers", REPO / "tests" / "test_distill_teacher_gen.py")  # scripted policies, tasks
BT = _load("_remote_build_helpers", REPO / "tests" / "test_distill_build.py")    # build.py fixtures

KEY = "f" * 16 + "0123456789abcdef" * 3   # a test value, never a real key
SHA = "bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372"
ALIAS = "qwen3.8-27b-ud-q4_k_xl"
ET = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def _restore_state_env(monkeypatch):
    for var in ("PROMETHEUS_LOGS_DIR", "PROMETHEUS_CONFIG_DIR", "PROMETHEUS_DATA_DIR", "PROMETHEUS_ENV_FILE",
                *tg.PROXY_ENV, "NO_PROXY", "no_proxy"):
        old = os.environ.get(var)
        monkeypatch.setenv(var, old if old is not None else "")
        if old is None:
            monkeypatch.delenv(var)


# ── a fake llama-server with an API key (the 4090's teacher) ──────────────────


class KeyedServer:
    """llama-server's shape: /health and /v1/models public, everything else needs ``Bearer <key>``
    (unless ``open``); ``loading`` answers 503 like a server loading its model; ``stop``/``start``
    take the port away and bring it back (refused connections in between)."""

    def __init__(self, key: str = KEY, slots: int = 4, policy: Any = None, open_: bool = False) -> None:
        self.key, self.slots, self.open = key, slots, open_
        self.policy = policy or H.scripted({})
        self.loading = False
        self.tools_status = 403      # llama-server answers GET /tools 403 while its tools are off
        self.cors_proxy = False
        self.lock = threading.Lock()
        self.auth_seen: list[str | None] = []      # Authorization header of every keyed request
        self.chat_paths: list[str] = []
        self.on_chat: Any = None                   # called with the count of chat requests so far
        self.port = 0
        self.server: ThreadingHTTPServer | None = None
        self.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _handler(self) -> type:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def _json(self, code: int, obj: Any) -> None:
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _authorized(self) -> bool:
                got = self.headers.get("Authorization")
                with outer.lock:
                    outer.auth_seen.append(got)
                return outer.open or got == f"Bearer {outer.key}"

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    self._json(503 if outer.loading else 200,
                               {"error": "Loading model"} if outer.loading else {"status": "ok"})
                elif outer.loading:
                    self._json(503, {"error": {"message": "Loading model"}})
                elif self.path == "/v1/models":
                    self._json(200, {"object": "list", "data": [{"id": ALIAS}]})
                elif self.path == "/props":
                    if not self._authorized():
                        self._json(401, {"error": {"message": "Invalid API Key"}})
                        return
                    self._json(200, {"total_slots": outer.slots, "model_path": "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf",
                                     "chat_template": "{{ base }}", "cors_proxy_enabled": outer.cors_proxy,
                                     "ui": False, "endpoint_slots": False, "endpoint_props": False,
                                     "endpoint_metrics": False,
                                     "default_generation_settings": {"n_ctx": 32768,
                                                                     "params": {"temperature": 1.0, "top_k": 20}}})
                elif self.path == "/tools":
                    if not self._authorized():
                        self._json(401, {"error": {"message": "Invalid API Key"}})
                        return
                    self._json(outer.tools_status, [{"name": "exec_shell_command"}] if outer.tools_status == 200
                               else {"error": {"message": "tools are disabled"}})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if not self._authorized():
                    self._json(401, {"error": {"message": "Invalid API Key"}})
                    return
                if outer.loading:
                    self._json(503, {"error": {"message": "Loading model"}})
                    return
                with outer.lock:
                    outer.chat_paths.append(self.path)
                    count = len(outer.chat_paths)
                if outer.on_chat is not None:
                    outer.on_chat(count)
                msgs = body.get("messages") or []
                if msgs and str(msgs[0].get("content", "")).startswith("You are a strict grader"):
                    reply: dict[str, Any] = {"content": json.dumps(
                        {"verdict": "pass", "completed": True, "tools_ok": True, "grounded": True,
                         "recovery_ok": None, "reason": "fine", "outcome": "done"})}
                else:
                    reply = outer.policy(body) if msgs else {"content": "ok"}
                if not body.get("stream"):
                    self._json(200, {"model": ALIAS, "choices": [
                        {"message": {"role": "assistant", "content": reply.get("content")}, "finish_reason": "stop"}]})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks: list[dict[str, Any]] = []
                if reply.get("content"):
                    chunks.append({"choices": [{"delta": {"content": reply["content"]}}]})
                for i, (name, args) in enumerate(reply.get("tool_calls") or []):
                    chunks.append({"choices": [{"delta": {"tool_calls": [{
                        "index": i, "id": f"call_{count}_{i}",
                        "function": {"name": name, "arguments": json.dumps(args)}}]}}]})
                chunks.append({"choices": [{"delta": {}, "finish_reason": "tool_calls" if reply.get("tool_calls")
                                            else "stop"}]})
                chunks.append({"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
                for c in chunks:
                    c["model"] = ALIAS
                    self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        return Handler

    def start(self) -> None:
        srv = ThreadingHTTPServer(("127.0.0.1", self.port), self._handler())
        srv.daemon_threads = True
        self.server = srv
        self.port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    def close(self) -> None:
        self.stop()


def key_file(tmp_path: Path, mode: int = 0o600, key: str = KEY) -> Path:
    p = tmp_path / "secrets" / "teacher-4090.key"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(key + "\n")
    p.chmod(mode)
    return p


def remote_settings(tmp_path: Path, url: str, **kw: Any) -> Any:
    s = tg.Settings(out=tmp_path / "out", teacher_url=url, sandbox_root=tmp_path / "sbx", request_timeout=20.0,
                    progress=False, task_source=tg.ORIGIN_TASKSET, teacher_key_file=key_file(tmp_path), teacher_key=KEY, teacher_host="gpu-tower",
                    teacher_quant="UD-Q4_K_XL", teacher_sha256=SHA, tool_host="gym-host", **kw)
    # the fake server is on loopback: stand in for the tailnet check this run already passed
    s._remote_checked, s._remote_url = url, url
    return s


# ── the endpoint, the key file, the host checks ───────────────────────────────


def test_a_remote_teacher_is_reached_over_the_tailnet_only(monkeypatch):
    addrs: dict[str, list[str]] = {"gpu-tower": ["100.64.0.39"], "lan": ["192.168.1.20"],
                                   "pub": ["203.0.113.7"], "lo": ["127.0.0.1"], "me": ["100.64.0.21"],
                                   "v6": ["fd7a:115c:a1e0::1"]}
    monkeypatch.setattr(tg, "_host_addresses", lambda h: addrs[h])
    monkeypatch.setattr(tg, "is_local_address", lambda a: a == "100.64.0.21")
    assert tg.check_tailnet_endpoint("http://gpu-tower:18094/") == "http://gpu-tower:18094"
    assert tg.check_tailnet_endpoint("http://v6:18094") == "http://v6:18094"
    for bad in ("http://lan:18094", "http://pub:18094", "http://lo:18094"):
        with pytest.raises(tg.EndpointError, match="not a tailnet address"):
            tg.check_tailnet_endpoint(bad)
    with pytest.raises(tg.EndpointError, match="is this host"):
        tg.check_tailnet_endpoint("http://me:18094")
    with pytest.raises(tg.EndpointError, match="scheme://host:port only"):
        tg.check_tailnet_endpoint("http://gpu-tower:18094/v1")
    with pytest.raises(tg.EndpointError, match="credentials"):
        tg.check_tailnet_endpoint("http://u:p@gpu-tower:18094")


def test_the_key_file_must_be_a_private_regular_file(tmp_path):
    good = key_file(tmp_path)
    assert tg.load_teacher_key(good) == KEY
    good.chmod(0o644)
    with pytest.raises(tg.TeacherKeyError, match="0600") as e:
        tg.load_teacher_key(good)
    assert KEY not in str(e.value)
    good.chmod(0o600)
    link = tmp_path / "link.key"
    link.symlink_to(good)
    with pytest.raises(tg.TeacherKeyError, match="symlink"):
        tg.load_teacher_key(link)
    short = tmp_path / "short.key"
    short.write_text("abc\n")
    short.chmod(0o600)
    with pytest.raises(tg.TeacherKeyError, match="at least"):
        tg.load_teacher_key(short)


def _run_args(tmp_path: Path, *extra: str) -> list[str]:
    return ["run", "--teacher", "local", "--teacher-url", "http://gpu-tower:18094",
            "--teacher-key-file", str(key_file(tmp_path)), "--teacher-host", "gpu-tower",
            "--teacher-quant", "UD-Q4_K_XL", "--teacher-sha256", SHA, "--tool-host", tg.local_hostname(),
            "--out", str(tmp_path / "out"), "--until", "2099-01-01T00:00:00+00:00",
            "--tasks", str(tmp_path / "tasks.jsonl"), "--injection-manifest", str(tmp_path / "manifest.json"), "--eval-tools", str(tmp_path / "eval.jsonl"),
            *extra]


def test_tools_run_only_under_bwrap_and_only_on_the_tool_host(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tg, "resolve_jail", lambda mode: "sandbox-exec")
    assert tg.main(_run_args(tmp_path)) == tg.EXIT_USAGE
    assert "run only under bwrap" in capsys.readouterr().err
    a = _run_args(tmp_path)
    a[a.index("--tool-host") + 1] = "some-other-host"
    assert tg.main(a) == tg.EXIT_USAGE
    assert "is not this host" in capsys.readouterr().err
    a = _run_args(tmp_path)
    a[a.index("--teacher-host") + 1] = tg.local_hostname()
    assert tg.main(a) == tg.EXIT_USAGE
    assert "is this host" in capsys.readouterr().err
    assert tg.main(_run_args(tmp_path, "--jail", "off")) == tg.EXIT_USAGE
    assert "--jail off is refused" in capsys.readouterr().err
    assert tg.main(_run_args(tmp_path, "--judge-url", "http://127.0.0.1:9")) == tg.EXIT_USAGE
    assert tg.main(_run_args(tmp_path, "--probe-guard-url", "http://127.0.0.1:9")) == tg.EXIT_USAGE
    # bwrap: the settings pass, the endpoint check is next (the fake name resolves nowhere here)
    monkeypatch.setattr(tg, "resolve_jail", lambda mode: "bwrap")
    monkeypatch.setattr(tg, "check_tailnet_endpoint", lambda url, **kw: (_ for _ in ()).throw(
        tg.EndpointError("gpu-tower is not a tailnet address")))
    assert tg.main(_run_args(tmp_path)) == tg.EXIT_USAGE
    assert "not a tailnet address" in capsys.readouterr().err


def test_the_server_must_enforce_the_key_and_accept_ours(tmp_path):
    srv = KeyedServer(open_=True)
    try:
        s = remote_settings(tmp_path, srv.url)
        with pytest.raises(tg.EndpointError, match="without a key"):
            asyncio.run(tg.remote_props(s, 5))
        srv.open = False
        assert asyncio.run(tg.remote_props(s, 5))["total_slots"] == 4
        s.teacher_key = "0" * 64
        with pytest.raises(tg.EndpointError, match="refused our key"):
            asyncio.run(tg.remote_props(s, 5))
    finally:
        srv.close()


def test_a_teacher_with_llama_servers_tools_or_mcp_proxy_on_is_refused_and_stops_a_resumed_run(tmp_path):
    """Its built-in tools (exec_shell_command, write_file, ...) would run on the teacher host, which has no
    jail: refused at the start (GET /tools answers 200, or /props says cors_proxy_enabled), and a server that
    comes back from a pause that way stops the run (TeacherChanged), like one serving another model."""
    srv = KeyedServer()
    try:
        s = remote_settings(tmp_path, srv.url)
        s.teacher_served = ALIAS
        assert asyncio.run(tg.remote_props(s, 5))["total_slots"] == 4      # 403 on /tools: off
        srv.tools_status = 404                                              # a build without the route
        assert asyncio.run(tg.remote_props(s, 5))["total_slots"] == 4
        srv.tools_status = 200
        with pytest.raises(tg.EndpointError, match="built-in tools or MCP servers"):
            asyncio.run(tg.remote_props(s, 5))
        with pytest.raises(tg.TeacherChanged, match="built-in tools"):
            asyncio.run(tg.remote_identity(s, 5))
        srv.tools_status, srv.cors_proxy = 403, True
        with pytest.raises(tg.EndpointError, match="MCP proxy on"):
            asyncio.run(tg.remote_props(s, 5))
    finally:
        srv.close()


def test_concurrency_never_exceeds_the_servers_slots():
    assert tg.remote_concurrency(None, 4) == 4
    assert tg.remote_concurrency(0, 4) == 4
    assert tg.remote_concurrency(6, 4) == 4
    assert tg.remote_concurrency(2, 4) == 2
    assert tg.remote_concurrency(None, 20) == tg.REMOTE_MAX_CONCURRENCY
    assert tg.remote_concurrency(None, 0) == 1


# ── the stop time ─────────────────────────────────────────────────────────────


def test_the_stop_time_defaults_to_monday_1030_utc_and_hhmm_follows_the_dst_change(tmp_path, monkeypatch):
    utc = dt.timezone.utc
    assert tg.parse_until(tg.REMOTE_DEFAULT_UNTIL) == dt.datetime(2026, 11, 2, 10, 30, tzinfo=utc).timestamp()
    # Saturday evening EDT: 05:30 next is Sunday 05:30 EST (after the 02:00 fall-back), not EDT
    sat = dt.datetime(2026, 10, 31, 22, 0, tzinfo=ET)
    assert tg.parse_until("05:30", now=sat) == dt.datetime(2026, 11, 1, 10, 30, tzinfo=utc).timestamp()
    sun = dt.datetime(2026, 11, 1, 20, 0, tzinfo=ET)
    assert tg.parse_until("05:30", now=sun, max_ahead_hours=12) == \
        dt.datetime(2026, 11, 2, 10, 30, tzinfo=utc).timestamp()
    # a fixed offset stays fixed (the old behaviour for callers that pass one)
    fixed = dt.datetime(2026, 10, 31, 22, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
    assert tg.parse_until("05:30", now=fixed) == dt.datetime(2026, 11, 1, 9, 30, tzinfo=utc).timestamp()
    # the remote default is applied when --until is absent, and refused once it has passed
    a = tg.parse_args([x for x in _run_args(tmp_path) if x not in ("--until", "2099-01-01T00:00:00+00:00")])
    monkeypatch.setattr(tg.time, "time", lambda: dt.datetime(2026, 11, 1, 12, 0, tzinfo=utc).timestamp())
    remote = tg._remote_args(a)
    assert a.until == tg.REMOTE_DEFAULT_UNTIL and remote["teacher_sha256"] == SHA
    a = tg.parse_args([x for x in _run_args(tmp_path) if x not in ("--until", "2099-01-01T00:00:00+00:00")])
    monkeypatch.setattr(tg.time, "time", lambda: dt.datetime(2026, 11, 2, 11, 0, tzinfo=utc).timestamp())
    with pytest.raises(ValueError, match="has passed"):
        tg._remote_args(a)


# ── a real task against the keyed server: the key, the tags, build.py ─────────


def _wiki_task() -> Any:
    return H.task(prompt="Look it up in the wiki: which host runs cobalt-intake?", domain="memory/wiki",
                  origin=tg.ORIGIN_TASKSET, meta={"provenance": tg.TASKSET_PROVENANCE},
                  fixtures={"wiki": {"cobalt-intake": "# cobalt-intake\n- Runs on: node-bravo\n"}},
                  score={"expect_tool": "wiki_query", "expect_text_contains": "node-bravo"})


def test_every_teacher_request_carries_the_key_and_the_rows_carry_the_route(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    srv = KeyedServer(policy=H.scripted({"cobalt-intake": [("call", "wiki_query", {"query": "cobalt-intake host"}),
                                                          "echo"]}))
    try:
        s = remote_settings(tmp_path, srv.url)
        s.teacher_served = ALIAS
        out = asyncio.run(tg.run_task(_wiki_task(), s))
        ok, reasons, record = tg.deterministic_checks(out, s)
        assert ok, reasons
        record["judge"] = asyncio.run(tg.judge(out, s))
        assert record["judge"]["pass"] and record["judge"]["kind"] == "self-judge"
        record["verified"] = True
        rows = tg.rows_for(out, s, record)
        assert rows
        # the teacher, its tool loop and the self-judge: every request to the server carried the key
        assert srv.chat_paths and set(srv.auth_seen) == {f"Bearer {KEY}"}
        for r in rows:
            m = r["_meta"]
            assert m["source"] == "qwen38-27b-gym" and m["model"] == tg.DEFAULT_MODEL_NAME
            assert (m["teacher_route"], m["teacher_host"], m["teacher_quant"], m["teacher_sha256"], m["tool_host"]) \
                == ("tailnet", "gpu-tower", "UD-Q4_K_XL", SHA, "gym-host")
        # the key is in no row, record, repr or log line
        assert KEY not in json.dumps(rows) and KEY not in json.dumps(record, default=str)
        assert KEY not in repr(s) and KEY not in caplog.text
        fp = s.fingerprint()["remote_teacher"]
        assert fp == {"teacher_route": "tailnet", "teacher_host": "gpu-tower", "teacher_quant": "UD-Q4_K_XL",
                      "teacher_sha256": SHA}
    finally:
        srv.close()
    # build.py's reader takes rows with these tags as plain local-teacher gym rows (train source)
    tags = {k: rows[0]["_meta"][k] for k in ("teacher_route", "teacher_host", "teacher_quant", "teacher_sha256",
                                             "tool_host")}
    golden = BT.corpus(tmp_path)
    gym_rows = []
    for t in range(40):
        name = "gym" + BT.word(t)
        gym_rows.append(BT.gym(name, "bash", {"command": f"make {name}"}, category=["coding", "other"][t % 2], **tags))
    gym_rows += [BT.gym_chat("chat" + BT.word(t), **tags) for t in range(8)]
    b = BT.make_builder(tmp_path, train_sources=(BT.GYM,), eval_task_frac=0.15, val_frac=0.05,
                        holdout_max_group_share=0.5, val_share_range=(0.0, 1.0))
    BT.feed(b, golden)
    BT.feed(b, BT.write_jsonl(tmp_path / "remote-gym.jsonl", gym_rows), "extra")
    b.split()
    train = b.final["train"]
    assert train and all(e.source == BT.GYM and e.train_ok and not e.max_derived for e in train)
    assert not [x for x in b.excluded if "remote-gym" in x[0]]


# ── the pause while the server is gone or loading, then the resume ────────────


def test_a_gone_or_loading_server_pauses_the_run_and_it_resumes_with_the_new_slots(tmp_path, monkeypatch):
    monkeypatch.setattr(tg, "REMOTE_BACKOFF_S", (0.05, 0.2))
    monkeypatch.setattr(tg, "REMOTE_WAIT_POLL_S", 0.02)
    monkeypatch.setattr(tg, "REMOTE_HEALTH_TIMEOUT_S", 1.0)
    srv = KeyedServer(slots=3)
    phases: list[str] = []
    inflight = {"now": 0, "max_after_resume": 0}
    resumed = threading.Event()

    def outage(count: int) -> None:
        if count != 5:
            return

        def go() -> None:
            srv.stop()                       # connections refused (the 4090 starts training, say)
            phases.append("gone")
            time.sleep(0.4)
            srv.loading, srv.slots = True, 1  # back on the same port, still loading its model
            srv.start()
            phases.append("loading")
            time.sleep(0.4)
            srv.loading = False
            phases.append("ready")

        threading.Thread(target=go, daemon=True).start()

    srv.on_chat = outage
    import httpx

    async def fake_process(task: Any, settings: Any, worker: int) -> tuple[str, list, list[str], dict[str, Any]]:
        inflight["now"] += 1
        if resumed.is_set():
            inflight["max_after_resume"] = max(inflight["max_after_resume"], inflight["now"])
        try:
            await asyncio.sleep(0.05)
            async with httpx.AsyncClient(timeout=2, trust_env=False, headers=settings.teacher_headers) as c:
                r = await c.post(f"{settings.teacher_endpoint}/v1/chat/completions",
                                 json={"messages": [{"role": "user", "content": task.prompt}]})
            if r.status_code != 200:
                return "error", [], ["transport_error"], {"error": f"HTTP {r.status_code}"}
            return "rejected", [], ["test_done"], {"duration_s": 0.05}
        except httpx.HTTPError as exc:
            return "error", [], ["transport_error"], {"error": type(exc).__name__}
        finally:
            inflight["now"] -= 1

    monkeypatch.setattr(tg, "process", fake_process)
    real_identity = tg.remote_identity

    async def identity(settings: Any, timeout: float) -> int:
        slots = await real_identity(settings, timeout)
        resumed.set()
        return slots

    monkeypatch.setattr(tg, "remote_identity", identity)
    try:
        s = remote_settings(tmp_path, srv.url, concurrency=3, max_consecutive_errors=2)
        s.teacher_served = ALIAS
        tasks = [H.task(id=f"t-{i:05d}", prompt=f"remote task number {i}") for i in range(14)]
        store = tg.Store(s.out, s.name, s.fingerprint(), s.coding_cap)
        report = asyncio.run(asyncio.wait_for(tg.run(s, tg.FixedGenerator(tasks), store, None), 60))
    finally:
        srv.close()
    assert phases == ["gone", "loading", "ready"]
    # not stopped by the errors of the outage: every task finished, none lost, the pause counted
    assert report["stop"] == "no_tasks", report
    assert report["remote_pauses"] >= 1 and report["remote_paused_s"] > 0
    assert {t.id for t in tasks} == {tid for tid, e in store.status.items() if e["status"] == "rejected"}
    # back with 1 slot: never more than one task in flight after the resume
    assert resumed.is_set() and inflight["max_after_resume"] == 1
    # the ledger: the errors were the tasks in flight at the outage only, each rerun afterwards
    ledger = [json.loads(x) for x in (s.out / f"{s.name}.tasks.jsonl").read_text().splitlines()]
    errors = [e for e in ledger if e["status"] == "error"]
    assert 1 <= len(errors) <= 3 * report["remote_pauses"]


def test_a_server_that_comes_back_as_another_model_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(tg, "REMOTE_BACKOFF_S", (0.05, 0.1))
    monkeypatch.setattr(tg, "REMOTE_WAIT_POLL_S", 0.02)
    srv = KeyedServer(slots=2)
    srv.on_chat = lambda n: setattr(srv, "loading", True) if n == 2 else None
    import httpx

    async def fake_process(task: Any, settings: Any, worker: int) -> tuple[str, list, list[str], dict[str, Any]]:
        async with httpx.AsyncClient(timeout=2, trust_env=False, headers=settings.teacher_headers) as c:
            r = await c.post(f"{settings.teacher_endpoint}/v1/chat/completions", json={"messages": []})
        if r.status_code != 200:
            # restarted with another key: loading for a moment, then ready but refusing ours
            srv.key = "another-key-" + "0" * 40
            threading.Timer(0.2, lambda: setattr(srv, "loading", False)).start()
            return "error", [], ["transport_error"], {}
        return "rejected", [], ["test_done"], {}

    monkeypatch.setattr(tg, "process", fake_process)
    try:
        s = remote_settings(tmp_path, srv.url, concurrency=1)
        s.teacher_served = ALIAS
        tasks = [H.task(id=f"t-{i:05d}", prompt=f"another task {i}") for i in range(6)]
        store = tg.Store(s.out, s.name, s.fingerprint(), s.coding_cap)
        report = asyncio.run(asyncio.wait_for(tg.run(s, tg.FixedGenerator(tasks), store, None), 30))
    finally:
        srv.close()
    assert report["stop"] == "teacher_changed" and "refused our key" in report["stop_detail"]
    assert KEY not in report["stop_detail"]


# ── --skip-done-in ────────────────────────────────────────────────────────────


def test_tasks_another_run_finished_are_skipped_by_id_or_prompt(tmp_path):
    led = tmp_path / "local" / "qwen38-27b-gym.tasks.jsonl"
    led.parent.mkdir()
    a, b, c = (H.task(id=f"x-{i}", prompt=f"please look at the {w} service") for i, w in
               enumerate(("amber", "cobalt", "harbor")))
    assert len({a.task_hash, b.task_hash, c.task_hash}) == 3
    led.write_text("\n".join(json.dumps(e) for e in [
        {"task_id": "x-0", "task_hash": "zzz", "status": "kept"},
        {"task_id": "other-id", "task_hash": b.task_hash, "status": "rejected"},
        {"task_id": c.id, "task_hash": c.task_hash, "status": "error"},
    ]) + "\n{torn")
    ids, hashes = tg.skip_keys_from([str(tmp_path / "local" / "*.tasks.jsonl")])
    left = [t.id for t in (a, b, c) if t.id not in ids and t.task_hash not in hashes]
    assert left == ["x-2"]  # an errored task still runs


# ── gen_runner.py remote / remote-setup ───────────────────────────────────────

FAKE_REMOTE_TG = r'''"""Stand-in for teacher_gen.py in remote mode (RemoteGate, --teacher-key-file, --teacher-sha256)."""
import json, os, sys
from pathlib import Path
state = Path(os.environ["FAKE_STATE"])
argv = sys.argv[1:]
with open(state / "remote-calls.jsonl", "a") as f:
    key = os.environ.get("FAKE_KEY", "-")
    f.write(json.dumps({"argv": argv, "env_has_key": any(key in v for k, v in os.environ.items() if k != "FAKE_KEY")}) + "\n")
n = sum(1 for _ in open(state / "remote-calls.jsonl"))
out = Path(argv[argv.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
if n == 1:
    print("error: http://gpu-tower:18094/v1/models is unreachable (ConnectError)", file=sys.stderr)
    sys.exit(3)
(out / "qwen38-27b-gym.report.json").write_text(json.dumps({"stop": "no_tasks", "rows_total": 0}))
sys.exit(1)
'''


def _remote_rig(tmp_path: Path) -> tuple[Any, Path, dict[str, str]]:
    GR = _load("_remote_gen_runner_tests", REPO / "tests" / "test_distill_gen_runner.py")
    rig = GR.Rig(tmp_path)
    code = tmp_path / "code-remote"
    dist = code / "scripts" / "distill"
    dist.mkdir(parents=True)
    (code / "src" / "prometheus").mkdir(parents=True)
    (code / "COMMIT").write_text("d" * 40 + "\n")
    (dist / "teacher_gen.py").write_text(FAKE_REMOTE_TG)
    shutil.copy(GR.FIX / "fake_taskset_gen.py", dist / "taskset_gen.py")
    shutil.copy(_DIR / "artifact_policy.py", dist / "artifact_policy.py")
    key = key_file(tmp_path)
    tasks = tmp_path / "in" / "real-v3-excl.jsonl"
    tasks.write_text("{}\n")
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "bwrap").write_text("#!/bin/sh\nexit 0\n")
    (stub / "bwrap").chmod(0o755)
    with open(rig.conf, "a") as f:
        f.write("\n".join([f"GEN_REMOTE_CODE={code}", f"GEN_REMOTE_KEY_FILE={key}", f"GEN_REMOTE_TASKS={tasks}",
                           "GEN_REMOTE_URL=http://gpu-tower:18094", "GEN_REMOTE_HOST=gpu-tower",
                           "GEN_REMOTE_WAIT_S=0.2", "GEN_REMOTE_WAIT_MAX_S=0.4", ""]))
    env = {**rig.env, "PATH": f"{stub}:{os.environ.get('PATH', '')}", "FAKE_KEY": KEY}
    return rig, key, env


def test_the_remote_runner_retries_an_absent_teacher_then_ends_with_the_taskset(tmp_path):
    rig, key, env = _remote_rig(tmp_path)
    try:
        until = (dt.datetime.now(ET) + dt.timedelta(seconds=60)).isoformat(timespec="seconds")
        dry = subprocess.run([sys.executable, "-I", str(REPO / "scripts/distill/mini_window/gen_runner.py"),
                              "--conf", str(rig.conf), "remote", "--until", until], env=env, capture_output=True,
                             text=True, timeout=120)
        assert dry.returncode == 0, dry.stdout + dry.stderr
        assert "--teacher-key-file" in dry.stdout and str(key) in dry.stdout and KEY not in dry.stdout
        cp = subprocess.run([sys.executable, "-I", str(REPO / "scripts/distill/mini_window/gen_runner.py"),
                             "--conf", str(rig.conf), "remote", "--execute", "--until", until], env=env,
                            capture_output=True, text=True, timeout=120)
        assert cp.returncode == 0, cp.stdout + cp.stderr
        calls = [json.loads(x) for x in (rig.state / "remote-calls.jsonl").read_text().splitlines()]
        assert len(calls) == 2, "exit 3 (not up yet) is retried; the report's no_tasks ends it"
        for c in calls:
            a = c["argv"]
            assert a[:3] == ["run", "--teacher", "local"]
            assert a[a.index("--teacher-key-file") + 1] == str(key) and KEY not in json.dumps(a)
            assert a[a.index("--jail") + 1] == "require" and a[a.index("--teacher-sha256") + 1] == SHA
            assert a[a.index("--tool-host") + 1] == socket.gethostname().split(".")[0].lower()
            assert a[a.index("--skip-done-in") + 1] == str(rig.runs / "local-*" / "*.tasks.jsonl")
            assert a[a.index("--tasks") + 1].endswith("real-v3-excl.jsonl")
            assert a[a.index("--out") + 1] == str(rig.runs / "remote-gpu-tower-ud-q4_k_xl")
            assert not c["env_has_key"]
        prog = json.loads((rig.runs / "state" / "remote-progress.json").read_text())
        assert prog["state"] == "finished" and prog["remote"]["waits"] == 1
        logs = "".join(p.read_text() for p in (rig.runs / "logs").glob("*.log"))
        assert KEY not in logs and KEY not in cp.stdout + cp.stderr
        notes = (rig.state / "notify.log").read_text()
        assert "remote gen started" in notes and KEY not in notes
    finally:
        rig.close()


def test_the_remote_runner_refuses_a_key_file_others_can_read(tmp_path):
    rig, key, env = _remote_rig(tmp_path)
    try:
        key.chmod(0o644)
        until = (dt.datetime.now(ET) + dt.timedelta(seconds=60)).isoformat(timespec="seconds")
        cp = subprocess.run([sys.executable, "-I", str(REPO / "scripts/distill/mini_window/gen_runner.py"),
                             "--conf", str(rig.conf), "remote", "--execute", "--until", until], env=env,
                            capture_output=True, text=True, timeout=120)
        assert cp.returncode == 2 and "must be 0600" in cp.stdout
        assert not (rig.state / "remote-calls.jsonl").exists()
    finally:
        rig.close()


def test_remote_setup_pulls_the_key_privately_and_pushes_the_pinned_template(tmp_path, monkeypatch):
    sys.path.insert(0, str(_DIR / "mini_window"))
    import gen_runner as G  # noqa: E402

    tmpl = tmp_path / "base-chat_template.jinja"
    tmpl.write_text("{{ the base template }}\n")
    calls: list[tuple[list[str], Any]] = []

    def fake_run(argv: list[str], **kw: Any) -> Any:
        calls.append((argv, kw.get("input")))
        return subprocess.CompletedProcess(argv, 0, stdout=(KEY + "\n").encode() if "cat --" in argv[-1] else b"",
                                           stderr=b"")

    dest = tmp_path / "sec" / "teacher-4090.key"
    c = {**G.GEN_DEFAULTS, "GEN_REMOTE_KEY_FILE": str(dest), "GEN_REMOTE_TEMPLATE_SRC": str(tmpl),
         "GEN_REMOTE_SSH": "tower@gpu-tower"}
    assert G.cmd_remote_setup({**c, "GEN_REMOTE_SSH": ""}, True, out=lambda _s: None, run=fake_run) == 2
    out: list[str] = []
    assert G.cmd_remote_setup(c, True, out=out.append, run=fake_run) == 1, "an unpinned template is refused"
    assert not calls and not dest.exists()
    monkeypatch.setattr(G, "REMOTE_TEMPLATE_SHA256", hashlib.sha256(tmpl.read_bytes()).hexdigest())
    assert G.cmd_remote_setup(c, True, out=out.append, run=fake_run) == 0, out
    assert dest.read_text().strip() == KEY and (dest.stat().st_mode & 0o777) == 0o600
    assert (dest.parent.stat().st_mode & 0o777) == 0o700
    pull, push = calls
    assert pull[0][:3] == ["ssh", "-o", "BatchMode=yes"] and pull[0][-1] == "cat -- distill/secrets/teacher-serve.key"
    assert push[1] == tmpl.read_bytes() and "distill/teacher-serve/chat_template.jinja" in push[0][-1]
    assert all(KEY not in " ".join(a) for a, _ in calls) and KEY not in "\n".join(out)


# ── teacher_ab.py: arm B on the remote teacher; mini_ab.py's flag ─────────────


def test_a_remote_arm_b_records_its_server_differences_and_the_report_decides_with_them(tmp_path, monkeypatch):
    srv = KeyedServer(slots=2)
    try:
        paths = ab.Paths(str(tmp_path / "ab"))
        paths.out.mkdir(parents=True)
        conf = {"model_name": "qwen3.8-27b", "judge_model": None, "judge_seed": 7, "judge_max_chars": 30000,
                "allow_lan": False, "concurrency": 1, "min_task_seconds": 1.0, "max_tokens": 512,
                "max_tool_iterations": 4, "tool_timeout": 10.0, "task_timeout": 60.0, "request_timeout": 10.0,
                "compaction_history_tokens": 1500, "system_prompt": None, "jail": "auto",
                "teacher_id": "*qwen3*8*27b*", "judge_id": "*qwen3*8*27b*", "sampling": {"temperature": 0.7},
                "ctx_may_differ": True, "labels": {"A": "IQ3_XXS", "B": "UD-Q4_K_XL"}, "max_consecutive_errors": 3,
                "threshold_pts": 5, "generator": tg.GENERATOR_VERSION}
        a_server = {"model_path": "/m/iq3.gguf", "model_file": "iq3.gguf", "n_ctx": 57344, "total_slots": 2,
                    "chat_template_sha": "aaaa", "sampler_sha": "same", "sampler": {}}
        paths.arm("A").write_text(json.dumps({"task_id": "t1", "server": a_server, "served": "iq3"}) + "\n")
        remote = {"teacher_key_file": key_file(tmp_path), "teacher_host": "gpu-tower", "teacher_quant": "UD-Q4_K_XL",
                  "teacher_sha256": SHA, "tool_host": tg.local_hostname(), "jail": "require"}
        monkeypatch.setattr(tg, "resolve_jail", lambda mode: "bwrap")
        settings = ab.remote_settings(conf, paths, srv.url, remote)
        settings._remote_checked, settings._remote_url = srv.url, srv.url
        monkeypatch.setattr(ab, "remote_settings", lambda *a, **k: settings)
        monkeypatch.setattr(ab, "load_tasks", lambda p: [])
        with pytest.raises(ab.InputError, match="record-server-diff"):
            asyncio.run(ab.run_arm(paths, conf, "B", srv.url, None, remote, record_diff=False))
        res = asyncio.run(ab.run_arm(paths, conf, "B", srv.url, None, remote, record_diff=True))
        # every keyed request carried our key; the only bare ones are the key-enforcement probes of /props
        assert res["served"] == ALIAS and {x for x in srv.auth_seen if x} == {f"Bearer {KEY}"}
        server = asyncio.run(ab.server_props(srv.url, 5, settings))
        assert ab.server_differences(server, a_server, True) == ["chat_template_sha", "sampler_sha"]
        # an open server is refused for a remote arm
        srv.open = True
        with pytest.raises(ab.ServerError, match="without a key"):
            asyncio.run(ab.server_props(srv.url, 5, settings))
    finally:
        srv.close()
    with pytest.raises(ab.InputError, match="--teacher-key-file"):
        ab.remote_kwargs(ab.parse_args(["arm", "--out", "x", "--arm", "B", "--url", "u", "--teacher-host", "h"]))
    with pytest.raises(ab.InputError, match="--teacher-key-file"):
        ab.remote_kwargs(ab.parse_args(["arm", "--out", "x", "--arm", "B", "--url", "u", "--record-server-diff"]))


def test_mini_ab_with_arm_b_remote_runs_arm_a_only_and_never_reads_the_q4(tmp_path):
    MA = _load("_remote_mini_ab_tests", REPO / "tests" / "test_distill_mini_ab.py")
    nt, host, abx, g, out = MA.night(tmp_path, AB_ARM_B="remote")
    Path(nt.c["AB_Q4_MODEL"]).unlink()          # not needed: arm B is not run here
    nt.layers = None
    rc = nt.run()
    assert rc in (0, 1)
    assert "AB_ARM_B=remote" in nt.q4_refused
    assert {s["kind"] for s in abx.steps} == {"arm-A"}
    assert not [u for _, u in MA._systemctl(host, "start") if u == MA.Q4P]
    sel = nt.ab.select_args()
    assert sel[sel.index("--label-b") + 1] == "UD-Q4_K_XL"
    with pytest.raises(ValueError, match="local or remote"):
        MA.AB.arm_b_remote({"AB_ARM_B": "elsewhere"})


# ── tower/serve_teacher.sh ────────────────────────────────────────────────────

SERVE = _DIR / "tower" / "serve_teacher.sh"


def _serve_rig(tmp_path: Path, *, tail_addr: str = "100.64.0.39", free_mib: int = 24000,
               business: str = "inactive") -> tuple[dict[str, str], Path]:
    stub = tmp_path / "bin"
    stub.mkdir(parents=True)
    (stub / "ip").write_text(f'#!/bin/sh\necho "5: tailscale0    inet {tail_addr}/32 scope global tailscale0"\n')
    # the 4090's card: 24,564 total, 491 reserved by the driver (never shown as free)
    (stub / "nvidia-smi").write_text(f'#!/bin/sh\ncase "$*" in *memory.reserved*) echo "24564, 491" ;; '
                                     f'*) echo {free_mib} ;; esac\n')
    (stub / "systemctl").write_text(f'#!/bin/sh\n[ "{business}" = active ]\n')
    for f in stub.iterdir():
        f.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 60)
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps({"files": [{"sha256": SHA}]}, indent=1))
    tmpl = tmp_path / "chat_template.jinja"
    tmpl.write_text("{{ base }}\n")
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\nexit 0\n")
    server.chmod(0o755)
    key = tmp_path / "secrets" / "teacher-serve.key"
    env = {"PATH": f"{stub}:/usr/bin:/bin", "HOME": str(tmp_path), "TEACHER_LLAMA_SERVER": str(server),
           "TEACHER_MODEL": str(model), "TEACHER_MODEL_BYTES": "64", "TEACHER_MANIFEST": str(manifest),
           "TEACHER_TEMPLATE": str(tmpl),
           "TEACHER_TEMPLATE_SHA256": hashlib.sha256(tmpl.read_bytes()).hexdigest(),
           "TEACHER_KEY_FILE": str(key)}
    return env, key


def _serve(env: dict[str, str], *args: str, **extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SERVE), *args], env={**env, **extra}, capture_output=True, text=True,
                          timeout=60)


def test_serve_teacher_binds_the_tailnet_address_only(tmp_path):
    env, key = _serve_rig(tmp_path)
    assert _serve(env, "keygen").returncode == 0
    k = key.read_text().strip()
    assert len(k) == 64 and (key.stat().st_mode & 0o777) == 0o600
    ok = _serve(env, "exec", SERVE_TEACHER_DRY="1")
    assert ok.returncode == 0, ok.stderr
    argv = ok.stdout.split()
    assert argv[argv.index("--host") + 1] == "100.64.0.39" and argv[argv.index("--port") + 1] == "18094"
    assert argv[argv.index("--api-key-file") + 1] == str(key) and "--api-key" not in argv
    assert k not in ok.stdout + ok.stderr
    assert "--no-mmproj" in argv and "--no-slots" in argv and "--jinja" in argv and "--no-kv-unified" in argv
    assert argv[argv.index("--parallel") + 1] == "4" and argv[argv.index("-c") + 1] == str(4 * 32768)
    assert argv[argv.index("--chat-template-file") + 1] == env["TEACHER_TEMPLATE"]
    assert argv[argv.index("-ctk") + 1] == "q8_0" and argv[argv.index("--reasoning-budget") + 1] == "2048"
    for bind, why in (("0.0.0.0", "tailnet address only"), ("127.0.0.1", "tailnet address only"),
                      ("192.168.1.20", "not a tailnet address"), ("100.64.0.40", "does not hold it"),
                      ("10.0.0.5", "not a tailnet address")):
        r = _serve(env, "exec", SERVE_TEACHER_DRY="1", TEACHER_BIND=bind)
        assert r.returncode == 1 and why in r.stderr, (bind, r.stderr)
    env2, _ = _serve_rig(tmp_path / "lan", tail_addr="192.168.1.20")
    _serve(env2, "keygen")
    r = _serve(env2, "check")
    assert r.returncode == 1 and "not a tailnet address" in r.stderr


def test_serve_teacher_refuses_a_loose_key_the_business_server_and_too_little_vram(tmp_path):
    env, key = _serve_rig(tmp_path, business="active", free_mib=20000)
    r = _serve(env, "check")
    assert r.returncode == 1 and "no key file" in r.stderr
    _serve(env, "keygen")
    key.chmod(0o644)
    r = _serve(env, "check")
    assert "must be 0600" in r.stderr
    assert "llama-server.service is active" in r.stderr
    assert "only 20000 MiB VRAM free, needs 23153" in r.stderr   # 22,553 projected (whole MiB) + 600
    assert key.read_text().strip() not in r.stdout + r.stderr
    plan = _serve(env, "plan")
    assert "slots 4 x ctx 32768: projected 22553 MiB" in plan.stdout


def test_serve_teacher_clears_llama_servers_environment_before_the_exec(tmp_path):
    """LLAMA_ARG_TOOLS / LLAMA_ARG_AGENT from the user manager would start llama-server's built-in tools
    (exec_shell_command) unjailed on the 4090; LLAMA_API_KEY would add a second key; AIP_MODE moves the
    port and routes. None of them reaches the server (it starts with `env -i` and a short allow-list), the
    command line turns the tools, the agent and MCP off explicitly (llama.cpp applies it after its env and
    config files), and `check` names the cleared variables without their values."""
    env, key = _serve_rig(tmp_path)
    _serve(env, "keygen")
    dump = tmp_path / "server-env.txt"
    Path(env["TEACHER_LLAMA_SERVER"]).write_text(
        f'#!/bin/sh\nenv > "{dump}"\nfor a in "$@"; do echo "ARG:$a" >> "{dump}"; done\nexit 0\n')
    second = "second" + "-value" * 5      # a stand-in, built so it is no literal
    hostile = {"LLAMA_ARG_TOOLS": "all", "LLAMA_ARG_AGENT": "1", "LLAMA_ARG_MCP_SERVERS_JSON": "{}",
               "LLAMA_ARG_MCP_SERVERS_CONFIG": "/tmp/mcp.json", "LLAMA_ARG_TOOLS_RUNTIME": "none",
               "LLAMA_ARG_UI_MCP_PROXY": "1", "LLAMA_ARG_STATIC_PATH": "/", "LLAMA_ARG_ENDPOINT_PROPS": "1",
               "LLAMA_API_KEY": second, "AIP_MODE": "PREDICTION",
               "HF_TOKEN": "hf-token-value"}
    chk = _serve(env, "check", **hostile)
    assert chk.returncode == 0, chk.stderr
    secrets = (second, "hf-token-value")
    assert all(name in chk.stderr for name in hostile)
    assert not any(v in chk.stdout + chk.stderr for v in secrets)
    run = _serve(env, "exec", **hostile, UNRELATED_VAR="x", CUDA_VISIBLE_DEVICES="0")
    assert run.returncode == 0, run.stderr
    seen = dump.read_text()
    assert not any(f"{name}=" in seen for name in hostile) and not any(v in seen for v in secrets)
    names = {line.split("=", 1)[0] for line in seen.splitlines() if "=" in line and not line.startswith("ARG:")}
    assert names <= {"HOME", "PATH", "XDG_CONFIG_HOME", "CUDA_VISIBLE_DEVICES", "PWD", "SHLVL", "_"}, names
    assert "UNRELATED_VAR" not in names and "CUDA_VISIBLE_DEVICES" in names
    cfg_home = tmp_path / "distill" / "teacher-serve" / "llama-config-empty"
    assert f"XDG_CONFIG_HOME={cfg_home}" in seen and cfg_home.is_dir() and not any(cfg_home.iterdir())
    args = [line[4:] for line in seen.splitlines() if line.startswith("ARG:")]
    for flag in ("--no-webui", "--no-agent", "--no-ui-mcp-proxy", "--api-key-file"):
        assert flag in args, flag
    for flag in ("--mcp-servers-config", "--mcp-servers-json"):
        assert args[args.index(flag) + 1] == "", flag
    assert not any(a in args for a in ("--tools", "--agent", "--tools-runtime", "--api-key", "--props",
                                       "--metrics", "--path", "--ui-mcp-proxy", "--webui"))
    dry = _serve(env, "exec", SERVE_TEACHER_DRY="1")
    assert "--no-webui" in dry.stdout.split() and "--no-agent" in dry.stdout.split()


def test_serve_teacher_refuses_while_a_llama_cpp_config_file_exists(tmp_path):
    """llama.cpp's config.ini (system-wide, or under XDG_CONFIG_HOME) can set the same options as the
    environment, api-key included (the command line cannot take a key away): the check refuses either."""
    env, _ = _serve_rig(tmp_path)
    _serve(env, "keygen")
    system_cfg = tmp_path / "etc-llama.cpp" / "config.ini"
    env["TEACHER_SYSTEM_LLAMA_CONFIG"] = str(system_cfg)
    assert _serve(env, "check").returncode == 0
    system_cfg.parent.mkdir()
    system_cfg.write_text("[*]\napi-key = another\n")
    r = _serve(env, "exec", SERVE_TEACHER_DRY="1")
    assert r.returncode == 1 and "system-wide config" in r.stderr and "another" not in r.stderr, r.stderr
    system_cfg.unlink()
    user_cfg = tmp_path / "distill" / "teacher-serve" / "llama-config-empty" / "llama.cpp" / "config.ini"
    user_cfg.parent.mkdir(parents=True)
    user_cfg.write_text("[*]\ntools = all\n")
    r = _serve(env, "check")
    assert r.returncode == 1 and "config directory must stay empty" in r.stderr, r.stderr


def test_serve_teacher_reports_a_failing_nvidia_smi_and_a_key_file_of_comments(tmp_path):
    """Under set -e these used to exit at once with nothing printed (the unit failed with an empty journal)."""
    env, key = _serve_rig(tmp_path)
    _serve(env, "keygen")
    stub = tmp_path / "bin" / "nvidia-smi"
    stub.write_text("#!/bin/sh\nexit 9\n")
    r = _serve(env, "check")
    assert r.returncode == 1 and "nvidia-smi did not report free VRAM" in r.stderr, (r.returncode, r.stderr)
    stub.write_text('#!/bin/sh\ncase "$*" in *memory.reserved*) echo "24564, 491" ;; *) echo 24000 ;; esac\n')
    key.write_text("# only a comment\n\n   \n")
    key.chmod(0o600)
    r = _serve(env, "check")
    assert r.returncode == 1 and "holds no key of at least 32 characters" in r.stderr, (r.returncode, r.stderr)
    r = _serve(env, "exec", SERVE_TEACHER_DRY="1")
    assert r.returncode == 1 and "holds no key" in r.stderr


def test_serve_teacher_plans_against_the_usable_vram_and_refuses_a_shape_the_card_cannot_hold(tmp_path):
    """24,564 MiB total - 491 reserved = 24,073 usable: 4 x 32K leaves ~1.5 GB; 5 x 32K needs 24,391 with the
    margin, which this card never reports free, so it is refused as not fitting (not as 'too little free')."""
    env, _ = _serve_rig(tmp_path, free_mib=24073)
    _serve(env, "keygen")
    plan = _serve(env, "plan")
    assert "slots 4 x ctx 32768: projected 22553 MiB of 24073 usable (spare 1520): fits" in plan.stdout, plan.stdout
    assert "slots 5 x ctx 32768: projected 23791 MiB of 24073 usable (spare 282): does NOT fit" in plan.stdout
    assert _serve(env, "check").returncode == 0
    r = _serve(env, "check", TEACHER_SLOTS="5")
    assert r.returncode == 1 and "need 24391 MiB" in r.stderr and "usable 24073" in r.stderr, r.stderr
    assert "only 24073 MiB VRAM free" not in r.stderr
    ok = _serve(env, "check", TEACHER_SLOTS="5", TEACHER_CTX_PER_SLOT="24576")
    assert ok.returncode == 0, ok.stderr
    # no nvidia-smi answer for total/reserved: TEACHER_USABLE_MIB
    (tmp_path / "bin" / "nvidia-smi").write_text("#!/bin/sh\necho 24073\n")
    plan = _serve(env, "plan", TEACHER_USABLE_MIB="24000")
    assert "projected 22553 MiB of 24000 usable (spare 1447)" in plan.stdout


# ── the checked address is the one every request goes to ─────────────────────


def test_the_checked_tailnet_address_is_pinned_for_the_whole_run(tmp_path, monkeypatch):
    addrs: dict[str, list[str]] = {"gpu-tower": ["fd7a:115c:a1e0::39", "100.64.0.39"], "v6": ["fd7a:115c:a1e0::1"]}
    looked: list[str] = []

    def resolve(h: str) -> list[str]:
        looked.append(h)
        return addrs[h]

    monkeypatch.setattr(tg, "_host_addresses", resolve)
    monkeypatch.setattr(tg, "is_local_address", lambda a: False)
    assert tg.check_tailnet_endpoint("http://gpu-tower:18094", pin=True) == "http://100.64.0.39:18094"
    assert tg.check_tailnet_endpoint("http://v6:18094/", pin=True) == "http://[fd7a:115c:a1e0::1]:18094"
    assert tg.check_tailnet_endpoint("http://gpu-tower:18094") == "http://gpu-tower:18094"
    s = remote_settings(tmp_path, "http://gpu-tower:18094")
    s._remote_checked, s._remote_url = None, ""
    looked.clear()
    assert s.teacher_endpoint == "http://100.64.0.39:18094" and s.judge_endpoint == s.teacher_endpoint
    # the name later resolves elsewhere (MagicDNS down, a LAN search domain): never looked up again
    addrs["gpu-tower"] = ["192.168.1.20"]
    for _ in range(3):
        assert s.teacher_endpoint == "http://100.64.0.39:18094"
    assert looked == ["gpu-tower"]
    assert tg.teacher_provider(s)._base_url == "http://100.64.0.39:18094"


def test_no_keyed_request_goes_through_a_proxy(tmp_path, monkeypatch):
    """The provider's httpx clients honour HTTP(S)_PROXY / ALL_PROXY (trust_env): through a proxy, the
    Authorization header and the trace text would leave the two hosts. Every entry point drops them
    (isolate_state_dirs: teacher_gen main, teacher_ab arm and judge), and the remote endpoint check drops
    them again before the first keyed request."""
    from prometheus.providers.llama_cpp import LlamaCppProvider

    dead = "http://127.0.0.1:9"  # nothing listens: a request sent through it fails
    srv = KeyedServer()
    try:
        for var in tg.PROXY_ENV:
            monkeypatch.setenv(var, dead)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        keyed = LlamaCppProvider(base_url=srv.url, headers={"Authorization": f"Bearer {KEY}"})
        # the control: with the proxy set, the request does go to it (so the assertions below bite)
        assert asyncio.run(keyed.detect_loaded_model()) is None  # it logs the failed connection, returns None
        tg.isolate_state_dirs(tmp_path / "work")
        assert not any(v in os.environ for v in tg.PROXY_ENV) and os.environ["NO_PROXY"] == "*"
        assert asyncio.run(keyed.detect_loaded_model()) == ALIAS
        # the remote endpoint check drops them too (an entry point that forgot isolate_state_dirs)
        for var in tg.PROXY_ENV:
            monkeypatch.setenv(var, dead)
        monkeypatch.setattr(tg, "_host_addresses", lambda h: ["100.64.0.39"])
        monkeypatch.setattr(tg, "is_local_address", lambda a: False)
        s = remote_settings(tmp_path, "http://gpu-tower:18094")
        s._remote_checked, s._remote_url = None, ""
        assert s.teacher_endpoint == "http://100.64.0.39:18094"
        assert not any(v in os.environ for v in tg.PROXY_ENV)
    finally:
        srv.close()


# ── the tail of a remote run ──────────────────────────────────────────────────


def test_a_task_that_errors_at_the_tail_while_the_teacher_is_gone_reruns_before_the_run_ends(tmp_path, monkeypatch):
    """Worker 0 runs out of tasks while worker 1's task is still in flight; the 4090's teacher then stops
    (training) and that task ends in error. It used to be lost: worker 0 had already ended the run
    'no_tasks', so the pause's monitor quit at once and the runner took the run as done."""
    monkeypatch.setattr(tg, "REMOTE_BACKOFF_S", (0.05, 0.2))
    monkeypatch.setattr(tg, "REMOTE_WAIT_POLL_S", 0.02)
    monkeypatch.setattr(tg, "REMOTE_HEALTH_TIMEOUT_S", 1.0)
    srv = KeyedServer(slots=2)
    attempts: dict[str, int] = {}

    async def fake_process(task: Any, settings: Any, worker: int) -> tuple[str, list, list[str], dict[str, Any]]:
        attempts[task.id] = attempts.get(task.id, 0) + 1
        if task.id == "t-00001" and attempts[task.id] == 1:
            await asyncio.sleep(0.4)          # the others finish and worker 0 finds no next task meanwhile
            srv.stop()                         # the teacher stops for training
            threading.Timer(0.4, srv.start).start()
            return "error", [], ["transport_error"], {"error": "ConnectError"}
        await asyncio.sleep(0.02)
        return "rejected", [], ["test_done"], {"duration_s": 0.02}

    monkeypatch.setattr(tg, "process", fake_process)
    try:
        s = remote_settings(tmp_path, srv.url, concurrency=2, max_consecutive_errors=2)
        s.teacher_served = ALIAS
        tasks = [H.task(id=f"t-{i:05d}", prompt=f"tail task number {i}") for i in range(4)]
        store = tg.Store(s.out, s.name, s.fingerprint(), s.coding_cap)
        report = asyncio.run(asyncio.wait_for(tg.run(s, tg.FixedGenerator(tasks), store, None), 60))
    finally:
        srv.close()
    assert report["stop"] == "no_tasks" and report["remote_pauses"] == 1, report
    assert attempts["t-00001"] == 2, attempts
    assert {t.id for t in tasks} == {tid for tid, e in store.status.items() if e["status"] == "rejected"}


def test_serve_teacher_refuses_to_start_when_an_edit_would_turn_tools_agent_or_mcp_on(tmp_path):
    """check_no_tools runs on the FINAL argv and environment: a later edit of server_argv that adds an
    enabling flag, drops an off switch or gives MCP a config, or one of SERVER_ENV_KEEP that passes a
    LLAMA_* variable through, makes check and exec refuse (exec never reaches llama-server)."""
    env, _ = _serve_rig(tmp_path)
    _serve(env, "keygen")
    ran = tmp_path / "server-ran.txt"
    Path(env["TEACHER_LLAMA_SERVER"]).write_text(f'#!/bin/sh\necho ran > "{ran}"\n')
    src = SERVE.read_text()
    assert _serve(env, "check").returncode == 0
    edits = {
        "--tools": src.replace("--no-webui --no-agent", "--no-webui --no-agent --tools all"),
        "--agent": src.replace("--no-webui --no-agent", "--no-webui --agent"),
        "lacks --no-agent": src.replace("--no-webui --no-agent", "--no-webui"),
        "lacks --no-ui-mcp-proxy": src.replace("--no-agent --no-ui-mcp-proxy --mcp", "--no-agent --mcp"),
        "sets --mcp-servers-json": src.replace("--mcp-servers-json ''", "--mcp-servers-json '{\"x\":{}}'"),
        "lacks --mcp-servers-config": src.replace("--no-ui-mcp-proxy --mcp-servers-config ''", "--no-ui-mcp-proxy"),
        "--api-key": src.replace('--api-key-file "$TEACHER_KEY_FILE"',
                                 '--api-key-file "$TEACHER_KEY_FILE" --api-key=inline'),
        "--webui": src.replace("--no-webui --no-agent", "--webui --no-agent"),
        "environment would hold LLAMA_ARG_TOOLS": src.replace(
            "SERVER_ENV_KEEP=(HOME PATH", "SERVER_ENV_KEEP=(LLAMA_ARG_TOOLS HOME PATH").replace(
            "      LLAMA_*|AIP_*|HF_*|HUGGING_FACE_*) SCRUBBED+=", "      NO_SUCH_*) SCRUBBED+="),
    }
    for why, text in edits.items():
        assert text != src and "TOOL_OFF_FLAGS=(--no-agent --no-ui-mcp-proxy --no-webui --no-slots)" in text, why
        script = tmp_path / f"edited-{abs(hash(why))}.sh"
        script.write_text(text)
        extra = {"LLAMA_ARG_TOOLS": "all"} if "LLAMA_ARG_TOOLS" in why else {}
        for cmd in ("check", "exec"):
            r = subprocess.run(["bash", str(script), cmd], env={**env, **extra}, capture_output=True, text=True,
                               timeout=60)
            assert r.returncode == 1 and "REFUSED" in r.stderr and why in r.stderr, (why, cmd, r.stderr)
        assert not ran.exists(), why


def test_serve_teacher_verify_checks_the_running_server_and_never_shows_the_key(tmp_path):
    """The unit's ExecStartPost: /health up, the key enforced and ours accepted, the slots as asked, no MCP
    proxy, web UI or extra endpoints, GET /tools refused. Any failure exits 1 (systemd then stops the unit)."""
    srv = KeyedServer()
    try:
        env, _ = _serve_rig(tmp_path)
        kf = key_file(tmp_path)
        env.update({"TEACHER_BIND": "127.0.0.1", "TEACHER_PORT": str(srv.port), "TEACHER_KEY_FILE": str(kf),
                    "TEACHER_PYTHON": sys.executable, "TEACHER_VERIFY_WAIT_S": "2"})
        ok = _serve(env, "verify")
        assert ok.returncode == 0 and "verified" in ok.stdout, ok.stderr
        assert set(srv.auth_seen) >= {None, f"Bearer {KEY}"}
        for attr, value, why in (("tools_status", 200, "built-in tools or MCP servers are ON"),
                                 ("tools_status", 500, "403 expected"),
                                 ("cors_proxy", True, "cors_proxy_enabled=true"),
                                 ("open", True, "the key is not enforced"),
                                 ("key", "9" * 64, "refused this host's own key"),
                                 ("slots", 3, "3 slots, not the 4 asked for"),
                                 ("loading", True, "did not answer 200 within 2 s")):
            before = getattr(srv, attr)
            setattr(srv, attr, value)
            try:
                r = _serve(env, "verify")
            finally:
                setattr(srv, attr, before)
            assert r.returncode == 1 and why in r.stderr, (attr, r.stderr)
            assert KEY not in r.stdout + r.stderr
        assert KEY not in ok.stdout + ok.stderr
    finally:
        srv.close()
    unit = (SERVE.parent / "distill-teacher-serve.service").read_text()
    assert "ExecStartPost=%h/distill/teacher-serve/serve_teacher.sh verify" in unit
    assert "TimeoutStartSec=600" in unit
