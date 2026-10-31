"""Tests for scripts/distill/teacher_gen.py (Phase 2b-style generation with a local teacher).

A fake OpenAI-compatible teacher (an HTTP server on loopback that speaks the
llama.cpp streaming wire) drives the real agent loop. No real model, no
network beyond 127.0.0.1, no live Prometheus state.
"""

from __future__ import annotations

import asyncio
import collections
import importlib.util
import json
import os
import re
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

_DIR = Path(__file__).resolve().parent.parent / "scripts" / "distill"


def _load(name: str, file: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _DIR / file)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tg = _load("distill_teacher_gen", "teacher_gen.py")
ev = _load("distill_build_eval_sets", "build_eval_sets.py")


# ── the fake teacher ───────────────────────────────────────────────────────────


def _last_user(messages: list[dict[str, Any]]) -> tuple[int, str]:
    for k in range(len(messages) - 1, -1, -1):
        m = messages[k]
        if m.get("role") == "user" and not str(m.get("content") or "").startswith("[Compacted summary"):
            return k, tg.content_text(m.get("content"))
    return 0, ""


def echo_reply(body: dict[str, Any]) -> dict[str, Any]:
    """Final answer quoting every tool result of this turn and the prompt."""
    msgs = body["messages"]
    k, prompt = _last_user(msgs)
    results = [tg.content_text(m.get("content")) for m in msgs[k + 1:] if m.get("role") == "tool"]
    return {"content": "Here is what I found.\n" + "\n".join(results) + "\n\nRequest: " + prompt[-3000:]}


def scripted(plans: dict[str, list[Any]]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """A policy: the plan whose key is in the last user prompt; step = assistant
    turns since that prompt. A step is ("call", name, args), ("calls", [(name, args), ...]),
    ("text", str) or "echo"."""

    def policy(body: dict[str, Any]) -> dict[str, Any]:
        msgs = body["messages"]
        k, prompt = _last_user(msgs)
        step = sum(1 for m in msgs[k + 1:] if m.get("role") == "assistant")
        plan = next((v for key, v in plans.items() if key in prompt), ["echo"])
        action = plan[step] if step < len(plan) else "echo"
        if action == "echo":
            return echo_reply(body)
        if action[0] == "call":
            return {"tool_calls": [(action[1], action[2])]}
        if action[0] == "calls":
            return {"tool_calls": list(action[1])}
        return {"content": action[1]}

    return policy


class FakeTeacher:
    def __init__(self, policy: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
                 verdict: dict[str, Any] | None = None) -> None:
        self.policy = policy or scripted({})
        self.verdict = verdict or {"verdict": "pass", "completed": True, "tools_ok": True,
                                   "grounded": True, "recovery_ok": None, "reason": "fine"}
        self.requests: list[dict[str, Any]] = []
        self.kinds: collections.Counter = collections.Counter()
        self.fail_with: int | None = None
        #: (status, body) answers for the next agent requests, in order (then the policy)
        self.fail_agent: list[tuple[int, Any]] = []
        teacher = self

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

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/v1/models":
                    self._json(200, {"object": "list", "data": [{"id": "fake-teacher"}]})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                teacher.requests.append(body)
                assert "authorization" not in {k.lower() for k in self.headers.keys()}
                if teacher.fail_with:
                    self._json(teacher.fail_with, {"error": "boom"})
                    return
                msgs = body.get("messages") or []
                if msgs and str(msgs[0].get("content", "")).startswith("You are a strict grader"):
                    teacher.kinds["judge"] += 1
                    reply: dict[str, Any] = {"content": json.dumps(teacher.verdict)}
                elif any("You are a conversation compactor" in tg.content_text(m.get("content"))
                         for m in msgs):
                    teacher.kinds["summary"] += 1
                    reply = {"content": "Earlier: the user read several files and gave instructions."}
                else:
                    teacher.kinds["agent"] += 1
                    if teacher.fail_agent:
                        code, err = teacher.fail_agent.pop(0)
                        self._json(code, err)
                        return
                    reply = teacher.policy(body)
                if body.get("stream"):
                    self._stream(reply)
                else:
                    self._json(200, {"model": "fake-teacher", "choices": [
                        {"message": {"role": "assistant", "content": reply.get("content")},
                         "finish_reason": "stop"}]})

            def _stream(self, reply: dict[str, Any]) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks: list[dict[str, Any]] = []
                if reply.get("content"):
                    chunks.append({"choices": [{"delta": {"content": reply["content"]}}]})
                for i, (name, args) in enumerate(reply.get("tool_calls") or []):
                    chunks.append({"choices": [{"delta": {"tool_calls": [{
                        "index": i, "id": f"call_{len(teacher.requests)}_{i}",
                        "function": {"name": name, "arguments": json.dumps(args)}}]}}]})
                finish = "tool_calls" if reply.get("tool_calls") else "stop"
                chunks.append({"choices": [{"delta": {}, "finish_reason": finish}]})
                chunks.append({"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
                for c in chunks:
                    c["model"] = "fake-teacher"
                    self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "FakeTeacher":
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture(autouse=True)
def _restore_state_env(monkeypatch):
    """``cmd_run`` points the Prometheus state roots at its own work dir; undo it after each test."""
    for var in ("PROMETHEUS_LOGS_DIR", "PROMETHEUS_CONFIG_DIR", "PROMETHEUS_DATA_DIR", "PROMETHEUS_ENV_FILE"):
        monkeypatch.setenv(var, os.environ.get(var, "")) if os.environ.get(var) else monkeypatch.delenv(var, raising=False)


@pytest.fixture
def no_outbound(monkeypatch):
    """Any connection that is not to loopback fails the test."""
    real = socket.socket.connect

    def guarded(self, address):  # noqa: ANN001
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"outbound connection attempted to {address!r}")
        return real(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


def settings_for(tmp_path: Path, url: str, **kw: Any) -> Any:
    return tg.Settings(out=tmp_path / "out", teacher_url=url, sandbox_root=tmp_path / "sbx",
                       request_timeout=20.0, progress=False, **kw)


def task(**kw: Any) -> Any:
    base = dict(id="t-00001", kind="agentic", domain="other", family="test", prompt="hello")
    base.update(kw)
    t = tg.GenTask(**base)
    t.validate()
    return t


def run_and_verify(t: Any, s: Any) -> tuple[Any, bool, list[str], dict[str, Any]]:
    out = asyncio.run(tg.run_task(t, s))
    ok, reasons, record = tg.deterministic_checks(out, s)
    return out, ok, reasons, record


# ── end to end ─────────────────────────────────────────────────────────────────


def test_end_to_end_agentic_task_writes_a_verified_tagged_row(tmp_path, no_outbound):
    t = task(prompt="Look it up in the wiki: which host runs cobalt-intake?", domain="memory/wiki",
             fixtures={"wiki": {"cobalt-intake": "# cobalt-intake\n- Runs on: node-bravo\n"}},
             score={"expect_tool": "wiki_query", "expect_text_contains": "node-bravo"})
    with FakeTeacher(scripted({"cobalt-intake": [("call", "wiki_query", {"query": "cobalt-intake host"}),
                                                 "echo"]})) as fake:
        s = settings_for(tmp_path, fake.url)
        out, ok, reasons, record = run_and_verify(t, s)
        assert ok, reasons
        record["judge"] = asyncio.run(tg.judge(out, s))
        assert record["judge"]["pass"] and record["judge"]["kind"] == "self-judge"
        judge_req = next(r for r in fake.requests if not r.get("stream"))
        assert judge_req["temperature"] == 0 and judge_req["seed"] == tg.DEFAULT_JUDGE_SEED
        rows = tg.rows_for(out, s, record)
    assert len(rows) == 1
    row = rows[0]
    meta = row["_meta"]
    assert meta["source"] == "qwen38-27b-gym" and meta["model"] == tg.DEFAULT_MODEL_NAME
    assert meta["category"] == "memory/wiki" and meta["task_category"] == "agentic"
    assert meta["task_id"] == "t-00001" and meta["verdict"] == "pass"
    assert meta["verification"]["tool_success"] is True and meta["tool_name"] == "wiki_query"
    msgs = row["messages"]
    assert msgs[0]["role"] == "system" and "# Workspace" in msgs[0]["content"]
    assert msgs[-1]["role"] == "assistant" and msgs[-1]["tool_calls"][0]["function"]["name"] == "wiki_query"
    assert {t["function"]["name"] for t in row["tools"]} == tg.SANDBOX_TOOLS


# ── the sandbox ────────────────────────────────────────────────────────────────


def _env(tmp_path: Path, t: Any | None = None) -> tuple[Any, Any]:
    root = tmp_path / "ws"
    root.mkdir(exist_ok=True)
    env = tg.SandboxEnv(root, t or task())
    return env, tg.build_sandbox_registry(env)


def _call(env: Any, reg: Any, name: str, args: dict[str, Any]) -> Any:
    from prometheus.tools.base import ToolExecutionContext

    tool = reg.get(name)
    parsed = tool.input_model.model_validate(args)
    return asyncio.run(tool.execute(parsed, ToolExecutionContext(cwd=env.root, metadata={})))


def test_registry_is_the_allow_list_with_the_runtime_schemas(tmp_path):
    env, reg = _env(tmp_path)
    names = {t.name for t in reg.list_tools()}
    assert names == tg.SANDBOX_TOOLS
    # nothing that spawns agents, asks the user, downloads, drives a browser or speaks
    assert not names & {"agent", "sessions_spawn", "sessions_send", "ask_user", "download_file", "browser",
                        "tts", "dashboard", "image_generate", "github_search", "skill"}
    for t in reg.list_tools():
        assert (t.impl is not None) == (t.name in tg.MOCKED_TOOLS or t.name == "bash"), t.name
    snap = json.loads((_DIR / "eval_tool_schemas.json").read_text())["tools"]
    snap = {x["function"]["name"]: x["function"] for x in snap}
    shared = names & set(snap)
    assert {"bash", "memory", "cron_create", "web_fetch", "message", "read_file"} <= shared
    for name in shared:
        assert reg.get(name).to_openai_schema()["function"]["parameters"] == snap[name]["parameters"], name


def test_mocked_tools_never_run_the_real_tool(tmp_path, monkeypatch, no_outbound):
    env, reg = _env(tmp_path, task(fixtures={
        "wiki": {"cobalt-intake": "runs on node-bravo"}, "cron": [],
        "tasks": [{"id": "tabc1234", "type": "local_bash", "status": "running", "description": "x",
                   "command": "c", "return_code": None, "output": "o"}],
        "web": {"results": [{"title": "t", "url": "https://docs.x.example.org/a", "snippet": "s"}],
                "pages": {"https://docs.x.example.org/a": "page"}}}))
    for t in reg.list_tools():
        if t.name in tg.MOCKED_TOOLS:
            def boom(*a, **k):  # noqa: ANN001, ANN002, ANN003
                raise AssertionError("real tool executed")
            monkeypatch.setattr(type(t.inner), "execute", boom)
    calls = {
        "memory": {"operation": "add", "entry": "x likes y"}, "wiki_query": {"query": "cobalt-intake"},
        "wiki_compile": {}, "cron_create": {"name": "j", "schedule": "every day at noon", "command": "echo hi"},
        "cron_list": {}, "cron_delete": {"name": "j"}, "task_create": {"description": "d", "command": "echo"},
        "task_list": {}, "task_get": {"task_id": "tabc1234"}, "task_output": {"task_id": "tabc1234"},
        "task_stop": {"task_id": "tabc1234"}, "web_search": {"query": "x"},
        "web_fetch": {"url": "https://docs.x.example.org/a"},
        "message": {"platform": "slack", "content": "hi", "recipient": "#ops"},
        "sentinel_status": {}, "lcm_grep": {"query": "hello"},
    }
    assert set(calls) == tg.MOCKED_TOOLS
    for name, args in calls.items():
        res = _call(env, reg, name, args)
        assert not res.is_error, (name, res.output)
    assert env.outbox == [{"platform": "slack", "recipient": "#ops", "content": "hi"}]
    assert "Created cron job 'j' [0 12 * * *]" in _call(env, reg, "cron_create",
                                                        {"name": "j", "schedule": "every day at noon",
                                                         "command": "echo hi"}).output


REFUSED = [
    "curl https://example.com/x", "ssh node-alpha uptime", "python3 -c 'print(1)'", "rm -rf /",
    "cat /etc/passwd", "cat ~/.ssh/id_rsa", "ls ..", "cd / && ls", "echo $HOME", "echo `id`",
    "echo $(id)", "sleep 100 &", "(ls)", "echo hi > /tmp/escape.txt", "sed -i s/a/b/ f.txt",
    "awk 'BEGIN{system(\"id\")}'", "find . -delete", "find . -exec rm {} ;", "git push",
    "systemctl --user restart cobalt-intake", "systemctl --user status x && rm -rf .", "FOO=1 ls",
    "./run.sh", "xargs rm", "pip install requests", "cat <<EOF", "wget http://127.0.0.1/x",
    "cp -s /etc/passwd p", "ls\nrm -rf .", "nc -l 9999", "sed -n '1w out' f.txt", "env",
    "ls | sh", "grep -r x ../..", "tee /tmp/x",
    # a value attached to a short option is a path too
    "sort -o/tmp/x f.txt", "cp -t/tmp f.txt", "mv -t/tmp f.txt", "sort -ro/tmp/x f.txt", "grep -f/etc/passwd f.txt",
    "grep -f /etc/passwd f.txt", "date -f/etc/passwd", "du -X/etc/passwd .", "touch -r/etc/passwd f.txt",
    "tee -a/tmp/x", "ls -I/etc",
    # awk: options before the program, program files, includes, pipes, the argument list
    "awk -fprog.awk f.txt", "awk --file=prog.awk f.txt", "awk '@include \"prog.awk\"' f.txt",
    "awk -F , 'BEGIN{system(\"id\")}' f.txt", "awk -v x=1 'BEGIN{system(\"id\")}' f.txt",
    "awk -F , 'BEGIN{while((getline l < \"/etc/passwd\")>0) print l}' f.txt",
    "awk --source='BEGIN{system(\"id\")}' f.txt", "awk -E prog f.txt", "awk -W exec f.txt",
    "awk -v c=sh 'BEGIN{print \"id\" | c}'", "awk 'BEGIN{ARGV[1]=\"/etc/passwd\";ARGC=2}{print}'",
    "awk 'BEGIN{f=\"sys\" \"tem\"; @f(\"id\")}'", "awk -- 'BEGIN{system(\"id\")}'",
    # sed: every script form, no in-place or script files under any spelling
    "sed -e'1e id' p", "sed -ne'1e id' p", "sed --expression='1e id' f.txt", "sed -n -e p -e '1e id' f.txt",
    "sed --in s/a/b/ f.txt", "sed --fi=s.sed f.txt",
    # programs that run other programs or read file lists
    "sort -S 1K --compress-program=sh f.txt", "sort --compress sh f.txt", "sort --files0-from=list",
    "du --files0-from=list", "wc --files0=list", "file -f names", "sha256sum -c list",
    "jq -f/etc/passwd f.json", "jq -L /x '.'", "jq 'import \"x\" as y; .' f.json",
    "jq --rawfile x /etc/passwd '.'", "cp -rs a b", "cp --sym a b", "find . -files0-from l",
    # shell expansion that hides a path: braces (bash), wildcards matching . or .., tildes
    "cat {/etc/passwd,f.txt}", "cp f.txt {/tmp,}", "cat .*/x", "cat .?/x", "cat [.][.]/x", "cat ~/x",
    # bash-only operators dash would read as a background job
    "echo hi &> f.txt", "ls |& cat",
    # redirects in every form
    "cat < /etc/passwd", "cat 0</etc/passwd", "echo x 1>/tmp/y", "ls >&/tmp/z",
    "grep x -- /etc/passwd", "cat -- ../x",
]

ALLOWED = [
    "grep -rn 'TODO' notes/", "cut -d/ -f2 paths.txt", "sort -t, -k2,2n data.csv", "awk -F/ '{print $2}' p.txt",
    "awk -F, '$3 > 5 {print $1}' d.csv", "awk '/ERROR|WARN/ {print}' log", "awk '{s+=$1} END {print s/NR}' n",
    "awk 'NR==1 || NR==3' f", "awk -v n=2 'NR==n' f", "sed -n '1,5p' f", "sed -E 's#/x#/y#' f",
    "sed -n -e '/start/,/end/p' f", "jq -r '.items[] | .name' a.json", "jq --arg n x '.[$n]' a.json",
    "find notes -maxdepth 1 -name '*.md'", "wc -l notes/*.md", "tail -n +2 f", "date -u '+%H:%M'",
    "printf '%s\\n' a/b", "tr '/' '_' < f", "paste -sd, f", "column -t -s, f", "seq -s, 1 5", "grep -A2 -B1 x f",
    "grep -e '-x' f", "od -An -tx1 f", "sort f -o sorted.txt", "grep -rl --include='*.py' foo .",
    "cat f 2>&1 | head", "ls 2>/dev/null", "basename /usr/bin/env", "echo /etc/passwd", "cd notes && ls",
]


@pytest.mark.parametrize("cmd", ALLOWED)
def test_bash_still_allows_ordinary_text_work(tmp_path, cmd):
    env, _reg = _env(tmp_path)
    env.plan_shell(cmd, env.root)


@pytest.mark.parametrize("cmd", REFUSED)
def test_bash_refuses_anything_with_effects_outside_the_sandbox(tmp_path, cmd):
    env, _reg = _env(tmp_path)
    with pytest.raises(tg.SandboxRefusal):
        env.plan_shell(cmd, env.root)


def test_bash_runs_text_utilities_inside_the_sandbox(tmp_path, no_outbound):
    env, reg = _env(tmp_path)
    (env.root / "notes").mkdir()
    (env.root / "notes" / "a.md").write_text("alpha\nbeta\ngamma\n")
    res = _call(env, reg, "bash", {"command": "grep -c a notes/a.md && sed -n '2p' notes/a.md | tr a-z A-Z"})
    assert not res.is_error and res.output.split() == ["3", "BETA"]
    res = _call(env, reg, "bash", {"command": "echo done > out.txt 2>&1 && cat out.txt"})
    assert res.output == "done" and (env.root / "out.txt").read_text() == "done\n"
    res = _call(env, reg, "bash", {"command": "cat /etc/hosts"})
    assert res.is_error and res.output.startswith("Refused:")
    # runaway output is cut off, not buffered
    res = _call(env, reg, "bash", {"command": "seq 1 100000000"})
    assert res.output.endswith("...[truncated]...")
    assert [r.outcome for r in env.log] == ["ok", "ok", "refused", "error"]


def test_attached_option_values_and_awk_options_cannot_escape(tmp_path, monkeypatch):
    """The reviewer's escapes, run for real through the bash tool: nothing lands
    or runs outside the workspace, and each call is a refusal, not a success."""
    monkeypatch.setattr(tg, "resolve_jail", lambda mode: "none")   # the command checks alone
    env, reg = _env(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (env.root / "a.md").write_text("b\na\n")
    (env.root / "prog.awk").write_text(f'BEGIN{{system("echo escaped > {outside}/awk.txt")}}\n')
    for cmd in (f"sort -o{outside}/sorted.txt a.md", "awk -fprog.awk a.md",
                f"awk -F , 'BEGIN{{system(\"touch {outside}/flag\")}}' a.md",
                f"cat {{{outside}/secret.txt,a.md}}"):
        res = _call(env, reg, "bash", {"command": cmd})
        assert res.is_error and res.output.startswith("Refused:"), (cmd, res.output)
    assert not list(outside.iterdir())
    assert [r.outcome for r in env.log] == ["refused"] * 4


def test_shell_has_no_brace_expansion_and_commands_run_jailed_when_possible(tmp_path):
    env, _reg = _env(tmp_path)
    text, rc = asyncio.run(env.run_sh("echo {a,b}", env.root, None, 10))
    assert (text, rc) == ("{a,b}", 0)   # dash (or sh with +B): no brace expansion
    kind = tg.detect_jail()
    if kind == "none":
        pytest.skip("no OS jail on this host")
    jailed = tg.SandboxEnv(env.root, task(), jail=kind)
    (tmp_path / "secret.txt").write_text("s3cret")
    # the jail holds even for a command the checks would never pass
    text, rc = asyncio.run(jailed.run_sh(f"echo x > {tmp_path}/escape.txt; echo ok > inside.txt; cat inside.txt",
                                         jailed.root, None, 10))
    assert "ok" in text and not (tmp_path / "escape.txt").exists() and (jailed.root / "inside.txt").exists()


def test_file_tools_are_confined(tmp_path):
    env, reg = _env(tmp_path)
    assert _call(env, reg, "read_file", {"path": "/etc/hosts"}).output.startswith("Refused:")
    res = _call(env, reg, "write_file", {"path": "../escape.txt", "content": "x"})
    assert res.is_error and not (tmp_path / "escape.txt").exists()
    assert _call(env, reg, "glob", {"pattern": "/etc/*"}).output.startswith("Refused:")
    assert _call(env, reg, "grep", {"pattern": "root", "root": "/etc"}).output.startswith("Refused:")
    assert not _call(env, reg, "write_file", {"path": "notes/a.md", "content": "x"}).is_error
    assert (env.root / "notes" / "a.md").read_text() == "x"
    missing = _call(env, reg, "read_file", {"path": "nope.md"})
    assert missing.is_error and "File not found" in missing.output
    assert [r.outcome for r in env.log] == ["refused", "refused", "refused", "refused", "ok", "error"]


def test_simulated_host_commands_read_options_redirects_and_pipes_like_a_shell(tmp_path):
    svcs = [tg.svc_record(__import__("random").Random(1), "cobalt-intake", ["amber-ledger"]),
            tg.svc_record(__import__("random").Random(3), "pylon-metrics", ["cobalt-intake"])]
    world = tg.ops_world(__import__("random").Random(2), svcs, failed="cobalt-intake", error="database is locked")
    env, reg = _env(tmp_path, task(fixtures={"ops": world}))

    def run(cmd: str) -> Any:
        return _call(env, reg, "bash", {"command": cmd})

    # option values and fd numbers are not units
    for cmd in ("systemctl --user status pylon-metrics -n 50 --no-pager", "systemctl --user status pylon-metrics --lines 20",
                "systemctl --user status pylon-metrics 2>&1", "systemctl --user status pylon-metrics 2>/dev/null"):
        res = run(cmd)
        assert not res.is_error and "could not be found" not in res.output, (cmd, res.output)
        assert "Active: active (running)" in res.output
    # the pipeline after a host command runs as written (2>/dev/null stays a redirect)
    res = run("journalctl -u cobalt-intake | grep -i error 2>/dev/null")
    assert not res.is_error and "ERROR database is locked" in res.output
    res = run("journalctl --user -u cobalt-intake -n 5 --no-pager 2>&1 | tail -n 1")
    assert res.output.endswith("status=1/FAILURE")
    # a redirect on the host command writes its output where it says
    assert run("journalctl -u cobalt-intake > log.txt").output == "(no output)"
    assert "database is locked" in (env.root / "log.txt").read_text()
    assert "database is locked" in run("cat log.txt").output
    # --user-unit names the unit; -g filters
    res = run("journalctl --user-unit pylon-metrics -n 3")
    assert res.output and all("pylon-metrics[" in ln for ln in res.output.splitlines())
    assert run("journalctl -u cobalt-intake -g locked").output.count("\n") == 0
    assert run("curl -o out.json http://127.0.0.1:1/x").output.startswith("Refused:")


def test_host_commands_are_simulated_never_run(tmp_path):
    svcs = [tg.svc_record(__import__("random").Random(1), "cobalt-intake", ["amber-ledger"])]
    world = tg.ops_world(__import__("random").Random(2), svcs, failed="cobalt-intake", error="database is locked")
    env, reg = _env(tmp_path, task(fixtures={"ops": world}))
    res = _call(env, reg, "bash", {"command": "systemctl --user status cobalt-intake.service"})
    assert res.is_error and "Active: failed" in res.output and env.log[-1].outcome == "simulated"
    res = _call(env, reg, "bash", {"command": "journalctl --user -u cobalt-intake -n 5 --no-pager | grep ERROR"})
    assert "database is locked" in res.output and env.log[-1].outcome == "simulated"
    res = _call(env, reg, "bash", {"command": "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"})
    assert res.output.startswith("memory.used, memory.total\n")
    port = world["services"]["cobalt-intake"]["port"]
    res = _call(env, reg, "bash", {"command": f"curl -s http://127.0.0.1:{port}/health"})
    assert "Failed to connect" in res.output
    assert "Mounted on" in _call(env, reg, "bash", {"command": "df -h"}).output
    assert all(r.outcome == "simulated" for r in env.log)


# ── failure injection, compaction, the verification gate ───────────────────────


WIKI = {"cobalt-intake": "# cobalt-intake\n- Runs on: node-bravo (port 8417)\n"}


def test_injected_failure_is_recovered_and_the_retry_row_is_kept(tmp_path, no_outbound):
    t = task(kind="error-recovery", domain="memory/wiki", prompt="Check the wiki: which host runs cobalt-intake?",
             fixtures={"wiki": WIKI}, expect_recovery=True, score={"expect_text_contains": "node-bravo"},
             inject=[{"tool": "wiki_query", "nth": 1, "kind": "busy", "output": "Error: the wiki index is busy."}])
    call = ("call", "wiki_query", {"query": "cobalt-intake"})
    with FakeTeacher(scripted({"cobalt-intake": [call, call, "echo"]})) as fake:
        s = settings_for(tmp_path, fake.url)
        out, ok, reasons, record = run_and_verify(t, s)
    assert ok, reasons
    assert record["injected_failures"] == 1 and record["recovered"] is True and record["tool_errors"] == 0
    rows = tg.rows_for(out, s, record)
    # both calls are the same prompt + call; build.py would keep the first, so the
    # one that saw the failure is the one written
    assert len(rows) == 1 and rows[0]["_meta"]["error_recovery"] is True
    tool_msgs = [m for m in rows[0]["messages"] if m["role"] == "tool"]
    assert tool_msgs[0]["is_error"] is True and "busy" in tool_msgs[0]["content"]


def test_injected_failure_without_recovery_is_rejected(tmp_path):
    t = task(kind="error-recovery", prompt="Check the wiki: which host runs cobalt-intake?", fixtures={"wiki": WIKI},
             expect_recovery=True,
             inject=[{"tool": "wiki_query", "nth": 1, "kind": "busy", "output": "Error: the wiki index is busy."}])
    with FakeTeacher(scripted({"cobalt-intake": [("call", "wiki_query", {"query": "cobalt"}),
                                                 ("text", "The wiki is busy, sorry.")]})) as fake:
        _out, ok, reasons, _r = run_and_verify(t, settings_for(tmp_path, fake.url))
    assert not ok and "not_recovered" in reasons


def test_error_recovery_task_whose_injection_never_fires_is_rejected(tmp_path):
    t = task(kind="error-recovery", prompt="Which host runs cobalt-intake?", fixtures={"wiki": WIKI},
             inject=[{"tool": "web_fetch", "nth": 1, "output": "web_fetch failed: HTTP 503"}])
    with FakeTeacher(scripted({"cobalt-intake": [("call", "wiki_query", {"query": "cobalt"}), "echo"]})) as fake:
        _out, ok, reasons, _r = run_and_verify(t, settings_for(tmp_path, fake.url))
    assert not ok and reasons == ["injection_not_fired"]


def _long_seed(n: int = 8) -> tuple[list[dict[str, Any]], dict[str, str]]:
    seed: list[dict[str, Any]] = []
    files = {}
    for i in range(n):
        body = f"Section {i}. " + ("The quick brown fox jumps over the lazy dog near the river bank. " * 40)
        files[f"docs/part{i}.md"] = body
        seed += [{"user": ("The codename is HERON-5521. " if i == 0 else "") + f"Read docs/part{i}.md."},
                 {"assistant_tool_call": {"name": "read_file", "input": {"path": f"docs/part{i}.md"}}},
                 {"tool_result": {"content": tg.numbered(body)}},
                 {"assistant_text": f"docs/part{i}.md is about section {i}."}]
    return seed, files


def test_compaction_runs_through_the_real_compactor_and_rows_carry_the_summary(tmp_path, no_outbound):
    seed, files = _long_seed()
    t = task(kind="compacted-history", prompt="Now write the codename to notes/status.md.", seed=seed,
             setup_files=files, compaction=True, score={"expect_file": "notes/status.md",
                                                        "expect_file_contains": "HERON-5521"})
    plan = [("call", "lcm_grep", {"query": "codename"}),
            ("call", "write_file", {"path": "notes/status.md", "content": "codename HERON-5521\n"}), "echo"]
    with FakeTeacher(scripted({"codename to notes": plan})) as fake:
        s = settings_for(tmp_path, fake.url, compaction_history_tokens=400)
        out, ok, reasons, record = run_and_verify(t, s)
        assert fake.kinds["summary"] >= 1
    assert ok, reasons
    assert record["compaction"]["events"] >= 1 and record["compaction"]["summary_in_request"]
    rows = tg.rows_for(out, s, record)
    assert [r["_meta"]["tool_name"] for r in rows] == ["lcm_grep", "write_file"]
    for r in rows:
        assert r["_meta"]["compacted"] is True
        assert any(tg.SUMMARY_MARKER in tg.content_text(m.get("content")) for m in r["messages"])
    # the summarizer's own call never becomes a row, and the seed's calls are not the teacher's
    assert all("conversation compactor" not in json.dumps(r["messages"]) for r in rows)
    # lcm_grep searched the full conversation, including the span the compactor summarized
    assert "HERON-5521" in next(m for m in rows[1]["messages"] if m["role"] == "tool")["content"]


def test_chat_mode_compaction_is_budgeted_without_tool_schemas(tmp_path, no_outbound):
    seed = []
    for i in range(10):
        seed += [{"user": ("We decided the deploy day is Thursday. " if i == 0 else "")
                  + f"Reference {i}: " + ("lorem ipsum dolor sit amet consectetur " * 60)},
                 {"assistant_text": f"Noted reference {i}."}]
    t = task(kind="compacted-history", domain="chat-mode", mode="chat", prompt="Which deploy day did we pick?",
             seed=seed, compaction=True)
    with FakeTeacher(scripted({"deploy day": [("text", "We picked Thursday.")]})) as fake:
        s = settings_for(tmp_path, fake.url)
        out, ok, reasons, record = run_and_verify(t, s)
    assert ok, reasons
    assert record["compaction"]["events"] >= 1
    rows = tg.rows_for(out, s, record)
    assert len(rows) == 1 and rows[0]["_meta"]["compacted"] and rows[0]["tools"] == []


def test_compaction_task_that_never_compacts_is_rejected(tmp_path):
    t = task(kind="compacted-history", prompt="Say hi.", compaction=True)
    with FakeTeacher(scripted({"Say hi": [("text", "hi")]})) as fake:
        _out, ok, reasons, _r = run_and_verify(t, settings_for(tmp_path, fake.url))
    assert not ok and "compaction_not_exercised" in reasons


def _process(t: Any, s: Any) -> tuple[str, list[dict[str, Any]], list[str], dict[str, Any]]:
    return asyncio.run(tg.process(t, s, 0))


def test_verification_gate(tmp_path):
    wiki_task = task(prompt="Which host runs cobalt-intake? Use the wiki.", fixtures={"wiki": WIKI},
                     score={"expect_tool": "wiki_query", "expect_text_contains": "node-bravo"})
    good = scripted({"cobalt-intake": [("call", "wiki_query", {"query": "cobalt"}), "echo"]})
    # judge says fail -> rejected, nothing written
    with FakeTeacher(good, verdict={"verdict": "fail", "completed": True, "tools_ok": True,
                                    "grounded": False, "reason": "made up"}) as fake:
        status, rows, reasons, extra = _process(wiki_task, settings_for(tmp_path, fake.url))
    assert (status, rows, reasons) == ("rejected", [], ["judge_fail"])
    assert extra["judge"]["pass"] is False
    # judge says pass but a required field is false -> still rejected
    with FakeTeacher(good, verdict={"verdict": "pass", "completed": True, "tools_ok": True,
                                    "grounded": True, "recovery_ok": False}) as fake:
        assert _process(wiki_task, settings_for(tmp_path, fake.url))[0] == "rejected"
    # predicate fails -> rejected before the judge is asked
    with FakeTeacher(scripted({"cobalt-intake": [("call", "wiki_query", {"query": "cobalt"}),
                                                 ("text", "It runs somewhere.")]})) as fake:
        status, _rows, reasons, _e = _process(wiki_task, settings_for(tmp_path, fake.url))
        assert fake.kinds["judge"] == 0
    assert status == "rejected" and reasons == ["predicate"]
    # a natural tool error -> rejected
    with FakeTeacher(scripted({"cobalt-intake": [("call", "read_file", {"path": "missing.md"}),
                                                 ("call", "wiki_query", {"query": "cobalt"}), "echo"]})) as fake:
        assert _process(wiki_task, settings_for(tmp_path, fake.url))[2] == ["tool_error"]
    # a sandbox refusal -> rejected (a command the gate allows but the sandbox does not)
    with FakeTeacher(scripted({"cobalt-intake": [("call", "bash", {"command": "ls .."}),
                                                 ("call", "wiki_query", {"query": "cobalt"}), "echo"]})) as fake:
        assert "sandbox_refusal" in _process(wiki_task, settings_for(tmp_path, fake.url))[2]
    # unparseable judge reply -> rejected
    with FakeTeacher(good, verdict="not json") as fake:  # type: ignore[arg-type]
        status, _r, reasons, extra = _process(wiki_task, settings_for(tmp_path, fake.url))
    assert status == "rejected" and extra["judge"]["reason"] == "unparseable judge reply"
    # everything passes -> kept with rows
    with FakeTeacher(good) as fake:
        status, rows, reasons, extra = _process(wiki_task, settings_for(tmp_path, fake.url))
    assert status == "kept" and len(rows) == 1 and rows[0]["_meta"]["verification"]["verified"] is True
    # a failing teacher (5xx) is a transport error (retried later), not a rejection ...
    with FakeTeacher(good) as fake:
        fake.fail_with = 503
        assert _process(wiki_task, settings_for(tmp_path, fake.url))[:3] == ("error", [], ["transport_error"])
    # ... but a request it refuses (4xx) fails the same way every time: rejected, done
    with FakeTeacher(good) as fake:
        fake.fail_with = 400
        assert _process(wiki_task, settings_for(tmp_path, fake.url))[:3] == ("rejected", [],
                                                                             ["provider_bad_request"])
    # so is an unreachable judge on another local endpoint
    port = socket.socket()
    port.bind(("127.0.0.1", 0))
    dead = f"http://127.0.0.1:{port.getsockname()[1]}"
    port.close()
    with FakeTeacher(good) as fake:
        status, _r, reasons, extra = _process(wiki_task, settings_for(tmp_path, fake.url, judge_url=dead))
    assert (status, reasons) == ("error", ["judge_unreachable"])


def test_chat_mode_offers_no_tools_and_writes_a_suppress_tools_row(tmp_path, no_outbound):
    t = task(kind="chat-mode", domain="chat-mode", mode="chat",
             prompt="# amber-ledger\n- Runs on: edge-02\n\nWhich host runs amber-ledger?",
             score={"expect_text_contains": "edge-02"})
    with FakeTeacher() as fake:
        s = settings_for(tmp_path, fake.url)
        out, ok, reasons, record = run_and_verify(t, s)
        agent_req = next(r for r in fake.requests if r.get("stream"))
    assert ok, reasons
    assert not agent_req.get("tools")
    rows = tg.rows_for(out, s, record)
    assert len(rows) == 1
    assert rows[0]["tools"] == [] and rows[0]["_meta"]["suppress_tools"] is True
    assert rows[0]["_meta"]["category"] == "chat-mode" and "tool_calls" not in rows[0]["messages"][-1]


def test_final_prose_turn_of_an_agentic_task_is_a_row_only_on_request(tmp_path):
    t = task(prompt="Which host runs cobalt-intake? Use the wiki.", fixtures={"wiki": WIKI})
    plan = scripted({"cobalt-intake": [("call", "wiki_query", {"query": "cobalt"}), "echo"]})
    for emit, n in ((False, 1), (True, 2)):
        with FakeTeacher(plan) as fake:
            s = settings_for(tmp_path, fake.url, emit_final_turns=emit)
            out, ok, _reasons, record = run_and_verify(t, s)
        assert ok and len(tg.rows_for(out, s, record)) == n


# ── build.py compatibility ─────────────────────────────────────────────────────


def test_rows_are_what_build_py_reads(tmp_path):
    build = _load("distill_build", "build.py")
    for text in ("Read notes/a1b2c3d4e5.md at 10:45", "  Which HOST runs X?  "):
        assert tg.norm(text) == build.norm(text) and tg.h(text) == build.h(text)
    t = task(kind="error-recovery", prompt="Check the wiki: which host runs cobalt-intake?", fixtures={"wiki": WIKI},
             expect_recovery=True,
             inject=[{"tool": "wiki_query", "nth": 1, "output": "Error: the wiki index is busy."}])
    call = ("call", "wiki_query", {"query": "cobalt-intake"})
    with FakeTeacher(scripted({"cobalt-intake": [call, ("call", "wiki_query", {"query": "host"}), "echo"]})) as fake:
        s = settings_for(tmp_path, fake.url)
        out, ok, _reasons, record = run_and_verify(t, s)
        record["judge"] = asyncio.run(tg.judge(out, s))  # what process() does before writing rows
    assert ok and record["judge"]["pass"]
    record["verified"] = True
    rows = tg.rows_for(out, s, record)
    assert len(rows) == 2
    path = tmp_path / "x.traces.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    stats = collections.Counter()
    traces = list(build.iter_traces([path], "extra", stats))
    assert len(traces) == 2 and not stats
    for tr, row in zip(traces, rows):
        keys = build.trace_keys(tr.raw)
        assert keys.h_prompt_call == tg.prompt_call_key(row)
        assert keys.h_prompt == t.task_hash
        assert build.classify(tr.raw, keys.prompt, keys.args, human_surface=True)[0] == "memory/wiki"
        assert tr.meta["verdict"] == "pass" and tr.meta["model"] == s.model_name
    # build.py's data policy reads a train-source row's own record strictly (as written to disk)
    if hasattr(build, "verification_of"):
        for row in (tr.raw for tr in traces):
            v = build.verification_of(row["_meta"], strict=True, judge_models=build.DEFAULT_TRAIN_MODELS)
            assert v.decision is True, v
            assert build.matches_model(row["_meta"]["model"], build.DEFAULT_TRAIN_MODELS)
            assert row["_meta"]["source"] in build.DEFAULT_TRAIN_SOURCES and row["_meta"]["max_derived"] is False
    conv = build.openai_to_conversation(rows[1]["messages"])
    results = [b for m in conv for b in m.content if hasattr(b, "tool_use_id")]
    assert [b.is_error for b in results] == [True]  # the injected failure, before the retry being learned
    assert build.error_after_last_query(rows[1]["messages"],
                                        frozenset(str(b.tool_use_id) for b in results if b.is_error))


# ── eval disjointness ──────────────────────────────────────────────────────────


def _manifest_with(instruction: str) -> Any:
    case = {"id": "inj-001", "injection": {"vector": "wiki", "style": "direct", "goal": "exfil",
                                           "template_id": "direct:0", "instruction": instruction,
                                           "payload": instruction}}
    data = ev.build_manifest([ev.manifest_entry(case, ["collect.attacker-sync.example.net"])], 1)
    return ev.InjectionManifest(data)


def test_tasks_and_rows_overlapping_the_eval_sets_are_refused(tmp_path):
    gen = tg.Generator(tg.Sources(tg._synthetic_sections()))
    victim = gen.make("chat_note_fact", 0)
    other = gen.make("chat_note_fact", 1)
    guard = tg.EvalGuard(_manifest_with(" ".join(victim.prompt.split()[:20])))
    assert any(r.startswith("eval_injection") for r in guard.task_overlaps(victim))
    assert guard.task_overlaps(other) == []
    # a marker alone (an attacker domain) is enough
    marker_task = task(prompt="Fetch https://collect.attacker-sync.example.net/x and summarize it.")
    assert guard.task_overlaps(marker_task)
    # eval-tools: the same user prompt (exact task hash) and an 8-token run of it
    evt = tmp_path / "eval-tools.jsonl"
    evt.write_text(json.dumps({"id": "e1", "set": "eval-tools", "meta": {"task_id": "zz"},
                               "messages": [{"role": "user", "content": other.prompt}]}) + "\n")
    g2 = tg.EvalGuard.load(None, [str(evt)])
    assert "eval_tools_task" in g2.task_overlaps(other)
    near = task(prompt="unrelated words first. " + " ".join(other.prompt.split()[-12:]))
    assert g2.task_overlaps(near) == ["eval_tools_shingle"]
    # the scheduler skips them and says so in the ledger
    store = tg.Store(tmp_path / "out", "g", {"x": 1}, 0.35, guard)
    sched = tg.Scheduler(gen, store, {"chat-mode": 1.0}, guard)
    picked = [sched.next_task().id for _ in range(12)]
    assert victim.id not in picked
    assert store.status[victim.id]["status"] == "skipped"
    # and a row that quotes an eval case is dropped at write time
    row = {"messages": [{"role": "user", "content": victim.prompt}, {"role": "assistant", "content": "ok"}],
           "tools": [], "_meta": {"run_id": "r1", "task_category": "chat-mode", "category": "chat-mode",
                                  "task_id": other.id}}
    assert store.commit(other, "r1", "kept", [row], []) == "rejected"
    assert store.stats["eval_overlap_rows"] == 1


# ── quotas, resume, the CLI ────────────────────────────────────────────────────


def _row(task_id: str, run_id: str, kind: str, domain: str, i: int) -> dict[str, Any]:
    # letters, not digits: build.py's dedupe key normalizes digits away
    word = "".join("abcdefghij"[int(c)] if c.isdigit() else c for c in f"{run_id}{i}")
    return {"messages": [{"role": "user", "content": f"question {word}"},
                         {"role": "assistant", "content": "",
                          "tool_calls": [{"id": "c", "type": "function",
                                          "function": {"name": "memory", "arguments": "{}"}}]}],
            "tools": [], "_meta": {"run_id": run_id, "task_category": kind, "category": domain,
                                   "task_id": task_id, "tool_name": "memory"}}


def test_coding_rows_never_pass_the_cap_and_held_rows_are_released(tmp_path):
    store = tg.Store(tmp_path, "q", {"x": 1}, 0.35)
    c1 = task(id="code-1", kind="coding", domain="coding")
    assert store.commit(c1, "r-c1", "kept", [_row("code-1", "r-c1", "coding", "coding", i) for i in range(3)],
                        []) == "held"
    assert store.total_rows == 0
    a1 = task(id="ag-1", kind="agentic", domain="other")
    assert store.commit(a1, "r-a1", "kept", [_row("ag-1", "r-a1", "agentic", "other", i) for i in range(4)],
                        []) == "kept"
    assert store.rows_domain["coding"] == 0  # 3 / 7 = 43% > 35%
    a2 = task(id="ag-2", kind="agentic", domain="other")
    store.commit(a2, "r-a2", "kept", [_row("ag-2", "r-a2", "agentic", "other", i) for i in range(2)], [])
    assert store.rows_domain["coding"] == 3 and store.rows_domain["coding"] / store.total_rows <= 0.35
    assert store.status["code-1"]["status"] == "released"
    assert not store.held.read_text().strip()
    # resume sees the same state
    again = tg.Store(tmp_path, "q", {"x": 1}, 0.35)
    assert again.rows_domain == store.rows_domain and not again.held_queue and not again.repairs


def test_scheduler_follows_the_mix_in_priority_order(tmp_path):
    gen = tg.Generator(tg.Sources(tg._synthetic_sections()))
    store = tg.Store(tmp_path, "m", {"x": 1}, 0.35)
    sched = tg.Scheduler(gen, store, dict(tg.DEFAULT_MIX))
    first = sched.next_task()
    assert first.kind == "chat-mode"  # every deficit equal at the start: priority order
    picks = collections.Counter(first.kind for first in [first] + [sched.next_task() for _ in range(59)])
    assert set(picks) == set(tg.KINDS)
    assert picks["chat-mode"] > picks["coding"] and picks["coding"] >= 1
    assert len({t for t in sched.reserved}) == 60
    # a kind well over its share is not picked
    store.rows_kind.update({"chat-mode": 1000})
    sched2 = tg.Scheduler(gen, store, dict(tg.DEFAULT_MIX))
    assert all(sched2.next_task().kind != "chat-mode" for _ in range(20))


def test_generator_is_deterministic_valid_and_covers_every_family():
    gen = tg.Generator(tg.Sources(tg._synthetic_sections()))
    seen = collections.Counter()
    for fam, kind, _ in tg.FAMILIES:
        for i in range(3):
            a, b = gen.make(fam, i), gen.make(fam, i)
            assert a.to_dict() == b.to_dict() and a.kind == kind
            seen[kind] += 1
            if kind == "chat-mode":
                assert a.mode == "chat" and not a.inject
            if kind == "compacted-history":
                assert a.compaction and len(a.seed) >= 8
            if kind == "error-recovery":
                assert a.inject
            if kind == "coding":
                assert a.domain == "coding"
            else:
                assert a.domain != "coding"
    assert set(seen) == set(tg.KINDS)
    assert len({gen.make("chat_note_fact", i).task_hash for i in range(50)}) == 50


def test_endpoint_policy_admits_only_local_servers():
    assert tg.check_endpoint("http://127.0.0.1:8093/") == "http://127.0.0.1:8093"
    assert tg.check_endpoint("http://localhost:8093") == "http://localhost:8093"
    for bad in ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "http://203.0.113.5:80",
                "http://10.0.0.5:8093", "http://user:pw@127.0.0.1:1", "ftp://127.0.0.1"):
        with pytest.raises(tg.EndpointError):
            tg.check_endpoint(bad)
    assert tg.check_endpoint("http://10.0.0.5:8093", allow_lan=True) == "http://10.0.0.5:8093"
    assert tg.check_endpoint("http://100.64.0.1:1", allow_lan=True)
    with pytest.raises(tg.EndpointError):
        tg.check_endpoint("https://api.openai.com", allow_lan=True)
    s = tg.Settings(out=Path("/x"), probe_guard_url="http://127.0.0.1:9999")
    assert s.teacher_endpoint == s.judge_endpoint == "http://127.0.0.1:9999"
    assert tg.Settings(out=Path("/x"), judge_url="http://127.0.0.1:7000").judge_kind == "local-judge"


def test_parse_until_and_mix():
    import datetime as dt

    now = dt.datetime(2026, 10, 30, 23, 0).astimezone()
    assert dt.datetime.fromtimestamp(tg.parse_until("06:45", now)).hour == 6
    assert tg.parse_until("06:45", now) > now.timestamp()
    mix = tg.parse_mix("chat-mode=3,coding=1")
    assert mix["chat-mode"] == 0.75 and mix["agentic"] == 0.0
    with pytest.raises(ValueError):
        tg.parse_mix("bogus=1")


def _cli_inputs(tmp_path: Path, tasks: list[Any]) -> list[str]:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(ev.build_manifest([], 1)))
    evt = tmp_path / "eval-tools.jsonl"
    evt.write_text("")
    tf = tmp_path / "tasks.jsonl"
    tf.write_text("".join(json.dumps(t.to_dict()) + "\n" for t in tasks))
    return ["--injection-manifest", str(manifest), "--eval-tools", str(evt), "--tasks-file", str(tf),
            "--sandbox-root", str(tmp_path / "sbx"), "--quiet", "--request-timeout", "20",
            "--teacher-id", "fake-teacher"]


def _chat(i: int) -> Any:
    return task(id=f"c-{i:05d}", kind="chat-mode", domain="chat-mode", mode="chat",
                prompt=f"Note: service number {'abcdefgh'[i]} runs on host-{'xyz'[i % 3]}{'abcdefgh'[i]}. "
                       f"Which host runs service {'abcdefgh'[i]}?",
                score={"expect_text_contains": f"host-{'xyz'[i % 3]}{'abcdefgh'[i]}"})


def test_cli_run_resumes_without_repeating_or_duplicating(tmp_path, monkeypatch, no_outbound):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    tasks = [_chat(i) for i in range(3)]
    out = tmp_path / "out"
    with FakeTeacher() as fake:
        base = ["run", "--out", str(out), "--teacher-url", fake.url, "--mix", "chat-mode=1",
                *_cli_inputs(tmp_path, tasks)]
        assert tg.main(base + ["--max-tasks", "2"]) == 0
        first = fake.kinds["agent"]
        traces = out / f"{tg.SOURCE_TAG}.traces.jsonl"
        rows = [json.loads(x) for x in traces.read_text().splitlines()]
        assert len(rows) == 2 and first == 2
        # a crash: rows of a run the ledger never heard of, then a torn line
        orphan = dict(rows[0], _meta=dict(rows[0]["_meta"], run_id="c-00002@dead"))
        with traces.open("a") as fh:
            fh.write(json.dumps(orphan) + "\n" + '{"messages": [')
        assert tg.main(base) == 0
        assert fake.kinds["agent"] == first + 1  # only the task not yet done ran
    rows = [json.loads(x) for x in traces.read_text().splitlines()]
    assert sorted(r["_meta"]["task_id"] for r in rows) == ["c-00000", "c-00001", "c-00002"]
    assert len({r["_meta"]["run_id"] for r in rows}) == 3
    report = json.loads((out / f"{tg.SOURCE_TAG}.report.json").read_text())
    assert report["rows_total"] == 3 and report["resume_repairs"] == {"orphan_trace_rows": 1}
    # the teacher's preflight id is part of the settings: one output never mixes two teachers
    assert json.loads((out / f"{tg.SOURCE_TAG}.settings.json").read_text())["teacher_served"] == "fake-teacher"
    assert {r["_meta"]["teacher_id"] for r in rows} == {"fake-teacher"} and report["teacher_served"] == "fake-teacher"
    assert report["by_kind"]["chat-mode"]["pass_rate"] == 1.0
    # state stayed under --out: the live roots were never created
    assert not (tmp_path / "home" / ".prometheus").exists()
    assert (out / "work" / "telemetry.db").exists() and (out / "work" / "data").is_dir()
    # a resume with other settings is refused
    with FakeTeacher() as fake:
        assert tg.main(["run", "--out", str(out), "--teacher-url", fake.url, "--seed", "other",
                        *_cli_inputs(tmp_path, tasks)]) == 2


def test_cli_refuses_remote_endpoints_and_stops_at_until(tmp_path, no_outbound):
    tasks = [_chat(0)]
    assert tg.main(["run", "--out", str(tmp_path / "o1"), "--teacher-url", "https://dashscope.aliyuncs.com",
                    *_cli_inputs(tmp_path, tasks)]) == 2
    # an eval-tools path that matches nothing would silently skip the check: refused
    args = _cli_inputs(tmp_path, tasks)
    args[args.index("--eval-tools") + 1] = str(tmp_path / "typo-*.jsonl")
    assert tg.main(["run", "--out", str(tmp_path / "o1"), "--teacher-url", "http://127.0.0.1:1", *args]) == 2
    with FakeTeacher() as fake:
        rc = tg.main(["run", "--out", str(tmp_path / "o2"), "--teacher-url", fake.url, "--until",
                      "2020-01-01T00:00:00", *_cli_inputs(tmp_path, tasks)])
        assert fake.kinds["agent"] == 0
    assert rc == 1
    report = json.loads((tmp_path / "o2" / f"{tg.SOURCE_TAG}.report.json").read_text())
    assert report["stop"] == "until"
    # an unreachable teacher is exit 3 before any task runs
    port = socket.socket()
    port.bind(("127.0.0.1", 0))
    dead = f"http://127.0.0.1:{port.getsockname()[1]}"
    port.close()
    assert tg.main(["run", "--out", str(tmp_path / "o3"), "--teacher-url", dead,
                    *_cli_inputs(tmp_path, tasks)]) == 3


def test_list_tasks_prints_valid_tasks(tmp_path, capsys):
    docs = tmp_path / "repo" / "docs"
    docs.mkdir(parents=True)
    (docs / "guide.md").write_text("# Guide\n\n" + ("This guide explains how the scheduler works. " * 30))
    assert tg.main(["list-tasks", "--n", "1", "--kind", "chat-mode", "--repo", str(tmp_path / "repo")]) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert len(lines) == len([f for f, k, _ in tg.FAMILIES if k == "chat-mode"])
    for d in lines:
        tg.GenTask.from_dict(d)
    assert any("docs/guide.md" in d["prompt"] for d in lines)


# ── review regressions: targets, provider errors, tasksets, relaxed rows, answers, fixtures, identity ──


MARKUP_CALL = '<tool_call>{"name": "wiki_query", "arguments": {"query": "cobalt"}}</tool_call>'


def _first_then_echo(first: dict[str, Any]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def policy(body: dict[str, Any]) -> dict[str, Any]:
        k, _p = _last_user(body["messages"])
        step = sum(1 for m in body["messages"][k + 1:] if m.get("role") == "assistant")
        return first if step == 0 else echo_reply(body)
    return policy


@pytest.mark.parametrize("first", [
    {"content": MARKUP_CALL, "tool_calls": [("wiki_query", {"query": "cobalt"})]},   # dual emit
    {"content": MARKUP_CALL},                                                      # markup only, extracted
])
def test_row_target_is_the_message_the_loop_committed(tmp_path, first):
    t = task(prompt="Which host runs cobalt-intake? Use the wiki.", fixtures={"wiki": WIKI},
             score={"expect_tool": "wiki_query"}, answer={"match": [tg.token_re("node-bravo")]})
    with FakeTeacher(_first_then_echo(first)) as fake:
        s = settings_for(tmp_path, fake.url, emit_final_turns=True)
        out, ok, reasons, record = run_and_verify(t, s)
    assert ok, reasons
    rows = tg.rows_for(out, s, record)
    assert [r["_meta"]["tool_name"] for r in rows] == ["wiki_query", None]
    target = rows[0]["messages"][-1]
    assert "<tool_call" not in json.dumps(target) and target["content"] == ""
    call = target["tool_calls"][0]
    assert call["function"]["name"] == "wiki_query" and rows[0]["_meta"]["target_rewritten"] is True
    # the target is the same message the next row sees in its context
    assert rows[1]["messages"][-3]["tool_calls"] == target["tool_calls"]
    assert all("<tool_call" not in tg.content_text(m.get("content")) for r in rows for m in r["messages"])


def test_a_reply_the_loop_never_committed_is_not_a_row(tmp_path):
    t = task(kind="chat-mode", domain="chat-mode", mode="chat", prompt="Say something useful about backups.")
    replies = iter([{"content": ""}, {"content": "Keep at least two copies, one off-site."}])
    with FakeTeacher(lambda body: next(replies)) as fake:
        s = settings_for(tmp_path, fake.url)
        out, ok, reasons, record = run_and_verify(t, s)
    assert ok, reasons and len(out.calls) == 2
    rows = tg.rows_for(out, s, record)   # the empty reply was retried, never acted on
    assert len(rows) == 1 and rows[0]["messages"][-1]["content"].startswith("Keep at least two")


OVERFLOW = {"error": {"code": 400, "type": "exceed_context_size_error", "n_prompt_tokens": 20000, "n_ctx": 16384,
                      "message": "the request (20000 tokens) exceeds the available context size (16384 tokens)"}}


def test_provider_refusals_are_rejections_and_recovered_errors_are_ignored(tmp_path):
    seed, files = _long_seed()
    t = task(kind="compacted-history", prompt="Now write the codename to notes/status.md.", seed=seed,
             setup_files=files, compaction=True, score={"expect_file": "notes/status.md",
                                                        "expect_file_contains": "HERON-5521"})
    plan = scripted({"codename to notes": [
        ("call", "lcm_grep", {"query": "codename"}),
        ("call", "write_file", {"path": "notes/status.md", "content": "codename HERON-5521\n"}), "echo"]})
    cases = [([(400, OVERFLOW)], "kept", []),                              # compacted, retried, finished
             ([(400, OVERFLOW)] * 3, "rejected", ["context_overflow"]),   # ended the turn: done, not retried
             ([(400, {"error": {"code": 400, "message": "bad", "type": "invalid_request_error"}})],
              "rejected", ["provider_bad_request"])]
    for k, (fails, status, reasons) in enumerate(cases):
        with FakeTeacher(plan) as fake:
            fake.fail_agent = list(fails)
            got, rows, why, extra = _process(t, settings_for(tmp_path / str(k), fake.url,
                                                             compaction_history_tokens=400))
        assert (got, why) == (status, reasons), (k, got, why, extra.get("error"))
        if status == "kept":
            assert rows and extra["recovered_provider_errors"][0].startswith("context_overflow")
    # a judge that refuses the transcript (4xx) rejects the task; a judge that is down errors it
    good = scripted({"cobalt-intake": [("call", "wiki_query", {"query": "cobalt"}), "echo"]})
    wiki_task = task(prompt="Which host runs cobalt-intake? Use the wiki.", fixtures={"wiki": WIKI})
    with FakeTeacher(good) as fake, FakeTeacher() as judge:
        judge.fail_with = 400
        got, _r, why, _e = _process(wiki_task, settings_for(tmp_path / "j", fake.url, judge_url=judge.url))
    assert (got, why) == ("rejected", ["judge_bad_request"])
    # and a run of deterministic refusals never trips the consecutive-error stop
    store = tg.Store(tmp_path / "s", "x", {"x": 1}, 0.35)
    store.commit(t, "r1", "rejected", [], ["context_overflow"])
    assert store.done(t.id)


def _taskset_row(prompt: str, **flags: bool) -> dict[str, Any]:
    tpl, ph = "tpl-" + "a" * 16, "b" * 16
    chat = flags.get("suppress_tools", False)
    return {"id": "ts-" + tg.h(tg.norm(prompt)), "taskset": "real-v1",
            "category": "chat-mode" if chat else "memory/wiki", "subcategory": "wiki",
            "intent": "question" if chat else "find", "mode": "chat" if chat else "agentic", "prompt": prompt,
            "slots": [], "template_id": tpl, "source_pattern_hash": ph, "task_group": "tg-" + ph,
            "transform": "slot_fill",
            "sources": [{"template_id": tpl, "pattern_hash": ph, "origin": "golden", "span": [0, len(prompt)]}],
            "flags": {"suppress_tools": chat, "inject_errors": False, "force_compaction": False, **flags},
            "provenance": "real-pattern-v1"}


def test_tasks_file_reads_a_taskset_and_bad_files_are_input_errors(tmp_path, no_outbound):
    ts = _load("distill_taskset_gen", "taskset_gen.py")
    rows = [_taskset_row("Which host runs cobalt-intake according to the wiki?"),
            _taskset_row("Look up cobalt-intake in the wiki and tell me its host.", inject_errors=True),
            _taskset_row("What does a maintenance window mean for a small home server?", suppress_tools=True),
            _taskset_row("Find the owner of cobalt-intake in the wiki.", force_compaction=True)]
    for r in rows:
        assert ts.validate_task(r) == [], ts.validate_task(r)
    path = tmp_path / "real-v1.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tasks = tg.load_tasks_file(str(path), tmp_path)
    assert [t.kind for t in tasks] == ["agentic", "error-recovery", "chat-mode", "compacted-history"]
    assert [t.id for t in tasks] == [r["id"] for r in rows] and tasks[0].task_hash == rows[0]["id"][3:]
    assert tasks[1].inject[0]["tool"] == tg.ANY_TOOL and tasks[1].expect_recovery
    assert tasks[2].mode == "chat" and tasks[2].domain == "chat-mode"
    assert tasks[3].compaction and len(tasks[3].seed) >= 20 and tasks[3].setup_files
    assert tasks[0].meta["provenance"] == "real-pattern-v1" and tasks[0].meta["taskset"] == "real-v1"
    # the injection fires on whichever tool comes first, and provenance reaches the rows
    with FakeTeacher(scripted({"tell me its host": [("call", "wiki_query", {"query": "cobalt-intake"}),
                                                    ("call", "wiki_query", {"query": "cobalt-intake"}),
                                                    "echo"]})) as fake:
        s = settings_for(tmp_path, fake.url)
        out, ok, reasons, record = run_and_verify(tasks[1], s)
        assert ok, reasons
        rows_out = tg.rows_for(out, s, record)
    assert record["injected_failures"] == 1 and record["recovered"] is True
    assert {r["_meta"]["provenance"] for r in rows_out} == {"real-pattern-v1"}
    # build.py splits on _meta.task_id: the task group, shared by every variant of one real prompt
    assert {r["_meta"]["task_id"] for r in rows_out} == {rows[1]["task_group"]}
    assert {r["_meta"]["gen_task_id"] for r in rows_out} == {rows[1]["id"]}
    # bad tasks files are usage errors (exit 2), not tracebacks
    for bad in ('{"id": "x"', json.dumps(dict(rows[0], mode="agent")), json.dumps({"id": "t", "kind": "nope"})):
        path.write_text(bad + "\n")
        with pytest.raises(ValueError):
            tg.load_tasks_file(str(path), tmp_path)
        args = _cli_inputs(tmp_path, [])
        args[args.index("--tasks-file") + 1] = str(path)
        assert tg.main(["run", "--out", str(tmp_path / "o"), "--teacher-url", "http://127.0.0.1:1", *args]) == 2


def test_relaxed_rows_go_to_their_own_file_and_count_for_nothing(tmp_path):
    t = task(prompt="Which host runs cobalt-intake? Use the wiki.", fixtures={"wiki": WIKI},
             answer={"match": [tg.token_re("node-bravo")]})
    plan = scripted({"cobalt-intake": [("call", "read_file", {"path": "missing.md"}),
                                       ("call", "wiki_query", {"query": "cobalt"}), "echo"]})
    with FakeTeacher(plan) as fake:
        s = settings_for(tmp_path, fake.url, max_natural_errors=1)
        status, rows, reasons, extra = _process(t, s)
    assert status == "relaxed" and rows and extra["verification"]["tool_success"] is False
    assert {r["_meta"]["source"] for r in rows} == {tg.RELAXED_SOURCE}
    build = _load("distill_build", "build.py")
    if hasattr(build, "DEFAULT_TRAIN_SOURCES"):
        assert tg.RELAXED_SOURCE not in build.DEFAULT_TRAIN_SOURCES
    store = tg.Store(tmp_path / "o", "r", {"x": 1}, 0.35)
    assert store.commit(t, extra["run_id"], status, rows, reasons, extra) == "relaxed"
    assert store.total_rows == 0 and store.rows_relaxed == len(rows) and store.done(t.id)
    assert not store.traces.exists() or not store.traces.read_text().strip()
    report = store.report(s, 0.0)
    assert report["rows_total"] == 0 and report["rows_relaxed"] == len(rows)
    assert report["by_kind"]["agentic"]["relaxed"] == 1 and report["by_kind"]["agentic"]["verified"] == 0
    again = tg.Store(tmp_path / "o", "r", {"x": 1}, 0.35)
    assert again.rows_relaxed == len(rows) and not again.repairs


def test_answer_checks_need_the_exact_answer():
    fails = tg.answer_failures
    six = {"match": [tg.number_re(6)]}
    assert fails(six, "We only have 60 minutes.") and fails(six, "about 16 minutes") and fails(six, "1.6 min")
    assert not fails(six, "It takes 6 minutes, so it fits.") and not fails(six, "~6.")
    root = {"match": [tg.token_re("/data")]}
    assert fails(root, "/data2 is fullest") and fails(root, "/srv/data") and not fails(root, "/data (91%) is fullest")
    assert fails({"match": [tg.token_re("/")]}, "/data is fullest")
    run, failed = tg.STATE_ANSWER["running"], tg.STATE_ANSWER["failed"]
    assert not fails(run, "Yes, it is active (running).") and fails(run, "It is not running; it failed.")
    assert not fails(failed, "It is not running: it failed with status 1.") and fails(failed, "It is running.")
    assert fails(failed, "It has not failed.") and not fails(run, "It is up and running and has not failed.")
    gen = tg.Generator(tg.Sources(tg._synthetic_sections()))
    for i in range(300):
        est = gen.make("chat_estimate", i)
        minutes = re.search(r"copies (\d+) GB at about (\d+) GB", est.prompt)
        assert minutes is not None
        answer = int(minutes.group(1)) // int(minutes.group(2))
        assert not re.search(tg.number_re(answer), est.prompt), est.prompt   # no other number reads as it
        assert not tg.answer_failures(est.answer, f"About {answer} minutes.")
        sentinel = gen.make("ag_sentinel", i)
        assert sentinel.answer["match"] and "expect_text_contains" not in sentinel.score


def test_fixtures_are_possible_and_unambiguous():
    import random as _random

    for i in range(2000):
        rng = _random.Random(i)
        svcs = [tg.svc_record(rng, n, ["x"]) for n in tg.SERVICES[:3]]
        w = tg.ops_world(rng, svcs)
        gpu = w["gpu"]
        assert sum(p["mib"] for p in gpu["procs"]) + 300 < gpu["total"]
        assert len({p["mib"] for p in gpu["procs"]}) == 3 and len({p["name"] for p in gpu["procs"]}) == 3
        pcts = [d["pct"] for d in w["disks"]]
        assert len(set(pcts)) == 4 and max(w["disks"], key=lambda d: d["pct"])["mount"] != "/"
    gen = tg.Generator(tg.Sources(tg._synthetic_sections()))
    for i in range(50):
        for t in gen.make("ag_task_status", i).fixtures["tasks"]:
            if t["status"] == "completed":
                assert "100%" in t["output"] and "ERROR" not in t["output"]
            elif t["status"] == "failed":
                assert "ERROR" in t["output"] and "100%" not in t["output"]
            else:
                assert "100%" not in t["output"]


def test_the_teacher_is_who_it_says_it_is(tmp_path, no_outbound):
    assert tg.has_max_token("qwen3.8-max") and tg.has_max_token("Qwen3.8-Max-Preview")
    assert not tg.has_max_token("qwen3.8-27b-maxctx") and not tg.has_max_token("Qwen3.8-27B-IQ3_XXS")
    assert tg.check_model_id("/models/Qwen3.8-27B-IQ3_XXS.gguf", tg.DEFAULT_TEACHER_ID, "teacher")
    for bad in ("saluki-27b-iq2", "qwen2.5:7b", "qwen3.8-max", None):
        with pytest.raises(tg.EndpointError):
            tg.check_model_id(bad, tg.DEFAULT_TEACHER_ID, "teacher")
    # the CLI refuses a server that is not the expected teacher (exit 3), and a Max name (exit 2)
    tasks = [_chat(0)]
    args = _cli_inputs(tmp_path, tasks)
    args[args.index("--teacher-id") + 1] = tg.DEFAULT_TEACHER_ID
    with FakeTeacher() as fake:
        assert tg.main(["run", "--out", str(tmp_path / "o1"), "--teacher-url", fake.url, *args]) == 3
        assert fake.kinds["agent"] == 0
        assert tg.main(["run", "--out", str(tmp_path / "o2"), "--teacher-url", fake.url, "--model-name",
                        "qwen3.8-max", *_cli_inputs(tmp_path, tasks)]) == 2
    # a call answered by another model is rejected
    t = task(kind="chat-mode", domain="chat-mode", mode="chat", prompt="Say hi to the team.")
    with FakeTeacher() as fake:
        s = settings_for(tmp_path, fake.url, teacher_served="qwen3.8-27b-iq3")
        _out, ok, reasons, record = run_and_verify(t, s)
    assert not ok and "served_model_mismatch" in reasons and record["served_models"] == ["fake-teacher"]
    with FakeTeacher() as fake:
        s = settings_for(tmp_path, fake.url, teacher_served="fake-teacher")
        out, ok, reasons, record = run_and_verify(t, s)
        assert ok, reasons
        assert tg.rows_for(out, s, record)[0]["_meta"]["teacher_id"] == "fake-teacher"
