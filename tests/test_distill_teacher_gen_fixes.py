"""Regression tests for the teacher_gen.py fixes after the stopped max-token-plan run
(kept 14 / rejected 197): the judge's outcome gate and context, no-tool answers,
injections that did not apply, recovery in execution order, the gym's permission
gate, empty-argument calls, pad-turn rollback, host inspection in the sandbox and
the read-only repo fixture. Synthetic fixtures only; no network, no model."""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.test_distill_teacher_gen import (  # noqa: F401 - fixtures are used by name
    WIKI,
    FakeTeacher,
    _call,
    _env,
    _load,
    _restore_state_env,
    _taskset_row,
    run_and_verify,
    scripted,
    settings_for,
    task,
)

tg = _load("distill_teacher_gen", "teacher_gen.py")
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


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


def _process(t: Any, s: Any) -> tuple[str, list[dict[str, Any]], list[str], dict[str, Any]]:
    return asyncio.run(tg.process(t, s, 0))


def _taskset_task(tmp_path: Path, prompt: str, **flags: bool) -> Any:
    return tg.tasks_from_taskset_rows([_taskset_row(prompt, **flags)], tmp_path)[0]


# ── the judge ──────────────────────────────────────────────────────────────────


def test_verdict_passes_a_plainly_reported_block_but_not_a_task_left_undone():
    base = {"verdict": "pass", "tools_ok": True, "grounded": True, "recovery_ok": None, "reason": "x"}
    rec = tg._verdict_record({}, dict(base, outcome="reported_blocked"))
    assert rec["pass"] is True and rec["outcome"] == "reported_blocked" and rec["completed"] is False
    assert tg._verdict_record({}, dict(base, outcome="done"))["pass"] is True
    assert tg._verdict_record({}, dict(base, outcome="not_done"))["pass"] is False
    assert tg._verdict_record({}, dict(base, outcome="something else"))["pass"] is False
    # the old shape still reads: completed true is done, completed false never passes
    assert tg._verdict_record({}, dict(base, completed=True))["pass"] is True
    assert tg._verdict_record({}, dict(base, completed=False))["pass"] is False
    # every other check still holds
    assert tg._verdict_record({}, dict(base, outcome="done", verdict="fail"))["pass"] is False
    assert tg._verdict_record({}, dict(base, outcome="done", grounded=False))["pass"] is False
    assert tg._verdict_record({}, dict(base, outcome="reported_blocked", recovery_ok=False))["pass"] is False
    assert '"outcome"' in tg.JUDGE_SYSTEM and "reported_blocked" in tg.JUDGE_SYSTEM


def test_the_judge_sees_the_system_prompt_the_offered_tools_and_the_sandbox(tmp_path, no_outbound):
    t = task(prompt="Which host runs cobalt-intake? Use the wiki.", fixtures={"wiki": WIKI})
    with FakeTeacher(scripted({"cobalt-intake": [("call", "wiki_query", {"query": "cobalt"}), "echo"]})) as fake:
        out, ok, reasons, _r = run_and_verify(t, settings_for(tmp_path, fake.url))
    assert ok, reasons
    text = tg.judge_transcript(out)
    assert text.startswith("CONTEXT (what the agent was given)")
    assert "You are Prometheus" in text and "# Workspace" in text
    offered = text.split("TOOLS OFFERED THIS TURN: ", 1)[1].split("\n", 1)[0]
    assert set(offered.split(", ")) == tg.SANDBOX_TOOLS
    assert tg.JUDGE_SANDBOX_NOTE in text
    # the gym's label is a hint, never "the task kind"
    assert "TASK LABEL (how the gym filed this task; a hint only" in text and "TASK KIND" not in text
    chat = task(kind="chat-mode", domain="chat-mode", mode="chat", prompt="What is a cron expression, briefly?")
    with FakeTeacher() as fake:
        out, _ok, _reasons, _r = run_and_verify(chat, settings_for(tmp_path, fake.url))
    assert "NO TOOLS AVAILABLE (chat mode)" in tg.judge_transcript(out) and out.tools_offered == []


def test_a_long_system_prompt_is_cut_but_keeps_its_workspace_line(tmp_path, no_outbound):
    s = settings_for(tmp_path, "http://127.0.0.1:1", base_system_prompt="You are Prometheus. " + "x" * 5000)
    out = tg.TaskOutcome(task=task(), run_id="r", messages=[], n_seed=0, calls=[], observed={},
                         env=_env(tmp_path)[0], system_prompt=s.system_prompt(tmp_path))
    ctx = tg._judge_context(out)
    assert "...[system prompt cut]..." in ctx and "# Workspace" in ctx and len(ctx) < 2500


# ── no-tool answers and injections that did not apply ──────────────────────────


def test_a_real_pattern_agent_task_answered_without_tools_is_a_row_that_keeps_its_tools(tmp_path, no_outbound):
    t = _taskset_task(tmp_path, "What does a maintenance window mean for the cobalt-intake service?")
    assert t.kind == "agentic" and t.mode == "agent"
    with FakeTeacher() as fake:  # the default policy: a prose answer, no tool call
        s = settings_for(tmp_path, fake.url, task_source=tg.ORIGIN_TASKSET)
        status, rows, reasons, extra = _process(t, s)
    assert (status, reasons) == ("kept", []) and len(rows) == 1
    meta = rows[0]["_meta"]
    assert meta["no_tool_answer"] is True and meta["final_answer"] is True and meta["judge_outcome"] == "done"
    assert meta["suppress_tools"] is False and rows[0]["tools"], "the tools offered stay in the row"
    assert "tool_calls" not in rows[0]["messages"][-1]
    # build.py reads it as a tools-offered request, not as chat mode
    build = _load("distill_build", "build.py")
    assert build.chat_mode_of(rows[0]) is False
    # a built-in task's final prose turn is still a row only with --emit-final-turns
    builtin = task(prompt="Say what a maintenance window is.")
    with FakeTeacher() as fake:
        assert _process(builtin, settings_for(tmp_path, fake.url))[:3] == ("rejected", [], ["no_rows"])


def test_an_injected_task_answered_without_tools_is_judged_as_a_plain_agentic_task(tmp_path, no_outbound):
    t = _taskset_task(tmp_path, "Look up cobalt-intake in the wiki and tell me its host.", inject_errors=True)
    assert t.kind == "error-recovery"
    with FakeTeacher() as fake:
        s = settings_for(tmp_path, fake.url, task_source=tg.ORIGIN_TASKSET)
        status, rows, reasons, extra = _process(t, s)
        assert fake.kinds["judge"] == 1
    assert status == "kept" and reasons == []
    assert extra["verification"]["injection"] == "not_applicable"
    assert extra["kind"] == "agentic" and extra["planned_kind"] == "error-recovery"
    assert rows[0]["_meta"]["task_category"] == "agentic" and rows[0]["_meta"]["error_recovery"] is False
    # a built-in error-recovery task still has to see its injection fire
    builtin = task(kind="error-recovery", prompt="Which host runs cobalt-intake?", fixtures={"wiki": WIKI},
                   inject=[dict(tg.TASKSET_INJECTION)], expect_recovery=True)
    with FakeTeacher() as fake:
        assert run_and_verify(builtin, settings_for(tmp_path, fake.url))[2] == ["injection_not_fired"]


def test_recovery_is_checked_in_execution_order(tmp_path, no_outbound):
    # one turn, two calls: the loop runs the read-only read_file before the write, so the
    # injection hits the call listed second and the write that follows it is the recovery
    (tmp_path / "x").mkdir()
    t = task(kind="error-recovery", prompt="Save a note about cobalt-intake.", expect_recovery=True,
             setup_files={"notes/a.md": "alpha\n"}, inject=[dict(tg.TASKSET_INJECTION)])
    plan = [("calls", [("write_file", {"path": "notes/b.md", "content": "cobalt\n"}),
                       ("read_file", {"path": "notes/a.md"})]), "echo"]
    with FakeTeacher(scripted({"cobalt-intake": plan})) as fake:
        out, ok, reasons, record = run_and_verify(t, settings_for(tmp_path, fake.url))
    events = tg.tool_events(out)
    assert [e["name"] for e in events] == ["write_file", "read_file"]
    by = {e["name"]: e for e in events}
    assert by["read_file"]["outcome"] == "injected" and by["write_file"]["seq"] > by["read_file"]["seq"]
    assert ok, reasons
    assert record["recovered"] is True


def test_giving_up_after_the_injected_failure_is_still_rejected(tmp_path, no_outbound):
    t = _taskset_task(tmp_path, "Look up cobalt-intake in the wiki and tell me its host.", inject_errors=True)
    plan = [("call", "wiki_query", {"query": "cobalt-intake"}), ("text", "The wiki timed out, sorry.")]
    with FakeTeacher(scripted({"cobalt-intake": plan})) as fake:
        _out, ok, reasons, record = run_and_verify(t, settings_for(tmp_path, fake.url))
    assert not ok and "not_recovered" in reasons and record["recovered"] is False


# ── the gym's permission gate ──────────────────────────────────────────────────


def test_loopback_curl_reaches_the_sandbox_and_ssh_is_refused_not_an_error(tmp_path, no_outbound):
    svcs = [tg.svc_record(random.Random(1), "cobalt-intake", ["amber-ledger"])]
    world = tg.ops_world(random.Random(2), svcs)
    port = world["services"]["cobalt-intake"]["port"]
    t = task(prompt="Is cobalt-intake answering on its port?", fixtures={"ops": world})
    plan = [("call", "bash", {"command": f"curl -s http://127.0.0.1:{port}/health"}), "echo"]
    with FakeTeacher(scripted({"cobalt-intake": plan})) as fake:
        out, ok, reasons, record = run_and_verify(t, settings_for(tmp_path, fake.url))
    assert ok, reasons
    assert [e["outcome"] for e in tg.tool_events(out)] == ["simulated"]
    assert '"status": "ok"' in tg.tool_events(out)[0]["result"]
    # ssh: approved at the gate, refused by the sandbox (a refusal, never a natural error)
    plan = [("call", "bash", {"command": "ssh node-bravo uptime"}), "echo"]
    with FakeTeacher(scripted({"cobalt-intake": plan})) as fake:
        out, ok, reasons, record = run_and_verify(t, settings_for(tmp_path, fake.url))
    assert reasons == ["sandbox_refusal"] and record["tool_errors"] == 0
    # the gate's floor still denies, and that counts as a refusal too
    plan = [("call", "bash", {"command": "rm -rf /"}), "echo"]
    with FakeTeacher(scripted({"cobalt-intake": plan})) as fake:
        out, ok, reasons, record = run_and_verify(t, settings_for(tmp_path, fake.url))
    assert "sandbox_refusal" in reasons and record["tool_errors"] == 0
    assert out.env.gate_refused and not out.env.log
    assert asyncio.run(tg.gym_permission_prompt("write_file", "outside the workspace")) is False


# ── calls with no arguments ───────────────────────────────────────────────────


def test_a_call_with_no_arguments_is_its_own_defect_not_a_tool_error(tmp_path, no_outbound):
    t = task(prompt="Read notes/a.md about cobalt-intake.", setup_files={"notes/a.md": "alpha\n"})
    plan = [("call", "read_file", {}), ("call", "read_file", {"path": "notes/a.md"}), "echo"]
    with FakeTeacher(scripted({"cobalt-intake": plan})) as fake:
        out, ok, reasons, record = run_and_verify(t, settings_for(tmp_path, fake.url))
    assert [e["outcome"] for e in tg.tool_events(out)] == ["empty_args", "ok"]
    assert reasons == ["empty_tool_args"] and record["tool_errors"] == 0 and record["empty_arg_calls"] == 1
    # a tool with nothing required may be called with {} (todo_write needs todos; tool_search a query)
    from prometheus.engine.messages import ConversationMessage, ToolUseBlock

    msg = ConversationMessage(role="assistant", content=[ToolUseBlock(id="a", name="noargs", input={}),
                                                         ToolUseBlock(id="b", name="needs", input={})])
    req = type("R", (), {"tools": [{"name": "noargs", "input_schema": {"type": "object", "properties": {}}},
                                   {"name": "needs", "input_schema": {"type": "object", "required": ["x"]}}]})()
    assert tg.empty_arg_call_ids(msg, req) == ("b",)


# ── compaction padding ────────────────────────────────────────────────────────


def test_a_failed_pad_turn_is_rolled_back_and_replaced(tmp_path, no_outbound):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "backups.md").write_text(
        "# Backups\n\n" + "Backups run nightly and are kept for two weeks; test a restore monthly. " * 8)
    prompts = ["Which host runs cobalt-intake according to the wiki notes?",
               "Find the owner of amber-ledger in the wiki notes.",
               "List the dependencies of harbor-sync from the wiki notes.",
               "Look up the port quill-indexer uses in the wiki notes.",
               "Summarize the backup policy for the home server from the docs."]
    rows = [_taskset_row(p) for p in prompts[:4]] + [_taskset_row(prompts[4], force_compaction=True)]
    tasks = tg.tasks_from_taskset_rows(rows, tmp_path)
    padded = tasks[-1]
    assert padded.compaction and len(padded.pad) == 4
    bad = padded.pad[0]["prompt"]
    plans = {bad[:40]: [("call", "bash", {"command": "cat /etc/hosts"}), "echo"]}
    plans.update({p[:40]: [("call", "read_file", {"path": "docs/backups.md"}), "echo"]
                  for p in prompts if p != bad})
    with FakeTeacher(scripted(plans)) as fake:
        s = settings_for(tmp_path, fake.url, compaction_history_tokens=150)
        out, ok, reasons, record = run_and_verify(padded, s)
    assert not out.pad_reasons
    assert out.pad["skipped"] == [{"id": padded.pad[0]["id"], "reason": "pad_sandbox_refusal"}]
    assert padded.pad[0]["id"] not in out.pad["task_ids"] and out.pad["turns"] >= tg.PAD_MIN_TURNS
    # nothing of the rolled-back turn is history: its prompt and its refused call are gone
    history = json.dumps([m.model_dump(mode="json") for m in out.messages])
    assert bad not in history and "/etc/hosts" not in history
    assert "pad_sandbox_refusal" not in reasons


def test_too_few_clean_pad_turns_still_reject_the_task(tmp_path, no_outbound):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "backups.md").write_text("# Backups\n\n" + "Backups run nightly. " * 40)
    prompts = ["Which host runs cobalt-intake according to the wiki notes?",
               "Find the owner of amber-ledger in the wiki notes.",
               "List the dependencies of harbor-sync from the wiki notes.",
               "Summarize the backup policy for the home server from the docs."]
    rows = [_taskset_row(p) for p in prompts[:3]] + [_taskset_row(prompts[3], force_compaction=True)]
    padded = tg.tasks_from_taskset_rows(rows, tmp_path)[-1]
    plans = {p[:40]: [("call", "bash", {"command": "cat /etc/hosts"}), "echo"] for p in prompts[:3]}
    with FakeTeacher(scripted(plans)) as fake:
        out = asyncio.run(tg.run_task(padded, settings_for(tmp_path, fake.url, compaction_history_tokens=150)))
    assert out.pad_reasons == ["pad_too_few_turns"] and len(out.pad["skipped"]) == 3


# ── the class caps ─────────────────────────────────────────────────────────────


def _row(task_id: str, run_id: str, i: int, **meta: Any) -> dict[str, Any]:
    # letters, not digits: build.py's dedupe key normalizes digits away
    word = "".join("abcdefghij"[int(c)] if c.isdigit() else c for c in task_id)
    return {"messages": [{"role": "user", "content": f"question {word}"}, {"role": "assistant", "content": "ok"}],
            "_meta": {"task_id": task_id, "gen_task_id": task_id, "run_id": run_id, "task_category": "agentic",
                      "category": "other", **meta}}


def test_reported_blocked_and_no_tool_rows_are_written_tagged_with_their_shares(tmp_path):
    # the Max trial: 13 verified, 12 held behind run-level caps, 1 row written. Now every
    # verified row is written, tagged; build.py enforces the caps over the dataset.
    store = tg.Store(tmp_path / "s", "x", {"x": 1}, 1.0, class_caps={"reported_blocked": 0.15,
                                                                    "no_tool_answer": 0.20})
    blocked = [_row("b1", "rb1", 0, judge_outcome="reported_blocked")]
    assert store.commit(task(id="b1"), "rb1", "kept", blocked, []) == "kept"
    for k in range(3):
        assert store.commit(task(id=f"n{k}"), f"rn{k}", "kept",
                            [_row(f"n{k}", f"rn{k}", k, no_tool_answer=True, judge_outcome="done")], []) == "kept"
    assert store.commit(task(id="d0"), "rd0", "kept", [_row("d0", "rd0", 0, judge_outcome="done")], []) == "kept"
    rows = [json.loads(x) for x in store.traces.read_text().splitlines()]
    assert len(rows) == 5 and not store.held.exists() and not store.held_queue
    assert [r["_meta"]["judge_outcome"] for r in rows] == ["reported_blocked", "done", "done", "done", "done"]
    assert sum(bool(r["_meta"].get("no_tool_answer")) for r in rows) == 3
    report = store.report(settings_for(tmp_path, "http://127.0.0.1:1"), 0.0)
    assert report["by_class"]["reported_blocked"] == {"rows": 1, "cap": 0.15, "share": 0.2}
    assert report["shares"]["classes"]["reported_blocked"]["over_cap"]
    assert report["shares"]["classes"]["no_tool_answer"] == {"rows": 3, "cap": 0.2, "share": 0.6,
                                                             "over_cap": True}
    again = tg.Store(tmp_path / "s", "x", {"x": 1}, 1.0)
    assert again.rows_class["reported_blocked"] == 1 and again.rows_class["no_tool_answer"] == 3
    assert tg.row_classes(_row("n", "r", 0, no_tool_answer=True)) == ["no_tool_answer"]


def _held_output(out: Path, *, torn_release: bool = False) -> None:
    """An output dir as the run-level-cap generator left it: one kept run, two held."""
    out.mkdir(parents=True)
    (out / "x.settings.json").write_text(json.dumps({"x": 1}, indent=1) + "\n")
    kept = [_row("k1", "rk1", 0, judge_outcome="done")]
    held1 = [_row("h1", "rh1", 0, judge_outcome="reported_blocked"),
             _row("h1", "rh1", 1, judge_outcome="reported_blocked")]
    held1[1]["messages"][0]["content"] += " again"
    held2 = [_row("h2", "rh2", 0, no_tool_answer=True)]
    events = [{"task_id": "k1", "run_id": "rk1", "kind": "agentic", "status": "kept", "rows": 1},
              {"task_id": "h1", "run_id": "rh1", "kind": "agentic", "status": "held", "rows": 2},
              {"task_id": "h2", "run_id": "rh2", "kind": "agentic", "status": "held", "rows": 1}]
    traces = kept + (held1 if torn_release else [])  # a crash after the append, before the event
    (out / "x.tasks.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (out / "x.traces.jsonl").write_text("".join(json.dumps(r) + "\n" for r in traces))
    (out / "x.held.jsonl").write_text("".join(json.dumps(r) + "\n" for r in held1 + held2))


@pytest.mark.parametrize("torn_release", [False, True])
def test_a_held_rows_file_from_an_earlier_run_is_released_on_resume(tmp_path, torn_release):
    out = tmp_path / "s"
    _held_output(out, torn_release=torn_release)
    store = tg.Store(out, "x", {"x": 1}, 1.0)
    rows = [json.loads(x) for x in store.traces.read_text().splitlines()]
    # every row once, in held order, the held file emptied, a released event per held run
    assert [(r["_meta"]["run_id"], r["messages"][0]["content"]) for r in rows] == [
        ("rk1", "question kb"), ("rh1", "question hb"), ("rh1", "question hb again"), ("rh2", "question hc")]
    assert not store.held.read_text().strip() and not store.held_queue
    assert store.total_rows == 4 and store.rows_class["reported_blocked"] == 2
    assert {t: store.status[t]["status"] for t in ("k1", "h1", "h2")} == {
        "k1": "kept", "h1": "released", "h2": "released"}
    ledger = [json.loads(x) for x in (out / "x.tasks.jsonl").read_text().splitlines()]
    assert [e["task_id"] for e in ledger if e["status"] == "released"] == ["h1", "h2"]
    assert store.repairs.get("orphan_trace_rows", 0) == (2 if torn_release else 0)
    # the next open finds nothing to release or repair
    again = tg.Store(out, "x", {"x": 1}, 1.0)
    assert again.total_rows == 4 and not again.repairs
    assert len(again.traces.read_text().splitlines()) == 4


def test_a_release_cut_off_after_its_event_drops_the_stale_held_copy(tmp_path):
    # a crash after h1's rows and its released event were written, before the held
    # file was rewritten: h1's rows are in both files; the held copy is the orphan
    out = tmp_path / "s"
    _held_output(out, torn_release=True)
    with (out / "x.tasks.jsonl").open("a") as fh:
        fh.write(json.dumps({"task_id": "h1", "status": "released", "run_id": "rh1"}) + "\n")
    store = tg.Store(out, "x", {"x": 1}, 1.0)
    rows = [json.loads(x) for x in store.traces.read_text().splitlines()]
    assert [r["_meta"]["run_id"] for r in rows] == ["rk1", "rh1", "rh1", "rh2"]
    assert store.repairs == {"orphan_held_rows": 2} and not store.held.read_text().strip()
    ledger = [json.loads(x) for x in (out / "x.tasks.jsonl").read_text().splitlines()]
    assert [e["task_id"] for e in ledger if e["status"] == "released"] == ["h1", "h2"]
    assert store.total_rows == 4


# ── host inspection in the sandbox ────────────────────────────────────────────


def _bash(env: Any, reg: Any, cmd: str) -> Any:
    return _call(env, reg, "bash", {"command": cmd})


def test_host_inspection_is_answered_from_the_fixture(tmp_path):
    env, reg = _env(tmp_path)
    w = env.ops()
    assert "MOUNTPOINTS" in _bash(env, reg, "lsblk").output
    res = _bash(env, reg, "df -h; lsblk; mount")
    assert not res.is_error and "Mounted on" in res.output and " on /data type ext4" in res.output
    res = _bash(env, reg, "echo ---; nvidia-smi")
    assert res.output.startswith("---\n") and "Memory-Usage" in res.output
    assert _bash(env, reg, "hostname").output == w["host"]
    assert _bash(env, reg, "uname -n").output == w["host"]
    res = _bash(env, reg, "cat /proc/meminfo | grep MemTotal")
    assert res.output.startswith("MemTotal:") and str(w["mem"]["total"] * 1024 * 1024) in res.output
    assert "/data" in _bash(env, reg, "grep data /proc/mounts").output
    assert _bash(env, reg, "grep -c processor /proc/cpuinfo").output == "16"
    read = _call(env, reg, "read_file", {"path": "/etc/os-release"})
    assert not read.is_error and read.output.startswith("     1\tPRETTY_NAME=")
    assert _bash(env, reg, "which lsblk").output == "/usr/bin/lsblk"
    which = _bash(env, reg, "which python3")
    assert which.is_error and which.output == "(no output)"
    assert _bash(env, reg, "command -v grep").output == "/usr/bin/grep"
    assert not _bash(env, reg, "sleep 600 && uptime -p").is_error
    assert _bash(env, reg, "systemctl status nope || echo fallback").output.endswith("fallback")
    port = next(iter(w["services"].values()))["port"]
    assert f"127.0.0.1:{port}" in _bash(env, reg, "ss -tlnp").output
    # the host's own files are never opened, and these still refuse
    for cmd in ("mount /dev/sdb1 /mnt", "cat /proc/self/environ", "cat /etc/shadow", "command ls",
                "ls /proc", "rm /proc/meminfo"):
        assert _bash(env, reg, cmd).output.startswith("Refused:"), cmd
    assert {r.outcome for r in env.log[:-6]} <= {"simulated", "ok"}


def test_tilde_is_the_sandbox_root(tmp_path):
    env, reg = _env(tmp_path)
    res = _bash(env, reg, "echo hi > ~/note.txt && cat ~/note.txt")
    assert not res.is_error and res.output == "hi" and (env.root / "note.txt").read_text() == "hi\n"
    read = _call(env, reg, "read_file", {"path": "~/note.txt"})
    assert not read.is_error and read.output.endswith("\thi")
    assert not _call(env, reg, "write_file", {"path": "~/sub/x.md", "content": "x"}).is_error
    assert (env.root / "sub" / "x.md").read_text() == "x"
    for path in ("~root/x", "~/../escape"):
        assert _call(env, reg, "read_file", {"path": path}).output.startswith("Refused:"), path


# ── the read-only repo fixture ───────────────────────────────────────────────


def _git(repo: Path, *args: str) -> None:
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                    "-c", "commit.gpgsign=false", *args], check=True, capture_output=True, env=env)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("needs git")
    r = tmp_path / "repo"
    (r / "docs").mkdir(parents=True)
    (r / "docs" / "guide.md").write_text("# Guide\n\nRun the daemon with `prometheus daemon`.\n")
    (r / "README.md").write_text("# Prometheus\n")
    _git(r, "init", "-q", "-b", "main")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "first: the guide")
    (r / "README.md").write_text("# Prometheus\n\nA personal agent.\n")
    _git(r, "commit", "-q", "-am", "second: the readme")
    _git(r, "update-ref", "refs/remotes/origin/main", "HEAD")
    # never in the fixture: a local edit, an untracked file, a local commit past the ref
    (r / "README.md").write_text("LOCAL EDIT\n")
    (r / "private-notes.md").write_text("untracked\n")
    return r


@needs_git
def test_every_agent_task_gets_a_read_only_clone_at_the_published_ref(tmp_path, repo, no_outbound):
    s = settings_for(tmp_path, "http://127.0.0.1:1", repo=repo, repo_fixture="on")
    info = tg.prepare_repo_store(s)
    assert info and info["ref"] == "refs/remotes/origin/main" and len(info["commit"]) == 40
    assert s.fingerprint()["repo_fixture"] == info["commit"]
    root = (tmp_path / "ws").resolve()
    root.mkdir()
    dest = tg.clone_repo_fixture(s.repo_store, root)
    assert (dest / "README.md").read_text() == "# Prometheus\n\nA personal agent.\n"
    assert not (dest / "private-notes.md").exists()
    assert not (dest / ".git").is_dir() and (tg.repo_git_dir(root) / "HEAD").is_file()  # git dir outside
    env = tg.SandboxEnv(root, task(), jail=tg.resolve_jail("auto"), git_dir=tg.repo_git_dir(root))
    reg = tg.build_sandbox_registry(env)
    log = _bash(env, reg, "cd Prometheus && git log --oneline")
    assert not log.is_error and "second: the readme" in log.output and "first: the guide" in log.output
    for cmd, want in (("git -C Prometheus status --short", None), ("git -C Prometheus branch -a", "main"),
                      ("git -C Prometheus show HEAD:docs/guide.md", "prometheus daemon"),
                      ("git -C Prometheus grep -n daemon", "docs/guide.md:3"),
                      ("git -C Prometheus log -n 1 --format=%s", "second: the readme")):
        res = _bash(env, reg, cmd)
        assert not res.is_error, (cmd, res.output)
        assert want is None or want in res.output, (cmd, res.output)
    assert "origin" not in _bash(env, reg, "git -C Prometheus remote -v").output
    assert _bash(env, reg, "which git").output == "/usr/bin/git"
    for cmd in ("git -C Prometheus fetch", "git -C Prometheus push", "git -C Prometheus branch new",
                "git -C Prometheus tag v1", "git -C Prometheus remote add x /tmp/x", "git -c core.pager=sh log",
                "git -C Prometheus log --output=out.txt", "git -C Prometheus grep -Osh x",
                "git -C Prometheus diff --ext-diff", "git --git-dir=/tmp/x log", "git -C / log",
                "git -C Prometheus diff --no-index /etc/hosts README.md", "git -C Prometheus config user.name",
                "git -C Prometheus checkout -b x", "git -C Prometheus commit -m x"):
        assert _bash(env, reg, cmd).output.startswith("Refused:"), cmd


@needs_git
def test_the_repo_fixture_is_only_for_real_pattern_agent_tasks(tmp_path, repo):
    s = settings_for(tmp_path, "http://127.0.0.1:1", repo=repo, repo_fixture="auto")
    assert tg.prepare_repo_store(s)
    agent = _taskset_task(tmp_path, "Which host runs cobalt-intake according to the wiki?")
    chat = tg.tasks_from_taskset_rows([_taskset_row("What is a maintenance window, in plain words?",
                                                    suppress_tools=True)], tmp_path)[0]
    assert tg.wants_repo_fixture(agent, s) and not tg.wants_repo_fixture(chat, s)
    assert not tg.wants_repo_fixture(task(), s)  # a built-in smoke task
    # auto stays off where it cannot be built; on refuses to run without it
    plain = tmp_path / "plain"
    plain.mkdir()
    off = settings_for(tmp_path, "http://127.0.0.1:1", repo=plain, repo_fixture="auto")
    assert tg.prepare_repo_store(off) is None and off.repo_store is None
    with pytest.raises(tg.RepoFixtureError):
        tg.prepare_repo_store(settings_for(tmp_path, "http://127.0.0.1:1", repo=plain, repo_fixture="on"))
    # a local branch can hold unpublished commits: only a remote-tracking ref or a tag
    for ref in ("HEAD", "main"):
        bad_ref = settings_for(tmp_path, "http://127.0.0.1:1", repo=repo, repo_fixture="on", repo_fixture_ref=ref)
        with pytest.raises(tg.RepoFixtureError):
            tg.prepare_repo_store(bad_ref)


def _plant_repo(env: Any, reg: Any, d: str) -> None:
    """A repository written into the workspace whose config makes git run a program."""
    for path, content in ((f"{d}/.git/HEAD", "ref: refs/heads/main\n"),
                          (f"{d}/.git/config", "[core]\n\trepositoryformatversion = 0\n"
                                               "[diff]\n\texternal = /bin/sh -c 'id > PLANTED' x\n"),
                          (f"{d}/.git/objects/.keep", ""), (f"{d}/.git/refs/.keep", ""),
                          (f"{d}/a", "one\n"), (f"{d}/b", "two\n")):
        assert not _call(env, reg, "write_file", {"path": path, "content": content}).is_error, path


@needs_git
def test_git_never_reads_a_repository_the_agent_wrote(tmp_path, repo, no_outbound):
    """A repository's config can make git run any program (diff.external, textconv,
    filters): git reads only the fixture's git directory, outside the workspace."""
    # no fixture (a built-in task, a chat row): no git at all
    (tmp_path / "plain").mkdir()
    env, reg = _env(tmp_path / "plain")
    _plant_repo(env, reg, "z")
    res = _bash(env, reg, "cd z && git diff --no-index a b")
    assert res.output.startswith("Refused:") and not (env.root / "z" / "PLANTED").exists()
    assert _bash(env, reg, "which git").is_error
    # with the fixture: pinned to its git dir, wherever the command runs
    s = settings_for(tmp_path, "http://127.0.0.1:1", repo=repo, repo_fixture="on")
    tg.prepare_repo_store(s)
    root = (tmp_path / "ws").resolve()
    root.mkdir()
    tg.clone_repo_fixture(s.repo_store, root)
    env = tg.SandboxEnv(root, task(), jail=tg.resolve_jail("auto"), git_dir=tg.repo_git_dir(root))
    reg = tg.build_sandbox_registry(env)
    _plant_repo(env, reg, "z")
    _plant_repo(env, reg, "Prometheus/sub")
    for cmd in ("cd z && git diff --no-index a b", "git -C z diff --no-index a b",
                "cd Prometheus/sub && git diff --no-index a b"):
        _bash(env, reg, cmd)
        assert not list(root.rglob("PLANTED")), cmd
    # replacing the clone's .git file does not move git either
    assert not _call(env, reg, "write_file", {"path": "Prometheus/.git",
                                              "content": f"gitdir: {root / 'z' / '.git'}\n"}).is_error
    _bash(env, reg, "cd Prometheus && git diff --no-index README.md docs/guide.md")
    assert not list(root.rglob("PLANTED"))
    assert "second: the readme" in _bash(env, reg, "cd Prometheus && git log --oneline").output
    # and the fixture's git dir is out of every tool's reach
    gd = tg.repo_git_dir(root)
    assert _call(env, reg, "write_file", {"path": str(gd / "config"), "content": "x"}).output.startswith("Refused:")
    assert _bash(env, reg, f"echo x >> {gd / 'config'}").output.startswith("Refused:")
    assert "external" not in (gd / "config").read_text()


def test_the_jail_lets_git_read_the_fixture_git_dir_and_never_write_it(tmp_path):
    root, gd = tmp_path / "ws", tmp_path / "ws.repo-git"
    argv = tg.jail_prefix("bwrap", root, root, (gd,))
    at = argv.index(str(gd))
    assert argv[at - 1: at + 2] == ["--ro-bind", str(gd), str(gd)] and argv.count(str(gd)) == 2
    # mounted after the /tmp tmpfs (the default sandbox root is under /tmp), never writable
    assert argv.index("/tmp") < at < argv.index("--bind")
    profile = tg.jail_prefix("sandbox-exec", root, root, (gd,))[2]
    assert f'(require-not (subpath "{gd}"))' in profile.split("(deny file-read*")[1]
    assert str(gd) not in profile.split("(deny file-read*")[0]  # writes stay confined to the workspace


def test_a_list_part_after_a_failed_cd_never_runs_in_another_directory(tmp_path):
    """A part checked against the directory a `cd` named never runs elsewhere: from
    the workspace, a path that stayed inside it from that directory could leave it."""
    env, reg = _env(tmp_path)
    (tmp_path / "outside.txt").write_text("SECRET\n")
    assert env.root.parent == tmp_path.resolve()
    res = _bash(env, reg, "cd nodir; df -h; cat ../outside.txt")
    assert "SECRET" not in res.output and "can't cd to nodir" in res.output and res.is_error
    ok = _bash(env, reg, "mkdir sub; cd sub; df -h | head -1; pwd")
    assert not ok.is_error and ok.output.splitlines()[-1].endswith("/sub")


# ── chat mode: no tools offered, and the system prompt says so ─────────────────


def test_chat_mode_turn_offers_no_tools_and_its_system_prompt_says_so(tmp_path, no_outbound):
    t = task(kind="chat-mode", domain="chat-mode", mode="chat",
             prompt="# amber-ledger\n- Runs on: edge-02\n\nWhich host runs amber-ledger?",
             score={"expect_text_contains": "edge-02"})
    with FakeTeacher() as fake:
        s = settings_for(tmp_path, fake.url)
        out, ok, reasons, record = run_and_verify(t, s)
        agent = [r for r in fake.requests if r.get("stream")]
    assert ok, reasons
    for req in agent:
        assert not req.get("tools") and "tool_choice" not in req
        system = req["messages"][0]
        assert system["role"] == "system" and system["content"].rstrip().endswith(tg.CHAT_MODE_NOTE)
    assert out.system_prompt.endswith(tg.CHAT_MODE_NOTE) and "chat_requests_with_tools" not in record
    # an agent turn's prompt has no such note
    t2 = task(prompt="Which host runs cobalt-intake? Use the wiki.", fixtures={"wiki": WIKI})
    with FakeTeacher(scripted({"cobalt-intake": [("call", "wiki_query", {"query": "cobalt"}), "echo"]})) as fake:
        out2, _ok, _r, _rec = run_and_verify(t2, settings_for(tmp_path / "b", fake.url))
        first = next(r for r in fake.requests if r.get("stream"))
    assert tg.CHAT_MODE_NOTE not in first["messages"][0]["content"] and first.get("tools")
    assert tg.CHAT_MODE_NOTE not in out2.system_prompt


def test_chat_mode_call_is_still_rejected_and_tools_on_the_wire_are_caught(tmp_path, monkeypatch):
    t = task(kind="chat-mode", domain="chat-mode", mode="chat", prompt="Which host runs amber-ledger?")
    with FakeTeacher(scripted({"amber-ledger": [("call", "wiki_query", {"query": "amber"}), "echo"]})) as fake:
        out, ok, reasons, _record = run_and_verify(t, settings_for(tmp_path, fake.url))
        agent = [r for r in fake.requests if r.get("stream")]
    assert all(not r.get("tools") for r in agent)
    assert not ok and "chat_mode_tool_call" in reasons and "chat_mode_tools_offered" not in reasons
    # a provider that put tools on the wire anyway: the row is refused, not trusted
    t2 = task(kind="chat-mode", domain="chat-mode", mode="chat",
              prompt="# amber-ledger\n- Runs on: edge-02\n\nWhich host runs amber-ledger?")
    with FakeTeacher() as fake:
        s = settings_for(tmp_path / "b", fake.url)
        out2 = asyncio.run(tg.run_task(t2, s))
    real = out2.provider._build_request_payload
    monkeypatch.setattr(out2.provider, "_build_request_payload",
                        lambda req: {**real(req), "tools": [{"type": "function", "function": {"name": "bash"}}]})
    ok2, reasons2, record2 = tg.deterministic_checks(out2, s)
    assert not ok2 and "chat_mode_tools_offered" in reasons2 and record2["chat_requests_with_tools"] >= 1


# ── compaction: a system prompt larger than the small window ───────────────────


def _big_prompt(tokens: int) -> str:
    line = "Policy: answer from the conversation and keep replies short and plain. "
    return "You are a test agent.\n" + line * (tokens * 4 // len(line))


def _chat_compaction_task() -> Any:
    seed = []
    for i in range(10):
        seed += [{"user": ("We decided the deploy day is Thursday. " if i == 0 else "")
                  + f"Reference {i}: " + ("lorem ipsum dolor sit amet consectetur " * 60)},
                 {"assistant_text": f"Noted reference {i}."}]
    return task(kind="compacted-history", domain="chat-mode", mode="chat", prompt="Which deploy day did we pick?",
                seed=seed, compaction=True)


def test_a_system_prompt_over_the_small_window_gets_a_window_the_compactor_can_converge_in(
        tmp_path, no_outbound, caplog):
    from prometheus.context.compactor import ContextCompactor
    from prometheus.context.token_estimation import estimate_tokens

    big = _big_prompt(40_000)
    t = _chat_compaction_task()
    with FakeTeacher(scripted({"deploy day": [("text", "We picked Thursday.")]})) as fake:
        s = settings_for(tmp_path, fake.url, base_system_prompt=big)
        out, ok, reasons, record = run_and_verify(t, s)
        assert fake.kinds["summary"] >= 1
    assert ok, reasons
    # the compactor's own word: no pass landed over its line
    assert not [r for r in caplog.records if "did NOT converge" in r.getMessage()]
    win = record["compaction"]["window"]
    sys_tokens = estimate_tokens(out.system_prompt)
    hist, room = s.compaction_history_tokens, s.max_tokens
    assert win["resized"] and win["converged"] and win["passes"] >= 1 and win["resized_passes"] >= 1
    assert win["configured"] == sys_tokens + hist + room          # the plain small window
    assert win["system_tokens"] == sys_tokens and win["history_tokens"] == hist
    assert win["line"] == sys_tokens + hist and win["effective"] == sys_tokens + hist + room
    assert sys_tokens + 512 > win["configured_line"]               # why: no room under its line
    rows = tg.rows_for(out, s, record)
    assert len(rows) == 1 and rows[0]["_meta"]["compacted"] and rows[0]["_meta"]["compaction_window"] == win
    # the plain compactor at the configured window cannot get under its own line
    plain = ContextCompactor(provider=None, model=s.model_name, effective_limit=win["configured"],
                             threshold_pct=tg.COMPACTION_THRESHOLD_PCT, reserve_tokens=room)
    floor = sys_tokens + 512
    assert floor > min(plain.threshold_for()[0], plain.limit_for() - room)


def test_the_window_follows_the_compactors_calibration_and_is_unchanged_when_it_fits(tmp_path):
    from prometheus.context.token_estimation import estimate_tokens

    from prometheus.engine.messages import ConversationMessage

    s = settings_for(tmp_path, "http://127.0.0.1:1")
    msgs = [ConversationMessage.from_user_text("hello")]
    # a small prompt: the configured window, untouched
    small = "You are a test agent."
    comp = tg._make_compactor(s, small, None, {"n": 0}, chat=True, provider=object())
    sys_t, window, line = comp.size_for(msgs, session_id="system", system_prompt=small, model=s.model_name,
                                        max_tokens=s.max_tokens)
    assert comp.sized is None and window == comp.configured_window and line == comp.configured_line
    # a prompt that fits at the bare estimate but not once the server counted 1.4x
    mid = _big_prompt(24_000)
    comp = tg._make_compactor(s, mid, None, {"n": 0}, chat=True, provider=object())
    comp.size_for(msgs, session_id="system", system_prompt=mid, model=s.model_name, max_tokens=s.max_tokens)
    assert comp.sized is None
    comp.observe_prompt_tokens(session_id="system", model=s.model_name, estimated=1000, reported=1400)
    sys_t, window, line = comp.size_for(msgs, session_id="system", system_prompt=mid, model=s.model_name,
                                        max_tokens=s.max_tokens)
    assert sys_t == int(estimate_tokens(mid) * 1.4)
    assert comp.sized == (window, line) and line == sys_t + s.compaction_history_tokens
    assert window == line + s.max_tokens
    assert comp.limit_for(s.model_name) == window and comp.threshold_for(s.model_name)[0] == line
