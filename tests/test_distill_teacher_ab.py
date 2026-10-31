"""Tests for the teacher integration: teacher_gen.py's real-pattern task source,
and teacher_ab.py / teacher_ab_run.sh (the IQ3 vs Q4_K_M teacher A/B).

Fake OpenAI-compatible servers on loopback stand in for llama-server; the real
agent loop, sandbox, compactor and judge code run against them. No real model,
no GPU, no network beyond 127.0.0.1, no live Prometheus state.
"""

from __future__ import annotations

import asyncio
import collections
import datetime as dt
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.test_distill_build import feed, make_pack, make_student
from tests.test_distill_teacher_gen import (  # noqa: F401 - fixtures are used by name
    FakeTeacher,
    _last_user,
    _load,
    _restore_state_env,
    echo_reply,
    run_and_verify,
    settings_for,
    task,
    tg,
)

ab = _load("distill_teacher_ab", "teacher_ab.py")
build = _load("distill_build", "build.py")
ts_gen = _load("distill_taskset_gen", "taskset_gen.py")
_REPO = Path(__file__).resolve().parent.parent
RUN_SH = _REPO / "scripts" / "distill" / "teacher_ab_run.sh"
FAKE_LLAMA = _REPO / "tests" / "fixtures" / "teacher_ab" / "fake_llama_server.py"
SAMPLING = {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.0}


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


# ── taskset rows ───────────────────────────────────────────────────────────────

SUBCAT = {"chat-mode": "general", "memory/wiki": "wiki", "sentinel/daemon": "daemon-ops", "coding": "coding",
          "other": "research"}


def ts_row(prompt: str, category: str = "memory/wiki", *, group: str | None = None,
           recombine_with: str | None = None, **flags: bool) -> dict[str, Any]:
    """A taskset_gen.py row (passes its validate_task). *group* shares a task
    group between variants of one real prompt."""
    chat = category == "chat-mode"
    ph = tg.h("pattern|" + (group or prompt))
    tpl = "tpl-" + tg.h("tpl|" + (group or prompt))
    sources = [{"template_id": tpl, "pattern_hash": ph, "origin": "golden", "span": [0, len(prompt)]}]
    if recombine_with:
        sources.append({"template_id": "tpl-" + tg.h("tpl|" + recombine_with),
                        "pattern_hash": tg.h("pattern|" + recombine_with), "origin": "lcm", "span": [0, 4]})
    row = {"id": "ts-" + tg.h(tg.norm(prompt)), "taskset": "real-test", "category": category,
           "subcategory": SUBCAT[category], "intent": "question" if chat else "find",
           "mode": "chat" if chat else "agentic", "prompt": prompt, "slots": [], "template_id": tpl,
           "source_pattern_hash": ph, "task_group": "tg-" + ph,
           "transform": "recombine" if recombine_with else "slot_fill", "sources": sources,
           "flags": {"suppress_tools": chat, "inject_errors": False, "force_compaction": False, **flags},
           "provenance": "real-pattern-v1"}
    assert ts_gen.validate_task(row) == [], ts_gen.validate_task(row)
    return row


def pad_rows() -> list[dict[str, Any]]:
    """Plain real-pattern rows a force_compaction row pads with (other task groups)."""
    return [ts_row("Which team owns harbor-sync according to the wiki?"),
            ts_row("What is a good way to keep notes for an on-call handoff?", "chat-mode"),
            ts_row("Find the port quill-indexer listens on in the wiki.")]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


DOC_PARAS = [
    "Backups run every night and are kept for two weeks. Before a restore, stop the service that writes to the "
    "store, copy the snapshot back into place, start the service again and check that its health endpoint answers. "
    "A restore that has not been tested is a hope rather than a plan, so the checklist asks for one test restore "
    "every month into a scratch directory, with the file counts and checksums compared against the source.",
    "Log rotation keeps the disk from filling up: each log file is renamed once it passes its size limit, the "
    "older copies are compressed, and the oldest are deleted after the retention period. When a disk does fill "
    "up anyway, look for a log that is no longer rotated, a cache that grew without a cap, or a core dump left "
    "behind by a crash, and clear the cause before clearing the space so that it does not come straight back.",
    "The scheduler runs jobs from a plain table of times and commands. A job that fails sends one message with "
    "its exit code and the last lines of its output, and a job that runs longer than its limit is stopped. To "
    "change a schedule, edit the table, check the next run time it reports, and leave a note in the change log "
    "so that the person on call knows why a job moved and who to ask about it.",
]


def write_docs(root: Path) -> Path:
    """A repo with real-looking git-untracked docs (read through the rglob fallback):
    what a force_compaction task pads with. Returns *root*."""
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    for i, para in enumerate(DOC_PARAS):
        (docs / f"guide-{i}.md").write_text(f"# Guide {i}\n\n{para}\n\n{para}\n", encoding="utf-8")
    return root


def write_gguf(path: Path, *, template: str = "{% for m in messages %}{{ m.content }}{% endfor %}",
               file_type: int = 15, pad: int = 0) -> Path:
    """A minimal GGUF v3 header (no tensors): a string, a u32, a string array and a
    float array (skipped), and the chat template; *pad* bytes of zeros after it."""
    import struct

    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    kvs = [s("general.architecture") + struct.pack("<I", 8) + s("qwen35"),
           s("general.file_type") + struct.pack("<I", 4) + struct.pack("<I", file_type),
           s("general.sampling.top_k") + struct.pack("<I", 5) + struct.pack("<i", 20),
           s("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", 3)
           + s("a") + s("bb") + s("<|im_end|>"),
           s("tokenizer.ggml.scores") + struct.pack("<I", 9) + struct.pack("<I", 6) + struct.pack("<Q", 3)
           + struct.pack("<3f", 0.0, -1.0, -2.0),
           s("tokenizer.chat_template") + struct.pack("<I", 8) + s(template)]
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
    path.write_bytes(head + b"".join(kvs) + b"\0" * pad)
    return path


def base_inputs(tmp_path: Path, teacher_id: str = "fake-teacher") -> list[str]:
    ev = _load("distill_build_eval_sets", "build_eval_sets.py")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(ev.build_manifest([], 1)))
    evt = tmp_path / "eval-tools.jsonl"
    evt.write_text("")
    return ["--injection-manifest", str(manifest), "--eval-tools", str(evt), "--sandbox-root",
            str(tmp_path / "sbx"), "--quiet", "--request-timeout", "20", "--teacher-id", teacher_id,
            "--repo", str(tmp_path)]


def tool_then_echo(fail: tuple[str, ...] = ()) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """A teacher: with tools offered, two wiki_query calls then an answer quoting
    the results; without tools, the answer. A prompt holding a *fail* text gets
    "No." (too short for a chat task, no call for an agentic one)."""

    def policy(body: dict[str, Any]) -> dict[str, Any]:
        msgs = body["messages"]
        k, prompt = _last_user(msgs)
        if any(f in prompt for f in fail):
            return {"content": "No."}
        step = sum(1 for m in msgs[k + 1:] if m.get("role") == "assistant")
        if body.get("tools") and step < 2:
            return {"tool_calls": [("wiki_query", {"query": f"service host {step}"})]}
        return echo_reply(body)

    return policy


def gym_builder(d: Path, rows: list[dict[str, Any]], **settings: Any) -> Any:
    """build.py's Builder whose runtime catalog offers the tools the rows were offered."""
    d.mkdir(parents=True, exist_ok=True)
    pack = make_pack()
    seen: dict[str, dict[str, Any]] = {}
    for r in rows:
        for t in r.get("tools") or []:
            fn = t["function"]
            seen[fn["name"]] = {"name": fn["name"], "description": fn.get("description", ""),
                                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}}}
    tools = list(pack["tools"]["full"]) + [t for n, t in sorted(seen.items())
                                           if n not in {x["name"] for x in pack["tools"]["full"]}]
    pack["tools"] = {"full": tools, "stubs": sorted(tools, key=lambda t: t["name"])}
    s = build.Settings(train_range=(1, 100000), eval_min=1, min_presence=0, **settings)
    return build.Builder(student=make_student(), pack=pack, catalog="stubs", settings=s, index={}, verdicts={},
                         lcm=None, spool=d / "spool.jsonl", verdict_checks=None)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ── teacher_gen: the task source ───────────────────────────────────────────────


def test_cli_needs_exactly_one_task_source(tmp_path, no_outbound, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    taskset = write_jsonl(tmp_path / "real.jsonl", [ts_row("Which host runs cobalt-intake per the wiki?")])
    builtin = write_jsonl(tmp_path / "builtin.jsonl", [task(id="b-1", prompt="Say hello").to_dict()])
    dead = "http://127.0.0.1:1"
    run = ["run", "--out", str(tmp_path / "o"), "--teacher-url", dead, *base_inputs(tmp_path)]
    for extra, why in (([], "exactly one"), (["--tasks", str(taskset), "--builtin-smoke"], "exactly one"),
                       (["--tasks-file", str(builtin)], "needs --builtin-smoke"),
                       (["--tasks", str(builtin)], "not a taskset_gen.py row"),
                       (["--builtin-smoke", "--tasks-file", str(taskset)], "run real-pattern tasks with --tasks")):
        assert tg.main(run + extra) == 2, extra
        assert why in capsys.readouterr().err
    # a list-tasks row cannot claim to be a real-pattern task
    with pytest.raises(ValueError, match="built-in task"):
        tg.GenTask.from_dict(dict(task(id="b-2").to_dict(), origin="taskset"))
    # a bad sampling value is a usage error
    assert tg.main(run + ["--tasks", str(taskset), "--top-p", "1.5"]) == 2
    # a task built from a taskset row is a real-pattern task; nothing else is
    (real,) = tg.load_taskset(str(taskset), tmp_path)
    assert real.origin == tg.ORIGIN_TASKSET and tg.load_builtin_tasks(str(builtin))[0].origin == "builtin"


def test_tasks_run_writes_train_source_rows_that_build_py_admits(tmp_path, no_outbound, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    write_docs(tmp_path)  # --repo (base_inputs): what the force_compaction task pads with
    rows = [ts_row("What does a maintenance window mean for a small home server?", "chat-mode"),
            ts_row("Which host runs cobalt-intake according to the wiki?"),
            ts_row("Look up cobalt-intake in the wiki and tell me its host.", inject_errors=True),
            ts_row("Find the owner of amber-ledger in the wiki notes.", "other", force_compaction=True),
            ts_row("Check the wiki for harbor-sync and then list its port.", recombine_with="list its port")]
    taskset = write_jsonl(tmp_path / "real-test.jsonl", rows)
    out = tmp_path / "out"
    with FakeTeacher(tool_then_echo()) as fake:
        rc = tg.main(["run", "--tasks", str(taskset), "--out", str(out), "--teacher-url", fake.url,
                      "--compaction-history-tokens", "150", "--temperature", "0.7", "--top-k", "20",
                      "--top-p", "0.8", "--min-p", "0", *base_inputs(tmp_path)])
        requests = list(fake.requests)
    assert rc == 0
    traces = out / f"{tg.SOURCE_TAG}.traces.jsonl"
    got = [json.loads(x) for x in traces.read_text().splitlines()]
    ledger = [json.loads(x) for x in (out / f"{tg.SOURCE_TAG}.tasks.jsonl").read_text().splitlines()]
    assert {e["task_id"]: e["status"] for e in ledger} == {r["id"]: "kept" for r in rows}, ledger
    by_task = collections.defaultdict(list)
    for r in got:
        by_task[r["_meta"]["gen_task_id"]].append(r)
    assert set(by_task) == {r["id"] for r in rows}
    for row in rows:
        metas = [r["_meta"] for r in by_task[row["id"]]]
        assert {m["source"] for m in metas} == {tg.SOURCE_TAG}
        assert {m["task_origin"] for m in metas} == {"taskset"}
        assert {m["task_id"] for m in metas} == {row["task_group"]}  # build.py splits on the group
        assert {m["provenance"] for m in metas} == {"real-pattern-v1"} and {m["taskset"] for m in metas} == {"real-test"}
        assert all(m["sampling"] == SAMPLING for m in metas)
    chat, _plain, inject, comp, recomb = (by_task[r["id"]] for r in rows)
    assert chat[0]["tools"] == [] and chat[0]["_meta"]["suppress_tools"] is True
    assert any(r["_meta"]["error_recovery"] for r in inject)
    assert inject[0]["_meta"]["verification"]["injected_failures"] == 1
    # the padding is other real-pattern prompts of this taskset, answered by the same teacher
    others = {r["id"] for i, r in enumerate(rows) if i != 3}
    for r in comp:
        pad = r["_meta"]["compaction_pad"]
        assert r["_meta"]["compacted"] and pad["version"] == tg.PAD_VERSION and pad["answered_by"] == "teacher"
        assert pad["turns"] >= tg.PAD_MIN_TURNS and set(pad["task_ids"]) <= others
    # the sandbox's docs are the repo's own, never the synthetic runbooks written in teacher_gen.py
    comp_task = tg.tasks_from_taskset_rows(rows, tmp_path)[3]
    assert comp_task.setup_files and all("guide-" in f and "runbooks" not in f for f in comp_task.setup_files)
    assert all(any(p[:60] in text for p in DOC_PARAS) for text in comp_task.setup_files.values())
    assert {r["_meta"]["partner_task_id"] for r in recomb} == {"tg-" + rows[4]["sources"][1]["pattern_hash"]}
    assert all("partner_task_id" not in r["_meta"] for r in chat + inject + comp)
    # flags reached the wire: the chat request offered no tools; every teacher and
    # summarizer request carried the explicit sampling; the judge stayed at temperature 0
    chat_req = next(b for b in requests if b.get("stream") and rows[0]["prompt"] in json.dumps(b["messages"]))
    assert not chat_req.get("tools") and not chat_req.get("tool_choice")
    streamed = [b for b in requests if b.get("stream")]
    assert streamed and all({k: b.get(k) for k in SAMPLING} == SAMPLING for b in streamed)
    judges = [b for b in requests if not b.get("stream")]
    assert len(judges) == len(rows) and all(b["temperature"] == 0 and "top_k" not in b for b in judges)
    report = json.loads((out / f"{tg.SOURCE_TAG}.report.json").read_text())
    assert report["source"] == tg.SOURCE_TAG and report["task_source"] == "taskset" and report["sampling"] == SAMPLING

    # build.py's own reader, strict (a train-source row), on the rows as written
    for r in got:
        rec = r["_meta"]["verification"]
        assert rec["sandboxed"] is True and rec["side_effects"] is False and rec["verified"] is True
        v = build.verification_of(r["_meta"], strict=True, judge_models=build.DEFAULT_TRAIN_MODELS)
        assert (v.decision, v.basis) == (True, "judge-pass")
        # the check is real: the same record without either field fails closed
        for drop, why in (("sandboxed", "no_sandbox_record"), ("side_effects", "no_side_effects_record")):
            cut = dict(r["_meta"], verification={k: x for k, x in rec.items() if k != drop})
            assert build.verification_of(cut, strict=True, judge_models=build.DEFAULT_TRAIN_MODELS).basis == why
    # and the builder: every row is a train source, verified by its own record
    b = gym_builder(tmp_path / "b1", got, train_sources=build.DEFAULT_TRAIN_SOURCES)
    feed(b, traces, "extra")
    assert not [k for k in b.stats if k.startswith("excluded:")], dict(b.stats)
    assert len(b.examples) == len(got)
    assert all(e.train_ok and not e.max_derived and e.source == tg.SOURCE_TAG for e in b.examples)
    assert b.stats["verification:pass"] == len(got)

    # a smoke run of the built-in families: the same machinery, never a train source
    smoke = write_jsonl(tmp_path / "smoke.jsonl", [task(
        id="b-1", prompt="Which host runs cobalt-intake? Use the wiki.", domain="memory/wiki",
        fixtures={"wiki": {"cobalt-intake": "# cobalt-intake\n- Runs on: node-bravo\n"}},
        score={"expect_tool": "wiki_query"}).to_dict()])
    with FakeTeacher(tool_then_echo()) as fake:
        rc = tg.main(["run", "--builtin-smoke", "--tasks-file", str(smoke), "--out", str(tmp_path / "smoke-out"),
                      "--teacher-url", fake.url, *base_inputs(tmp_path)])
    smoke_rows = [json.loads(x) for x in (tmp_path / "smoke-out" / "builtin-smoke.traces.jsonl").read_text().splitlines()]
    assert rc == 0 and smoke_rows and {r["_meta"]["source"] for r in smoke_rows} == {tg.BUILTIN_SOURCE}
    assert not (tmp_path / "smoke-out" / f"{tg.SOURCE_TAG}.traces.jsonl").exists()
    for patterns in (build.DEFAULT_TRAIN_SOURCES, ("*",)):  # even a match-all pattern never trains on them
        b = gym_builder(tmp_path / f"b-{len(patterns[0])}", smoke_rows, train_sources=patterns)
        feed(b, tmp_path / "smoke-out" / "builtin-smoke.traces.jsonl", "extra")
        assert not b.examples and b.stats["excluded:non_teacher_model"] == len(smoke_rows)


def test_model_requests_never_go_through_a_proxy(tmp_path, no_outbound, monkeypatch):
    """The endpoint policy checks the URL; a proxy in the environment would carry every
    teacher, summarizer and judge request (the real-pattern prompts) elsewhere."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    write_docs(tmp_path)
    rows = [ts_row("Which host runs cobalt-intake according to the wiki?"),
            ts_row("Check the status page for lumen-cache in the wiki.", "sentinel/daemon", force_compaction=True),
            *pad_rows()]
    taskset = write_jsonl(tmp_path / "real.jsonl", rows)
    with FakeTeacher(tool_then_echo()) as teacher, FakeTeacher() as proxy:
        for var in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.setenv(var, proxy.url)
        for var in ("NO_PROXY", "no_proxy"):
            monkeypatch.setenv(var, "example.invalid")
        rc = tg.main(["run", "--tasks", str(taskset), "--out", str(tmp_path / "o"), "--teacher-url", teacher.url,
                      "--compaction-history-tokens", "150", *base_inputs(tmp_path)])
        assert rc == 0 and teacher.kinds["agent"] >= 2 and teacher.kinds["summary"] >= 1
        assert not proxy.requests
    assert not any(v in os.environ for v in tg.PROXY_ENV) and os.environ["NO_PROXY"] == "*"


def test_force_compaction_never_pads_with_synthetic_text(tmp_path, no_outbound, monkeypatch, capsys):
    """A repo with no docs (a staged tree with only scripts/ and src/, say) is refused
    for a real-pattern run with a force_compaction task: its padding would otherwise be
    teacher_gen.py's own synthetic runbooks under a 'repo docs' label, in train rows."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    bare = tmp_path / "staged"
    (bare / "scripts").mkdir(parents=True)
    (bare / "src").mkdir()
    comp = ts_row("Find the owner of amber-ledger in the wiki notes.", "other", force_compaction=True)
    plain = ts_row("Which host runs cobalt-intake according to the wiki?")
    with pytest.raises(tg.NoRepoDocs, match="real repo docs"):
        tg.tasks_from_taskset_rows([plain, comp, *pad_rows()], bare)
    assert tg.tasks_from_taskset_rows([plain], bare)[0].origin == "taskset"  # nothing to pad: no docs needed
    with pytest.raises(tg.NoRepoDocs):  # sources that fell back to the synthetic runbooks
        tg.task_from_taskset(comp, tg.Sources.from_repo(bare))
    assert tg.Sources.from_repo(bare).synthetic  # a --builtin-smoke run still has its fallback
    assert not tg.Sources.from_repo(write_docs(tmp_path / "repo")).synthetic
    # the CLI: exit 2 before any request, nothing written
    taskset = write_jsonl(tmp_path / "real.jsonl", [plain, comp])
    with FakeTeacher(tool_then_echo()) as fake:
        rc = tg.main(["run", "--tasks", str(taskset), "--out", str(tmp_path / "o"), "--teacher-url", fake.url,
                      *base_inputs(tmp_path), "--repo", str(bare)])
        assert rc == 2 and not fake.requests
    assert "real repo docs" in capsys.readouterr().err
    # teacher_ab select refuses it too (the A/B pads like generation does)
    out = tmp_path / "ab"
    assert ab.main(["select", "--tasks", str(taskset), "--out", str(out), "--repo", str(bare), "--n", "2"]) == 2
    assert not (out / "ab_tasks.jsonl").exists()


def test_row_source_needs_both_a_taskset_run_and_a_taskset_task(tmp_path, no_outbound):
    (real,) = tg.tasks_from_taskset_rows([ts_row("Which host runs cobalt-intake per the wiki?")], tmp_path)
    builtin = task(prompt="Which host runs cobalt-intake? Use the wiki.")
    for t, source, want in ((real, tg.ORIGIN_TASKSET, tg.SOURCE_TAG), (real, "builtin", tg.BUILTIN_SOURCE),
                            (builtin, tg.ORIGIN_TASKSET, tg.BUILTIN_SOURCE)):
        with FakeTeacher(tool_then_echo()) as fake:
            s = settings_for(tmp_path, fake.url, task_source=source)
            out, ok, reasons, record = run_and_verify(t, s)
            assert ok, reasons
            assert {r["_meta"]["source"] for r in tg.rows_for(out, s, record)} == {want}
    # a relaxed taskset row keeps the relaxed tag (never train), a relaxed smoke row stays smoke
    s = settings_for(tmp_path, "http://127.0.0.1:1", task_source=tg.ORIGIN_TASKSET)
    assert (s.row_source, s.relaxed_source) == (tg.SOURCE_TAG, tg.RELAXED_SOURCE)
    s.task_source = "builtin"
    assert (s.row_source, s.relaxed_source) == (tg.BUILTIN_SOURCE, tg.BUILTIN_SOURCE)


def test_task_flags_are_honoured(tmp_path, no_outbound):
    write_docs(tmp_path)
    chat, inject, both = tg.tasks_from_taskset_rows([
        ts_row("What is a sensible backup rotation for a home server?", "chat-mode"),
        ts_row("Look up cobalt-intake in the wiki and tell me its host.", inject_errors=True),
        ts_row("Read the wiki page on amber-ledger and give me its owner.", inject_errors=True,
               force_compaction=True), *pad_rows()], tmp_path)[:3]
    assert (chat.kind, inject.kind, both.kind) == ("chat-mode", "error-recovery", "compacted-history")
    assert both.inject and both.compaction and both.pad and not both.seed
    # suppress_tools: a chat-mode request, no tools on the wire
    with FakeTeacher(tool_then_echo()) as fake:
        s = settings_for(tmp_path, fake.url, task_source=tg.ORIGIN_TASKSET)
        out, ok, reasons, record = run_and_verify(chat, s)
        assert ok, reasons
        assert all(not b.get("tools") for b in fake.requests if b.get("stream"))
    # inject_errors: the first tool call fails, the run recovers; no tool call at all fails the task
    with FakeTeacher(tool_then_echo()) as fake:
        out, ok, reasons, record = run_and_verify(inject, settings_for(tmp_path, fake.url))
    assert ok and record["injected_failures"] == 1 and record["recovered"] is True
    with FakeTeacher(tool_then_echo(fail=("cobalt-intake",))) as fake:
        assert "injection_not_fired" in run_and_verify(inject, settings_for(tmp_path, fake.url))[2]
    # both flags: compacted AND the injection has to fire (a compacted-history task too)
    with FakeTeacher(tool_then_echo()) as fake:
        s = settings_for(tmp_path, fake.url, compaction_history_tokens=150)
        out, ok, reasons, record = run_and_verify(both, s)
        assert fake.kinds["summary"] >= 1
    assert ok, reasons
    assert record["compaction"]["events"] >= 1 and record["compaction"]["summary_in_request"]
    assert record["injected_failures"] == 1 and record["recovered"] is True
    with FakeTeacher(tool_then_echo(fail=("amber-ledger",))) as fake:
        reasons = run_and_verify(both, settings_for(tmp_path, fake.url, compaction_history_tokens=150))[2]
    assert "injection_not_fired" in reasons


def test_judge_url_and_probe_guard_routing(tmp_path, no_outbound):
    s = tg.Settings(out=tmp_path, teacher_url="http://127.0.0.1:8093", probe_guard_url="http://127.0.0.1:18592")
    assert s.judge_endpoint == "http://127.0.0.1:18592" and s.judge_kind == "self-judge"
    s.judge_url = "http://127.0.0.1:8093/"  # the teacher named again: still through the guard
    assert s.judge_endpoint == "http://127.0.0.1:18592" and s.judge_kind == "self-judge"
    s.judge_url = "http://127.0.0.1:18080"  # another server: called directly
    assert s.judge_endpoint == "http://127.0.0.1:18080" and s.judge_kind == "local-judge"
    # the teacher's (or the guard's) port under another loopback spelling is still the teacher:
    # through the guard, never straight to the teacher's port
    for spelling in ("http://localhost:8093", "http://[::1]:8093/", "http://127.0.0.2:8093",
                     "https://127.0.0.1:8093", "http://LOCALHOST:8093/v1", "http://localhost:18592"):
        s.judge_url = spelling
        assert s.judge_endpoint == "http://127.0.0.1:18592" and s.judge_kind == "self-judge", spelling
    s.judge_url = "http://localhost:18081"  # another port: another server
    assert s.judge_endpoint == "http://localhost:18081" and s.judge_kind == "local-judge"
    assert tg.same_server("http://127.0.0.1", "http://localhost:80/") and not tg.same_server("http://a:1", None)
    write_docs(tmp_path)
    t = task(prompt="Which host runs cobalt-intake? Use the wiki.", fixtures={"wiki": {"cobalt-intake": "host x"}})
    # a separate judge: generation on the teacher only, judging on the judge only
    with FakeTeacher(tool_then_echo()) as teacher, FakeTeacher() as judge:
        status, rows, _r, extra = asyncio.run(tg.process(t, settings_for(tmp_path, teacher.url,
                                                                            judge_url=judge.url), 0))
        assert status == "kept" and extra["judge"]["kind"] == "local-judge"
        assert teacher.kinds["judge"] == 0 and teacher.kinds["agent"] >= 1
        assert judge.kinds == collections.Counter({"judge": 1})
    # the probe guard carries every teacher, summarizer and (self-)judge request; the
    # teacher's own port (dead here) is never called
    dead = f"http://127.0.0.1:{free_port()}"
    comp = tg.tasks_from_taskset_rows([ts_row("Read the wiki page on amber-ledger and give me its owner.",
                                              force_compaction=True), *pad_rows()], tmp_path)[0]
    for judge_url in (None, dead):
        with FakeTeacher(tool_then_echo()) as guard:
            s = settings_for(tmp_path, dead, probe_guard_url=guard.url, judge_url=judge_url,
                             compaction_history_tokens=150)
            status, _rows, reasons, _e = asyncio.run(tg.process(comp, s, 0))
            assert status == "kept", reasons
            assert guard.kinds["agent"] >= 1 and guard.kinds["summary"] >= 1 and guard.kinds["judge"] == 1


def test_cli_checks_a_separate_judge_is_the_expected_model(tmp_path, no_outbound, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    taskset = write_jsonl(tmp_path / "real.jsonl", [ts_row("Which host runs cobalt-intake per the wiki?")])
    with FakeTeacher(tool_then_echo()) as teacher, FakeTeacher() as judge:
        args = ["run", "--tasks", str(taskset), "--out", str(tmp_path / "o"), "--teacher-url", teacher.url,
                "--judge-url", judge.url, *base_inputs(tmp_path)]
        assert tg.main(args + ["--judge-id", "qwen3*8*27b*"]) == 3  # it serves "fake-teacher"
        assert teacher.kinds["agent"] == 0
        assert tg.main(args + ["--judge-id", "fake-*"]) == 0
        assert judge.kinds["judge"] == 1 and teacher.kinds["judge"] == 0


# ── teacher_ab.py ──────────────────────────────────────────────────────────────


class ArmServer:
    """A fake llama-server for one arm: one model id everywhere (/v1/models and every
    completion), the tool_then_echo teacher (failing *fail* prompts), the strict
    grader's pass verdict, the compactor's summary, and a log of every request."""

    def __init__(self, model_id: str, fail: tuple[str, ...] = (), *, model_path: str | None = None,
                 template: str = "{% for m in messages %}{{ m.content }}{% endfor %}") -> None:
        self.model_id = model_id
        self.policy = tool_then_echo(fail)
        self.requests: list[dict[str, Any]] = []
        self.kinds: collections.Counter = collections.Counter()
        #: (status, body) answers for the next agent requests, in order (then the policy);
        #: (200, "truncate") cuts the reply off mid-stream as llama-server does on a server error
        self.fail_agent: list[tuple[int, Any]] = []
        #: seconds every judge reply waits
        self.judge_delay = 0.0
        #: llama-server's /props: the model file, per-slot context, template and sampler defaults
        self.props = {"model_path": model_path or f"/models/{model_id}.gguf", "chat_template": template,
                      "total_slots": 2, "default_generation_settings": {"n_ctx": 16384, "params": {
                          "temperature": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.0, "repeat_penalty": 1.0,
                          "samplers": ["penalties", "top_k", "top_p", "min_p", "temperature"]}}}
        srv = self

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
                    self._json(200, {"object": "list", "data": [{"id": srv.model_id}]})
                elif self.path == "/props":
                    self._json(200, dict(srv.props, model_alias=srv.model_id))
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                srv.requests.append(body)
                msgs = body.get("messages") or []
                if msgs and str(msgs[0].get("content", "")).startswith("You are a strict grader"):
                    srv.kinds["judge"] += 1
                    if srv.judge_delay:
                        time.sleep(srv.judge_delay)
                    verdict = {"verdict": "pass", "completed": True, "tools_ok": True, "grounded": True,
                               "recovery_ok": None, "reason": "fine"}
                    self._json(200, {"model": srv.model_id, "choices": [
                        {"message": {"role": "assistant", "content": json.dumps(verdict)}, "finish_reason": "stop"}]})
                    return
                if any("You are a conversation compactor" in tg.content_text(m.get("content")) for m in msgs):
                    srv.kinds["summary"] += 1
                    reply: dict[str, Any] = {"content": "Earlier: the user read several files."}
                else:
                    srv.kinds["agent"] += 1
                    if srv.fail_agent:
                        code, err = srv.fail_agent.pop(0)
                        if err == "truncate":  # a partial reply, then llama-server's mid-stream error event
                            self.send_response(200)
                            self.send_header("Content-Type", "text/event-stream")
                            self.end_headers()
                            for c in ({"choices": [{"delta": {"content": "Here is what I fo"}}],
                                       "model": srv.model_id},
                                      {"error": {"code": 500, "message": "Context size has been exceeded.",
                                                 "type": "server_error"}}):
                                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                            self.wfile.flush()
                            return
                        self._json(code, err)
                        return
                    reply = srv.policy(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks: list[dict[str, Any]] = []
                if reply.get("content"):
                    chunks.append({"choices": [{"delta": {"content": reply["content"]}}]})
                for i, (name, args) in enumerate(reply.get("tool_calls") or []):
                    chunks.append({"choices": [{"delta": {"tool_calls": [{
                        "index": i, "id": f"call_{len(srv.requests)}_{i}",
                        "function": {"name": name, "arguments": json.dumps(args)}}]}}]})
                chunks.append({"choices": [{"delta": {}, "finish_reason": "tool_calls" if reply.get("tool_calls")
                                            else "stop"}]})
                chunks.append({"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
                for c in chunks:
                    c["model"] = srv.model_id
                    self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "ArmServer":
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.server.shutdown()
        self.server.server_close()


AB_ROWS = [
    ts_row("What does a maintenance window mean for a small home server?", "chat-mode"),
    ts_row("Explain in plain words how log rotation keeps a disk from filling up.", "chat-mode"),
    ts_row("Give me a short plan for testing restores of last week's backups.", "chat-mode"),
    ts_row("Which host runs cobalt-intake according to the wiki?"),
    ts_row("Look up amber-ledger in the wiki and tell me its owner.", inject_errors=True),
    ts_row("Find the wiki notes on harbor-sync and list its dependencies."),
    ts_row("Check the research notes on quill-indexer and summarize the findings.", "other"),
    ts_row("Check the status page for lumen-cache in the wiki.", "sentinel/daemon", force_compaction=True),
]


def select(tmp_path: Path, rows: list[dict[str, Any]], out: Path, *extra: str) -> int:
    write_docs(tmp_path)  # --repo: the force_compaction task pads with these docs
    taskset = write_jsonl(tmp_path / "real-ab.jsonl", rows)
    return ab.main(["select", "--tasks", str(taskset), "--out", str(out), "--repo", str(tmp_path),
                    "--teacher-id", "qwen3*8*27b*", "--compaction-history-tokens", "150",
                    "--request-timeout", "20", "--min-task-seconds", "5", *extra])


def test_selection_is_stratified_seeded_and_recorded(tmp_path, capsys):
    rows = ([ts_row(f"Explain how backup step {w} works on a home server.", "chat-mode") for w in "abcdefghijklmnop"]
            + [ts_row(f"Find the wiki page for service {w} and give me its host.") for w in "abcdefgh"]
            + [ts_row(f"Find the wiki page for service {w} and give me its owner.", group=f"wiki-{w}")
               for w in "ab"]  # variants sharing a group with nothing else: still distinct groups
            + [ts_row(f"Check the status of daemon job {w} in the wiki.", "sentinel/daemon") for w in "abcdef"]
            + [ts_row(f"Find where function {w} is defined in the repo.", "coding") for w in "abcd"]
            + [ts_row(f"Research the release notes of library {w}.", "other") for w in "abcd"])
    out = tmp_path / "ab1"
    assert select(tmp_path, rows, out, "--n", "10") == 0
    sel = json.loads((out / "ab_selection.json").read_text())
    assert sel["n"] == 10 and len(set(sel["ids"])) == 10 and sel["seed"] == ab.DEFAULT_SEED
    strata = {k: v["picked"] for k, v in sel["strata"].items()}
    assert strata == {"chat-mode|chat": 4, "coding|agentic": 1, "memory/wiki|agentic": 3, "other|agentic": 1,
                      "sentinel/daemon|agentic": 1}
    assert sel["task_groups"] == 10 and sel["taskset_sha256"] and sel["taskset_rows"] == len(rows)
    tasks = [json.loads(x) for x in (out / "ab_tasks.jsonl").read_text().splitlines()]
    assert [t["task"]["id"] for t in tasks] == sel["ids"] and {t["task"]["origin"] for t in tasks} == {"taskset"}
    settings = json.loads((out / "ab_settings.json").read_text())
    assert settings["sampling"] == SAMPLING and settings["labels"] == {"A": "IQ3_XXS", "B": "Q4_K_M"}
    # seeded: the same seed picks the same ids, another seed others
    assert select(tmp_path, rows, tmp_path / "ab2", "--n", "10") == 0
    assert json.loads((tmp_path / "ab2" / "ab_selection.json").read_text())["ids"] == sel["ids"]
    assert select(tmp_path, rows, tmp_path / "ab3", "--n", "10", "--seed", "other") == 0
    assert json.loads((tmp_path / "ab3" / "ab_selection.json").read_text())["ids"] != sel["ids"]
    # resume: the same selection is a no-op; another one into the same --out is refused
    capsys.readouterr()
    assert select(tmp_path, rows, out, "--n", "10") == 0 and "already selected" in capsys.readouterr().out
    assert select(tmp_path, rows, out, "--n", "10", "--seed", "other") == 2
    assert select(tmp_path, rows, out, "--n", "10", "--top-k", "40") == 2
    # only a real-pattern taskset is accepted
    lt = write_jsonl(tmp_path / "lt.jsonl", [task(id="b-1", prompt="hi").to_dict()])
    assert ab.main(["select", "--tasks", str(lt), "--out", str(tmp_path / "ab4")]) == 2
    # allocation edges
    assert ab.allocate({"a": 3, "b": 1}, 10) == {"a": 3, "b": 1}
    assert ab.allocate({"a": 8, "b": 1, "c": 1}, 2) == {"a": 2, "b": 0, "c": 0}
    assert sum(ab.allocate({"a": 5, "b": 5, "c": 5}, 4).values()) == 4


def test_ab_runs_both_arms_with_identical_settings_and_one_judge(tmp_path, no_outbound, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    out = tmp_path / "ab"
    assert select(tmp_path, AB_ROWS, out, "--n", "8") == 0
    # arm A (IQ3) fails three tasks, arm B (Q4) one of the same
    fail_a = ("log rotation", "amber-ledger", "quill-indexer")
    fail_b = ("log rotation",)
    with ArmServer("qwen3.8-27b-iq3_xxs", fail_a) as a_srv, ArmServer("qwen3.8-27b-q4_k_m", fail_b) as b_srv:
        assert ab.main(["arm", "--out", str(out), "--arm", "A", "--url", a_srv.url]) == 0
        assert ab.main(["status", "--out", str(out), "--need", "arm-A"]) == 0
        assert ab.main(["status", "--out", str(out), "--need", "judge"]) == 1
        assert ab.main(["arm", "--out", str(out), "--arm", "B", "--url", b_srv.url]) == 0
        # no arm was judged in its own run
        assert a_srv.kinds["judge"] == 0 and b_srv.kinds["judge"] == 0
        assert ab.main(["report", "--out", str(out)]) == 1  # nothing judged yet: no decision
        assert (out / "ab_summary.txt").read_text().startswith("Teacher A/B: INCOMPLETE")
        # the judge is arm B's model: IQ3 grading both arms (its own traces included) is refused
        assert ab.main(["judge", "--out", str(out), "--judge-url", a_srv.url]) == 2
        assert a_srv.kinds["judge"] == 0 and not (out / "judge.jsonl").exists()
        assert ab.main(["judge", "--out", str(out), "--judge-url", b_srv.url]) == 0
        # every trace of BOTH arms went to the one judge, none to the other server
        assert b_srv.kinds["judge"] == 16 and a_srv.kinds["judge"] == 0
        assert ab.main(["status", "--out", str(out), "--need", "judge"]) == 0
        rc = ab.main(["report", "--out", str(out)])
        a_reqs, b_reqs = list(a_srv.requests), list(b_srv.requests)
    assert rc == 0
    # identical request settings on both servers: explicit sampling on every teacher and
    # summarizer request, the same everything else (only the messages differ)
    def shape(b: dict[str, Any]) -> str:
        return json.dumps({k: v for k, v in b.items() if k not in ("messages", "tools", "tool_choice")},
                          sort_keys=True)
    gen_a = {shape(b) for b in a_reqs if b.get("stream")}
    gen_b = {shape(b) for b in b_reqs if b.get("stream")}
    assert gen_a == gen_b and all(json.loads(x)["temperature"] == 0.7 and json.loads(x)["min_p"] == 0.0
                                  and json.loads(x)["top_k"] == 20 and json.loads(x)["top_p"] == 0.8 for x in gen_a)
    judge_reqs = [b for b in b_reqs if not b.get("stream")]
    assert len(judge_reqs) == 16 and {(b["temperature"], b["seed"]) for b in judge_reqs} == {(0, tg.DEFAULT_JUDGE_SEED)}
    rep = json.loads((out / "ab_report.json").read_text())
    A, B = rep["arms"]["A"], rep["arms"]["B"]
    assert (A["served"], B["served"]) == ("qwen3.8-27b-iq3_xxs", "qwen3.8-27b-q4_k_m")
    assert (A["pass"], B["pass"], A["pairs"]) == (5, 7, 8)
    assert rep["paired"]["table"] == {"a": 5, "b": 2, "c": 0, "d": 1}
    assert rep["paired"]["diff_pts"] == -25.0 and rep["paired"]["ci_pts"][0] < -25.0 < rep["paired"]["ci_pts"][1]
    assert rep["decision"]["verdict"] == ab.DECISION_Q4 and rep["decision"]["a_minus_b_pts"] == -25.0
    assert rep["judge"]["served"] == "qwen3.8-27b-q4_k_m" and rep["judge"]["is_arm_b_model"]
    assert rep["judge"]["model_file"] == ["qwen3.8-27b-q4_k_m.gguf"]
    assert rep["settings"]["identical_across_arms"] and rep["settings"]["sampling"] == SAMPLING
    # the servers behind the ids were recorded: two files, one template, one sampler set, one context
    srv_a, srv_b = rep["settings"]["servers"]["A"], rep["settings"]["servers"]["B"]
    assert (srv_a["model_file"], srv_b["model_file"]) == ("qwen3.8-27b-iq3_xxs.gguf", "qwen3.8-27b-q4_k_m.gguf")
    assert all(srv_a[k] == srv_b[k] for k in ab.SERVER_SAME_KEYS) and srv_a["n_ctx"] == 16384
    assert A["fail_reasons"] and "injection_not_fired" in A["fail_reasons"]
    summary = (out / "ab_summary.txt").read_text().splitlines()
    assert len(summary) == 5 and summary[0] == "Teacher A/B decision: use Q4_K_M (less context)"
    assert "IQ3_XXS 5/8" in summary[1] and "Q4_K_M 7/8" in summary[1] and "-25.0 pts" in summary[2]


SERVER_A = {"model_path": "/m/iq3.gguf", "model_file": "iq3.gguf", "n_ctx": 16384, "chat_template_sha": "t1",
            "sampler_sha": "s1", "sampler": {}}
SERVER_B = dict(SERVER_A, model_path="/m/q4.gguf", model_file="q4.gguf")


def _write_ab(out: Path, n: int, pass_a: int, pass_b: int, *, sha_b: str | None = None,
              served_b: str = "q4", judges: tuple[str, ...] = ("q4",), server_b: dict[str, Any] | None = None,
              no_server: bool = False, judge_file: str = "/m/q4.gguf") -> dict[str, Any]:
    """An A/B directory written by hand: n tasks, the first pass_a pass in A and the
    first pass_b in B (so every A pass is also a B pass when pass_a <= pass_b)."""
    out.mkdir(parents=True)
    conf = {"schema": ab.SETTINGS_SCHEMA, "generator": tg.GENERATOR_VERSION, "sampling": SAMPLING,
            "model_name": "qwen3.8-27b", "teacher_id": "*", "judge_id": "*", "judge_model": None, "judge_seed": 7,
            "judge_max_chars": 30000, "max_tokens": 4096, "max_tool_iterations": 12, "tool_timeout": 60,
            "task_timeout": 900, "request_timeout": 300, "compaction_history_tokens": 1500, "concurrency": 2,
            "jail": "auto", "min_task_seconds": 120, "system_prompt": "x", "allow_lan": False,
            "labels": dict(ab.DEFAULT_LABELS), "threshold_pts": ab.THRESHOLD_PTS, "max_consecutive_errors": 5}
    ids = [f"ts-{i:016x}" for i in range(n)]
    (out / "ab_settings.json").write_text(json.dumps(conf))
    (out / "ab_selection.json").write_text(json.dumps({"ids": ids, "n": n, "seed": "s", "taskset": "t",
                                                      "taskset_sha256": "0", "strata": {}, "flags": {}}))
    (out / "ab_tasks.jsonl").write_text("")
    sha = ab.settings_sha(conf)
    servers = {"A": SERVER_A, "B": dict(SERVER_B, **(server_b or {}))}
    for arm, k, served, s in (("A", pass_a, "iq3", sha), ("B", pass_b, served_b, sha_b or sha)):
        write_jsonl(out / f"arm-{arm}.jsonl", [
            {"arm": arm, "task_id": t, "category": "chat-mode", "served": served, "settings_sha": s,
             "det_ok": True, "n_rows": 1, "tool_success": True, "predicates_pass": True, "reasons": [],
             "transcript": "t", **({} if no_server and arm == "B" else {"server": servers[arm]})} for t in ids])
    write_jsonl(out / "judge.jsonl", [
        {"arm": arm, "task_id": t, "judge_served": judges[(i + j) % len(judges)], "judge_model_path": judge_file,
         "judge": {"pass": i < (pass_a if arm == "A" else pass_b), "served_model": "q4"}}
        for j, arm in enumerate(ab.ARMS) for i, t in enumerate(ids)])
    return conf


def test_decision_rule_edges(tmp_path):
    d = ab.decide
    assert d(14, 15, 20) == ab.DECISION_IQ3  # exactly 5 points lower: not MORE than 5 -> keep IQ3
    assert d(57, 60, 60) == ab.DECISION_IQ3  # 5.0 exactly again (a float would say 5.000000000000001)
    assert d(13, 15, 20) == ab.DECISION_Q4  # 10 points
    assert d(29, 30, 30) == ab.DECISION_IQ3 and d(28, 30, 30) == ab.DECISION_Q4  # 3.3 vs 6.7 points
    assert d(30, 20, 30) == ab.DECISION_IQ3  # IQ3 better
    assert d(0, 0, 0) == ab.DECISION_INCOMPLETE
    for pass_a, want, diff in ((14, ab.DECISION_IQ3, -5.0), (13, ab.DECISION_Q4, -10.0), (15, ab.DECISION_IQ3, 0.0)):
        out = tmp_path / f"r{pass_a}"
        conf = _write_ab(out, 20, pass_a, 15)
        rep = ab.build_report(ab.Paths(out), conf)
        assert rep["decision"]["verdict"] == want and rep["decision"]["a_minus_b_pts"] == diff
        assert rep["paired"]["diff_pts"] == diff and rep["paired"]["table"]["c"] == 0
        lo, hi = rep["paired"]["ci_pts"]
        assert lo <= diff <= hi
    # no decision when the comparison is not like for like
    for i, (kw, why) in enumerate((
            ({"sha_b": "other"}, "settings differ"), ({"served_b": "iq3"}, "same served model"),
            ({"judges": ("q4", "iq3")}, "more than one judge"),
            # IQ3 grading all 60 (its own 30 included): the plan is the bigger quant as the one judge
            ({"judges": ("iq3",)}, "is not arm B's model"),
            ({"judge_file": "/m/iq3.gguf"}, "judge's model file"),
            # the servers behind the ids: one file under two aliases (or swapped files), another
            # chat template, other sampler defaults, another per-slot context, nothing recorded
            ({"server_b": {"model_path": "/m/iq3.gguf"}}, "same model file"),
            ({"server_b": {"chat_template_sha": "t2"}}, "servers differ: chat_template_sha"),
            ({"server_b": {"sampler_sha": "s2"}}, "servers differ: sampler_sha"),
            ({"server_b": {"n_ctx": 8192}}, "servers differ: n_ctx"),
            ({"no_server": True}, "were not recorded"))):
        out = tmp_path / f"bad-{i}"
        rep = ab.build_report(ab.Paths(out), _write_ab(out, 20, 10, 15, **kw))
        assert rep["decision"]["verdict"] == ab.DECISION_INVALID
        assert any(why in p for p in rep["decision"]["problems"]), (why, rep["decision"]["problems"])
        assert ab.summary_lines(rep)[0].startswith("Teacher A/B: INVALID")
        if why.startswith("servers differ"):  # client settings equal is not enough to call them identical
            assert not rep["settings"]["identical_across_arms"]
    # a missing pair: incomplete unless --min-pairs allows fewer
    out = tmp_path / "partial"
    conf = _write_ab(out, 20, 15, 15)
    rows = [json.loads(x) for x in (out / "arm-B.jsonl").read_text().splitlines()][:-1]
    write_jsonl(out / "arm-B.jsonl", rows)
    rep = ab.build_report(ab.Paths(out), conf)
    assert rep["decision"]["verdict"] == ab.DECISION_INCOMPLETE and rep["decision"]["pairs"] == 19
    assert ab.build_report(ab.Paths(out), conf, min_pairs=19)["decision"]["verdict"] == ab.DECISION_IQ3


def test_ab_resumes_and_keeps_its_time_box(tmp_path, no_outbound, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    import prometheus.providers.retry as retry

    monkeypatch.setattr(retry, "BASE_DELAY", 0.0)  # the provider's own retries, without the wait
    out = tmp_path / "ab"
    rows = AB_ROWS[:4]
    assert select(tmp_path, rows, out, "--n", "4", "--concurrency", "1") == 0
    with ArmServer("qwen3.8-27b-iq3_xxs") as srv:
        # a time box already over: nothing starts
        assert ab.main(["arm", "--out", str(out), "--arm", "A", "--url", srv.url,
                        "--until", "2020-01-01T00:00:00"]) == 1
        assert srv.kinds["agent"] == 0 and not (out / "arm-A.jsonl").exists()
        # the server fails (503, every retry too) for the first task: not recorded, the rest are
        srv.fail_agent = [(503, {"error": "loading"})] * (retry.MAX_RETRIES + 1)
        assert ab.main(["arm", "--out", str(out), "--arm", "A", "--url", srv.url]) == 1
        recs = [json.loads(x) for x in (out / "arm-A.jsonl").read_text().splitlines()]
        assert len(recs) == 3
        missing = ({r["id"] for r in rows} - {r["task_id"] for r in recs}).pop()
        # a crash left a torn line; the rerun cuts it and runs only the missing task
        with (out / "arm-A.jsonl").open("a") as fh:
            fh.write('{"arm": "A", "task_id": "ts-')
        before = len(srv.requests)
        assert ab.main(["arm", "--out", str(out), "--arm", "A", "--url", srv.url]) == 0
        new = srv.requests[before:]
        prompt = next(r["prompt"] for r in rows if r["id"] == missing)
        assert new and all(prompt in json.dumps(b["messages"]) for b in new if b.get("stream"))
        recs = [json.loads(x) for x in (out / "arm-A.jsonl").read_text().splitlines()]
        assert sorted(r["task_id"] for r in recs) == sorted(r["id"] for r in rows)
        assert ab.main(["arm", "--out", str(out), "--arm", "A", "--url", srv.url]) == 0  # nothing left
        assert len(srv.requests) == before + len(new)
    # resuming an arm on another model is refused
    with ArmServer("qwen3.8-27b-q4_k_m") as other:
        (out / "arm-A.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs[:-1]))
        assert ab.main(["arm", "--out", str(out), "--arm", "A", "--url", other.url]) == 2
        assert other.kinds["agent"] == 0
    with ArmServer("qwen3.8-27b-iq3_xxs") as srv:
        assert ab.main(["arm", "--out", str(out), "--arm", "A", "--url", srv.url]) == 0
    # the judge: every trace on one judge; a resume on another judge is refused
    with ArmServer("qwen3.8-27b-q4_k_m") as b_srv:
        assert ab.main(["arm", "--out", str(out), "--arm", "B", "--url", b_srv.url]) == 0
        assert ab.main(["judge", "--out", str(out), "--judge-url", b_srv.url, "--until",
                        "2020-01-01T00:00:00"]) == 1
        assert b_srv.kinds["judge"] == 0
        assert ab.main(["judge", "--out", str(out), "--judge-url", b_srv.url]) == 0
    judged = [json.loads(x) for x in (out / "judge.jsonl").read_text().splitlines()]
    assert len(judged) == 8 and {j["judge_served"] for j in judged} == {"qwen3.8-27b-q4_k_m"}
    (out / "judge.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judged[:-1]))
    with ArmServer("qwen3.8-27b-iq3_xxs") as wrong:
        assert ab.main(["judge", "--out", str(out), "--judge-url", wrong.url]) == 2
        assert wrong.kinds["judge"] == 0
    assert ab.main(["report", "--out", str(out)]) == 1
    assert (out / "ab_summary.txt").read_text().startswith("Teacher A/B: INCOMPLETE (3 of 4 pairs")
    # an arm on a hosted API or a Max model is refused before any task
    assert ab.main(["arm", "--out", str(out), "--arm", "B", "--url", "https://dashscope.aliyuncs.com"]) == 2
    with ArmServer("qwen3.8-max") as maxed:
        assert ab.main(["arm", "--out", str(out), "--arm", "B", "--url", maxed.url]) == 3
        assert maxed.kinds["agent"] == 0


def test_an_arm_refuses_a_server_that_is_not_like_for_like(tmp_path, no_outbound, monkeypatch):
    """The arms' servers are compared, not only the requests: arm B on a server with
    another chat template, other sampler defaults, another per-slot context, or the
    same model file as arm A is refused before its first task."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    out = tmp_path / "ab"
    assert select(tmp_path, AB_ROWS[:2], out, "--n", "2", "--concurrency", "1") == 0
    with ArmServer("qwen3.8-27b-iq3_xxs") as a_srv:
        assert ab.main(["arm", "--out", str(out), "--arm", "A", "--url", a_srv.url]) == 0
    a_file = "/models/qwen3.8-27b-iq3_xxs.gguf"
    for change in ({"chat_template": "{{ other }}"}, {"model_path": a_file},
                   {"default_generation_settings": {"n_ctx": 8192, "params": {"repeat_penalty": 1.1}}}):
        with ArmServer("qwen3.8-27b-q4_k_m") as b_srv:
            b_srv.props.update(change)
            assert ab.main(["arm", "--out", str(out), "--arm", "B", "--url", b_srv.url]) == 2, change
            assert b_srv.kinds["agent"] == 0 and not (out / "arm-B.jsonl").exists()
    # a server without /props (not llama-server) cannot be compared: refused as a server problem
    with ArmServer("qwen3.8-27b-q4_k_m") as b_srv:
        b_srv.props = {}
        assert ab.main(["arm", "--out", str(out), "--arm", "B", "--url", b_srv.url]) == 3
        assert b_srv.kinds["agent"] == 0
    # resuming arm A on another file under the same alias is refused too
    with ArmServer("qwen3.8-27b-iq3_xxs", model_path="/models/elsewhere.gguf") as a2:
        (out / "arm-A.jsonl").write_text((out / "arm-A.jsonl").read_text().splitlines()[0] + "\n")
        assert ab.main(["arm", "--out", str(out), "--arm", "A", "--url", a2.url]) == 2
        assert a2.kinds["agent"] == 0
    with ArmServer("qwen3.8-27b-q4_k_m") as b_srv:  # like for like: runs
        assert ab.main(["arm", "--out", str(out), "--arm", "B", "--url", b_srv.url]) == 0


def test_a_reply_cut_off_mid_stream_is_a_server_failure(tmp_path, no_outbound, monkeypatch):
    """llama-server ends a failed generation (a full KV cache, say) with an error event
    and no finish_reason; the partial reply is never verified, recorded or a row."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    out = tmp_path / "ab"
    rows = [ts_row("Which host runs cobalt-intake according to the wiki?")]
    assert select(tmp_path, rows, out, "--n", "1", "--concurrency", "1") == 0
    with ArmServer("qwen3.8-27b-iq3_xxs") as srv:
        srv.fail_agent = [(200, "truncate")]
        assert ab.main(["arm", "--out", str(out), "--arm", "A", "--url", srv.url]) == 1
        assert not (out / "arm-A.jsonl").exists() or not (out / "arm-A.jsonl").read_text().strip()
        assert ab.main(["arm", "--out", str(out), "--arm", "A", "--url", srv.url]) == 0  # rerun: recorded
        (rec,) = [json.loads(x) for x in (out / "arm-A.jsonl").read_text().splitlines()]
        assert rec["det_ok"], rec["reasons"]
    # teacher_gen: the same cut-off is a transport error (retried next run), never a kept row
    t = tg.tasks_from_taskset_rows(rows, tmp_path)[0]
    with ArmServer("fake-teacher") as srv:
        srv.fail_agent = [(200, "truncate")]
        status, got, reasons, _extra = asyncio.run(tg.process(t, settings_for(tmp_path, srv.url,
                                                                           task_source=tg.ORIGIN_TASKSET), 0))
    assert status == "error" and not got and "transport_error" in reasons


def test_judge_call_is_cancelled_at_until(tmp_path, no_outbound, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    out = tmp_path / "ab"
    assert select(tmp_path, AB_ROWS[:1], out, "--n", "1", "--concurrency", "1") == 0
    with ArmServer("qwen3.8-27b-iq3_xxs") as a_srv:
        assert ab.main(["arm", "--out", str(out), "--arm", "A", "--url", a_srv.url]) == 0
    monkeypatch.setattr(ab, "JUDGE_MARGIN_S", 0.5)
    with ArmServer("qwen3.8-27b-q4_k_m") as b_srv:
        assert ab.main(["arm", "--out", str(out), "--arm", "B", "--url", b_srv.url]) == 0
        b_srv.judge_delay = 20.0  # a stalled judge: the call must not outlive --until
        until = (dt.datetime.now().astimezone() + dt.timedelta(seconds=2)).isoformat()
        t0 = time.monotonic()
        assert ab.main(["judge", "--out", str(out), "--judge-url", b_srv.url, "--until", until]) == 1
        assert time.monotonic() - t0 < 10 and b_srv.kinds["judge"] >= 1
    assert not (out / "judge.jsonl").exists() or not (out / "judge.jsonl").read_text().strip()


def test_until_is_resolved_once(tmp_path, monkeypatch, capsys):
    """An HH:MM --until resolved again after it has passed would mean tomorrow: `run`
    resolves it once and hands every phase the same absolute time; time-left gives the
    shell driver that time and refuses one too far ahead."""
    seen: list[str | None] = []
    monkeypatch.setattr(ab, "cmd_select", lambda a: 0)
    monkeypatch.setattr(ab, "cmd_arm", lambda a: seen.append(a.until) or 0)
    monkeypatch.setattr(ab, "cmd_judge", lambda a: seen.append(a.until) or 0)
    monkeypatch.setattr(ab, "cmd_report", lambda a: 0)
    hhmm = (dt.datetime.now().astimezone() + dt.timedelta(hours=1)).strftime("%H:%M")
    assert ab.main(["run", "--tasks", "x", "--out", str(tmp_path / "ab"), "--url-a", "http://127.0.0.1:1",
                    "--url-b", "http://127.0.0.1:2", "--judge-url", "http://127.0.0.1:2", "--until", hhmm]) == 0
    assert len(seen) == 3 and len(set(seen)) == 1 and "T" in str(seen[0])
    when = dt.datetime.fromisoformat(str(seen[0]))
    assert when.strftime("%H:%M") == hhmm and 0 < when.timestamp() - time.time() <= 3600
    # an HH:MM that passed a minute ago is ~24 h away: refused, never read as tomorrow
    past = (dt.datetime.now().astimezone() - dt.timedelta(minutes=1)).strftime("%H:%M")
    capsys.readouterr()
    assert ab.main(["time-left", "--until", past, "--min-seconds", "600"]) == 2
    assert "passed today" in capsys.readouterr().err
    with pytest.raises(ValueError, match="passed today"):
        tg.parse_until("06:20", dt.datetime(2026, 11, 1, 6, 20, 1).astimezone(), max_ahead_hours=12)
    assert tg.parse_until("06:20", dt.datetime(2026, 11, 1, 6, 20, 1).astimezone()) > 0  # no limit given
    with pytest.raises(ab.InputError, match="passed today"):
        ab.resolve_until(past)
    # --iso prints the absolute time; --max-seconds refuses one further ahead
    capsys.readouterr()
    assert ab.main(["time-left", "--until", hhmm, "--iso", "--max-seconds", "7200"]) == 0
    assert dt.datetime.fromisoformat(capsys.readouterr().out.strip()).strftime("%H:%M") == hhmm
    assert ab.main(["time-left", "--until", hhmm, "--max-seconds", "1800"]) == 2
    soon = (dt.datetime.now().astimezone() + dt.timedelta(minutes=5)).isoformat(timespec="seconds")
    assert ab.main(["time-left", "--until", soon]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["until_iso"] == soon


def test_models_check_reads_the_gguf_headers(tmp_path, capsys):
    iq3 = write_gguf(tmp_path / "iq3.gguf", file_type=23)
    q4 = write_gguf(tmp_path / "q4.gguf", pad=4096)
    meta = ab.gguf_metadata(q4)
    assert meta["general.file_type"] == 15 and meta["general.sampling.top_k"] == 20
    assert meta["tokenizer.ggml.tokens"] == "<array 3>" and "messages" in meta["tokenizer.chat_template"]
    assert ab.main(["models", "--a", str(iq3), "--b", str(q4)]) == 0
    assert ab.check_models(iq3, q4)[1] == []
    for a_, b_, why in ((q4, iq3, "not bigger"), (q4, q4, "same model file")):
        assert any(why in p for p in ab.check_models(a_, b_)[1]), why
    other = write_gguf(tmp_path / "other.gguf", template="{{ different }}", pad=8192)
    assert any("different chat templates" in p for p in ab.check_models(iq3, other)[1])
    assert ab.check_models(iq3, other, template_pinned=True)[1] == []
    (tmp_path / "junk.gguf").write_text("x" * 100)
    assert any("not a GGUF" in p for p in ab.check_models(iq3, tmp_path / "junk.gguf")[1])
    capsys.readouterr()
    assert ab.main(["models", "--a", str(q4), "--b", str(iq3)]) == 2 and "not bigger" in capsys.readouterr().err


# ── teacher_ab_run.sh ──────────────────────────────────────────────────────────


def _script_env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ, HOME=str(tmp_path / "home"))
    env.pop("PYTHONPATH", None)
    return env


def _models(tmp_path: Path) -> tuple[Path, Path]:
    """Two GGUF headers: the IQ3 file, and a bigger Q4_K_M file with the same template."""
    return write_gguf(tmp_path / "iq3.gguf", file_type=23), write_gguf(tmp_path / "q4.gguf", pad=4096)


def _descendants(pid: int) -> dict[int, str]:
    """Every process under *pid* (by parent PID), with its command line."""
    out: dict[int, str] = {}
    todo = [pid]
    while todo:
        res = subprocess.run(["pgrep", "-P", str(todo.pop())], capture_output=True, text=True)
        for child in (int(x) for x in res.stdout.split()):
            cmd = subprocess.run(["ps", "-o", "command=", "-p", str(child)], capture_output=True, text=True)
            out[child] = cmd.stdout.strip()
            todo.append(child)
    return out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(st) and not st.startswith("Z")


def test_run_script_dry_run_prints_the_plan_and_starts_nothing(tmp_path):
    iq3, q4 = _models(tmp_path)
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\nexit 1\n")
    server.chmod(0o755)
    taskset = write_jsonl(tmp_path / "ts.jsonl", AB_ROWS)
    out = tmp_path / "out"
    common = ["bash", str(RUN_SH), "--tasks", str(taskset), "--out", str(out), "--server", str(server),
              "--iq3", str(iq3), "--q4", str(q4), "--port", "18080", "--py", sys.executable]
    hhmm = (dt.datetime.now().astimezone() + dt.timedelta(hours=1)).strftime("%H:%M")
    res = subprocess.run(common + ["--until", hhmm], capture_output=True, text=True, env=_script_env(tmp_path),
                         timeout=60)
    assert res.returncode == 0, res.stdout + res.stderr
    text = res.stdout
    assert text.startswith("DRY RUN") and not out.exists() and "problem" not in text
    servers = [ln for ln in text.splitlines() if str(server) in ln]
    assert len(servers) == 2
    for ln, model in zip(servers, ("iq3.gguf", "q4.gguf")):
        assert f"-m {tmp_path / model}" in ln and "--host 127.0.0.1 --port 18080" in ln
        # 16,384 tokens for EACH of the 2 slots: no unified cache two requests could overrun together
        assert "-c 32768 --parallel 2 --no-kv-unified" in ln and "--kv-unified" not in ln.replace("--no-kv", "")
        assert "--no-mmproj" in ln and "--chat-template-file" not in ln
        assert "--temp 0.7 --top-k 20 --top-p 0.8 --min-p 0.0" in ln
    # the request sampling teacher_ab.py fixes is the same as the servers'
    assert "--temperature 0.7 --top-k 20 --top-p 0.8 --min-p 0.0" in text
    # --until is shown as the absolute time it resolves to (once, at start)
    assert f"--until {hhmm} is resolved once" in text
    iso_shown = text.split("now it would be ")[1].split(")")[0]
    assert dt.datetime.fromisoformat(iso_shown).strftime("%H:%M") == hhmm
    assert "arm --out" in text and "judge --out" in text and f"--until {iso_shown}" in text
    # a pinned template goes to both servers (and lets the embedded templates differ)
    tmpl = tmp_path / "chat.jinja"
    tmpl.write_text("{{ messages }}")
    write_gguf(q4, template="{{ other }}", pad=4096)
    res = subprocess.run(common, capture_output=True, text=True, env=_script_env(tmp_path), timeout=60)
    assert res.returncode == 2 and "different chat templates" in res.stdout
    res = subprocess.run(common + ["--chat-template-file", str(tmpl)], capture_output=True, text=True,
                         env=_script_env(tmp_path), timeout=60)
    assert res.returncode == 0, res.stdout
    assert sum(f"--chat-template-file {tmpl}" in ln for ln in res.stdout.splitlines() if str(server) in ln) == 2
    _models(tmp_path)
    # the model files: the same file twice, or swapped (Q4 must be the bigger, the judge)
    for iq3_arg, q4_arg, why in ((q4, q4, "same model file"), (q4, iq3, "not bigger")):
        res = subprocess.run(common + ["--iq3", str(iq3_arg), "--q4", str(q4_arg)], capture_output=True, text=True,
                             env=_script_env(tmp_path), timeout=60)
        assert res.returncode == 2 and why in res.stdout, res.stdout
    # a --repo without docs: force_compaction padding would have nothing real to read
    bare = tmp_path / "bare"
    (bare / "src").mkdir(parents=True)
    res = subprocess.run(common + ["--repo", str(bare)], capture_output=True, text=True, env=_script_env(tmp_path),
                         timeout=60)
    assert res.returncode == 2 and "has no docs/" in res.stdout
    # usage errors
    assert subprocess.run(["bash", str(RUN_SH)], capture_output=True, env=_script_env(tmp_path)).returncode == 2
    res = subprocess.run(common + ["--execute"], capture_output=True, text=True, env=_script_env(tmp_path))
    assert res.returncode == 2 and "--until" in res.stderr and not out.exists()
    q4.unlink()
    res = subprocess.run(common, capture_output=True, text=True, env=_script_env(tmp_path))
    assert res.returncode == 2 and "Q4_K_M model not readable" in res.stdout


def test_run_script_execute_runs_one_server_at_a_time_and_stops_it_by_pid(tmp_path):
    if shutil.which("curl") is None:
        pytest.skip("curl is needed by the script")
    _models(tmp_path)
    server = tmp_path / "llama-server"
    server.write_text(f"#!{sys.executable}\n" + FAKE_LLAMA.read_text())
    server.chmod(0o755)
    rows = AB_ROWS[:3] + [AB_ROWS[3]]
    taskset = write_jsonl(tmp_path / "ts.jsonl", rows)
    out = tmp_path / "out"
    log = tmp_path / "requests.jsonl"
    env = _script_env(tmp_path)
    # each server cuts its first agent reply off mid-stream (llama-server's error event): that
    # task is a transport error, left for a rerun, and the script reruns the step at once
    env.update(FAKE_LLAMA_LOG=str(log), FAKE_LLAMA_FAIL_QWEN3_8_27B_IQ3_XXS="log rotation",
               FAKE_LLAMA_TRUNCATE_FIRST="1")
    # an HH:MM --until: resolved once at start, every step gets the absolute time
    until = (dt.datetime.now().astimezone() + dt.timedelta(hours=1)).strftime("%H:%M")
    res = subprocess.run(
        ["bash", str(RUN_SH), "--execute", "--tasks", str(taskset), "--out", str(out), "--server", str(server),
         "--iq3", str(tmp_path / "iq3.gguf"), "--q4", str(tmp_path / "q4.gguf"), "--port", str(free_port()),
         "--py", sys.executable, "--code", str(_REPO), "--n", "4", "--until", until, "--load-timeout", "60",
         "--min-free-mib-iq3", "0", "--min-free-mib-q4", "0", "--retry-pause", "0"],
        capture_output=True, text=True, env=env, timeout=300)
    assert res.returncode == 0, res.stderr[-4000:]
    run_log = (out / "run.log").read_text()
    resolved = run_log.split(f"--until {until} is ")[1].split()[0]
    assert dt.datetime.fromisoformat(resolved).strftime("%H:%M") == until and f"until={resolved} " in run_log
    assert "arm A left work (tasks_left); rerunning it (1 of 2)" in run_log
    assert "arm B left work (tasks_left); rerunning it (1 of 2)" in run_log
    reqs = [json.loads(x) for x in log.read_text().splitlines()]
    # one server at a time: every IQ3 request came before every Q4 request
    aliases = [r["alias"] for r in reqs]
    first_q4 = aliases.index("qwen3.8-27b-q4_k_m")
    assert set(aliases[:first_q4]) == {"qwen3.8-27b-iq3_xxs"} and set(aliases[first_q4:]) == {"qwen3.8-27b-q4_k_m"}
    # the judge was the Q4 server, for both arms' traces; never the IQ3 one
    judged = [r for r in reqs if r["kind"] == "judge"]
    assert len(judged) == 8 and {r["alias"] for r in judged} == {"qwen3.8-27b-q4_k_m"}
    gen = {json.dumps({k: r["body"].get(k) for k in SAMPLING}) for r in reqs if r["kind"] != "judge"}
    assert gen == {json.dumps(SAMPLING)}
    rep = json.loads((out / "ab_report.json").read_text())
    assert (rep["arms"]["A"]["pass"], rep["arms"]["B"]["pass"]) == (3, 4)
    assert rep["decision"]["verdict"] == ab.DECISION_Q4  # 25 points lower
    # the servers' /props: two files, 16,384 tokens per slot on both, one template
    servers = rep["settings"]["servers"]
    assert (servers["A"]["model_file"], servers["B"]["model_file"]) == ("iq3.gguf", "q4.gguf")
    assert servers["A"]["n_ctx"] == servers["B"]["n_ctx"] == 16384 and rep["judge"]["is_arm_b_model"]
    # both servers were stopped by their recorded PIDs; nothing is left behind
    assert not (out / "server.pid").exists() and not (out / "run.lock.d").exists()
    stopped = [int(ln.split("stopping server ")[1].split()[0]) for ln in (out / "run.log").read_text().splitlines()
               if "stopping server " in ln]
    assert len(stopped) == 2
    for pid in stopped:
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            os.kill(pid, signal.SIGKILL)
            pytest.fail(f"server {pid} still running")
    # a rerun has nothing left to do: no server is started again
    n_before = len(reqs)
    res = subprocess.run(
        ["bash", str(RUN_SH), "--execute", "--tasks", str(taskset), "--out", str(out), "--server", str(server),
         "--iq3", str(tmp_path / "iq3.gguf"), "--q4", str(tmp_path / "q4.gguf"), "--port", str(free_port()),
         "--py", sys.executable, "--code", str(_REPO), "--n", "4", "--until", resolved],
        capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stderr[-2000:]
    assert len(log.read_text().splitlines()) == n_before and "starting" not in res.stderr


def test_stop_only_touches_the_recorded_server(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    # a PID that is not our server (this test's own process) is never signalled
    (out / "server.pid").write_text(f"{os.getpid()}\nsome other start\n{tmp_path / 'q4.gguf'}\n")
    res = subprocess.run(["bash", str(RUN_SH), "--stop", "--execute", "--out", str(out),
                          "--server", str(tmp_path / "llama-server")],
                         capture_output=True, text=True, env=_script_env(tmp_path), timeout=60)
    assert res.returncode == 1 and "no longer the server" in res.stderr
    assert not (out / "server.pid").exists()
    # dry run of --stop says what it would do and does nothing
    (out / "server.pid").write_text("999999\nx\ny\n")
    res = subprocess.run(["bash", str(RUN_SH), "--stop", "--out", str(out)], capture_output=True, text=True,
                         env=_script_env(tmp_path), timeout=60)
    assert res.returncode == 0 and "would stop PID 999999" in res.stdout and (out / "server.pid").exists()



def test_run_subcommand_does_everything_against_servers_up_at_once(tmp_path, no_outbound, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    taskset = write_jsonl(tmp_path / "real-ab.jsonl", AB_ROWS[:4])
    with ArmServer("qwen3.8-27b-iq3_xxs", ("log rotation",)) as a_srv, ArmServer("qwen3.8-27b-q4_k_m") as b_srv:
        rc = ab.main(["run", "--tasks", str(taskset), "--out", str(tmp_path / "ab"), "--repo", str(tmp_path),
                      "--n", "4", "--teacher-id", "qwen3*8*27b*", "--min-task-seconds", "5",
                      "--url-a", a_srv.url, "--url-b", b_srv.url, "--judge-url", b_srv.url])
        assert a_srv.kinds["judge"] == 0 and b_srv.kinds["judge"] == 8
    assert rc == 0
    rep = json.loads((tmp_path / "ab" / "ab_report.json").read_text())
    assert (rep["arms"]["A"]["pass"], rep["arms"]["B"]["pass"]) == (3, 4)
    assert rep["decision"]["verdict"] == ab.DECISION_Q4  # 25 points lower


def test_time_left_and_the_script_never_starts_a_server_without_time(tmp_path, capsys):
    soon = (dt.datetime.now().astimezone() + dt.timedelta(minutes=2)).isoformat(timespec="seconds")
    assert ab.main(["time-left", "--until", soon, "--min-seconds", "60"]) == 0
    assert ab.main(["time-left", "--until", soon, "--min-seconds", "600"]) == 1
    assert ab.main(["time-left", "--until", "2020-01-01T00:00:00"]) == 1
    assert ab.main(["time-left", "--until", "half past six"]) == 2
    capsys.readouterr()
    _models(tmp_path)
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\necho started >> \"$0.ran\"\nexit 1\n")
    server.chmod(0o755)
    taskset = write_jsonl(tmp_path / "ts.jsonl", AB_ROWS)
    out = tmp_path / "out"
    base = ["bash", str(RUN_SH), "--execute", "--tasks", str(taskset), "--out", str(out), "--server", str(server),
            "--iq3", str(tmp_path / "iq3.gguf"), "--q4", str(tmp_path / "q4.gguf"), "--port", str(free_port()),
            "--py", sys.executable, "--code", str(_REPO), "--n", "4"]
    res = subprocess.run(base + ["--until", soon], capture_output=True, text=True, env=_script_env(tmp_path),
                         timeout=120)
    assert res.returncode == 1 and "not starting the IQ3 server" not in res.stderr
    assert "not starting anything" in res.stderr
    assert not Path(str(server) + ".ran").exists() and not (out / "server.pid").exists()
    assert (out / "ab_selection.json").exists() and not (out / "run.lock.d").exists()
    res = subprocess.run(base + ["--until", "half past six"], capture_output=True, text=True,
                         env=_script_env(tmp_path), timeout=120)
    assert res.returncode == 2 and not Path(str(server) + ".ran").exists()
    # started after its HH:MM (a late timer, a manual resume): refused, never read as tomorrow
    past = (dt.datetime.now().astimezone() - dt.timedelta(minutes=1)).strftime("%H:%M")
    res = subprocess.run(base + ["--until", past], capture_output=True, text=True, env=_script_env(tmp_path),
                         timeout=120)
    assert res.returncode == 2 and "passed today" in res.stderr and not Path(str(server) + ".ran").exists()
    # an end further ahead than --max-hours (a window is 6 h) is refused too
    late = (dt.datetime.now().astimezone() + dt.timedelta(hours=9)).isoformat(timespec="seconds")
    res = subprocess.run(base + ["--until", late], capture_output=True, text=True, env=_script_env(tmp_path),
                         timeout=120)
    assert res.returncode == 2 and "--max-hours" in res.stderr and not Path(str(server) + ".ran").exists()
    # an ISO end already past: nothing to do
    res = subprocess.run(base + ["--until", "2020-01-01T00:00:00"], capture_output=True, text=True,
                         env=_script_env(tmp_path), timeout=120)
    assert res.returncode == 1 and "already passed" in res.stderr and not Path(str(server) + ".ran").exists()


def _start_script(tmp_path: Path, env_extra: dict[str, str]) -> tuple[subprocess.Popen, Path]:
    _models(tmp_path)
    server = tmp_path / "llama-server"
    server.write_text(f"#!{sys.executable}\n" + FAKE_LLAMA.read_text())
    server.chmod(0o755)
    # 4 tasks on 2 slots: once the server is gone, an orphaned step would still be retrying
    # the queued ones (connection refused, with backoff) for seconds after the script ends
    taskset = write_jsonl(tmp_path / "ts.jsonl", AB_ROWS[3:7])
    out = tmp_path / "out"
    env = _script_env(tmp_path)
    env.update(env_extra)
    until = (dt.datetime.now().astimezone() + dt.timedelta(hours=1)).isoformat(timespec="seconds")
    proc = subprocess.Popen(
        ["bash", str(RUN_SH), "--execute", "--tasks", str(taskset), "--out", str(out), "--server", str(server),
         "--iq3", str(tmp_path / "iq3.gguf"), "--q4", str(tmp_path / "q4.gguf"), "--port", str(free_port()),
         "--py", sys.executable, "--code", str(_REPO), "--n", "4", "--until", until, "--load-timeout", "60",
         "--min-free-mib-iq3", "0", "--min-free-mib-q4", "0"],
        # files, not pipes: an orphan holding a pipe would keep communicate() waiting for it
        stdout=(tmp_path / "script.out").open("w"), stderr=(tmp_path / "script.err").open("w"), env=env)
    return proc, out


def _wait_for(proc: subprocess.Popen, pred: Callable[[dict[int, str]], bool], timeout: float = 60) -> dict[int, str]:
    end = time.time() + timeout
    while time.time() < end:
        tree = _descendants(proc.pid)
        if pred(tree):
            return tree
        assert proc.poll() is None, proc.args
        time.sleep(0.05)
    raise AssertionError("timed out waiting for the script")


@pytest.mark.parametrize("phase", ["server-loading", "arm-step"])
def test_a_signal_leaves_no_server_and_no_step_running(tmp_path, phase):
    """SIGTERM to the script (the window's end, an operator's Ctrl-C) stops what it
    started: the step's own Python process (not just a subshell around it) and the
    server, recorded the moment it was spawned (a signal while it loads included)."""
    if shutil.which("curl") is None or shutil.which("pgrep") is None:
        pytest.skip("curl and pgrep are needed")
    extra = ({"FAKE_LLAMA_HEALTH_DELAY": "60"} if phase == "server-loading"
             else {"FAKE_LLAMA_AGENT_DELAY": "60"})
    proc, out = _start_script(tmp_path, extra)
    tree: dict[int, str] = {}
    try:
        if phase == "server-loading":
            tree = _wait_for(proc, lambda t: any("--alias qwen3.8-27b" in c for c in t.values()))
        else:
            tree = _wait_for(proc, lambda t: any("teacher_ab.py arm" in c for c in t.values()))
            time.sleep(0.5)
            tree.update(_descendants(proc.pid))
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=120)
        err = (tmp_path / "script.err").read_text()
        assert proc.returncode != 0
        assert "signal: stopping" in err
        # the script waits for what it stops: at its exit nothing it started may still run
        # (an orphaned step would only give up later, after its own retries)
        deadline = time.time() + 1
        while time.time() < deadline and any(_alive(pid) for pid in tree):
            time.sleep(0.1)
        left = {pid: cmd for pid, cmd in tree.items() if _alive(pid)}
        assert not left, f"left running after the script ended: {left}"
        assert not (out / "server.pid").exists() and not (out / "run.lock.d").exists()
        assert any("--alias qwen3.8-27b" in c for c in tree.values())
        if phase == "arm-step":
            assert any("teacher_ab.py arm" in c for c in tree.values())
    finally:
        if proc.poll() is None:
            proc.kill()
        for pid in tree:  # by recorded PID only
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)
