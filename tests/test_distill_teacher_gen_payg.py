"""Tests for teacher_gen.py --teacher max-payg and the real-only compaction padding.

A fake Model Studio (an OpenAI-compatible, non-streaming server on loopback;
the paid client's checked https request is re-pointed at it by an injected
opener) stands in for Qwen Max, and a fake llama-server for the local 27B
judge and the local teacher. HOME is tmp_path, so the one canonical spend
ledger (~/distill/spend-ledger.csv), its STOP file and the notify command
live there. Fixture keys are fake. No real model, no paid call, no network
beyond 127.0.0.1.
"""

from __future__ import annotations

import asyncio
import collections
import csv
import datetime as dt
import json
import signal
import socket
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import pytest

from tests.test_distill_build import feed
from tests.test_distill_teacher_ab import gym_builder, pad_rows, ts_row, write_docs, write_jsonl
from tests.test_distill_teacher_gen import (  # noqa: F401 - fixtures are used by name
    FakeTeacher,
    _last_user,
    _load,
    _restore_state_env,
    echo_reply,
    tg,
)

spend = _load("distill_spend", "spend.py")
build = _load("distill_build", "build.py")

MAX = "qwen3.8-max"
KEY = "sk-" + "fixturepaygkey" + "0123456789"
PLAN_KEY = "sk-sp-" + "fixtureplankey0123456"
REF = "operator Q6b 2026-10-30"
VERDICT = {"verdict": "pass", "completed": True, "tools_ok": True, "grounded": True,
           "recovery_ok": None, "reason": "fine"}
SUMMARY = "Earlier turns: questions about services and notes, each answered."


@pytest.fixture(autouse=True)
def _home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """No ambient key; HOME (the canonical ledger, its STOP file, the notify command) in tmp_path."""
    for name in spend.SECRET_ENV:
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    (home / "distill").mkdir(parents=True)
    notify = home / "distill" / "notify.sh"
    notify.write_text('#!/bin/sh\ncat >> "$HOME/notified.txt"\n')
    notify.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("PROMETHEUS_ENV_FILE", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    return home


@pytest.fixture
def no_outbound(monkeypatch: pytest.MonkeyPatch) -> None:
    real = socket.socket.connect

    def guarded(self, address):  # noqa: ANN001
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"outbound connection attempted to {address!r}")
        return real(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


# ── fakes ──────────────────────────────────────────────────────────────────────


def read_then_echo(body: dict[str, Any]) -> dict[str, Any]:
    """A teacher policy: with tools offered, read a repo doc once, then answer
    quoting the results; without tools, answer."""
    msgs = body["messages"]
    k, _prompt = _last_user(msgs)
    step = sum(1 for m in msgs[k + 1:] if m.get("role") == "assistant")
    if body.get("tools") and step == 0:
        return {"tool_calls": [("read_file", {"path": "docs/guide-0.md"})]}
    return echo_reply(body)


def wiki_then_echo(body: dict[str, Any]) -> dict[str, Any]:
    msgs = body["messages"]
    k, _prompt = _last_user(msgs)
    step = sum(1 for m in msgs[k + 1:] if m.get("role") == "assistant")
    if body.get("tools") and step < 2:
        return {"tool_calls": [("wiki_query", {"query": f"service host {step}"})]}
    return echo_reply(body)


class FakeMax:
    """Model Studio's chat completions, non-streaming, on loopback. Every POST is
    recorded with its headers. Teacher, summarizer and judge requests are told
    apart the way the local fake does. *errors* maps a kind (agent, summary,
    judge) to the HTTP statuses its next requests get instead of a reply (one
    each, in order; a 429 says Throttling). *hold* (agent requests) blocks a
    reply until ``release`` is set (``held`` is set once one is waiting)."""

    def __init__(self, policy: Callable[[dict[str, Any]], dict[str, Any]] = wiki_then_echo, *,
                 errors: dict[str, list[int]] | None = None, hold: bool = False) -> None:
        self.policy = policy
        self.requests: list[dict[str, Any]] = []
        self.kinds: collections.Counter = collections.Counter()
        self.errors = {k: list(v) for k, v in (errors or {}).items()}
        self.hold = hold
        self.held = threading.Event()
        self.release = threading.Event()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) or b"{}"
                body = json.loads(raw)
                fake.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                msgs = body.get("messages") or []
                if msgs and str(msgs[0].get("content", "")).startswith("You are a strict grader"):
                    kind = "judge"
                elif any("You are a conversation compactor" in tg.content_text(m.get("content")) for m in msgs):
                    kind = "summary"
                else:
                    kind = "agent"
                fake.kinds[kind] += 1
                if fake.errors.get(kind):
                    status = fake.errors[kind].pop(0)
                    err = json.dumps({"error": {"code": "Throttling" if status == 429 else "InternalError",
                                                "message": "fixture error"}}).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                    return
                if kind == "judge":
                    message: dict[str, Any] = {"role": "assistant", "content": json.dumps(VERDICT)}
                elif kind == "summary":
                    message = {"role": "assistant", "content": SUMMARY}
                else:
                    if fake.hold:
                        fake.held.set()
                        fake.release.wait(60)
                    reply = fake.policy(body)
                    message = {"role": "assistant", "content": reply.get("content")}
                    if reply.get("tool_calls"):
                        message["tool_calls"] = [
                            {"id": f"call_{len(fake.requests)}_{i}", "type": "function",
                             "function": {"name": name, "arguments": json.dumps(args)}}
                            for i, (name, args) in enumerate(reply["tool_calls"])]
                # usage well under the worst case spend.py reserved (bytes as tokens)
                usage = {"prompt_tokens": max(1, len(raw) // 8), "completion_tokens": 25}
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                data = json.dumps({"id": "chatcmpl-fixture", "object": "chat.completion", "model": MAX,
                                   "choices": [{"index": 0, "message": message,
                                                "finish_reason": "tool_calls" if message.get("tool_calls")
                                                else "stop"}],
                                   "usage": usage}).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:  # the client gave up on a held reply
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "FakeMax":
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


class ToFake:
    """The paid client's opener: its (checked, https) request goes to the fake instead."""

    def __init__(self, fake: FakeMax) -> None:
        self.base = fake.url
        self.inner = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def open(self, request: urllib.request.Request, timeout: float | None = None) -> Any:
        request.full_url = self.base + urlsplit(request.full_url).path
        return self.inner.open(request, timeout=timeout)


# ── helpers ────────────────────────────────────────────────────────────────────


def inputs(tmp_path: Path) -> list[str]:
    ev = _load("distill_build_eval_sets", "build_eval_sets.py")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(ev.build_manifest([], 1)))
    evt = tmp_path / "eval-tools.jsonl"
    evt.write_text("")
    return ["--injection-manifest", str(manifest), "--eval-tools", str(evt), "--sandbox-root",
            str(tmp_path / "sbx"), "--quiet", "--request-timeout", "20", "--repo", str(tmp_path)]


def ledger_rows(home: Path) -> list[dict[str, str]]:
    path = home / "distill" / "spend-ledger.csv"
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def taskset(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    write_docs(tmp_path)
    rows = [ts_row("What does a maintenance window mean for a small home server?", "chat-mode"),
            ts_row("Which host runs cobalt-intake according to the wiki?"),
            ts_row("Look up cobalt-intake in the wiki and tell me its host.", inject_errors=True),
            ts_row("Find the owner of amber-ledger in the wiki notes.", "other", force_compaction=True),
            *pad_rows()]
    return write_jsonl(tmp_path / "real-test.jsonl", rows), rows


def payg_args(tmp_path: Path, ts: Path, out: Path, *extra: str) -> list[str]:
    return ["run", "--teacher", "max-payg", "--tasks", str(ts), "--out", str(out),
            "--compaction-history-tokens", "150", *inputs(tmp_path), *extra]


# ── the key: QWEN_PAYG_KEY only, refused cleanly until it is there ────────────


def test_max_payg_refuses_cleanly_before_any_call(tmp_path, monkeypatch, no_outbound, capsys, _home):
    ts, _rows = taskset(tmp_path)
    out = tmp_path / "gen-max-derived"
    with FakeMax() as fake:
        def run() -> int:
            return tg.main(payg_args(tmp_path, ts, out, "--budget-usd", "5"), payg_opener=ToFake(fake))

        # no key at all (the state today): exit 5, nothing sent, nothing reserved
        assert run() == tg.EXIT_PAYG_REFUSED
        assert "QWEN_PAYG_KEY is not set" in capsys.readouterr().err
        # the Token Plan key is never a fallback, whatever it is called
        monkeypatch.setenv("QWEN_API_KEY", PLAN_KEY)
        assert run() == tg.EXIT_PAYG_REFUSED
        assert "QWEN_PAYG_KEY is not set" in capsys.readouterr().err
        monkeypatch.setenv("QWEN_PAYG_KEY", PLAN_KEY)  # the Token Plan key under the PAYG name
        assert run() == tg.EXIT_PAYG_REFUSED
        assert "same key as QWEN_API_KEY" in capsys.readouterr().err
        monkeypatch.delenv("QWEN_API_KEY")
        assert run() == tg.EXIT_PAYG_REFUSED  # a sk-sp- key under the PAYG name
        assert "subscription-plan key" in capsys.readouterr().err
        monkeypatch.setenv("QWEN_PAYG_KEY", KEY)
        monkeypatch.setenv("QWEN_API_KEY", KEY)
        assert run() == tg.EXIT_PAYG_REFUSED
        assert "same key as QWEN_API_KEY" in capsys.readouterr().err
        monkeypatch.setenv("QWEN_API_KEY", PLAN_KEY)
        monkeypatch.setenv("QWEN_PAYG_BASE_URL", "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1")
        assert run() == tg.EXIT_PAYG_REFUSED
        assert "subscription-plan host" in capsys.readouterr().err
        monkeypatch.delenv("QWEN_PAYG_BASE_URL")
        # a valid key, but the STOP file is there: refused before anything
        stop = _home / "distill" / "spend-ledger.STOP"
        stop.write_text("2026-10-30T23:00:00Z distill spend STOP: test\n")
        assert run() == tg.EXIT_PAYG_REFUSED
        assert "stopped" in capsys.readouterr().err
        stop.unlink()
        # a valid key, but no notify command: the alert could not reach the operator
        (_home / "distill" / "notify.sh").unlink()
        assert run() == tg.EXIT_PAYG_REFUSED
        assert "notify command" in capsys.readouterr().err
        assert not fake.requests
    assert not [r for r in ledger_rows(_home) if r["note"].startswith(("reserve", "settle"))]
    assert not (out / f"{tg.PAYG_SOURCE}.traces.jsonl").exists()


def test_max_payg_usage_errors(tmp_path, monkeypatch, no_outbound, capsys, _home):
    monkeypatch.setenv("QWEN_PAYG_KEY", KEY)
    ts, _rows = taskset(tmp_path)
    tagged = tmp_path / "gen-max-derived"
    for out, extra, why in (
            (tagged, [], "--budget-usd"),
            (tagged, ["--budget-usd", "0"], "--budget-usd"),
            (tagged, ["--budget-usd", "151"], "cap"),
            (tmp_path / "gen", ["--budget-usd", "5"], "max-derived tag"),
            (tagged, ["--budget-usd", "5", "--teacher-url", "http://127.0.0.1:8093"], "local teacher"),
            (tagged, ["--budget-usd", "5", "--teacher-id", "*"], "local teacher"),
            (tagged, ["--budget-usd", "5", "--model-name", "qwen3.8-plus"], "priced model"),
            (tagged, ["--budget-usd", "5", "--min-p", "0"], "--min-p"),
            (tagged, ["--budget-usd", "5", "--concurrency", "5"], "--concurrency"),
            (tagged, ["--budget-usd", "5", "--max-tokens", "9000"], "--max-tokens"),
            (tagged, ["--budget-usd", "5", "--thinking-budget", "-1"], "--thinking-budget"),
            (tagged, ["--budget-usd", "5", "--judge-model", "qwen3.8-27b"], "LOCAL judge"),
            (tagged, ["--budget-usd", "5", "--probe-guard-url", "http://127.0.0.1:18592"], "--judge-url")):
        assert tg.main(payg_args(tmp_path, ts, out, *extra)) == tg.EXIT_USAGE, extra
        assert why in capsys.readouterr().err, extra
    # the paid flags mean nothing to the local teacher
    for extra in (["--budget-usd", "5"], ["--thinking-budget", "100"]):
        argv = ["run", "--tasks", str(ts), "--out", str(tmp_path / "local"), *inputs(tmp_path), *extra]
        assert tg.main(argv) == tg.EXIT_USAGE
        assert "--teacher max-payg" in capsys.readouterr().err
    # the paid key is for the real-pattern taskset only: never the hand-written smoke tasks
    listed = tmp_path / "smoke-tasks.jsonl"
    listed.write_text("")
    with FakeMax() as fake:
        for extra in (["--builtin-smoke"], ["--builtin-smoke", "--tasks-file", str(listed)],
                      ["--builtin-smoke", "--max-tasks", "1", "--judge-url", "http://127.0.0.1:9"]):
            argv = ["run", "--teacher", "max-payg", "--budget-usd", "5", "--out", str(tagged), *extra,
                    *inputs(tmp_path)]
            assert tg.main(argv, payg_opener=ToFake(fake)) == tg.EXIT_USAGE, extra
            assert "real-pattern taskset only" in capsys.readouterr().err, extra
        assert not fake.requests
    assert not ledger_rows(_home)


# ── a paid run end to end ─────────────────────────────────────────────────────


def test_max_payg_run_writes_max_derived_rows_build_admits_only_under_the_allow(
        tmp_path, monkeypatch, no_outbound, _home):
    monkeypatch.setenv("QWEN_PAYG_KEY", KEY)
    monkeypatch.setenv("QWEN_API_KEY", PLAN_KEY)  # present, and never sent
    ts, rows = taskset(tmp_path)
    out = tmp_path / "gen-max-derived"
    with FakeMax() as fake, FakeTeacher() as judge:
        rc = tg.main(payg_args(tmp_path, ts, out, "--budget-usd", "5", "--max-tokens", "2048",
                               "--judge-url", judge.url, "--judge-id", "fake-teacher",
                               "--temperature", "0.7", "--top-k", "20", "--top-p", "0.8"),
                     payg_opener=ToFake(fake))
        posts = list(fake.requests)
        judge_calls = judge.kinds["judge"]
    assert rc == tg.EXIT_DONE
    # every request: the PAYG key only, the spend.py fields only, bounded, thinking explicit
    assert posts and all(p["headers"].get("Authorization") == "Bearer " + KEY for p in posts)
    assert not any(PLAN_KEY in json.dumps(p) for p in posts)
    for p in posts:
        b = p["body"]
        assert set(b) <= spend.PAYLOAD_KEYS, set(b) - spend.PAYLOAD_KEYS
        assert b["model"] == MAX and b["stream"] is False and b["enable_thinking"] is False
        assert "thinking_budget" not in b and 1 <= b["max_tokens"] <= 2048
    teacher_posts = [p["body"] for p in posts if "You are a strict grader" not in json.dumps(p["body"])]
    assert all({k: b[k] for k in ("temperature", "top_k", "top_p")} == {"temperature": 0.7, "top_k": 20,
                                                                        "top_p": 0.8} for b in teacher_posts)
    # the local 27B judged (preferred when given); Max never judged
    # (a taskset this small reserves no pad pool: every row runs as a task)
    assert fake.kinds["judge"] == 0 and judge_calls == len(rows)
    # one ledger: every paid call reserved, then settled, under teacher_gen / 2b-teacher
    led = ledger_rows(_home)
    reserves = [r for r in led if r["note"].startswith("reserve")]
    settles = [r for r in led if r["note"].startswith("settle")]
    assert len(reserves) == len(settles) == len(posts)
    assert {(r["tool"], r["phase"], r["model"]) for r in reserves + settles} == {("teacher_gen", "2b-teacher", MAX)}
    assert {r["call_id"] for r in reserves} == {r["call_id"] for r in settles}
    assert all(float(s["usd"]) <= float(r["usd"]) for r, s in zip(
        sorted(reserves, key=lambda x: x["call_id"]), sorted(settles, key=lambda x: x["call_id"])))
    # the rows: Max-derived, tagged, verified by the local judge, recorded as such
    got = [json.loads(x) for x in (out / f"{tg.PAYG_SOURCE}.traces.jsonl").read_text().splitlines()]
    assert {r["_meta"]["gen_task_id"] for r in got} == {r["id"] for r in rows}
    for r in got:
        m = r["_meta"]
        assert m["source"] == "max-payg-gym" and m["model"] == MAX and m["max_derived"] is True
        assert m["tags"] == ["max-derived"] and m["teacher"] == "max-payg" and m["served_model"] == MAX
        assert m["paid_call_id"] in {s["call_id"] for s in settles} and m["cost_usd"] > 0
        assert m["task_origin"] == "taskset" and m["provenance"] == "real-pattern-v1"
        j = m["verification"]["judge"]
        assert (j["provider"], j["judge_key"], j["model"], j["endpoint"]) == ("local", "none", "qwen3.8-27b",
                                                                              judge.url)
        assert m["verification"]["sandboxed"] is True and m["verification"]["side_effects"] is False
    comp = [r for r in got if r["_meta"]["gen_task_id"] == rows[3]["id"]]
    assert comp and all(r["_meta"]["compacted"] for r in comp)
    pad = comp[0]["_meta"]["compaction_pad"]
    # the padding turns were Max's own answers, paid and counted like any teacher call
    assert pad["turns"] >= tg.PAD_MIN_TURNS and pad["teacher_calls"] >= pad["turns"]
    assert fake.kinds["summary"] >= 1
    report = json.loads((out / f"{tg.PAYG_SOURCE}.report.json").read_text())
    assert report["max_derived"] is True and report["tags"] == ["max-derived"] and report["private"] is True
    assert report["teacher"] == "max-payg" and report["source"] == "max-payg-gym"
    assert report["spend"]["phase_paid_calls"] == len(settles) and report["spend"]["budget_usd"] == 5.0

    # build.py: refused without the allow (no split at all, not even eval) ...
    traces = out / f"{tg.PAYG_SOURCE}.traces.jsonl"
    b = gym_builder(tmp_path / "b-no", got, train_sources=build.DEFAULT_TRAIN_SOURCES)
    feed(b, traces, "extra")
    assert not b.examples and b.stats["excluded:max_payg_gym_needs_allow"] == len(got)
    # ... accepted under it, trained as Max under the approval, counted as Max in the share
    b = gym_builder(tmp_path / "b-allow", got, train_sources=build.DEFAULT_TRAIN_SOURCES,
                    allow_max_derived=True, max_derived_approval=REF)
    feed(b, traces, "extra")
    assert not [k for k in b.stats if k.startswith("excluded:")], dict(b.stats)
    assert len(b.examples) == len(got)
    assert all(e.max_derived and e.train_ok and e.approval == REF and e.source == "max-payg-gym"
               for e in b.examples)
    info = b.split()
    report = b.report(info, b.gates(), catalog="stubs", head_tokens={}, inputs={}, elapsed=0.0)
    share = report["data_policy"]["share"]["by_split"]
    in_train_val = sum(share[k]["max"]["rows"] for k in ("train", "val"))
    assert in_train_val == len(b.final["train"]) + len(b.final["val"]) > 0
    assert all(share[k]["qwen38-27b-gym"]["rows"] == 0 for k in ("train", "val"))
    assert report["gates"]["data_policy"]["max_derived_covered_in_train_val"] == in_train_val
    assert report["max_derived"] is True


def test_max_payg_self_judge_is_the_pay_per_token_judge_build_reads(tmp_path, monkeypatch, no_outbound, _home):
    monkeypatch.setenv("QWEN_PAYG_KEY", KEY)
    write_docs(tmp_path)
    rows = [ts_row("Which host runs cobalt-intake according to the wiki?"),
            ts_row("What is a good way to keep notes for an on-call handoff?", "chat-mode")]
    ts = write_jsonl(tmp_path / "real-small.jsonl", rows)
    out = tmp_path / "self-max-derived"
    with FakeMax() as fake:
        rc = tg.main(payg_args(tmp_path, ts, out, "--budget-usd", "2"), payg_opener=ToFake(fake))
        judged = fake.kinds["judge"]
        judge_bodies = [p["body"] for p in fake.requests if "You are a strict grader" in json.dumps(p["body"])]
    assert rc == tg.EXIT_DONE and judged == len(rows)
    assert all(b["temperature"] == 0 and b["seed"] == tg.DEFAULT_JUDGE_SEED and b["enable_thinking"] is False
               and b["max_tokens"] == tg.JUDGE_MAX_TOKENS and set(b) <= spend.PAYLOAD_KEYS for b in judge_bodies)
    settled = {r["call_id"] for r in ledger_rows(_home) if r["note"].startswith("settle")}
    got = [json.loads(x) for x in (out / f"{tg.PAYG_SOURCE}.traces.jsonl").read_text().splitlines()]
    assert got
    for r in got:
        j = r["_meta"]["verification"]["judge"]
        assert (j["kind"], j["provider"], j["judge_key"], j["model"]) == ("self-judge", "max-payg", "payg", MAX)
        assert j["call_ids"] and set(j["call_ids"]) <= settled and j["pass"] is True
        # build.py: the pay-per-token judge verifies a Max row's record; nothing else naming Max does
        v = build.verification_of(r["_meta"], strict=True, judge_models=build.DEFAULT_TRAIN_MODELS, payg_judge=True)
        assert (v.decision, v.payg_judge) == (True, True)
        assert build.verification_of(r["_meta"], strict=True, judge_models=build.DEFAULT_TRAIN_MODELS).basis \
            == "judge_is_max"
    b = gym_builder(tmp_path / "b", got, train_sources=build.DEFAULT_TRAIN_SOURCES,
                    allow_max_derived=True, max_derived_approval=REF)
    feed(b, out / f"{tg.PAYG_SOURCE}.traces.jsonl", "extra")
    assert len(b.examples) == len(got) and b.stats["verification:payg_max_judge"] == len(got)
    # the strict record is required: a row without its sandbox record is refused, not eval-only
    cut = [dict(r, _meta=dict(r["_meta"], verification={k: v for k, v in r["_meta"]["verification"].items()
                                                         if k != "sandboxed"})) for r in got]
    bad = write_jsonl(tmp_path / "cut.traces.jsonl", cut)
    b = gym_builder(tmp_path / "b-cut", cut, train_sources=build.DEFAULT_TRAIN_SOURCES,
                    allow_max_derived=True, max_derived_approval=REF)
    feed(b, bad, "extra")
    assert not b.examples and b.stats["excluded:max_payg_gym_unverified:no_sandbox_record"] == len(cut)
    # and so is the Max-derived tag the producer writes
    untagged = [dict(r, _meta=dict(r["_meta"], max_derived=False)) for r in got]
    bad = write_jsonl(tmp_path / "untagged.traces.jsonl", untagged)
    b = gym_builder(tmp_path / "b-untagged", untagged, train_sources=build.DEFAULT_TRAIN_SOURCES,
                    allow_max_derived=True, max_derived_approval=REF)
    feed(b, bad, "extra")
    assert not b.examples and b.stats["excluded:max_payg_gym_not_max_derived"] == len(untagged)


# ── the budget and the STOP ────────────────────────────────────────────────────


def _prefill(home: Path, usd: float, *, alerted: bool = False) -> None:
    """The ledger with another tool's settled spend already in it (and, *alerted*, the
    $140 alert already sent once)."""
    path = home / "distill" / "spend-ledger.csv"
    base = {"ts": "2026-10-30T22:00:00Z", "tool": "judge", "phase": "2-judge", "model": MAX,
            "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=spend.LEDGER_FIELDS)
        w.writeheader()
        w.writerow(dict(base, call_id="prefill", usd=f"{usd:.6f}", cumulative_usd=f"{usd:.6f}", note="settle ok"))
        if alerted:
            w.writerow(dict(base, call_id="", usd="0", cumulative_usd=f"{usd:.6f}",
                            note="alert threshold 140.00 notify:sent committed spend crossed the alert"))
    path.chmod(0o600)


def _tasks_ledger(out: Path) -> list[dict[str, Any]]:
    path = out / f"{tg.PAYG_SOURCE}.tasks.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def test_max_payg_honours_the_phase_budget_the_cap_and_the_stop(tmp_path, monkeypatch, no_outbound, _home):
    monkeypatch.setenv("QWEN_PAYG_KEY", KEY)
    write_docs(tmp_path)
    rows = [ts_row("Which host runs cobalt-intake according to the wiki?"),
            ts_row("Which team owns harbor-sync according to the wiki?")]
    ts = write_jsonl(tmp_path / "real-small.jsonl", rows)

    # 1. this phase's --budget-usd is smaller than one call's worst case: refused before sending
    out = tmp_path / "budget-max-derived"
    with FakeMax() as fake:
        rc = tg.main(payg_args(tmp_path, ts, out, "--budget-usd", "0.001"), payg_opener=ToFake(fake))
        assert rc == tg.EXIT_SPEND_STOP and not fake.requests
    report = json.loads((out / f"{tg.PAYG_SOURCE}.report.json").read_text())
    assert report["stop"] == "spend_stop" and "budget" in report["stop_detail"]
    assert not _tasks_ledger(out)  # the task in flight is not recorded: it reruns next time
    assert not [r for r in ledger_rows(_home) if r["note"].startswith("reserve")]

    # 2. the global $150 cap, whatever the phase budget (the operator removed an earlier
    #    STOP to resume on purpose: the $140 alert is not sent again, the cap still holds)
    _prefill(_home, 149.99, alerted=True)
    out = tmp_path / "cap-max-derived"
    with FakeMax() as fake:
        rc = tg.main(payg_args(tmp_path, ts, out, "--budget-usd", "5"), payg_opener=ToFake(fake))
        assert rc == tg.EXIT_SPEND_STOP and not fake.requests
    assert "cap" in json.loads((out / f"{tg.PAYG_SOURCE}.report.json").read_text())["stop_detail"]

    # 3. the $140 alert: the call that crosses it goes out (it is under the cap), STOP is
    #    written and the operator told; the next call, from any tool, is refused
    _prefill(_home, 139.99)
    out = tmp_path / "stop-max-derived"
    with FakeMax() as fake:
        rc = tg.main(payg_args(tmp_path, ts, out, "--budget-usd", "5", "--concurrency", "1"),
                     payg_opener=ToFake(fake))
        assert rc == tg.EXIT_SPEND_STOP and len(fake.requests) == 1
    assert (_home / "distill" / "spend-ledger.STOP").exists()
    assert "STOPPED" in (_home / "notified.txt").read_text()
    assert json.loads((out / f"{tg.PAYG_SOURCE}.report.json").read_text())["stop"] == "spend_stop"
    assert not _tasks_ledger(out)
    led = ledger_rows(_home)
    assert [r["note"].split(" ", 1)[0] for r in led if r["tool"] == "teacher_gen"] == ["reserve", "alert", "settle"]
    # a rerun while STOP is there is refused before anything (exit 5)
    with FakeMax() as fake:
        assert tg.main(payg_args(tmp_path, ts, out, "--budget-usd", "5"), payg_opener=ToFake(fake)) \
            == tg.EXIT_PAYG_REFUSED
        assert not fake.requests


# ── --until and stop signals: no paid call is left open ───────────────────────


def _one_task(tmp_path: Path) -> Path:
    write_docs(tmp_path)
    return write_jsonl(tmp_path / "real-one.jsonl", [ts_row("Which host runs cobalt-intake according to the wiki?")])


def test_max_payg_calls_end_before_until_and_none_starts_too_close(tmp_path, monkeypatch, no_outbound, _home):
    """A paid call runs in a worker thread that cancelling the task does not stop: under --until
    its timeout is cut so it is over before --until (gen_runner.py stops its children then), and
    the process ends on time instead of waiting out --request-timeout."""
    monkeypatch.setenv("QWEN_PAYG_KEY", KEY)
    monkeypatch.setattr(tg, "PAYG_UNTIL_MARGIN_S", 1.0)
    monkeypatch.setattr(tg, "PAYG_MIN_CALL_S", 1.0)
    ts = _one_task(tmp_path)
    out = tmp_path / "until-max-derived"
    until = dt.datetime.now().astimezone() + dt.timedelta(seconds=4)
    with FakeMax(hold=True) as fake:  # Max never answers in time
        t0 = time.monotonic()
        rc = tg.main(payg_args(tmp_path, ts, out, "--budget-usd", "5", "--min-task-seconds", "0",
                               "--until", until.isoformat(timespec="seconds")), payg_opener=ToFake(fake))
        took = time.monotonic() - t0
        assert fake.held.is_set() and fake.kinds["agent"] == 1
    assert rc in (tg.EXIT_DONE, tg.EXIT_NOTHING)
    assert took < 8, took  # was: --request-timeout (20 s here; 600 s under gen_runner's probe guard)
    led = ledger_rows(_home)
    reserves = {r["call_id"] for r in led if r["note"].startswith("reserve")}
    settles = {r["call_id"]: r["note"] for r in led if r["note"].startswith("settle")}
    assert len(reserves) == 1 and reserves == set(settles)  # nothing left open
    assert list(settles.values()) == ["settle error:timeout charged-reservation"]
    assert not [e for e in _tasks_ledger(out) if e["status"] in ("kept", "rejected")]  # it reruns

    # a call with less than PAYG_MIN_CALL_S left is never started: nothing reserved, the run stops
    monkeypatch.setattr(tg, "PAYG_MIN_CALL_S", 60.0)
    out = tmp_path / "late-max-derived"
    until = dt.datetime.now().astimezone() + dt.timedelta(seconds=30)
    with FakeMax() as fake:
        rc = tg.main(payg_args(tmp_path, ts, out, "--budget-usd", "5", "--min-task-seconds", "0",
                               "--until", until.isoformat(timespec="seconds")), payg_opener=ToFake(fake))
        assert not fake.requests
    assert rc == tg.EXIT_NOTHING and len(ledger_rows(_home)) == len(led)
    report = json.loads((out / f"{tg.PAYG_SOURCE}.report.json").read_text())
    assert report["stop"] == "until" and not _tasks_ledger(out)


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="posix only")
def test_max_payg_sigterm_settles_the_calls_in_flight_before_exiting(tmp_path, monkeypatch, no_outbound, _home):
    """gen_runner.py (or systemd) stops a paid run with SIGTERM: no new call, the calls in flight
    get a moment to settle, the rest are settled as interrupted, then the process exits 143. Before,
    the default action killed it mid-call and the reservation stayed open in the ledger for good."""
    monkeypatch.setenv("QWEN_PAYG_KEY", KEY)
    monkeypatch.setattr(tg, "PAYG_SIGNAL_WAIT_S", 0.5)

    class Exited(BaseException):
        pass

    codes: list[int] = []

    def fake_exit(code: int) -> None:
        codes.append(code)
        raise Exited(code)

    monkeypatch.setattr(tg, "_hard_exit", fake_exit)
    ts = _one_task(tmp_path)
    out = tmp_path / "sig-max-derived"
    before = signal.getsignal(signal.SIGTERM)
    with FakeMax(hold=True) as fake:
        def stopper() -> None:
            assert fake.held.wait(30)
            time.sleep(0.2)
            signal.pthread_kill(threading.main_thread().ident, signal.SIGTERM)
            end = time.time() + 30
            while not codes and time.time() < end:
                time.sleep(0.05)
            fake.release.set()  # the worker thread's reply comes too late to count

        th = threading.Thread(target=stopper, daemon=True)
        th.start()
        with pytest.raises(Exited):
            tg.main(payg_args(tmp_path, ts, out, "--budget-usd", "5"), payg_opener=ToFake(fake))
        th.join(30)
    assert codes == [128 + signal.SIGTERM]
    assert signal.getsignal(signal.SIGTERM) == before  # the handler is put back
    led = ledger_rows(_home)
    reserves = [r for r in led if r["note"].startswith("reserve")]
    settles = [r for r in led if r["note"].startswith("settle")]
    assert len(reserves) == 1 and [r["call_id"] for r in settles] == [reserves[0]["call_id"]]
    assert settles[0]["note"] == "settle error:interrupted charged-reservation"
    assert float(settles[0]["usd"]) == float(reserves[0]["usd"])
    assert not [e for e in _tasks_ledger(out) if e["status"] in ("kept", "rejected")]


def test_paid_client_close_and_settle_open(tmp_path, _home):
    """spend.PaygClient: close refuses every later call; settle_open settles the calls in flight
    once (a call that settled itself meanwhile is left as it settled)."""
    with FakeMax(hold=True) as fake:
        creds = spend.resolve_payg({"QWEN_PAYG_KEY": KEY}, env_file=tmp_path / "none.env")
        client = spend.PaygClient(creds, spend.Ledger(), tool="teacher_gen", phase="2b-teacher", model=MAX,
                                  timeout=20, opener=ToFake(fake))
        body = {"model": MAX, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16,
                "stream": False, "enable_thinking": False}
        result: dict[str, Any] = {}

        def call() -> None:
            try:
                result["r"] = client.chat(body)
            except BaseException as exc:  # noqa: BLE001
                result["e"] = exc

        th = threading.Thread(target=call)
        th.start()
        assert fake.held.wait(10)
        assert client.open_calls() == 1
        client.close("test stop")
        with pytest.raises(spend.SpendRefused, match="closed"):
            client.chat(body)
        settled = client.settle_open()
        assert len(settled) == 1 and client.open_calls() == 0
        assert client.settle_open() == []  # once
        fake.release.set()
        th.join(10)
    assert isinstance(result.get("e"), spend.SpendError)  # its own settle found it settled already
    notes = [r["note"] for r in ledger_rows(_home)]
    assert notes == ["reserve " + creds.region, "settle error:interrupted charged-reservation"]
    # a per-call timeout below the client's own
    with FakeMax(hold=True) as fake:
        client = spend.PaygClient(creds, spend.Ledger(), tool="teacher_gen", phase="2b-teacher", model=MAX,
                                  timeout=20, opener=ToFake(fake))
        t0 = time.monotonic()
        with pytest.raises(spend.PaidCallError) as err:
            client.chat(body, timeout=1.0)
        assert err.value.kind == "timeout" and time.monotonic() - t0 < 5
        with pytest.raises(ValueError, match="timeout"):
            client.chat(body, timeout=0)


def test_a_passing_summarizer_error_is_retried_not_a_rejection(tmp_path, monkeypatch, no_outbound, _home):
    """The compactor swallows its summarizer's errors and sends the history uncompacted: a 429 on
    the one summary request of a chat-mode force_compaction task must leave the task to rerun
    (error), not reject it for good as compaction_not_exercised."""
    monkeypatch.setenv("QWEN_PAYG_KEY", KEY)
    write_docs(tmp_path)
    comp = ts_row("What does a maintenance window mean for a small home server?", "chat-mode",
                  force_compaction=True)
    chat_pad = [ts_row(q, "chat-mode") for q in ("Why do log files on a small server need rotating?",
                                                  "What should a checklist for restoring a backup include?",
                                                  "How often should a home server's backups be tested?")]
    ts = write_jsonl(tmp_path / "real-comp.jsonl", [comp, *pad_rows(), *chat_pad])
    out = tmp_path / "comp-max-derived"
    argv = payg_args(tmp_path, ts, out, "--budget-usd", "5", "--concurrency", "1")
    with FakeMax(errors={"summary": [429]}) as fake:
        assert tg.main(argv, payg_opener=ToFake(fake)) == tg.EXIT_DONE
        assert fake.kinds["summary"] == 2  # refused once, then answered when the task ran again
    events = [e for e in _tasks_ledger(out) if e["task_id"] == comp["id"]]
    assert [e["status"] for e in events] == ["error", "kept"], events  # was: ["rejected"], for good
    assert events[0]["reasons"] == ["transport_error"] and "rate_limit" in events[0]["error"]
    got = [json.loads(x) for x in (out / f"{tg.PAYG_SOURCE}.traces.jsonl").read_text().splitlines()]
    mine = [r for r in got if r["_meta"]["gen_task_id"] == comp["id"]]
    assert mine and all(r["_meta"]["compacted"] for r in mine)
    # a rerun has nothing left to do for it
    with FakeMax() as fake:
        tg.main(argv, payg_opener=ToFake(fake))
        assert fake.kinds["summary"] == 0
    # a lasting refusal (a 400) of the summary request still rejects: retrying would fail the same way
    out2 = tmp_path / "comp400-max-derived"
    with FakeMax(errors={"summary": [400]}) as fake:
        tg.main(payg_args(tmp_path, ts, out2, "--budget-usd", "5", "--concurrency", "1"), payg_opener=ToFake(fake))
    events = [e for e in _tasks_ledger(out2) if e["task_id"] == comp["id"]]
    assert [e["status"] for e in events] == ["rejected"] and "compaction_not_exercised" in events[0]["reasons"]


def test_a_refused_paid_call_stops_the_run_from_any_request(tmp_path, monkeypatch, _home):
    """A refusal inside the loop, the compactor or the judge is a PaygStop: never a
    retried 'error', never a task recorded as done."""
    monkeypatch.setenv("QWEN_PAYG_KEY", KEY)

    class Refusing:
        def chat(self, payload: dict[str, Any], **_kw: Any) -> Any:
            raise spend.StopActive("paid calls are stopped (test)")

    s = tg.Settings(out=tmp_path / "o-max-derived", teacher=tg.TEACHER_PAYG, model_name=MAX,
                    teacher_id=tg.PAYG_TEACHER_ID, teacher_served=MAX, budget_usd=1.0,
                    payg_client=Refusing(), sandbox_root=tmp_path / "sbx", progress=False,
                    task_source=tg.ORIGIN_TASKSET)
    (t,) = tg.tasks_from_taskset_rows([ts_row("Which host runs cobalt-intake according to the wiki?")], tmp_path)
    with pytest.raises(tg.PaygStop) as err:
        asyncio.run(tg.process(t, s, 0))
    assert err.value.kind == "spend_stop"
    with pytest.raises(tg.PaygStop):
        asyncio.run(tg.judge_text("TASK: x", s))
    # an auth or billing refusal by the API ends the run too (exit 5), it is not retried
    with pytest.raises(tg.PaygStop) as err:
        tg.payg_call(type("C", (), {"chat": lambda self, p: (_ for _ in ()).throw(
            spend.PaidCallError("billing", "arrearage"))})(), {"model": MAX, "messages": []})
    assert err.value.kind == "payg_fatal"
    with pytest.raises(ValueError, match="PAYLOAD_KEYS"):
        tg.payg_call(Refusing(), {"model": MAX, "enable_search": True})


# ── compaction padding: real content only ─────────────────────────────────────


def _openai(message: Any) -> dict[str, Any]:
    from prometheus.providers.base import ApiMessageRequest
    from prometheus.providers.stub import _build_openai_messages

    return _build_openai_messages(ApiMessageRequest(model="m", messages=[message]))[-1]


def _assert_padded_row_is_real(row: dict[str, Any], out: Any, prompts: set[str]) -> None:
    """Every user-role string traces to a taskset prompt (verbatim) or to the runtime
    compactor's summary of earlier turns, whose body is the teacher's own summarizer
    reply; every tool-role string is a sandbox tool result; every assistant turn is a
    reply the teacher sent. Nothing in the row was written by the harness."""
    from prometheus.engine.messages import ToolResultBlock

    summaries = {c.reply.text.strip() for c in out.summary_calls}
    replies = [_openai(c.reply) for c in out.calls]
    results = {b.content if isinstance(b.content, str) else str(b.content)
               for m in out.messages for b in m.content if isinstance(b, ToolResultBlock)}
    for m in row["messages"]:
        text = tg.content_text(m.get("content"))
        if m["role"] == "user":
            if text in prompts:
                continue
            assert text.startswith(tg.SUMMARY_MARKER), text[:200]
            body = text.split("] ", 1)[1]
            body = body.split("\n\n[The user's current request, verbatim:]\n", 1)
            assert body[0] in summaries, body[0][:200]
            assert len(body) == 1 or body[1] in prompts
        elif m["role"] == "tool":
            assert text in results, text[:200]
        elif m["role"] == "assistant":
            target = {"content": m.get("content") or "", "tool_calls": m.get("tool_calls")}
            assert any((r.get("content") or "") == target["content"] and r.get("tool_calls") == target["tool_calls"]
                       for r in replies), m
        else:
            assert m["role"] == "system"


def test_compaction_padding_is_other_taskset_prompts_answered_by_the_teacher(tmp_path, no_outbound):
    write_docs(tmp_path)
    rows = [ts_row("Find the owner of amber-ledger in the wiki notes.", "other", force_compaction=True),
            ts_row("How should I think about retention for nightly backups?", "chat-mode", force_compaction=True),
            ts_row("Which team owns harbor-sync according to the wiki?"),
            ts_row("Find the port quill-indexer listens on in the wiki."),
            ts_row("What is a good way to keep notes for an on-call handoff?", "chat-mode"),
            ts_row("What should a weekly capacity report cover for a home lab?", "chat-mode"),
            ts_row("How do I decide when to rotate an old log file?", "chat-mode")]
    agent, chat = tg.tasks_from_taskset_rows(rows, tmp_path)[:2]
    prompts = {r["prompt"] for r in rows}
    # the pads are other rows' prompts; a chat task pads with chat turns only
    assert {p["id"] for p in agent.pad} <= {r["id"] for r in rows[1:]} and len(agent.pad) >= tg.PAD_MIN_TURNS
    assert {p["mode"] for p in chat.pad} == {"chat"} and rows[2]["id"] not in {p["id"] for p in chat.pad}
    assert not agent.seed and not chat.seed and not chat.setup_files
    # an agent task's sandbox holds whole repo docs, as they are in the repo
    assert agent.setup_files and all(text == (tmp_path / rel).read_text() for rel, text in agent.setup_files.items())
    old_template = ("Here's a reference section first", "What does it cover?", "tell me what it covers",
                    "It opens with", "Some reading first", "This section (")
    for t in (agent, chat):
        with FakeTeacher(read_then_echo) as fake:
            s = tg.Settings(out=tmp_path / "o", teacher_url=fake.url, sandbox_root=tmp_path / "sbx",
                            request_timeout=20.0, progress=False, task_source=tg.ORIGIN_TASKSET,
                            compaction_history_tokens=120)
            out = asyncio.run(tg.run_task(t, s))
            ok, reasons, record = tg.deterministic_checks(out, s)
            assert ok, reasons
            assert fake.kinds["summary"] >= 1
            record["judge"] = asyncio.run(tg.judge(out, s))
            record["verified"] = True
            got = tg.rows_for(out, s, record)
        assert got and out.pad["turns"] >= tg.PAD_MIN_TURNS and record["compaction"]["events"] >= 1
        assert out.pad["prompt_hashes"] == [tg.h(tg.norm(p["prompt"])) for p in t.pad[:out.pad["turns"]]]
        # the pad turns ran through the teacher (calls before the task's own), and are never rows
        assert out.pad_calls >= out.pad["turns"] and all(r["_meta"]["step"] >= out.pad_calls for r in got)
        if t is agent:
            if "agent" in out.pad["modes"]:
                assert out.pad["reads"] >= 1  # a read during padding returned a real repo doc
            # a read in this sandbox returns the repo's own doc text, verbatim
            from prometheus.engine.messages import ToolResultBlock

            doc = (tmp_path / "docs" / "guide-0.md").read_text().splitlines()[2]
            reads = [b.content for m in out.messages for b in m.content if isinstance(b, ToolResultBlock)]
            assert reads and all(doc in str(c) for c in reads)
        for r in got:
            assert r["_meta"]["compaction_pad"] == out.pad and r["_meta"]["compacted"]
            _assert_padded_row_is_real(r, out, prompts)
            blob = json.dumps(r["messages"])
            assert not any(phrase in blob for phrase in old_template)


def test_the_pad_pool_is_reserved_shared_by_every_run_and_kept_clear_of_the_eval_sets(tmp_path):
    write_docs(tmp_path)
    # (distinct words, not just numbers: the task proxy normalizes digits away)
    words = ["amber", "basalt", "cedar", "drift", "ember", "fern", "harbor", "lumen", "moss", "orchid",
             "pylon", "quill", "slate", "tidal", "willow", "cobalt", "heron", "marlin", "osprey", "puffin",
             "sable", "tern", "wren", "lynx", "otter", "ibis", "egret", "plover", "gannet", "curlew"]
    rows = [ts_row(f"Which host runs the {w} service according to the wiki?") for w in words]
    rows += [ts_row(f"What is a sensible way to plan {w} maintenance for a home server?", "chat-mode")
             for w in words]
    rows += [ts_row("Find the owner of amber-ledger in the wiki notes.", "other", force_compaction=True),
             ts_row("How should I think about retention for nightly backups?", "chat-mode", force_compaction=True),
             ts_row("Look up cobalt-intake in the wiki and tell me its host.", inject_errors=True)]
    main, pool, summary = tg.pad_pool_split(rows)
    assert summary["reserved"] and summary["pool_groups"] >= tg.PAD_MIN_TURNS
    assert sum(1 for r in pool if r["mode"] == "chat") >= tg.PAD_MIN_TURNS  # a chat-mode row is padded
    assert {r["id"] for r in main}.isdisjoint({r["id"] for r in pool}) and len(main) + len(pool) == len(rows)
    assert not any(r["flags"]["force_compaction"] or r["flags"]["inject_errors"] for r in pool)
    # the same pool whatever the row order (keyed by the taskset's name)
    again = tg.pad_pool_split(list(reversed(rows)))[1]
    assert {r["id"] for r in again} == {r["id"] for r in pool}
    # load_taskset: pool rows never run as tasks; padded tasks pad only from the pool
    ts = write_jsonl(tmp_path / "real-test.jsonl", rows)
    tasks, info = tg.load_taskset_with_pool(str(ts), tmp_path)
    assert {t.id for t in tasks} == {r["id"] for r in main} and info["tasks"] == len(main)
    for t in tasks:
        assert {p["id"] for p in t.pad} <= {r["id"] for r in pool}
    # a pool prompt that overlaps an eval set is never padding
    # (one the padded tasks can spare: a chat prompt when the pool holds more than they need)
    chat_pool = [r for r in pool if r["mode"] == "chat"]
    spare = chat_pool if len(chat_pool) > tg.PAD_MIN_TURNS else [r for r in pool if r["mode"] != "chat"]
    banned = spare[0]["prompt"]
    guard = tg.EvalGuard(None, {tg.h(tg.norm(banned))}, set())
    tasks, info = tg.load_taskset_with_pool(str(ts), tmp_path, guard=guard)
    assert info["eval_overlap_dropped"] == 1
    assert all(p["prompt"] != banned for t in tasks for p in t.pad)
    # too few other prompts to pad with is an input error, never a canned seed
    with pytest.raises(tg.NoPadPrompts):
        tg.tasks_from_taskset_rows([rows[-3], rows[0]], tmp_path)
    # a task's pad turns are part of what the eval guard checks
    t = next(t for t in tg.load_taskset(str(ts), tmp_path) if t.pad)
    assert t.pad[0]["prompt"] in t.user_texts()
    assert tg.EvalGuard(None, {tg.h(tg.norm(t.pad[0]["prompt"]))}, set()).task_overlaps(t) == ["eval_tools_task"]
