"""Tests for scripts/distill/judge.py and scripts/distill/build_fail_set.py (distill Phase 2).

No network and no model: the judge's client is a scripted fake, the clock and
the sleep are injected, and every store is a synthetic SQLite file under
tmp_path. Fixtures use neutral names only. The paid (max-payg) and local
clients, the spend ledger and judge_run.sh are tested against a fake
OpenAI-compatible server in tests/test_distill_spend.py.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import importlib.util
import json
import sqlite3
import sys
import threading
import time
import types
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable

import pytest

_DIR = Path(__file__).resolve().parent.parent / "scripts" / "distill"


def _load(name: str, filename: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


spend = _load("distill_spend", "spend.py")
judge = _load("distill_judge", "judge.py")
failset = _load("distill_build_fail_set", "build_fail_set.py")


@pytest.fixture(autouse=True)
def _home_in_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The one spend ledger is ~/distill/spend-ledger.csv: HOME keeps it (and its STOP) in tmp_path."""
    monkeypatch.setenv("HOME", str(tmp_path))


# ── fixtures ───────────────────────────────────────────────────────────────

BASH_TOOL = {
    "type": "function",
    "function": {"name": "bash", "description": "Run a command.",
                 "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                                "required": ["command"]}},
}
BASH_SCHEMA = json.dumps(BASH_TOOL["function"])

NOW = datetime(2026, 10, 31, 14, 0, tzinfo=timezone.utc)

PASS_REPLY = json.dumps({
    "verdict": "pass", "reasons": ["right next step"],
    "rubric": {"right_tool": 2, "valid_args": 2, "sensible_order": 2, "task_progress": 2,
               "no_fabrication": 2},
})
FAIL_REPLY = json.dumps({
    "verdict": "fail", "reasons": ["repeats the previous step"],
    "rubric": {"right_tool": 2, "valid_args": 2, "sensible_order": 0, "task_progress": 1,
               "no_fabrication": 2},
})


def make_trace(i: int, *, prompt: str = "list the files in the demo folder",
               command: str = "ls demo", session: str = "beacon:s1") -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": f"{prompt} {i}"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"call_{i}", "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": f"{command} {i}"})},
            }]},
        ],
        "tools": [BASH_TOOL],
        "_meta": {"model": "teacher", "tool_name": "bash", "timestamp": 1000.0 + i,
                  "session_id": session},
    }


def make_item(i: int, label: str = "pass", category: str = "coding",
              outcome_text: str | None = None) -> dict[str, Any]:
    trace = make_trace(i)
    return {
        "id": f"{label}-{i:04d}", "label": label, "category": category, "stratum_weight": 1.0,
        "source": {"telemetry_rowid": i}, "messages": trace["messages"], "tools": trace["tools"],
        "meta": trace["_meta"],
        "outcome": {"status": "success" if label == "pass" else "error", "text": outcome_text},
    }


class FakeClient:
    """Scripted judge client: each call pops the next reply (a string) or raises (an exception)."""

    provider = "fake"
    judge_key = "none"
    region = ""
    price = None

    def __init__(self, script: list[Any] | None = None, *, default: Any = PASS_REPLY,
                 usage: tuple[int, int] = (1000, 100), cost: float = 0.01, on_call: Any = None) -> None:
        self.script = list(script or [])
        self.default = default
        self.usage = usage
        self.cost = cost
        self.model = "judge-model"
        self.calls: list[tuple[str, str, int]] = []
        self.on_call = on_call

    def budget_describe(self) -> str:
        return "fake budget"

    async def complete(self, system: str, user: str, *, max_tokens: int) -> Any:
        self.calls.append((system, user, max_tokens))
        if self.on_call:
            self.on_call()
        step = self.script.pop(0) if self.script else self.default
        if isinstance(step, BaseException):
            raise step
        return judge.CallResult(text=step, input_tokens=self.usage[0], output_tokens=self.usage[1],
                                served_model="judge-model-served", cost_usd=self.cost,
                                call_id=f"call-{len(self.calls)}")


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def write_queue(work: Path, items: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / judge.QUEUE_FILE).write_text("".join(json.dumps(it) + "\n" for it in items))
    (work / judge.QUEUE_META_FILE).write_text(json.dumps(meta or {}))


def make_run(tmp_path: Path, items: list[dict[str, Any]], client: Any, *,
             config: Any = None, clock: Any = None, probe: Any = None,
             sleeps: list[float] | None = None) -> Any:
    recorded = sleeps if sleeps is not None else []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    return judge.JudgeRun(
        items, client=client,
        results_path=tmp_path / judge.RESULTS_FILE, status_path=tmp_path / judge.STATUS_FILE,
        config=config or judge.RunConfig(jitter=0.0),
        clock=clock or Clock(NOW), sleep=fake_sleep, probe=probe,
    )


def results(tmp_path: Path) -> list[dict[str, Any]]:
    return judge.read_jsonl(tmp_path / judge.RESULTS_FILE)


def notify_script(tmp_path: Path) -> Path:
    path = tmp_path / "notify.sh"
    path.write_text("#!/bin/sh\ncat >> \"$(dirname \"$0\")/notified.txt\"\n")
    path.chmod(0o755)
    return path


# ── the CLI's guard rails ────────────────────────────────────────────────────


def test_cli_needs_a_provider_and_cannot_raise_the_cap_or_change_the_phase(tmp_path: Path) -> None:
    write_queue(tmp_path, [make_item(1)])
    built: list[bool] = []

    def factory() -> FakeClient:
        built.append(True)
        return FakeClient()

    base = ["--workdir", str(tmp_path), "run", "--no-live-probe"]
    # (--provider token-plan exists since the operator's 2026-10-31 override: tests/test_distill_token_plan.py)
    for extra in ([], ["--provider", "max-token-plan"], ["--provider", "max-payg", "--cap-usd", "500"],
                  ["--provider", "max-payg", "--ledger", str(tmp_path / "other.csv")],
                  ["--provider", "max-payg", "--alert-usd", "500"], ["--provider", "local", "--phase", "x"],
                  ["--provider", "local", "--ignore-window"]):
        with pytest.raises(SystemExit):
            judge.main(base + extra, client_factory=factory)
    # The paid provider needs this phase's budget, and nothing is built without it.
    assert judge.main(base + ["--provider", "max-payg", "--notify-cmd", str(notify_script(tmp_path))],
                      client_factory=factory) == judge.EXIT_FATAL
    assert built == []
    assert judge.main(base + ["--provider", "local"], client_factory=factory) == judge.EXIT_DONE
    assert built == [True]


def test_max_payg_refuses_while_the_stop_file_exists(tmp_path: Path) -> None:
    write_queue(tmp_path, [make_item(1)])
    ledger = spend.Ledger()   # the one ledger: ~/distill (HOME is tmp_path)
    ledger.stop_path.parent.mkdir(parents=True)
    ledger.stop_path.write_text("2026-10-31T00:00:00Z distill spend STOP: test\n")
    built: list[bool] = []
    code = judge.main(["--workdir", str(tmp_path), "run", "--provider", "max-payg", "--budget-usd", "5",
                       "--notify-cmd", str(notify_script(tmp_path))],
                      client_factory=lambda: built.append(True))
    assert code == judge.EXIT_BUDGET
    assert built == []
    status = json.loads((tmp_path / judge.STATUS_FILE).read_text())
    assert status["state"] == "budget_stop" and "STOP" in status["reason"]
    assert not (tmp_path / judge.RESULTS_FILE).exists()


def test_max_payg_needs_a_working_notify_command_and_low_concurrency(tmp_path: Path) -> None:
    write_queue(tmp_path, [make_item(1)])
    base = ["--workdir", str(tmp_path), "run", "--provider", "max-payg", "--budget-usd", "5"]
    built: list[bool] = []
    assert judge.main(base + ["--notify-cmd", str(tmp_path / "missing.sh")],
                      client_factory=lambda: built.append(True)) == judge.EXIT_FATAL
    assert judge.main(base + ["--notify-cmd", str(notify_script(tmp_path)), "--concurrency", "8"],
                      client_factory=lambda: built.append(True)) == judge.EXIT_FATAL
    assert built == []


def test_a_second_run_is_refused_while_one_holds_the_workdir(tmp_path: Path) -> None:
    work = tmp_path / "phase2"
    other = tmp_path / "phase2-rejudge"
    write_queue(work, [make_item(1)])
    write_queue(other, [make_item(2)])
    built: list[bool] = []

    def factory() -> FakeClient:
        built.append(True)
        return FakeClient()

    def run(workdir: Path) -> int:
        return judge.main(["--workdir", str(workdir), "run", "--provider", "local", "--no-live-probe"],
                          client_factory=factory)

    # A run in progress on the same workdir: the second run does nothing, not
    # even a status write. Another workdir runs (the ledger is shared under its own lock).
    with judge.acquire_locks([work / judge.LOCK_FILE]):
        assert run(work) == judge.EXIT_BUSY
        assert run(other) == judge.EXIT_DONE
    assert not (work / judge.STATUS_FILE).exists()
    assert len(built) == 1
    assert run(work) == judge.EXIT_DONE   # released: it runs
    assert len(built) == 2


def test_a_rejudge_queue_spends_under_its_own_phase(tmp_path: Path) -> None:
    write_queue(tmp_path, [make_item(1)], {"mode": "rejudge"})
    client = FakeClient()
    assert judge.main(["--workdir", str(tmp_path), "run", "--provider", "local", "--no-live-probe"],
                      client_factory=lambda: client) == 0
    assert {r["phase"] for r in results(tmp_path)} == {judge.PHASE_ALL}


# ── budget stops, backoff and pause ─────────────────────────────────────────


def test_a_budget_refusal_stops_the_run_before_the_next_call(tmp_path: Path) -> None:
    client = FakeClient([PASS_REPLY, judge.JudgeBudgetStop("the cap")])
    outcome = asyncio.run(make_run(tmp_path, [make_item(i) for i in range(5)], client).run())
    assert outcome.state == "budget_stop"
    assert outcome.exit_code == judge.EXIT_BUDGET
    assert len(client.calls) == 2 and outcome.calls == 1
    assert outcome.remaining == 4
    assert [r["status"] for r in results(tmp_path)] == ["ok"]
    status = json.loads((tmp_path / judge.STATUS_FILE).read_text())
    assert status["state"] == "budget_stop" and status["budget"] == "fake budget"


def test_a_ledger_fault_is_fatal(tmp_path: Path) -> None:
    client = FakeClient([judge.JudgeBudgetStop("ledger unreadable", state="fatal")])
    outcome = asyncio.run(make_run(tmp_path, [make_item(1), make_item(2)], client).run())
    assert outcome.state == "fatal" and outcome.exit_code == judge.EXIT_FATAL
    assert len(client.calls) == 1


def test_429_and_5xx_back_off_exponentially_then_succeed(tmp_path: Path) -> None:
    client = FakeClient([
        judge.JudgeCallError("rate_limit", status=429),
        judge.JudgeCallError("provider_error", status=503),
        PASS_REPLY,
    ])
    sleeps: list[float] = []
    outcome = asyncio.run(make_run(tmp_path, [make_item(1)], client, sleeps=sleeps,
                                   config=judge.RunConfig(jitter=0.0, backoff_base=20.0)).run())
    assert sleeps == [20.0, 40.0]
    assert outcome.state == "done"
    assert outcome.calls == 3
    assert [(r["status"], r["attempts"]) for r in results(tmp_path)] == [("ok", 3)]


def test_backoff_is_capped_and_honours_retry_after(tmp_path: Path) -> None:
    run = make_run(tmp_path, [], FakeClient(),
                   config=judge.RunConfig(jitter=0.0, backoff_base=20.0, backoff_max=600.0))
    assert [run.backoff_delay(a, None) for a in range(7)] == [20, 40, 80, 160, 320, 600, 600]
    assert run.backoff_delay(0, 120.0) == 120.0
    assert run.backoff_delay(0, 5000.0) == judge.RETRY_AFTER_MAX_S


def test_rate_limit_on_every_attempt_pauses_the_run(tmp_path: Path) -> None:
    client = FakeClient([judge.JudgeCallError("rate_limit", status=429)] * 10)
    outcome = asyncio.run(make_run(tmp_path, [make_item(1), make_item(2)], client,
                                   config=judge.RunConfig(jitter=0.0, max_attempts=3)).run())
    assert outcome.state == "paused"
    assert outcome.exit_code == judge.EXIT_PAUSED
    assert len(client.calls) == 3  # item 2 is never tried
    assert results(tmp_path) == []
    assert json.loads((tmp_path / judge.STATUS_FILE).read_text())["state"] == "paused"


def test_live_throttle_signal_pauses_before_any_call(tmp_path: Path) -> None:
    client = FakeClient()
    outcome = asyncio.run(make_run(tmp_path, [make_item(1)], client,
                                   probe=lambda: "live errors").run())
    assert client.calls == []
    assert outcome.state == "paused"


def test_a_429_rechecks_the_live_probe_before_retrying(tmp_path: Path) -> None:
    signals = iter([None, "live errors"])
    client = FakeClient([judge.JudgeCallError("rate_limit", status=429)])
    outcome = asyncio.run(make_run(tmp_path, [make_item(1)], client,
                                   probe=lambda: next(signals)).run())
    assert len(client.calls) == 1
    assert outcome.state == "paused"


def test_billing_and_auth_errors_are_fatal(tmp_path: Path) -> None:
    for kind, status in (("billing", 402), ("auth", 401)):
        work = tmp_path / kind
        client = FakeClient([judge.JudgeCallError(kind, status=status)])
        outcome = asyncio.run(make_run(work, [make_item(1), make_item(2)], client).run())
        assert outcome.state == "fatal"
        assert outcome.exit_code == judge.EXIT_FATAL
        assert len(client.calls) == 1


def test_non_retryable_error_is_recorded_and_the_run_moves_on(tmp_path: Path) -> None:
    client = FakeClient([judge.JudgeCallError("bad_request", status=400)])
    outcome = asyncio.run(make_run(tmp_path, [make_item(1), make_item(2)], client).run())
    assert [r["status"] for r in results(tmp_path)] == ["error", "ok"]
    assert outcome.state == "incomplete"
    assert outcome.exit_code == judge.EXIT_INCOMPLETE


def test_live_error_probe_counts_recent_errors_for_the_provider(tmp_path: Path) -> None:
    db = tmp_path / "live.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE responses (ts REAL, provider TEXT, response_kind TEXT)")
    conn.executemany("INSERT INTO responses VALUES (?, ?, ?)", [
        (10_000.0, "fake", "tool_call"),   # recent, not an error
        (9_000.0, "fake", "error"),        # an error, but 1000 s old
        (10_000.0, "other", "error"),      # another provider
    ])
    conn.commit()
    probe = judge.live_error_probe(db, "fake", window_s=900, clock=lambda: 10_050.0)
    assert probe() is None
    conn.execute("INSERT INTO responses VALUES (10000.0, 'fake', 'error')")
    conn.commit()
    conn.close()
    assert "1 fake provider error" in probe()
    assert judge.live_error_probe(tmp_path / "missing.db", "fake")() is None


# ── verdict rows ────────────────────────────────────────────────────────────


def test_a_verdict_row_records_provider_model_cost_and_a_verification_record(tmp_path: Path) -> None:
    item = make_item(1)
    item["source"] = {"file": "golden_traces_1.jsonl", "line": 0, "telemetry_rowid": 1}
    asyncio.run(make_run(tmp_path, [item, make_item(2, "fail")], FakeClient([PASS_REPLY, PASS_REPLY])).run())
    row, fail_row = results(tmp_path)
    assert row["status"] == "ok" and row["verdict"] == "pass" and row["agrees"] is True
    assert (row["provider"], row["judge_key"], row["model"]) == ("fake", "none", "judge-model")
    assert row["served_model"] == "judge-model-served"
    assert row["cost_usd"] == pytest.approx(0.01) and row["call_id"] == "call-1"
    assert row["source"] == item["source"]
    assert row["verification"] == {"verified": True, "judge": {
        "verdict": "pass", "model": "judge-model", "served_model": "judge-model-served", "provider": "fake"}}
    assert fail_row["agrees"] is False


def test_projection_counts_worst_and_typical_cost() -> None:
    items = [make_item(i) for i in range(10, 20)]
    cfg = judge.RunConfig(max_tokens=1000)
    price = spend.price_for("qwen3.8-max", "ap-southeast-1")
    out = judge.project_cost(items, cfg, price=price)
    assert out["items"] == 10 and out["output_basis"] == "assumed"
    assert out["output_tokens_bound"] == 10_000 and out["output_tokens_typical"] == 3000
    assert out["input_tokens_bound"] > out["input_tokens_typical"] > 0
    assert out["usd_worst_case"] == pytest.approx(
        (out["input_tokens_bound"] * 2.0 + 10_000 * 6.0) / 1e6, abs=1e-4)
    assert 0 < out["usd_typical"] < out["usd_worst_case"]
    prior = [{"status": "ok", "output_tokens": 50}] * 10
    assert judge.project_cost(items, cfg, price=price, prior_rows=prior)["output_per_call_typical"] == 50
    assert judge.project_cost(items, cfg)["usd_worst_case"] == 0.0   # a local judge


# ── resume ─────────────────────────────────────────────────────────────────


def test_resume_skips_judged_ids(tmp_path: Path) -> None:
    items = [make_item(i) for i in range(5)]
    first = FakeClient()
    out1 = asyncio.run(make_run(tmp_path, items, first,
                                config=judge.RunConfig(jitter=0.0, limit=2)).run())
    assert len(first.calls) == 2
    assert out1.state == "incomplete"
    assert out1.remaining == 3

    second = FakeClient()
    out2 = asyncio.run(make_run(tmp_path, items, second).run())
    assert len(second.calls) == 3
    assert out2.state == "done"
    ids = [r["id"] for r in results(tmp_path)]
    assert sorted(ids) == sorted(it["id"] for it in items)
    assert len(ids) == len(set(ids))


def test_unparseable_reply_is_retried_on_the_next_run_only(tmp_path: Path) -> None:
    items = [make_item(1)]
    first = FakeClient(["no json here"])
    out1 = asyncio.run(make_run(tmp_path, items, first).run())
    assert len(first.calls) == 1
    assert out1.unparseable == 1
    assert results(tmp_path)[0]["status"] == "unparseable"

    second = FakeClient()
    asyncio.run(make_run(tmp_path, items, second).run())
    assert len(second.calls) == 1
    assert [r["status"] for r in results(tmp_path)] == ["unparseable", "ok"]


def test_a_different_view_is_a_different_judgement(tmp_path: Path) -> None:
    items = [make_item(1)]
    asyncio.run(make_run(tmp_path, items, FakeClient()).run())
    blind = FakeClient()
    asyncio.run(make_run(tmp_path, items, blind, config=judge.RunConfig(jitter=0.0, view="blind")).run())
    assert len(blind.calls) == 1
    system, user, _ = blind.calls[0]
    assert "What the tool returned" not in user
    assert "what the tool returned" not in system


# ── prompt and parser ───────────────────────────────────────────────────────


def test_prompt_carries_context_call_and_outcome() -> None:
    item = make_item(7, outcome_text="file-a\nfile-b")
    text, truncated = judge.render_prompt(item, view="outcome")
    assert "list the files in the demo folder 7" in text
    assert "Tool called: bash" in text and "ls demo 7" in text
    assert "the tool reported success" in text and "file-b" in text
    assert text.startswith("BEGIN SESSION DATA") and text.rstrip().endswith("END SESSION DATA")
    assert truncated is False


def test_prompt_says_what_the_export_leaves_out() -> None:
    for view in judge.VIEWS:
        system = judge.system_prompt(view)
        assert "every earlier tool call and tool result" in system
        assert "not fabrication" in system
        assert "{" not in system.split("Reply with ONLY")[0]   # every placeholder filled
    assert "(no result text was recorded)" in judge.system_prompt("outcome")
    assert "no result text" not in judge.system_prompt("blind")
    item = make_item(1)
    item["tools"] = []
    text, _ = judge.render_prompt(item, view="blind")
    assert "the other tools offered are not shown" in text
    assert "(no schema was recorded for this call)" in text


def test_prompt_drops_oldest_messages_to_fit() -> None:
    item = make_item(1)
    item["messages"] = [{"role": "user", "content": f"old message {i} " + "x" * 900}
                        for i in range(30)] + item["messages"]
    text, truncated = judge.render_prompt(item, view="outcome", max_prompt_chars=5000)
    assert truncated is True
    assert len(text) <= 5000
    assert "old message 0 " not in text
    assert "list the files in the demo folder 1" in text  # the newest context survives


def test_parse_verdict_is_strict() -> None:
    ok = judge.parse_verdict("```json\n" + PASS_REPLY + "\n```")
    assert ok and ok["verdict"] == "pass" and ok["rubric"]["right_tool"] == 2
    assert judge.parse_verdict("<think>pondering {\"verdict\": 1}</think>" + FAIL_REPLY)["verdict"] == "fail"
    # A zero anywhere fails the call even when the judge wrote "pass".
    zero = json.loads(PASS_REPLY)
    zero["rubric"]["no_fabrication"] = 0
    parsed = judge.parse_verdict(json.dumps(zero))
    assert parsed["verdict"] == "fail" and parsed["verdict_model"] == "pass"
    for bad in ("", "pass", '{"verdict": "maybe"}',
                json.dumps({**json.loads(PASS_REPLY), "rubric": {"right_tool": 2}}),
                json.dumps({**json.loads(PASS_REPLY), "rubric": {k: 3 for k in judge.RUBRIC_KEYS}}),
                json.dumps({**json.loads(PASS_REPLY), "rubric": {k: True for k in judge.RUBRIC_KEYS}}),
                json.dumps({**json.loads(PASS_REPLY), "reasons": [1, 2]})):
        assert judge.parse_verdict(bad) is None, bad


# ── sampling ───────────────────────────────────────────────────────────────


def test_allocate_floor_then_proportional_largest_remainder() -> None:
    sizes = {"coding": 8958, "sentinel/daemon": 695, "other": 429, "memory/wiki": 113}
    assert judge.allocate(sizes, 300, 10) == {
        "coding": 238, "memory/wiki": 13, "other": 21, "sentinel/daemon": 28}
    assert judge.allocate(sizes, 300, 0) == {
        "coding": 264, "memory/wiki": 3, "other": 13, "sentinel/daemon": 20}
    assert judge.allocate({"a": 5, "b": 1000}, 100, 10) == {"a": 5, "b": 95}   # capped at the pool
    assert judge.allocate({"a": 3, "b": 4}, 50, 10) == {"a": 3, "b": 4}         # all of a small pool
    assert judge.allocate({"a": 3, "b": 3, "c": 3}, 4, 2) == {"a": 2, "b": 1, "c": 1}  # floors overflow
    assert judge.allocate({"a": 0, "b": 9}, 3, 1) == {"b": 3}


def _pool(cats: dict[str, int]) -> list[dict[str, Any]]:
    out = []
    for cat, count in cats.items():
        out += [{"id": f"{cat}-{i:05d}", "category": cat} for i in range(count)]
    return out


def test_stratified_sample_is_deterministic_and_matches_the_allocation() -> None:
    cats = {"coding": 900, "other": 60, "memory/wiki": 12}
    a, sizes, alloc = judge.stratified_sample(_pool(cats), 100, seed=7, label="pass", floor=10)
    b, _, _ = judge.stratified_sample(list(reversed(_pool(cats))), 100, seed=7, label="pass", floor=10)
    c, _, _ = judge.stratified_sample(_pool(cats), 100, seed=8, label="pass", floor=10)
    assert sorted(x["id"] for x in a) == sorted(x["id"] for x in b)   # input order does not matter
    assert sorted(x["id"] for x in a) != sorted(x["id"] for x in c)   # the seed does
    assert sizes == {"coding": 900, "memory/wiki": 12, "other": 60}
    got = {k: sum(1 for x in a if x["category"] == k) for k in cats}
    assert got == alloc and sum(got.values()) == 100 and min(got.values()) >= 10
    for x in a:
        assert x["stratum_weight"] == pytest.approx(sizes[x["category"]] / alloc[x["category"]])


def test_pass_pool_dedupes_teacher_rows_in_index_order() -> None:
    index = [
        {"file": "f1", "model": "teacher", "h_prompt_call": "k1", "category": "coding", "tool": "bash", "ts": 1},
        {"file": "f1", "model": "other", "h_prompt_call": "k2", "category": "coding", "tool": "bash", "ts": 2},
        {"file": "f1", "model": "teacher", "h_prompt_call": "k1", "category": "coding", "tool": "bash", "ts": 3},
        {"file": "f2", "model": "teacher", "h_prompt_call": "k3", "category": "other", "tool": "bash", "ts": 4},
    ]
    pool = judge.pass_pool(index, "teacher")
    assert [(p["source"]["file"], p["source"]["line"]) for p in pool] == [("f1", 0), ("f2", 0)]
    assert [p["category"] for p in pool] == ["coding", "other"]


def _phase1_stub() -> Any:
    def classify(trace: dict[str, Any], prompt: str, args: dict[str, Any]) -> tuple[str, str]:
        return ("memory/wiki", "stub") if "wiki" in prompt else ("coding", "stub")

    return types.SimpleNamespace(classify=classify, norm=lambda s: (s or "").lower().strip(),
                                 h=lambda s: hashlib.sha1(s.encode()).hexdigest()[:16])


def test_a_fail_trace_without_a_recorded_schema_is_not_chat_mode() -> None:
    def classify(trace: dict[str, Any], prompt: str, args: dict[str, Any]) -> tuple[str, str]:
        if not trace.get("tools"):
            return "chat-mode", "no tools"
        return "coding", "tool"

    phase1 = types.SimpleNamespace(classify=classify, norm=str.lower, h=lambda s: s)
    trace = make_trace(1)
    del trace["tools"]
    assert judge.phase1_keys(trace, phase1)[0] == "coding"
    assert "tools" not in trace  # the trace itself is left as it is


def test_build_queue_end_to_end(tmp_path: Path) -> None:
    traj = tmp_path / "traj"
    traj.mkdir()
    index, lines = [], []
    for i in range(40):
        trace = make_trace(i, prompt="fix the build" if i % 4 else "update the wiki page")
        trace["_meta"]["model"] = "teacher"
        lines.append(json.dumps(trace))
        index.append({"file": "golden_a.jsonl", "model": "teacher", "tool": "bash",
                      "ts": trace["_meta"]["timestamp"], "h_prompt_call": f"k{i}",
                      "category": "coding" if i % 4 else "memory/wiki", "rowid": i + 1})
    (traj / "golden_a.jsonl").write_text("\n".join(lines) + "\n")
    tele = tmp_path / "telemetry.db"
    conn = sqlite3.connect(tele)
    conn.execute("CREATE TABLE tool_calls (result_summary TEXT)")
    conn.executemany("INSERT INTO tool_calls (rowid, result_summary) VALUES (?, ?)",
                     [(i + 1, f"output {i}") for i in range(40)])
    conn.commit()
    conn.close()
    fails = []
    for i in range(12):
        trace = make_trace(100 + i, prompt="run the tests", command="pytest")
        trace["_meta"]["failure"] = {"error_type": "ToolError", "error_detail": "exit 1",
                                     "result_summary": None}
        fails.append(trace)
    fails.append(json.loads(json.dumps(fails[0])))  # an exact duplicate

    queue, meta = judge.build_queue(index, fails, _phase1_stub(), trajectories=traj, telemetry=tele,
                                    n_pass=20, n_fail=5, seed=3, floor=4, model="teacher")
    assert meta["n_pass"] == 20 and meta["n_fail"] == 5
    assert meta["fail_duplicates_dropped"] == 1
    assert meta["pass_pool"] == {"coding": 30, "memory/wiki": 10}
    passes = [q for q in queue if q["label"] == "pass"]
    assert all(q["outcome"]["text"].startswith("output ") for q in passes)
    assert all(q["messages"][-1]["tool_calls"] for q in queue)
    failq = [q for q in queue if q["label"] == "fail"]
    assert all(q["outcome"] == {"status": "error", "text": "ToolError exit 1"} for q in failq)
    assert all("failure" not in q["meta"] for q in failq)
    again, _ = judge.build_queue(index, fails, _phase1_stub(), trajectories=traj, telemetry=tele,
                                 n_pass=20, n_fail=5, seed=3, floor=4, model="teacher")
    assert [q["id"] for q in again] == [q["id"] for q in queue]

    index[0]["ts"] = 1.5  # the corpus no longer matches the audit's index
    with pytest.raises(judge.JudgeSetupError, match="does not match"):
        judge.build_queue(index, fails, _phase1_stub(), trajectories=traj, telemetry=tele,
                          n_pass=40, n_fail=5, seed=3, floor=4, model="teacher")


def test_sample_refuses_to_replace_a_queue(tmp_path: Path) -> None:
    write_queue(tmp_path, [make_item(1)])
    assert judge.main(["--workdir", str(tmp_path), "sample"]) == judge.EXIT_ERROR


def _corpus(tmp_path: Path, cats: dict[str, int]) -> tuple[list[dict[str, Any]], Path, Path]:
    """A synthetic teacher corpus: (index rows, trajectories dir, telemetry snapshot)."""
    traj = tmp_path / "traj"
    traj.mkdir()
    index, lines, i = [], [], 0
    for cat, count in cats.items():
        for _ in range(count):
            trace = make_trace(i)
            lines.append(json.dumps(trace))
            index.append({"file": "golden_a.jsonl", "model": "teacher", "tool": "bash",
                          "ts": trace["_meta"]["timestamp"], "h_prompt_call": f"k{i}",
                          "category": cat, "rowid": i + 1})
            i += 1
    (traj / "golden_a.jsonl").write_text("\n".join(lines) + "\n")
    tele = tmp_path / "telemetry.db"
    conn = sqlite3.connect(tele)
    conn.execute("CREATE TABLE tool_calls (result_summary TEXT)")
    conn.executemany("INSERT INTO tool_calls (rowid, result_summary) VALUES (?, ?)",
                     [(n + 1, f"output {n}") for n in range(i)])
    conn.commit()
    conn.close()
    return index, traj, tele


def test_rejudge_queue_is_the_unjudged_pass_pool_worst_category_first(tmp_path: Path) -> None:
    index, traj, tele = _corpus(tmp_path, {"coding": 30, "other": 6, "memory/wiki": 4})
    pool = judge.pass_pool(index, "teacher")
    coding_ids = [p["id"] for p in pool if p["category"] == "coding"]
    other_ids = [p["id"] for p in pool if p["category"] == "other"]
    judged = set(coding_ids[:10] + other_ids[:2])   # already judged in the Phase 2 sample
    priority = ["other", "coding"]   # memory/wiki was not ranked: it goes last
    queue, meta = judge.build_rejudge_queue(index, judged, priority, trajectories=traj,
                                            telemetry=tele, seed=5, model="teacher")
    assert [q["category"] for q in queue] == ["other"] * 4 + ["coding"] * 20 + ["memory/wiki"] * 4
    assert not judged & {q["id"] for q in queue}
    assert meta["n_pass"] == 28 and meta["already_judged"] == 12 and meta["n_fail"] == 0
    assert meta["priority"] == ["other", "coding", "memory/wiki"]
    assert all(q["label"] == "pass" and q["stratum_weight"] == 1.0 for q in queue)
    assert all(q["outcome"]["text"].startswith("output ") for q in queue)
    again, _ = judge.build_rejudge_queue(index, judged, priority, trajectories=traj,
                                         telemetry=tele, seed=5, model="teacher")
    assert [q["id"] for q in again] == [q["id"] for q in queue]
    coding = [q["id"] for q in queue if q["category"] == "coding"]
    assert coding != sorted(coding)   # shuffled within the category


def test_rejudge_sample_needs_a_re_judge_gate_and_its_own_workdir(tmp_path: Path) -> None:
    index, traj, tele = _corpus(tmp_path, {"coding": 6, "other": 3})
    (tmp_path / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))
    phase2 = tmp_path / "phase2"
    phase2.mkdir()
    pool = judge.pass_pool(index, "teacher")
    (phase2 / judge.RESULTS_FILE).write_text(
        "".join(json.dumps(r) + "\n" for r in _judged([], {pool[0]["id"]: "fail"})))

    def sample(workdir: Path, *extra: str) -> int:
        return judge.main(["--workdir", str(workdir), "sample", "--rejudge-from", str(phase2),
                           "--index", str(tmp_path / "index.jsonl"), "--trajectories", str(traj),
                           "--telemetry", str(tele), "--model", "teacher", *extra])

    report = {"view": "outcome", "rubric_version": judge.RUBRIC_VERSION,
              "gate": {"decision": "trust-labels", "rejudge_priority": []}}
    (phase2 / judge.REPORT_FILE).write_text(json.dumps(report))
    assert sample(tmp_path / "rejudge") == judge.EXIT_ERROR             # nothing to re-judge
    report["gate"] = {"decision": "re-judge", "rejudge_priority": ["other", "coding"]}
    (phase2 / judge.REPORT_FILE).write_text(json.dumps(report))
    assert sample(phase2, "--force") == judge.EXIT_ERROR                # would replace the sample
    assert not (phase2 / judge.QUEUE_FILE).exists()
    assert sample(tmp_path / "rejudge") == judge.EXIT_DONE
    queue = judge.read_jsonl(tmp_path / "rejudge" / judge.QUEUE_FILE)
    assert [q["category"] for q in queue] == ["other"] * 3 + ["coding"] * 5
    assert pool[0]["id"] not in {q["id"] for q in queue}
    meta = json.loads((tmp_path / "rejudge" / judge.QUEUE_META_FILE).read_text())
    assert meta["mode"] == "rejudge" and meta["view"] == "outcome"


# ── report and gate ─────────────────────────────────────────────────────────


def _judged(items: list[dict[str, Any]], verdicts: dict[str, str], view: str = "outcome") -> list[dict[str, Any]]:
    return [{"id": i, "status": "ok", "view": view, "rubric_version": judge.RUBRIC_VERSION,
             "verdict": v, "model": "m", "served_model": "m",
             "rubric": dict.fromkeys(judge.RUBRIC_KEYS, 2 if v == "pass" else 0)}
            for i, v in verdicts.items()]


def _gate_case(n_pass: int, false_passes: int, n_fail: int = 10) -> tuple[list, dict, list]:
    items = [make_item(i, "pass") for i in range(n_pass)] + [make_item(i, "fail") for i in range(n_fail)]
    verdicts = {it["id"]: ("fail" if it["label"] == "fail" or i < false_passes else "pass")
                for i, it in enumerate(items)}
    return items, {"pass_pool": {"coding": 10_000}}, _judged(items, verdicts)


def test_wilson_reference_values() -> None:
    lo, hi = judge.wilson(5, 100)
    assert (round(lo, 4), round(hi, 4)) == (0.0215, 0.1118)
    assert judge._ci(0, 100) == [0.0, 0.037]


def test_gate_trusts_labels_at_exactly_five_percent() -> None:
    items, meta, rows = _gate_case(300, 15)
    report = judge.build_report(items, meta, rows)
    assert report["false_pass"]["rate"] == 0.05
    assert report["false_pass"]["k_raw"] == 15 and report["false_pass"]["n_raw"] == 300
    assert report["gate"]["decision"] == "trust-labels"
    assert report["agreement"]["k"] == 285 + 10


def test_gate_decides_on_the_unrounded_rate() -> None:
    # The real pass pools and allocation: 12, 2, 0 and 1 judge fails weight back
    # to 0.050024, which rounds to 0.05 but is above the gate.
    pools = {"coding": 8958, "sentinel/daemon": 695, "other": 429, "memory/wiki": 113}
    sampled = {"coding": 238, "sentinel/daemon": 28, "other": 21, "memory/wiki": 13}
    fails = {"coding": 12, "sentinel/daemon": 2, "other": 0, "memory/wiki": 1}
    items, verdicts, i = [], {}, 0
    for cat, n in sampled.items():
        for j in range(n):
            item = make_item(i, "pass", cat)
            items.append(item)
            verdicts[item["id"]] = "fail" if j < fails[cat] else "pass"
            i += 1
    report = judge.build_report(items, {"pass_pool": pools}, _judged(items, verdicts))
    assert report["false_pass"]["rate"] == 0.05
    assert 0.05 < report["false_pass"]["rate_exact"] < 0.0501
    assert report["gate"]["decision"] == "re-judge"
    assert "5.002%" in report["gate"]["reason"]


def test_gate_asks_for_rejudging_above_five_percent_worst_category_first() -> None:
    items = ([make_item(i, "pass", "coding") for i in range(100)]
             + [make_item(100 + i, "pass", "other") for i in range(50)]
             + [make_item(200 + i, "pass", "memory/wiki") for i in range(50)])
    bad = {"coding": 5, "other": 20, "memory/wiki": 10}
    seen: dict[str, int] = {}
    verdicts = {}
    for it in items:
        seen[it["category"]] = seen.get(it["category"], 0) + 1
        verdicts[it["id"]] = "fail" if seen[it["category"]] <= bad[it["category"]] else "pass"
    meta = {"pass_pool": {"coding": 100, "other": 50, "memory/wiki": 50}}
    report = judge.build_report(items, meta, _judged(items, verdicts))
    assert report["false_pass"]["rate"] == pytest.approx(35 / 200, abs=1e-4)
    assert report["gate"]["decision"] == "re-judge"
    assert report["gate"]["rejudge_priority"] == ["other", "memory/wiki", "coding"]
    assert [c["category"] for c in report["by_category"]] == ["other", "memory/wiki", "coding"]
    assert "Re-judge order: other, memory/wiki, coding." in judge.render_markdown(report)


def test_rejudge_order_follows_the_pass_side_not_the_padded_fail_side() -> None:
    # coding: 3 of 10 passes fail (30%), padded by 20 fail items the outcome view
    # lets the judge agree with; other: 2 of 10 (20%), no fail items. Overall
    # agreement ranks other worse (80% vs 90%); the pass set, which is what gets
    # re-judged, is worse in coding.
    items = ([make_item(i, "pass", "coding") for i in range(10)]
             + [make_item(100 + i, "fail", "coding") for i in range(20)]
             + [make_item(200 + i, "pass", "other") for i in range(10)])
    verdicts = {it["id"]: "pass" for it in items}
    for it in items[:3] + items[30:32]:
        verdicts[it["id"]] = "fail"
    for it in items[10:30]:
        verdicts[it["id"]] = "fail"
    report = judge.build_report(items, {"pass_pool": {"coding": 100, "other": 100}},
                                _judged(items, verdicts))
    assert [c["category"] for c in report["by_category"]] == ["other", "coding"]
    assert report["gate"]["rejudge_priority"] == ["coding", "other"]


def test_fail_side_is_reported_from_the_blind_view_when_judged(tmp_path: Path) -> None:
    items = [make_item(i, "pass") for i in range(20)] + [make_item(i, "fail") for i in range(4)]
    outcome_rows = _judged(items, {it["id"]: it["label"] for it in items})   # full agreement
    report = judge.build_report(items, {"pass_pool": {"coding": 20}}, outcome_rows)
    assert report["fail_side"]["label_visible"] is True
    assert report["fail_side_blind"] == {"judge_fail_k": 0, "n": 0, "rate": None, "ci95": None}
    text = judge.render_markdown(report)
    assert "not a test of the judge" in text and "`run --view blind --only fail`" in text

    # The blind fail-side check: `run --only fail` judges just the fail items.
    write_queue(tmp_path, items, {"pass_pool": {"coding": 20}})
    (tmp_path / judge.RESULTS_FILE).write_text("".join(json.dumps(r) + "\n" for r in outcome_rows))
    client = FakeClient([FAIL_REPLY, PASS_REPLY, FAIL_REPLY, FAIL_REPLY])
    code = judge.main(["--workdir", str(tmp_path), "run", "--provider", "local", "--view", "blind",
                       "--only", "fail", "--no-live-probe"],
                      client_factory=lambda: client)
    assert code == judge.EXIT_DONE
    assert len(client.calls) == 4
    # Its own ledger phase, so its --budget-usd is not what the outcome run left of 2-judge's.
    assert {r["phase"] for r in results(tmp_path) if r["view"] == "blind"} == {judge.PHASE_BLIND}
    # Mixed views no longer need --view: the outcome view judged the pass side.
    report = judge.build_report(items, {"pass_pool": {"coding": 20}}, results(tmp_path))
    assert report["view"] == "outcome"
    assert report["fail_side_blind"]["judge_fail_k"] == 3 and report["fail_side_blind"]["n"] == 4
    assert "Fail side, blind view: 3/4 = 75.0%" in judge.render_markdown(report)


def test_false_pass_is_weighted_back_to_the_pool() -> None:
    items = ([make_item(i, "pass", "a") for i in range(10)]
             + [make_item(100 + i, "pass", "b") for i in range(10)])
    verdicts = {it["id"]: "pass" for it in items}
    for it in items[:1] + items[10:15]:   # 1 of 10 in a, 5 of 10 in b
        verdicts[it["id"]] = "fail"
    report = judge.build_report(items, {"pass_pool": {"a": 900, "b": 100}}, _judged(items, verdicts))
    fp = report["false_pass"]
    assert fp["rate"] == pytest.approx(0.9 * 0.1 + 0.1 * 0.5, abs=1e-4)
    assert fp["rate_raw"] == pytest.approx(0.3)
    assert fp["n_eff"] == pytest.approx(1e6 / (900 ** 2 / 10 + 100 ** 2 / 10), abs=0.1)
    lo, hi = fp["ci95"]
    assert lo < fp["rate"] < hi


def test_gate_is_incomplete_while_sampled_items_lack_a_verdict() -> None:
    items, meta, rows = _gate_case(20, 0)
    rows = rows[2:]
    rows.append({"id": items[0]["id"], "status": "unparseable", "view": "outcome",
                 "rubric_version": judge.RUBRIC_VERSION})
    report = judge.build_report(items, meta, rows)
    assert report["gate"]["decision"] == "incomplete"
    assert report["missing"] == 2 and report["missing_after_unparseable"] == 1
    assert judge.build_report(items, meta, rows, allow_missing=2)["gate"]["decision"] == "trust-labels"


def test_report_refuses_mixed_views_unless_one_is_named() -> None:
    items, meta, rows = _gate_case(4, 0, n_fail=0)
    rows += _judged(items[:1], {items[0]["id"]: "fail"}, view="blind")
    with pytest.raises(judge.ReportError, match="view"):
        judge.build_report(items, meta, rows)
    assert judge.build_report(items, meta, rows, view="blind")["judged"]["pass"] == 1


def test_report_cli_writes_aggregates(tmp_path: Path) -> None:
    items, meta, rows = _gate_case(10, 1, n_fail=2)
    write_queue(tmp_path, items, meta)
    (tmp_path / judge.RESULTS_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert judge.main(["--workdir", str(tmp_path), "report", "--allow-missing", "0"]) == judge.EXIT_DONE
    report = json.loads((tmp_path / judge.REPORT_FILE).read_text())
    assert report["gate"]["decision"] == "re-judge"   # 1/10 = 10%
    assert "list the files" not in (tmp_path / judge.REPORT_FILE).read_text()


# ── build_fail_set ─────────────────────────────────────────────────────────

SECRET_SHAPED = "sk-FAKEfixture0000000000"


def _make_lcm(path: Path) -> None:
    from prometheus.memory.lcm_conversation_store import LCMConversationStore
    from prometheus.memory.lcm_types import MessagePart

    store = LCMConversationStore(path)

    def add(session: str, role: str, text: str, ts: float, idx: int, provenance: str = "user") -> None:
        store.insert_message(MessagePart(role=role, content=text, timestamp=ts, session_id=session,
                                         turn_index=idx, provenance=provenance))

    add("beacon:good", "user", "please run the demo checks", 900.0, 0)
    add("beacon:good", "assistant", "running them now", 950.0, 1)
    add("beacon:good", "user", "AFTER the call: must not leak", 1100.0, 2)
    add("beacon:member", "user", "a member asks for something", 900.0, 0, provenance="member")
    add("beacon:onlyasst", "assistant", "preamble", 900.0, 0)
    add("web", "user", "shared surface text", 900.0, 0)
    store._conn.close()


def _make_telemetry(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE tool_calls (id TEXT, timestamp REAL, model TEXT, tool_name TEXT, success INTEGER,"
        " retries INTEGER, error_type TEXT, error_detail TEXT, raw_model_output TEXT,"
        " parsed_tool_call TEXT, session_id TEXT, tool_schema TEXT, turn_id TEXT, result_summary TEXT)")
    call = json.dumps({"name": "bash", "input": {"command": "make check"}})
    rows = [
        # (model, tool, success, session, parsed_call, error_detail)
        ("teacher", "bash", 0, "beacon:good", call, f"exit 2; token {SECRET_SHAPED}"),
        ("teacher", "bash", 0, None, call, "e"),
        ("teacher", "bash", 0, "web", call, "e"),
        ("teacher", "bash", 0, "beacon:good", None, "e"),
        ("teacher", "bash", 0, "beacon:empty", call, "e"),
        ("teacher", "bash", 0, "beacon:member", call, "e"),
        ("teacher", "bash", 0, "beacon:onlyasst", call, "e"),
        ("teacher", "bash", 0, "beacon:good", "{not json", "e"),
        ("teacher", "bash", 1, "beacon:good", call, None),              # a success: not read
        ("other-model", "bash", 0, "beacon:good", call, "e"),         # another model: not read
        ("teacher", "_loop_transition", 0, "beacon:good", None, "e"),  # bookkeeping: not read
    ]
    for i, (model, tool, ok, session, parsed, detail) in enumerate(rows):
        conn.execute(
            "INSERT INTO tool_calls VALUES (?, 1000.0, ?, ?, ?, 0, 'ToolError', ?, 'checking',"
            " ?, ?, ?, 't1', 'partial output')",
            (f"id{i}", model, tool, ok, detail, parsed, session, BASH_SCHEMA))
    conn.commit()
    conn.close()


def test_build_fail_set_rebuilds_through_the_exporter_and_counts_skips(tmp_path: Path) -> None:
    live = tmp_path / "live"
    live.mkdir()
    snap = tmp_path / "snap"
    snap.mkdir()
    _make_lcm(snap / "lcm.db")
    _make_telemetry(snap / "telemetry.db")
    out = tmp_path / "out" / "fail_traces.jsonl"

    summary = failset.build_fail_set(snap / "telemetry.db", snap / "lcm.db", out,
                                     model="teacher", live_dir=live)
    assert summary["failed_rows"] == 8
    assert summary["rebuilt"] == 1
    assert summary["skipped_by_reason"] == {
        "no_session_id": 1, "shared_surface_session": 1, "no_parsed_call": 1,
        "no_conversation_rows_before_call": 1, "group_member_conversation": 1,
        "no_usable_context": 1, "unparseable_call": 1,
    }
    [example] = judge.read_jsonl(out)
    assert [m["role"] for m in example["messages"]] == ["user", "assistant"]
    assert example["messages"][0]["content"] == "please run the demo checks"  # preamble trimmed
    assert "must not leak" not in json.dumps(example)
    target = example["messages"][-1]
    assert target["tool_calls"][0]["function"]["name"] == "bash"
    assert json.loads(target["tool_calls"][0]["function"]["arguments"]) == {"command": "make check"}
    assert example["tools"][0]["function"]["name"] == "bash"
    meta = example["_meta"]
    assert meta["label"] == "fail" and meta["model"] == "teacher"
    assert meta["failure"]["error_type"] == "ToolError"
    assert SECRET_SHAPED not in meta["failure"]["error_detail"]   # redacted like an export
    saved = json.loads(out.with_name("fail_traces.summary.json").read_text())
    assert saved["rebuilt"] == 1
    assert (out.stat().st_mode & 0o777) == 0o600


def test_build_fail_set_refuses_live_stores(tmp_path: Path) -> None:
    live = tmp_path / "live"
    live.mkdir()
    _make_telemetry(live / "telemetry.db")
    _make_lcm(live / "lcm.db")
    with pytest.raises(failset.FailSetError, match="live store"):
        failset.build_fail_set(live / "telemetry.db", live / "lcm.db", tmp_path / "o.jsonl",
                               model="teacher", live_dir=live)


def test_snapshot_destination_in_the_live_directory_is_refused_before_any_write(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live = tmp_path / "live"
    (live / "data").mkdir(parents=True)
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(live))
    live_db = live / "data" / "lcm.db"
    _make_lcm(live_db)
    daemon = sqlite3.connect(live_db)   # held open, as the daemon holds it
    daemon.execute("PRAGMA wal_autocheckpoint = 0")
    daemon.execute("INSERT INTO lcm_messages (id, session_id, turn_index, role, content, timestamp)"
                   " VALUES ('w1', 'beacon:wal', 0, 'user', 'only in the wal', 1.0)")
    daemon.commit()
    inode, main_before = live_db.stat().st_ino, live_db.read_bytes()
    snap = tmp_path / "snap"
    snap.mkdir()
    _make_telemetry(snap / "telemetry.db")
    listing = sorted(p.name for p in (live / "data").iterdir())

    # --lcm read as "which LCM": the live file itself, then another name beside it.
    for dst in (live_db, live / "data" / "copy.db"):
        code = failset.main(["--make-lcm-snapshot", "--lcm", str(dst), "--telemetry",
                             str(snap / "telemetry.db"), "--out", str(tmp_path / "o.jsonl"),
                             "--model", "teacher"])
        assert code == 2
    assert live_db.stat().st_ino == inode and live_db.read_bytes() == main_before
    assert sorted(p.name for p in (live / "data").iterdir()) == listing   # no .tmp, no copy
    daemon.close()

    # Outside the live directory, the source itself is still refused.
    other = tmp_path / "elsewhere.db"
    _make_lcm(other)
    with pytest.raises(failset.FailSetError, match="being copied"):
        failset.snapshot_sqlite(other, other, live_dir=live)
    with pytest.raises(failset.FailSetError, match="live config directory"):
        failset.snapshot_sqlite(other, live / "x.db", live_dir=live)
    assert not (live / "x.db").exists() and not (live / "x.db.tmp").exists()


def test_snapshot_never_writes_the_source_and_keeps_its_wal(tmp_path: Path) -> None:
    live = tmp_path / "live"
    live.mkdir()
    live_db = live / "lcm.db"
    _make_lcm(live_db)
    # A writer that stays open, as the daemon does, with a row only in the WAL.
    daemon = sqlite3.connect(live_db)
    daemon.execute("PRAGMA wal_autocheckpoint = 0")
    daemon.execute("INSERT INTO lcm_messages (id, session_id, turn_index, role, content, timestamp)"
                   " VALUES ('w1', 'beacon:wal', 0, 'user', 'only in the wal', 1.0)")
    daemon.commit()
    main_before = live_db.read_bytes()
    wal_before = (live / "lcm.db-wal").read_bytes()
    assert wal_before

    dst = tmp_path / "snap" / "lcm.db"
    assert failset.snapshot_sqlite(live_db, dst) > 0
    assert live_db.read_bytes() == main_before          # no write, no checkpoint
    assert (live / "lcm.db-wal").read_bytes() == wal_before
    daemon.close()

    conn = failset.open_ro(dst)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert conn.execute("SELECT COUNT(*) FROM lcm_messages").fetchone()[0] == 7
    assert conn.execute("SELECT content FROM lcm_messages WHERE id = 'w1'").fetchone()[0] == "only in the wal"
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM lcm_messages")
    conn.close()
    assert not dst.with_name("lcm.db-wal").exists()     # a later read-only open needs no side files


def test_build_fail_set_cli_snapshots_then_builds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    live = tmp_path / "live"
    live.mkdir()
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(live))
    _make_lcm(live / "lcm.db")
    snap = tmp_path / "snap"
    snap.mkdir()
    _make_telemetry(snap / "telemetry.db")
    code = failset.main(["--make-lcm-snapshot", "--live-lcm", str(live / "lcm.db"),
                         "--lcm", str(snap / "lcm.db"), "--telemetry", str(snap / "telemetry.db"),
                         "--out", str(tmp_path / "fail.jsonl"), "--model", "teacher"])
    assert code == 0
    assert json.loads(capsys.readouterr().out.split("\n", 1)[1])["rebuilt"] == 1
    # Pointing it at the live directory is refused.
    assert failset.main(["--lcm", str(live / "lcm.db"), "--telemetry", str(snap / "telemetry.db"),
                         "--out", str(tmp_path / "x.jsonl"), "--model", "teacher"]) == 2


# ── the local 27B judge served over the tailnet (--judge-url) ──────────────────

REMOTE_KEY = "a" * 16 + "0123456789abcdef" * 3   # a test value, never a real key
REMOTE_ALIAS = "qwen3.8-27b-ud-q4_k_xl"


class FakeJudgeServer:
    """llama-server's shape as serve_teacher.sh runs it: /health and /v1/models public, everything else
    needs ``Bearer <key>``; /tools refused (403) unless ``tools_status`` says otherwise; ``loading``
    answers 503; ``stop``/``start`` take the port away and bring it back. A chat reply's verdict is
    ``fail`` when the prompt holds ``JUDGE-FAIL``, else ``pass``."""

    def __init__(self, *, key: str = REMOTE_KEY, slots: int = 2, model: str = REMOTE_ALIAS,
                 model_path: str = "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf", delay: float = 0.05) -> None:
        self.key, self.slots, self.model, self.model_path, self.delay = key, slots, model, model_path, delay
        self.open = False
        self.loading = False
        self.tools_status = 403
        self.cors_proxy = False
        self.lock = threading.Lock()
        self.auth_seen: list[str | None] = []
        self.inflight = 0
        self.max_inflight = 0
        self.chats = 0
        self.on_chat: Any = None
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
                    self._json(503 if outer.loading else 200, {"status": "loading" if outer.loading else "ok"})
                elif outer.loading:
                    self._json(503, {"error": {"message": "Loading model"}})
                elif self.path == "/v1/models":
                    self._json(200, {"object": "list", "data": [{"id": outer.model}]})
                elif not self._authorized():
                    self._json(401, {"error": {"message": "Invalid API Key"}})
                elif self.path == "/props":
                    self._json(200, {"total_slots": outer.slots, "model_path": outer.model_path,
                                     "cors_proxy_enabled": outer.cors_proxy, "ui": False,
                                     "endpoint_slots": False, "endpoint_props": False, "endpoint_metrics": False})
                elif self.path == "/tools":
                    self._json(outer.tools_status, {"error": {"message": "tools off"}} if outer.tools_status != 200
                               else [{"name": "exec_shell_command"}])
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
                    outer.chats += 1
                    outer.inflight += 1
                    outer.max_inflight = max(outer.max_inflight, outer.inflight)
                    count = outer.chats
                try:
                    if outer.on_chat is not None:
                        outer.on_chat(count)
                    time.sleep(outer.delay)
                    user = body["messages"][-1]["content"]
                    verdict = FAIL_REPLY if "JUDGE-FAIL" in user else PASS_REPLY
                    self._json(200, {"model": outer.model, "choices": [
                        {"message": {"role": "assistant", "content": verdict}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1200, "completion_tokens": 90}})
                finally:
                    with outer.lock:
                        outer.inflight -= 1

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


@pytest.fixture
def judge_server(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A fake judge server; --judge-url's tailnet check stands in for the one a real name passes
    (the fake listens on loopback), and the pause timings are short."""
    srv = FakeJudgeServer()
    monkeypatch.setattr(judge, "check_tailnet_url", lambda url: srv.url)
    monkeypatch.setattr(judge, "REMOTE_WAIT_POLL_S", 0.02)
    monkeypatch.setattr(judge, "REMOTE_BACKOFF_S", (0.05, 0.2))
    yield srv
    srv.stop()


def remote_key(tmp_path: Path, mode: int = 0o600, key: str = REMOTE_KEY) -> Path:
    p = tmp_path / "secrets" / "judge-4090.key"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# the judge server's key\n" + key + "\n")
    p.chmod(mode)
    return p


def golden_item(i: int, *, fail: bool = False, category: str = "coding") -> dict[str, Any]:
    """A re-judge queue item located in a golden trace file (0-based line), as build_rejudge_queue writes it."""
    item = make_item(i, category=category)
    item["id"] = f"pass-{i:04d}"
    item["source"] = {"file": "golden_traces_1_1.jsonl", "line": i, "telemetry_rowid": 100 + i}
    if fail:
        item["messages"][0]["content"] += " JUDGE-FAIL"
    return item


def max_row(item: dict[str, Any], verdict: str, status: str = "ok") -> dict[str, Any]:
    row = {"id": item["id"], "label": item["label"], "category": item["category"], "source": item["source"],
           "view": "outcome", "rubric_version": judge.RUBRIC_VERSION, "provider": "max-token-plan",
           "judge_key": "token-plan", "model": "qwen3.8-max", "status": status}
    if status == "ok":
        row["verdict"] = verdict
    return row


def remote_args(work: Path, key: Path, *extra: str) -> list[str]:
    return ["--workdir", str(work), "run", "--provider", "local", "--judge-url", "http://gpu-tower:18094",
            "--judge-key-file", str(key), "--backoff-base", "0", *extra]


def test_judge_url_is_tailnet_only_and_pinned_to_the_checked_address(monkeypatch: pytest.MonkeyPatch) -> None:
    addrs = {"gpu-tower": ["fd7a:115c:a1e0::39", "100.64.0.39"], "v6": ["fd7a:115c:a1e0::1"],
             "lan": ["192.168.1.20"], "pub": ["203.0.113.7"], "lo": ["127.0.0.1"], "me": ["100.64.0.21"],
             "mixed": ["100.64.0.39", "192.168.1.20"]}
    looked: list[str] = []

    def resolve(h: str) -> list[str]:
        looked.append(h)
        return addrs[h]

    monkeypatch.setattr(judge, "host_addresses", resolve)
    monkeypatch.setattr(judge, "is_local_address", lambda a: a == "100.64.0.21")
    assert judge.check_tailnet_url("http://gpu-tower:18094") == "http://100.64.0.39:18094"
    assert judge.check_tailnet_url("http://v6:18094/") == "http://[fd7a:115c:a1e0::1]:18094"
    for bad, why in (("http://lan:18094", "not a tailnet address"), ("http://pub:18094", "not a tailnet"),
                     ("http://lo:18094", "not a tailnet"), ("http://mixed:18094", "not a tailnet"),
                     ("http://me:18094", "is this host"), ("https://gpu-tower:18094", "must be http"),
                     ("http://gpu-tower", "needs its port"), ("http://gpu-tower:18094/v1", "only"),
                     ("http://u:p@gpu-tower:18094", "credentials"),
                     ("http://dashscope.aliyuncs.com:80", "hosted API")):
        with pytest.raises(judge.JudgeSetupError, match=why):
            judge.check_tailnet_url(bad)


def test_judge_key_file_must_be_a_private_regular_file_and_is_never_echoed(tmp_path: Path) -> None:
    good = remote_key(tmp_path)
    assert judge.load_key_file(good) == REMOTE_KEY
    loose = remote_key(tmp_path / "loose", mode=0o644)
    short = remote_key(tmp_path / "short", key="abc")
    link = tmp_path / "link.key"
    link.symlink_to(good)
    comments = tmp_path / "c.key"
    comments.write_text("# nothing\n\n")
    comments.chmod(0o600)
    for path, why in ((loose, "must be 0600"), (short, "no key of at least"), (link, "symlink"),
                      (comments, "no key of at least"), (tmp_path / "none.key", "cannot open")):
        with pytest.raises(judge.JudgeSetupError, match=why) as info:
            judge.load_key_file(path)
        assert REMOTE_KEY not in str(info.value) and "abc" not in str(info.value).split(":")[-1]


def test_remote_judge_end_to_end_caps_concurrency_at_the_slots_and_records_its_server(
        tmp_path: Path, judge_server: Any, capsys: pytest.CaptureFixture[str]) -> None:
    key = remote_key(tmp_path)
    items = [golden_item(i, fail=i % 4 == 0) for i in range(12)]
    work = tmp_path / "local-v4"
    write_queue(work, items, {"mode": judge.MODE_LOCAL_JUDGE, "view": "outcome"})
    argv = remote_args(work, key, "--concurrency", "5", "--judge-host", "gpu-tower", "--warmup", "4")
    assert judge.main(argv) == judge.EXIT_DONE
    out, err = capsys.readouterr()
    assert "cut to 2: the judge server has 2 slot(s)" in err
    assert judge_server.max_inflight <= 2 and judge_server.chats == 12
    rows = judge.read_jsonl(work / judge.RESULTS_FILE)
    assert len(rows) == 12 and {r["status"] for r in rows} == {"ok"}
    for r in rows:
        assert (r["provider"], r["model"], r["judge_route"], r["judge_host"], r["judge_quant"]) == (
            "local", REMOTE_ALIAS, "remote", "gpu-tower", "UD-Q4_K_XL")
        assert r["judge_sha256"] == judge.REMOTE_SHA256 and r["judge_model_file"] == "Qwen3.8-27B-UD-Q4_K_XL.gguf"
        assert r["phase"] == judge.PHASE_LOCAL and r["elapsed_s"] >= 0
        assert r["verification"]["judge"] == {
            "verdict": r["verdict"], "model": REMOTE_ALIAS, "served_model": REMOTE_ALIAS, "provider": "local",
            "route": "remote", "host": "gpu-tower", "quant": "UD-Q4_K_XL", "sha256": judge.REMOTE_SHA256}
    assert sum(r["verdict"] == "fail" for r in rows) == 3
    # the key: only ever the Authorization header; never argv, output, rows, status or the ledger
    keyed = [a for a in judge_server.auth_seen if a]
    assert keyed and set(keyed) == {f"Bearer {REMOTE_KEY}"}
    blob = out + err + " ".join(argv) + (work / judge.RESULTS_FILE).read_text() + \
        (work / judge.STATUS_FILE).read_text() + (tmp_path / "distill" / "spend-ledger.csv").read_text()
    assert REMOTE_KEY not in blob
    summary = json.loads(out[out.rindex("\n{\n") + 1:])
    assert summary["verdicts"] == {"pass": 9, "fail": 3} and summary["judge_host"] == "gpu-tower"
    assert summary["concurrency"] == 2 and summary["projection"]["pending"] == 0
    warm = [json.loads(line) for line in out.splitlines() if line.startswith('{"projection"')]
    assert warm and warm[0]["projection"]["pending"] == 8 and warm[0]["projection"]["verdicts_per_hour"] > 0
    status = json.loads((work / judge.STATUS_FILE).read_text())
    assert status["state"] == "done" and status["judge_host"] == "gpu-tower" and status["concurrency"] == 2


def test_remote_judge_refuses_tools_an_open_server_a_wrong_quant_a_max_model_and_a_wrong_key(
        tmp_path: Path, judge_server: Any, capsys: pytest.CaptureFixture[str]) -> None:
    key = remote_key(tmp_path)
    work = tmp_path / "w"
    write_queue(work, [golden_item(1)], {"mode": judge.MODE_LOCAL_JUDGE})
    cases = [("tools_status", 200, "built-in tools or MCP"), ("cors_proxy", True, "MCP proxy on"),
             ("open", True, "without a key"), ("model_path", "/m/Qwen3.8-27B-IQ3_XXS.gguf", "not the UD-Q4_K_XL"),
             ("model", "qwen3.8-max", "never the local judge"), ("key", "b" * 64, "refused our key"),
             ("tools_status", 500, "HTTP 500")]
    for attr, value, why in cases:
        before = getattr(judge_server, attr)
        setattr(judge_server, attr, value)
        try:
            assert judge.main(remote_args(work, key)) == judge.EXIT_FATAL, attr
            assert why in capsys.readouterr().err, attr
        finally:
            setattr(judge_server, attr, before)
    assert judge_server.chats == 0 and not (work / judge.RESULTS_FILE).exists()
    # a server still loading: paused (exit 4), start it and run again
    judge_server.loading = True
    assert judge.main(remote_args(work, key)) == judge.EXIT_PAUSED
    assert "not ready (loading)" in capsys.readouterr().err
    judge_server.loading = False
    # the key file is required, and a loopback URL is --local-url's
    r = judge.main(["--workdir", str(work), "run", "--provider", "local", "--judge-url", "http://gpu-tower:18094"])
    assert r == judge.EXIT_FATAL and "needs --judge-key-file" in capsys.readouterr().err
    r = judge.main(["--workdir", str(work), "run", "--provider", "local", "--local-url", "http://127.0.0.1:9",
                    "--judge-key-file", str(key)])
    assert r == judge.EXIT_FATAL and "is for --judge-url" in capsys.readouterr().err
    # a local-judge queue is never judged by a Max route
    r = judge.main(["--workdir", str(work), "run", "--provider", "token-plan"])
    assert r == judge.EXIT_FATAL and "the local judge judges it" in capsys.readouterr().err


def _remote_run(tmp_path: Path, items: list[dict[str, Any]], srv: Any, **kw: Any) -> Any:
    ledger = spend.Ledger(notify_cmd=str(notify_script(tmp_path)))
    client = judge.RemoteJudgeClient("http://gpu-tower:18094", key_file=remote_key(tmp_path), ledger=ledger,
                                     phase=judge.PHASE_LOCAL, host="gpu-tower", timeout=5.0)
    client.preflight()
    config = judge.RunConfig(jitter=0.0, backoff_base=0.0,
                             concurrency=judge.remote_concurrency(kw.pop("concurrency", None), client.slots))
    return judge.JudgeRun(items, client=client, results_path=tmp_path / judge.RESULTS_FILE,
                          status_path=tmp_path / judge.STATUS_FILE, config=config, clock=Clock(NOW),
                          sleep=asyncio.sleep, **kw)


def test_a_server_that_goes_away_pauses_the_run_and_it_resumes_with_the_slots_it_comes_back_with(
        tmp_path: Path, judge_server: Any) -> None:
    items = [golden_item(i) for i in range(10)]

    def outage(count: int) -> None:
        if count == 4:
            def restart() -> None:
                judge_server.stop()
                time.sleep(0.6)
                judge_server.slots = 1
                judge_server.start()
            threading.Thread(target=restart, daemon=True).start()

    judge_server.on_chat = outage
    judge_server.delay = 0.15
    run = _remote_run(tmp_path, items, judge_server)
    assert run.limit == 2
    outcome = asyncio.run(asyncio.wait_for(run.run(), 60))
    assert outcome.state == "done" and outcome.remaining == 0 and outcome.errors == 0, outcome
    assert run.pauses >= 1 and run.limit == 1
    rows = results(tmp_path)
    assert {r["status"] for r in rows} == {"ok"} and {r["id"] for r in rows} == {it["id"] for it in items}


def test_a_server_that_comes_back_changed_stops_the_run(tmp_path: Path, judge_server: Any) -> None:
    items = [golden_item(i) for i in range(6)]

    def swap(count: int) -> None:
        if count == 2:
            def restart() -> None:
                judge_server.stop()
                time.sleep(0.3)
                judge_server.tools_status = 200      # someone started it with --tools all
                judge_server.start()
            threading.Thread(target=restart, daemon=True).start()

    judge_server.on_chat = swap
    judge_server.delay = 0.15
    outcome = asyncio.run(asyncio.wait_for(_remote_run(tmp_path, items, judge_server).run(), 60))
    assert outcome.state == "fatal" and "came back changed" in outcome.reason and "GET /tools" in outcome.reason
    assert outcome.exit_code == judge.EXIT_FATAL and outcome.remaining > 0


def test_a_server_gone_longer_than_max_pause_ends_the_run_paused_and_the_next_run_resumes(
        tmp_path: Path, judge_server: Any) -> None:
    items = [golden_item(i) for i in range(6)]
    judge_server.on_chat = lambda count: threading.Thread(target=judge_server.stop, daemon=True).start() \
        if count == 3 else None
    judge_server.delay = 0.1
    outcome = asyncio.run(asyncio.wait_for(_remote_run(tmp_path, items, judge_server, max_pause_s=0.5).run(), 60))
    assert outcome.state == "paused" and "run again to resume" in outcome.reason and outcome.remaining > 0
    assert not [r for r in results(tmp_path) if r["status"] != "ok"]     # nothing recorded as an error
    judge_server.on_chat = None
    judge_server.start()
    again = asyncio.run(asyncio.wait_for(_remote_run(tmp_path, items, judge_server).run(), 60))
    assert again.state == "done" and {r["id"] for r in results(tmp_path)} == {it["id"] for it in items}


def _build_helpers() -> Any:
    name = "distill_build_test_helpers"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("test_distill_build.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_remote_verdicts_are_read_by_build_a_fail_drops_the_row_and_a_pass_is_a_judge_pass(
        tmp_path: Path, judge_server: Any) -> None:
    bt = _build_helpers()
    build = bt.build
    ts = 1_790_000_000.0
    golden = bt.write_jsonl(tmp_path / "golden_traces_1_1.jsonl", [
        bt.trace(bt.word(i), "bash", {"command": f"make {bt.word(i)}"}, ts=ts + i) for i in range(3)])
    index = [{"file": golden.name, "model": "qwen3.8-max", "h_prompt_call": f"k{i}", "category": "coding",
              "rowid": 100 + i, "tool": "bash", "ts": ts + i} for i in range(3)]
    items = [judge.queue_item({**it, "trace": {"messages": [{"role": "user", "content": f"task {n}"
                                                             + (" JUDGE-FAIL" if n == 1 else "")},
                                                            {"role": "assistant", "content": "",
                                                             "tool_calls": [{"function": {"name": "bash",
                                                                                          "arguments": "{}"}}]}],
                                               "tools": [], "_meta": {}},
                               "stratum_weight": 1.0, "outcome": {"status": "success", "text": None}})
             for n, it in enumerate(judge.pass_pool(index))]
    asyncio.run(_remote_run(tmp_path, items, judge_server).run())
    vfile = tmp_path / judge.RESULTS_FILE
    stats: collections.Counter = collections.Counter()
    checks: dict[str, Any] = {}
    verdicts = build.load_verdicts([str(vfile)], stats, checks)
    assert verdicts == {"golden_traces_1_1.jsonl:1": "pass", "golden_traces_1_1.jsonl:2": "fail",
                        "golden_traces_1_1.jsonl:3": "pass"}
    assert stats["verdict_rows_max_judge"] == 0 and stats["verdict_rows_unusable"] == 0
    b = bt.make_builder(tmp_path, verdicts=verdicts, verdict_checks=checks,
                        index=build.load_index(str(bt.write_jsonl(tmp_path / "index.jsonl", index))))
    b.stats.update(stats)
    bt.feed(b, golden)
    assert {e.trace_id: e.label for e in b.examples} == {
        "golden_traces_1_1.jsonl:1": "judge-pass", "golden_traces_1_1.jsonl:3": "judge-pass"}
    assert b.stats["excluded:judge_fail"] == 1
    b.split()
    assert b.gates()["verdicts"]["pass"]
    # a CALIBRATION queue's verdicts are never applied: fed to build.py by mistake, they fail its gate
    cal = [{**it, "source": {"calibration_of": it["source"]}} for it in items]
    rows = [{**r, "source": c["source"]} for r, c in zip(judge.read_jsonl(vfile), cal)]
    st: collections.Counter = collections.Counter()
    assert build.load_verdicts([str(bt.write_jsonl(tmp_path / "cal.jsonl", rows))], st) == {}
    assert st["verdict_rows_unusable"] == 3


def _src_workdir(tmp_path: Path, items: list[dict[str, Any]], max_rows: list[dict[str, Any]],
                 name: str = "phase2-rejudge", meta: dict[str, Any] | None = None) -> Path:
    src = tmp_path / name
    write_queue(src, items, {"mode": "rejudge", "view": "outcome"} if meta is None else meta)
    (src / judge.RESULTS_FILE).write_text("".join(json.dumps(r) + "\n" for r in max_rows))
    return src


def write_calibration(work: Path, apply: list[str], *, decision: str = "use-local-judge",
                      schema: str | None = None) -> Path:
    """A calibrate report as ``subset --which unjudged --calibration`` reads it."""
    work.mkdir(parents=True, exist_ok=True)
    rep = {"schema": schema or judge.CALIBRATION_SCHEMA, "rubric_version": judge.RUBRIC_VERSION,
           "created": "2026-10-31T17:00:00Z", "apply_categories": apply,
           "decision": {"decision": decision, "reason": "test"},
           "by_category": [{"category": c, "n": 30, "agreement": 0.9} for c in apply]}
    if schema == "distill-judge-calibration/1":
        del rep["apply_categories"]
    (work / judge.CALIBRATION_FILE).write_text(json.dumps(rep))
    return work


def test_subset_cuts_the_calibration_set_and_the_unjudged_train_rows(tmp_path: Path,
                                                                     capsys: pytest.CaptureFixture[str]) -> None:
    bt = _build_helpers()
    items = [golden_item(i) for i in range(10)]
    max_rows = [max_row(items[i], "pass" if i % 3 else "fail") for i in range(5)]
    max_rows += [max_row(items[5], "", status="unparseable"),
                 {**max_row(items[6], "pass"), "provider": "local", "model": "qwen3.8-27b"}]  # not a Max verdict
    src = _src_workdir(tmp_path, items, max_rows)
    # build.py's own id for each item's train row
    for it in items:
        assert judge.train_row_id(it["source"]["file"], it["source"]["line"]) == \
            "p3-" + bt.build.h(f"{it['source']['file']}:{it['source']['line'] + 1}")
    train = [{"id": judge.train_row_id(it["source"]["file"], it["source"]["line"]),
              "label": "judge-pass" if i < 5 else ("tool-success" if i < 9 else "judge-pass")}
             for i, it in enumerate(items)]
    train.append({"id": "p3-0000000000000000", "label": "tool-success"})  # a gym row: no queue item
    tf = tmp_path / "train.jsonl"
    tf.write_text("".join(json.dumps(r) + "\n" for r in train))

    cal = tmp_path / "cal"
    assert judge.main(["--workdir", str(cal), "subset", "--from", str(src), "--which", "max-judged"]) == 0
    q = judge.read_jsonl(cal / judge.QUEUE_FILE)
    assert sorted(it["id"] for it in q) == [items[i]["id"] for i in range(5)]
    assert all(set(it["source"]) == {"calibration_of"} for it in q)
    meta = json.loads((cal / judge.QUEUE_META_FILE).read_text())
    assert meta["mode"] == judge.MODE_CALIBRATION and meta["max_verdicts"] == {"fail": 2, "pass": 3}
    assert judge.main(["--workdir", str(cal), "subset", "--from", str(src), "--which", "max-judged"]) == 1  # exists

    unj = tmp_path / "unj"
    # never without a calibration that decided use-local-judge
    assert judge.main(["--workdir", str(unj), "subset", "--from", str(src), "--which", "unjudged",
                       "--train-file", str(tf)]) == judge.EXIT_ERROR
    assert "--calibration" in capsys.readouterr().err and not (unj / judge.QUEUE_FILE).exists()
    cal_ok = write_calibration(tmp_path / "cal-ok", ["coding"])
    assert judge.main(["--workdir", str(unj), "subset", "--from", str(src), "--which", "unjudged",
                       "--train-file", str(tf), "--calibration", str(cal_ok)]) == 0
    q = judge.read_jsonl(unj / judge.QUEUE_FILE)
    assert [it["id"] for it in q] == [items[i]["id"] for i in (5, 6, 7, 8)]   # source order, sources kept
    assert all("file" in it["source"] for it in q)
    meta = json.loads((unj / judge.QUEUE_META_FILE).read_text())
    assert meta["mode"] == judge.MODE_LOCAL_JUDGE and meta["train_rows_with_label"] == 5
    assert meta["train_rows_matched"] == 4 and meta["train_rows_unmatched"] == 1
    assert meta["train_files"][0]["sha256"] == hashlib.sha256(tf.read_bytes()).hexdigest()
    with pytest.raises(SystemExit):
        judge.main(["--workdir", str(unj), "subset", "--from", str(src), "--which", "everything"])
    assert judge.main(["--workdir", str(src), "subset", "--from", str(src), "--which", "unjudged",
                       "--train-file", str(tf), "--calibration", str(cal_ok), "--force"]) == judge.EXIT_ERROR
    assert "its own --workdir" in capsys.readouterr().err


def test_cohen_kappa_reference_values() -> None:
    # 2x2 with a=20, b=5, c=10, d=15: po = 0.7, pe = 0.5 * 0.6 + 0.5 * 0.4 = 0.5, kappa = 0.4
    k, ci = judge.cohen_kappa(20, 5, 10, 15)
    assert k == pytest.approx(0.4) and ci[0] < 0.4 < ci[1]
    assert judge.cohen_kappa(10, 0, 0, 10)[0] == pytest.approx(1.0)
    assert judge.cohen_kappa(0, 0, 0, 0) == (None, None)


def test_calibration_reports_agreement_kappa_confusion_and_decides_at_85_percent(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    items = [golden_item(i, category="coding" if i < 60 else "other") for i in range(100)]
    # Max: 70 pass, 30 fail. Local agrees on 85 of 100: 10 Max passes it fails, 5 Max fails it passes.
    mx = {it["id"]: ("pass" if i < 70 else "fail") for i, it in enumerate(items)}
    flip = {items[i]["id"] for i in list(range(10)) + list(range(70, 75))}
    src = _src_workdir(tmp_path, items, [max_row(it, mx[it["id"]]) for it in items])
    cal = tmp_path / "cal"
    assert judge.main(["--workdir", str(cal), "subset", "--from", str(src), "--which", "max-judged"]) == 0
    queue = judge.read_jsonl(cal / judge.QUEUE_FILE)
    capsys.readouterr()

    def local_rows(ids: Iterable[str]) -> list[dict[str, Any]]:
        out = []
        for it in queue:
            if it["id"] not in ids:
                continue
            v = mx[it["id"]] if it["id"] not in flip else ("fail" if mx[it["id"]] == "pass" else "pass")
            out.append({"id": it["id"], "status": "ok", "verdict": v, "view": "outcome",
                        "rubric_version": judge.RUBRIC_VERSION, "provider": "local", "model": REMOTE_ALIAS,
                        "judge_host": "gpu-tower", "judge_quant": "UD-Q4_K_XL", "judge_sha256": "0" * 64})
        return out

    all_ids = {it["id"] for it in queue}
    (cal / judge.RESULTS_FILE).write_text("".join(json.dumps(r) + "\n" for r in local_rows(all_ids)))
    assert judge.main(["--workdir", str(cal), "calibrate"]) == judge.EXIT_DONE
    text = capsys.readouterr().out
    rep = json.loads((cal / judge.CALIBRATION_FILE).read_text())
    assert rep["set"] == 100 and rep["pairs"] == 100 and rep["agreement"]["k"] == 85
    assert rep["decision"]["decision"] == "use-local-judge" and "DECISION: use the local judge" in text
    assert rep["confusion"] == {"max_pass": {"local_pass": 60, "local_fail": 10},
                                "max_fail": {"local_pass": 5, "local_fail": 25}}
    # po = 0.85, pe = 0.7 * 0.65 + 0.3 * 0.35 = 0.56, kappa = 0.29 / 0.44
    assert rep["kappa"]["value"] == pytest.approx(round(0.29 / 0.44, 4))
    assert rep["false_pass"]["k"] == 5 and rep["false_pass"]["n"] == 30
    assert rep["false_pass"]["share_of_local_passes"] == pytest.approx(round(5 / 65, 4))
    assert rep["false_fail"]["k"] == 10 and rep["false_fail"]["n"] == 70
    lo, hi = rep["agreement"]["ci95"]
    assert lo < 0.85 < hi
    cats = {c["category"]: c for c in rep["by_category"]}
    assert cats["coding"]["n"] == 60 and cats["coding"]["agreement"] == pytest.approx(50 / 60, abs=1e-4)
    assert cats["other"]["agreement"] == pytest.approx(35 / 40, abs=1e-4)
    # 85% overall, but coding alone is 83%: only "other" (87.5% on 40 pairs) may be applied
    assert rep["apply_categories"] == ["other"] and not cats["coding"]["apply"] and cats["other"]["apply"]
    assert "Applied only in: other" in text
    assert rep["judge"]["hosts"] == {"gpu-tower": 100} and rep["judge"]["quants"] == {"UD-Q4_K_XL": 100}

    # one more disagreement: 84% < 85%: train on the dataset as it is
    flip.add(items[20]["id"])
    (cal / judge.RESULTS_FILE).write_text("".join(json.dumps(r) + "\n" for r in local_rows(all_ids)))
    assert judge.main(["--workdir", str(cal), "calibrate"]) == judge.EXIT_DONE
    assert "DECISION: do not use the local judge" in capsys.readouterr().out
    rep = json.loads((cal / judge.CALIBRATION_FILE).read_text())
    assert rep["decision"]["decision"] == "do-not-use" and rep["apply_categories"] == []

    # three items without a local verdict: more than the 2 (2%) allowed -> incomplete (exit 6)
    some = {it["id"] for it in queue[3:]}
    (cal / judge.RESULTS_FILE).write_text("".join(json.dumps(r) + "\n" for r in local_rows(some)))
    assert judge.main(["--workdir", str(cal), "calibrate"]) == judge.EXIT_INCOMPLETE
    assert "incomplete" in capsys.readouterr().out
    assert judge.main(["--workdir", str(cal), "calibrate", "--allow-missing", "3"]) == judge.EXIT_DONE
    capsys.readouterr()


def test_calibration_set_spans_every_max_judged_source_on_the_pass_side_interleaved_by_category(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The re-judge reached only coding and memory/wiki; the Phase 2 sample holds Max's verdicts in
    sentinel/daemon and other. Calibration takes both, pass side only, small categories first."""
    rejudge = [golden_item(i) for i in range(40)] + [golden_item(i, category="memory/wiki") for i in range(40, 44)]
    rj = _src_workdir(tmp_path, rejudge, [max_row(it, "pass") for it in rejudge])
    sample = [golden_item(i, category="sentinel/daemon") for i in range(100, 106)]
    sample += [golden_item(i, category="other") for i in range(106, 110)]
    fails = [make_item(i, label="fail", category="other") for i in range(110, 113)]  # fail side: error shown
    sample_rows = [max_row(it, "fail" if n % 2 else "pass") for n, it in enumerate(sample + fails)]
    sample_rows.append({**max_row(sample[0], "fail"), "view": "blind"})  # another view: not a verdict here
    p2 = _src_workdir(tmp_path, sample + fails, sample_rows, name="phase2", meta={})  # the sample's meta: no view
    cal = tmp_path / "cal"
    assert judge.main(["--workdir", str(cal), "subset", "--which", "max-judged",
                       "--from", str(rj), "--from", str(p2)]) == 0
    q = judge.read_jsonl(cal / judge.QUEUE_FILE)
    assert {it["id"] for it in q} == {it["id"] for it in rejudge + sample}   # no fail-side item
    assert collections.Counter(it["category"] for it in q[:16]) == {
        "coding": 4, "memory/wiki": 4, "other": 4, "sentinel/daemon": 4}     # every category early
    meta = json.loads((cal / judge.QUEUE_META_FILE).read_text())
    assert meta["subset_sources"] == [str(rj.resolve()), str(p2.resolve())] and meta["side"] == "pass"
    assert meta["max_verdict_files"] == [str(rj.resolve() / judge.RESULTS_FILE),
                                         str(p2.resolve() / judge.RESULTS_FILE)]
    assert meta["max_verdicts_by_category"]["sentinel/daemon"] == {"pass": 3, "fail": 3}
    assert meta["queued_by_source"][str(p2.resolve())] == {"other": 4, "sentinel/daemon": 6}
    # calibrate reads the Max verdicts of every source by default
    local = [{"id": it["id"], "status": "ok", "verdict": "pass", "view": "outcome",
              "rubric_version": judge.RUBRIC_VERSION, "provider": "local", "model": REMOTE_ALIAS} for it in q]
    (cal / judge.RESULTS_FILE).write_text("".join(json.dumps(r) + "\n" for r in local))
    assert judge.main(["--workdir", str(cal), "calibrate", "--min-category-pairs", "4"]) == judge.EXIT_DONE
    rep = json.loads((cal / judge.CALIBRATION_FILE).read_text())
    assert rep["set"] == 54 and {c["category"]: c["n"] for c in rep["by_category"]} == {
        "coding": 40, "memory/wiki": 4, "other": 4, "sentinel/daemon": 6}
    # --side both takes the fail side too; sources judged in different views are refused
    capsys.readouterr()
    both = tmp_path / "both"
    assert judge.main(["--workdir", str(both), "subset", "--which", "max-judged", "--side", "both",
                       "--from", str(rj), "--from", str(p2)]) == 0
    assert len(judge.read_jsonl(both / judge.QUEUE_FILE)) == 57
    (p2 / judge.QUEUE_META_FILE).write_text(json.dumps({"view": "blind"}))
    assert judge.main(["--workdir", str(tmp_path / "mixed"), "subset", "--which", "max-judged",
                       "--from", str(rj), "--from", str(p2)]) == judge.EXIT_ERROR
    assert "different views" in capsys.readouterr().err


def test_local_verdicts_are_used_only_in_categories_calibrated_with_enough_pairs(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The review's case: coding agrees 88% and carries the overall gate; sentinel/daemon agrees 60%
    and other has too few pairs. Only coding's rows get the local judge; the rest stay as they are."""
    cal_items = ([golden_item(i) for i in range(200)]
                 + [golden_item(i, category="sentinel/daemon") for i in range(200, 225)]
                 + [golden_item(i, category="other") for i in range(225, 235)])
    mx = {it["id"]: "pass" for it in cal_items}
    flip = ({cal_items[i]["id"] for i in range(24)}            # coding: 176/200 = 88%
            | {cal_items[i]["id"] for i in range(200, 210)}    # sentinel/daemon: 15/25 = 60%
            | {cal_items[225]["id"]})                          # other: 9/10, but 10 pairs < 20
    local = [{"id": it["id"], "status": "ok", "verdict": "fail" if it["id"] in flip else "pass",
              "view": "outcome", "rubric_version": judge.RUBRIC_VERSION, "provider": "local",
              "model": REMOTE_ALIAS} for it in cal_items]
    rep = judge.calibration_report(cal_items, local, [max_row(it, mx[it["id"]]) for it in cal_items])
    assert rep["agreement"]["k"] == 200 and rep["agreement"]["n"] == 235   # 85.1% overall
    assert rep["decision"]["decision"] == "use-local-judge" and rep["apply_categories"] == ["coding"]
    cats = {c["category"]: c for c in rep["by_category"]}
    assert not cats["sentinel/daemon"]["apply"] and "60.00% < 85%" in cats["sentinel/daemon"]["apply_reason"]
    assert not cats["other"]["apply"] and "10 pair(s) < 20" in cats["other"]["apply_reason"]
    text = judge.render_calibration(rep)
    assert "Applied only in: coding" in text and "| sentinel/daemon | 25/25 | 60.0% |" in text
    calw = tmp_path / "cal"
    calw.mkdir()
    (calw / judge.CALIBRATION_FILE).write_text(json.dumps(rep))

    # the unjudged train rows: coding, sentinel/daemon and other, none Max-judged
    target = ([golden_item(i) for i in range(300, 306)] + [golden_item(i, category="sentinel/daemon")
                                                           for i in range(306, 310)]
              + [golden_item(i, category="other") for i in range(310, 313)])
    src = _src_workdir(tmp_path, target, [])
    tf = tmp_path / "train.jsonl"
    tf.write_text("".join(json.dumps({"id": judge.train_row_id(it["source"]["file"], it["source"]["line"]),
                                      "label": "tool-success", "category": it["category"]}) + "\n"
                          for it in target))
    unj = tmp_path / "unj"
    assert judge.main(["--workdir", str(unj), "subset", "--which", "unjudged", "--from", str(src),
                       "--train-file", str(tf), "--calibration", str(calw)]) == 0
    assert [it["id"] for it in judge.read_jsonl(unj / judge.QUEUE_FILE)] == [it["id"] for it in target[:6]]
    meta = json.loads((unj / judge.QUEUE_META_FILE).read_text())
    assert meta["uncalibrated_by_category"] == {"other": 3, "sentinel/daemon": 4}
    assert meta["target_by_category"]["sentinel/daemon"] == {
        "rows": 4, "applied": False, "calibration_pairs": 25, "calibration_agreement": 0.6}
    assert meta["target_by_category"]["coding"]["applied"] and meta["calibration"]["apply_categories"] == ["coding"]

    # no category calibrated -> do-not-use, and a do-not-use or schema-1 report queues nothing
    rep2 = judge.calibration_report(cal_items, local, [max_row(it, mx[it["id"]]) for it in cal_items],
                                    min_category_pairs=500)
    assert rep2["decision"]["decision"] == "do-not-use" and rep2["apply_categories"] == []
    capsys.readouterr()
    for name, cal_dir in (("no", write_calibration(tmp_path / "no", ["coding"], decision="do-not-use")),
                          ("v1", write_calibration(tmp_path / "v1", ["coding"],
                                                   schema="distill-judge-calibration/1"))):
        assert judge.main(["--workdir", str(tmp_path / f"unj-{name}"), "subset", "--which", "unjudged",
                           "--from", str(src), "--train-file", str(tf), "--calibration", str(cal_dir)]) == 1
        assert not (tmp_path / f"unj-{name}" / judge.QUEUE_FILE).exists()
    err = capsys.readouterr().err
    assert "decided 'do-not-use'" in err and "no per-category decision" in err


def _report_of(b: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """What check-build reads of a build.py report.json, from a split Builder."""
    return {"gates": b.gates(), "counts": dict(sorted(b.stats.items())),
            "inputs": {"written": {"train.jsonl": len(b.final["train"])}}, "settings": {"seed": "phase3-v1"},
            **(extra or {})}


def test_rebuild_loop_keeps_every_max_file_and_judges_the_rows_a_rebuild_lets_in(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """End to end against build.py: base build (two Max verdict files) -> local judge on its
    tool-success rows -> rebuild. Leaving out a Max file the base used is caught (check-build exit 5).
    The capped coding category refills from past its old cut, so the rebuild holds tool-success rows
    the judge never saw; the loop judges them until `subset` exits 8 and none is left."""
    bt = _build_helpers()
    build = bt.build
    golden = bt.corpus(tmp_path)          # 120 coding rows over 40 tasks, 12 each daemon/other/memory
    feed_rows = judge.read_jsonl(golden)
    settings = {"eval_task_frac": 0.15, "val_frac": 0.05, "holdout_max_group_share": 0.5}

    def builder(name: str, files: list[Path]) -> Any:
        stats: collections.Counter = collections.Counter()
        verdicts = build.load_verdicts([str(f) for f in files], stats)
        (tmp_path / name).mkdir()
        b = bt.make_builder(tmp_path / name, verdicts=verdicts, **settings)
        b.stats.update(stats)
        bt.feed(b, golden)
        b.split()
        return b

    # the re-judge queue: every teacher row, in its build.py category
    probe = builder("probe", [])
    cat_of = {e.trace_id: e.category for e in probe.examples}
    items = []
    for line, row in enumerate(feed_rows):
        tid = f"{golden.name}:{line + 1}"
        if tid in cat_of:
            items.append({**golden_item(line, category=cat_of[tid]),
                          "source": {"file": golden.name, "line": line, "telemetry_rowid": 1000 + line}})
    by_id = {it["id"]: it for it in items}
    coding = [it for it in items if it["category"] == "coding"]
    # Max: the Phase 2 sample judged a few rows (two fails) plus a fail-side item; the re-judge 30 coding rows
    sample_rows = [max_row(coding[i], "fail" if i < 2 else "pass") for i in range(6)]
    sample_rows.append({**max_row(make_item(900, label="fail"), "fail")})
    p2 = _src_workdir(tmp_path, coding[:6], sample_rows, name="phase2", meta={})
    rj_rows = [max_row(it, "pass" if n % 5 else "fail") for n, it in enumerate(coding[6:36])]
    rj = _src_workdir(tmp_path, [it for it in items if it["id"] not in {c["id"] for c in coding[:6]}],
                      rj_rows)
    max_files = [p2 / judge.RESULTS_FILE, rj / judge.RESULTS_FILE]

    base = builder("base", max_files)
    assert base.gates()["verdicts"]["pass"] and base.stats["excluded:judge_fail"] == 2 + 6
    base_dir = tmp_path / "build-base"
    base_dir.mkdir()
    (base_dir / "report.json").write_text(json.dumps(_report_of(base)))

    def write_train(b: Any, d: Path) -> Path:
        d.mkdir(exist_ok=True)
        (d / "train.jsonl").write_text("".join(json.dumps({"id": e.id, "label": e.label, "category": e.category})
                                               + "\n" for e in b.final["train"]))
        return d / "train.jsonl"

    base_train = {e.id for e in base.final["train"]}
    assert any(e.label == "tool-success" for e in base.final["train"])
    cal = write_calibration(tmp_path / "cal", ["coding", "memory/wiki", "other", "sentinel/daemon"])

    def judge_round(n: int, train_file: Path) -> tuple[int, Path]:
        work = tmp_path / f"local-{n}"
        rc = judge.main(["--workdir", str(work), "subset", "--which", "unjudged", "--from", str(rj),
                         "--from", str(p2), "--train-file", str(train_file), "--calibration", str(cal)])
        if rc != 0:
            return rc, work
        rows = []
        for k, it in enumerate(judge.read_jsonl(work / judge.QUEUE_FILE)):
            verdict = "fail" if (it["category"] == "coding" and k % 3 == 0) else "pass"
            rows.append({"id": it["id"], "label": "pass", "category": it["category"], "source": it["source"],
                         "status": "ok", "verdict": verdict, "view": "outcome",
                         "rubric_version": judge.RUBRIC_VERSION, "provider": "local", "model": REMOTE_ALIAS})
        (work / judge.RESULTS_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
        return rc, work

    rc, l1 = judge_round(1, write_train(base, base_dir))
    assert rc == 0
    l1_rows = judge.read_jsonl(l1 / judge.RESULTS_FILE)
    max_judged = {r["id"] for r in rj_rows + sample_rows}
    assert {r["id"] for r in l1_rows} == {
        i for i, it in by_id.items()
        if judge.train_row_id(golden.name, it["source"]["line"]) in base_train and i not in max_judged}
    l1_fails = sum(1 for r in l1_rows if r["verdict"] == "fail")
    assert l1_fails > 0
    capsys.readouterr()

    # the reviewed command: the re-judge file and the local file, WITHOUT the Phase 2 sample file
    bad = builder("bad", [rj / judge.RESULTS_FILE, l1 / judge.RESULTS_FILE])
    bad_dir = tmp_path / "build-bad"
    write_train(bad, bad_dir)
    (bad_dir / "report.json").write_text(json.dumps(_report_of(bad)))
    assert judge.main(["check-build", "--base", str(base_dir), "--build", str(bad_dir),
                       "--local", str(l1)]) == judge.EXIT_FATAL
    out = json.loads(capsys.readouterr().out)
    assert not out["ok"] and any("pass every --verdicts file the base used" in p for p in out["problems"])
    assert out["expected"]["judge_fail"] == 8 + l1_fails and out["build"]["judge_fail"] == 6 + l1_fails

    # the right rebuild: every Max file the base used, in its order, the local file last; then loop
    files = [*max_files, l1 / judge.RESULTS_FILE]
    locals_ = [l1]
    newcomers_seen = False
    for n in range(2, 12):
        b = builder(f"re-{n}", files)
        d = tmp_path / f"build-{n}"
        train_file = write_train(b, d)
        (d / "report.json").write_text(json.dumps(_report_of(b)))
        capsys.readouterr()
        assert judge.main(["check-build", "--base", str(base_dir), "--build", str(d),
                           *[a for w in locals_ for a in ("--local", str(w))]]) == judge.EXIT_DONE
        out = json.loads(capsys.readouterr().out)
        assert out["ok"] and out["build"]["tokenplan_max_judge"] == out["base"]["tokenplan_max_judge"]
        left = {e.id for e in b.final["train"] if e.label == "tool-success"}
        assert out["tool_success_left"] == len(left)
        newcomers_seen |= bool(left - base_train)
        rc, work = judge_round(n, train_file)
        if rc == judge.EXIT_NOTHING_LEFT:
            assert not left and not (work / judge.QUEUE_FILE).exists()
            break
        assert rc == 0 and {it["id"] for it in judge.read_jsonl(work / judge.QUEUE_FILE)} <= {
            it["id"] for it in items if judge.train_row_id(golden.name, it["source"]["line"]) in left}
        files.append(work / judge.RESULTS_FILE)
        locals_.append(work)
    else:
        pytest.fail("the rebuild loop did not converge")
    assert newcomers_seen        # the coding cap let unjudged rows in on the first rebuild
    assert len(locals_) >= 2     # so one loop round was needed
    assert all(e.label != "tool-success" for e in b.final["train"])


def test_the_local_judge_uses_the_max_judges_rubric_and_prompt(judge_server: Any, tmp_path: Path) -> None:
    """Apples to apples: the remote judge sends exactly the Max judge's system prompt and user prompt."""
    seen: list[dict[str, Any]] = []
    orig = judge.RemoteJudgeClient._request

    def spy(self: Any, path: str, body: Any = None, **kw: Any) -> Any:
        if body is not None:
            seen.append(body)
        return orig(self, path, body, **kw)

    item = golden_item(3)
    item["outcome"] = {"status": "success", "text": "3 files"}
    run = _remote_run(tmp_path, [item], judge_server)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(judge.RemoteJudgeClient, "_request", spy)
        asyncio.run(run.run())
    body = seen[0]
    user, _ = judge.render_prompt(item, view="outcome")
    max_body = judge.judge_payload(judge.system_prompt("outcome"), user, model="qwen3.8-max", max_tokens=1024)
    assert body["messages"] == max_body["messages"]
    assert body["temperature"] == 0 and body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == max_body["max_tokens"]
