"""Tests for scripts/distill/judge.py and scripts/distill/build_fail_set.py (distill Phase 2).

No network and no model: the judge's client is a scripted fake, the clock and
the sleep are injected, and every store is a synthetic SQLite file under
tmp_path. Fixtures use neutral names only. The paid (max-payg) and local
clients, the spend ledger and judge_run.sh are tested against a fake
OpenAI-compatible server in tests/test_distill_spend.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sqlite3
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    for extra in ([], ["--provider", "token-plan"], ["--provider", "max-payg", "--cap-usd", "500"],
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
