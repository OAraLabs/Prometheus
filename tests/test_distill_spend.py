"""Tests for scripts/distill/spend.py, judge.py's paid and local providers, and judge_run.sh.

No real API and no real key: a fake OpenAI-compatible server on loopback
stands in for Model Studio (the paid client's requests are re-pointed at it by
an injected opener) and for the local llama-server. HOME is tmp_path in every
test, so the one canonical ledger (~/distill/spend-ledger.csv), its STOP file,
every env file and notify command live under tmp_path. Fixture keys are fake.
"""

from __future__ import annotations

import asyncio
import collections
import http.server
import importlib.util
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_DIR = _ROOT / "scripts" / "distill"


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

MAX = "qwen3.8-max"
SG = "ap-southeast-1"
KEY = "sk-" + "fixturepaygkey" + "0123456789"
PLAN_KEY = "sk-sp-" + "fixtureplankey0123456"
WS_KEY = "sk-ws-" + "fixtureworkspacekey0123"
WS_HOST = "ws-fixture01.ap-southeast-1.maas.aliyuncs.com"
WS_URL = f"https://{WS_HOST}/compatible-mode/v1"
LOCAL_ID = "qwen3.8-27b-fixture-q4"

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
USAGE = {"prompt_tokens": 1200, "completion_tokens": 80, "total_tokens": 1280}


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in spend.SECRET_ENV:
        monkeypatch.delenv(name, raising=False)
    # The canonical ledger is ~/distill/spend-ledger.csv: HOME points it (and its STOP) into tmp_path.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("LEDGER", raising=False)


# ── the fake API ────────────────────────────────────────────────────────────


def chat_body(text: str, *, model: str = MAX, usage: dict[str, Any] | None = USAGE) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-fixture", "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


class FakeAPI:
    """An OpenAI-compatible server on loopback. Each POST pops the next scripted step:
    ``{"body": ...}`` / ``{"status": 429, "body": ..., "headers": {...}}`` / ``{"raw": b"..."}``
    / ``{"action": "drop"}`` (read the request, close without a reply) / ``{"action": "hang"}``."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.model_id = LOCAL_ID
        self.default: dict[str, Any] = {"body": chat_body(PASS_REPLY)}
        api = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def _reply(self, status: int, payload: Any, headers: dict[str, str] | None = None) -> None:
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                api.requests.append({"method": "GET", "path": self.path, "headers": dict(self.headers)})
                if self.path.endswith("/v1/models"):
                    self._reply(200, {"object": "list", "data": [{"id": api.model_id}]})
                else:
                    self._reply(404, {"error": {"message": "not found"}})

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                api.requests.append({"method": "POST", "path": self.path, "headers": dict(self.headers),
                                     "body": body})
                step = api.steps.pop(0) if api.steps else api.default
                if step.get("action") == "drop":
                    self.close_connection = True
                    return
                if step.get("action") == "hang":
                    time.sleep(step.get("seconds", 1.0))
                    self.close_connection = True
                    return
                self._reply(step.get("status", 200), step["raw"] if "raw" in step else step.get("body"),
                            step.get("headers"))

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def posts(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["method"] == "POST"]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def api() -> Any:
    server = FakeAPI()
    yield server
    server.close()


class ToFake:
    """An opener that sends the paid client's (checked, https) request to the fake server instead."""

    def __init__(self, api: FakeAPI) -> None:
        self.base = api.url
        self.inner = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def open(self, request: urllib.request.Request, timeout: float | None = None) -> Any:
        request.full_url = self.base + urlsplit(request.full_url).path
        return self.inner.open(request, timeout=timeout)


# ── helpers ─────────────────────────────────────────────────────────────────


def write_env(tmp_path: Path, name: str = "payg.env", mode: int = 0o600, **values: str) -> Path:
    path = tmp_path / name
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    path.chmod(mode)
    return path


def creds(tmp_path: Path, **values: str) -> Any:
    return spend.resolve_payg({}, env_file=write_env(tmp_path, **values))


class Notes:
    """A notifier that records instead of running a command."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def __call__(self, command: str, message: str) -> str:
        self.sent.append(message)
        return "sent"


def ledger_path(tmp_path: Path) -> Path:
    """The canonical ledger (HOME is tmp_path)."""
    return tmp_path / "distill" / "spend-ledger.csv"


def ledger_at(tmp_path: Path, notes: Notes | None = None, **kw: Any) -> Any:
    return spend.Ledger(ledger_path(tmp_path), notify_cmd="notify", notifier=notes or Notes(), **kw)


def rows(ledger: Any) -> list[dict[str, str]]:
    return ledger.read_rows()


def kinds(ledger: Any) -> list[str]:
    return [r["note"].split(" ", 1)[0] for r in rows(ledger)]


def reserve(ledger: Any, *, input_tokens: int = 0, output_tokens: int = 0, tool: str = "judge",
            phase: str = "p", budget: float | None = None) -> Any:
    return ledger.reserve(tool=tool, phase=phase, model=MAX, region=SG, input_tokens=input_tokens,
                          output_tokens=output_tokens, phase_budget_usd=budget)


def notify_script(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "notify.sh"
    path.write_text('#!/bin/sh\nd=$(dirname "$0")\necho "$0 $*" >> "$d/notify-argv.txt"\n'
                    'env >> "$d/notify-env.txt"\ncat >> "$d/notify-stdin.txt"\n')
    path.chmod(0o755)
    return path


def make_item(i: int, label: str = "pass", category: str = "coding", *, file: str = "golden_a.jsonl",
              line: int | None = None) -> dict[str, Any]:
    msgs = [
        {"role": "user", "content": f"list the files in the demo folder {i}"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": f"call_{i}", "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": f"ls demo {i}"})}}]},
    ]
    tool = {"type": "function", "function": {"name": "bash", "description": "Run a command.",
                                             "parameters": {"type": "object"}}}
    source: dict[str, Any] = {"telemetry_rowid": i}
    if line is not None:
        source.update({"file": file, "line": line})
    return {"id": f"{label}-{i:04d}", "label": label, "category": category, "stratum_weight": 1.0,
            "source": source, "messages": msgs, "tools": [tool],
            "meta": {"model": "teacher", "tool_name": "bash", "timestamp": 1000.0 + i, "session_id": "beacon:s"},
            "outcome": {"status": "success" if label == "pass" else "error", "text": None}}


def write_queue(work: Path, items: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / judge.QUEUE_FILE).write_text("".join(json.dumps(it) + "\n" for it in items))
    (work / judge.QUEUE_META_FILE).write_text(json.dumps(meta or {}))


# ── the key and its host ─────────────────────────────────────────────────────


def test_a_plain_key_defaults_to_the_singapore_shared_host(tmp_path: Path) -> None:
    c = creds(tmp_path, QWEN_PAYG_KEY=KEY)
    assert (c.region, c.host_kind, c.key_source) == (SG, "shared", "env file")
    assert c.base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    us = spend.resolve_payg({"QWEN_PAYG_KEY": KEY,
                             "QWEN_PAYG_BASE_URL": "https://dashscope-us.aliyuncs.com/compatible-mode/v1/"},
                            env_file=tmp_path / "none.env")
    assert (us.region, us.key_source) == ("us-east-1", "environment")


def test_a_workspace_key_goes_with_its_workspace_host(tmp_path: Path) -> None:
    c = creds(tmp_path, QWEN_PAYG_KEY=WS_KEY, QWEN_PAYG_BASE_URL=WS_URL)
    assert (c.region, c.host_kind) == (SG, "workspace")
    with pytest.raises(spend.KeyRefused, match="needs QWEN_PAYG_BASE_URL"):
        creds(tmp_path, QWEN_PAYG_KEY=WS_KEY)


@pytest.mark.parametrize("key, url, match", [
    (PLAN_KEY, None, "subscription-plan key"),
    (PLAN_KEY, WS_URL, "subscription-plan key"),
    (KEY, "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1", "subscription-plan host"),
    (WS_KEY, "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1", "subscription-plan host"),
    (KEY, "https://coding-intl.dashscope.aliyuncs.com/v1", "subscription-plan host"),
])
def test_the_token_plan_key_and_hosts_are_refused(tmp_path: Path, key: str, url: str | None, match: str) -> None:
    values = {"QWEN_PAYG_KEY": key, **({"QWEN_PAYG_BASE_URL": url} if url else {})}
    with pytest.raises(spend.KeyRefused, match=match):
        creds(tmp_path, **values)


@pytest.mark.parametrize("key, url, match", [
    (WS_KEY, "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "only works on its workspace host"),
    (KEY, WS_URL, "needs its workspace's sk-ws- key"),
    (KEY, "https://ws-x1.mars-1.maas.aliyuncs.com/compatible-mode/v1", "not a Model Studio pay-as-you-go"),
    (KEY, "https://trial.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1", "not a Model Studio pay-as-you-go"),
    (KEY, "https://api.example.com/compatible-mode/v1", "not a Model Studio pay-as-you-go"),
    (KEY, "http://dashscope-intl.aliyuncs.com/compatible-mode/v1", "https"),
    (KEY, "https://dashscope-intl.aliyuncs.com/v1", "compatible-mode"),
    (KEY, "https://dashscope-intl.aliyuncs.com:8443/compatible-mode/v1", "port"),
    (KEY, "https://u:p@dashscope-intl.aliyuncs.com/compatible-mode/v1", "credentials"),
    ("pk-" + "fixture0123456789", None, "not a Model Studio key"),
    ("sk-has space in it", None, "not a Model Studio key"),
])
def test_the_key_prefix_must_match_the_host(tmp_path: Path, key: str, url: str | None, match: str) -> None:
    values = {"QWEN_PAYG_KEY": key, **({"QWEN_PAYG_BASE_URL": url} if url else {})}
    with pytest.raises(spend.KeyRefused, match=match):
        creds(tmp_path, **values)


def test_a_payg_key_equal_to_the_token_plan_key_is_refused(tmp_path: Path) -> None:
    with pytest.raises(spend.KeyRefused, match="same key as QWEN_API_KEY"):
        creds(tmp_path, QWEN_PAYG_KEY=KEY, QWEN_API_KEY=KEY)
    path = write_env(tmp_path, name="file.env", QWEN_API_KEY=KEY)
    with pytest.raises(spend.KeyRefused, match="same key as QWEN_API_KEY"):
        spend.resolve_payg({"QWEN_PAYG_KEY": KEY}, env_file=path)
    with pytest.raises(spend.KeyRefused, match="same key as QWEN_API_KEY"):
        spend.resolve_payg({"QWEN_PAYG_KEY": KEY, "QWEN_API_KEY": KEY}, env_file=tmp_path / "none.env")
    # A different Token Plan key beside it is fine.
    assert creds(tmp_path, QWEN_PAYG_KEY=KEY, QWEN_API_KEY=PLAN_KEY).region == SG


def test_a_missing_key_or_a_readable_env_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(spend.KeyRefused, match="QWEN_PAYG_KEY is not set"):
        spend.resolve_payg({}, env_file=tmp_path / "none.env")
    with pytest.raises(spend.KeyRefused, match="QWEN_PAYG_KEY is not set"):
        creds(tmp_path, QWEN_PAYG_KEY="")
    path = write_env(tmp_path, name="open.env", mode=0o644, QWEN_PAYG_KEY=KEY)
    with pytest.raises(spend.KeyRefused, match="chmod 600"):
        spend.resolve_payg({}, env_file=path)
    # Prometheus's own resolution of the env file path.
    assert spend.env_file_path({"PROMETHEUS_ENV_FILE": str(path)}) == path
    assert spend.env_file_path({"XDG_CONFIG_HOME": str(tmp_path)}) == tmp_path / "prometheus" / "env"


def test_credentials_never_show_the_key_or_the_workspace_host(tmp_path: Path) -> None:
    c = creds(tmp_path, QWEN_PAYG_KEY=WS_KEY, QWEN_PAYG_BASE_URL=WS_URL)
    for text in (repr(c), str(c), c.describe()):
        assert WS_KEY not in text and WS_HOST not in text
    assert c.scrub(f"bad key {WS_KEY} at {WS_URL}") == "bad key <key> at https://<host>/compatible-mode/v1"


# ── prices, usage, payloads ──────────────────────────────────────────────────


def test_the_price_table_is_keyed_by_model_and_region_and_refuses_the_unknown() -> None:
    p = spend.price_for(MAX, SG)
    assert (p.input_per_m, p.cached_input_per_m, p.output_per_m) == (2.0, 2.0, 6.0)
    assert p.source.startswith("https://www.alibabacloud.com/help/en/model-studio/") and p.checked
    assert spend.price_for(MAX, "us-east-1").output_per_m == 6.0   # the higher of its two scopes
    assert spend.price_for(MAX, "eu-central-1").input_per_m == 1.65
    for model, region in ((MAX, "mars-1"), ("qwen3.8-flash", SG), ("qwen3.8-max-latest", SG)):
        with pytest.raises(spend.PriceUnknown):
            spend.price_for(model, region)
    assert p.usd(1_000_000, 0, 1_000_000) == pytest.approx(8.0)
    with pytest.raises(spend.PriceUnknown, match="beyond the priced tier"):
        p.usd(1_000_001, 0, 0)
    assert spend.highest_price(MAX).output_per_m == 6.0


def test_cached_input_is_priced_at_its_own_rate() -> None:
    p = spend.Price(model="m", region="r", input_per_m=2.0, cached_input_per_m=0.5, output_per_m=6.0,
                    max_input_tokens=10**6, scope="x")
    assert p.usd(1000, 400, 100) == pytest.approx((600 * 2.0 + 400 * 0.5 + 100 * 6.0) / 1e6)
    assert p.usd(100, 400, 0) == pytest.approx(100 * 0.5 / 1e6)   # cached never exceeds the input


@pytest.mark.parametrize("usage, out, cached, note", [
    ({"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500,
      "completion_tokens_details": {"reasoning_tokens": 300}}, 500, 0, ""),          # included
    ({"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1500,
      "completion_tokens_details": {"reasoning_tokens": 300}}, 500, 0, "output-from-total"),
    ({"prompt_tokens": 1000, "completion_tokens": 200,
      "completion_tokens_details": {"reasoning_tokens": 300}}, 500, 0, "reasoning-added"),
    ({"prompt_tokens": 1000, "completion_tokens": 80, "prompt_tokens_details": {"cached_tokens": 600}},
     80, 600, ""),
    ({"input_tokens": 1000, "output_tokens": 80, "cached_tokens": 5000}, 80, 1000, ""),
])
def test_usage_counts_reasoning_tokens_as_output(usage: dict[str, Any], out: int, cached: int,
                                                 note: str) -> None:
    u = spend.parse_usage(usage)
    assert (u.input_tokens, u.cached_input_tokens, u.output_tokens, u.note) == (1000, cached, out, note)


def test_unusable_usage_is_none() -> None:
    for bad in (None, {}, {"prompt_tokens": 10}, {"prompt_tokens": -1, "completion_tokens": 2},
                {"prompt_tokens": True, "completion_tokens": 2}, "usage"):
        assert spend.parse_usage(bad) is None


def test_a_request_must_bound_its_own_cost() -> None:
    ok = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100, "enable_thinking": False}
    assert spend.worst_case_tokens(ok)[1] == 100
    thinking = {**ok, "enable_thinking": True, "thinking_budget": 400}
    assert spend.worst_case_tokens(thinking)[1] == 500
    tools = {**ok, "tools": [{"type": "function", "function": {"name": "bash"}}]}
    assert spend.worst_case_tokens(tools)[0] >= spend.worst_case_tokens(ok)[0] + spend.TOOLS_PREAMBLE_TOKENS
    for bad, match in (
        ({k: v for k, v in ok.items() if k != "max_tokens"}, "max_tokens"),
        ({k: v for k, v in ok.items() if k != "enable_thinking"}, "enable_thinking"),
        ({**ok, "enable_thinking": True}, "thinking_budget"),
        ({**ok, "stream": True}, "streaming"),
        ({**ok, "n": 2}, "n must be 1"),
        ({**ok, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:,"}}]}]}, "only text"),
        # Options whose cost the worst case does not cover: search adds billed input and a fee.
        ({**ok, "enable_search": True}, "enable_search are not supported"),
        ({**ok, "search_options": {"forced_search": True}}, "search_options are not supported"),
        ({**ok, "plugins": {"code_interpreter": {}}}, "plugins are not supported"),
        ({**ok, "max_completion_tokens": 10}, "max_completion_tokens are not supported"),
        ({**ok, "tools": [{"type": "web_search"}]}, "function tools"),
        ({**ok, "tools": {"type": "function"}}, "function tools"),
    ):
        with pytest.raises(spend.SpendRefused, match=match):
            spend.worst_case_tokens(bad)


# ── the ledger: cap, sub-budget, alert, STOP, lock ──────────────────────────


def test_the_cap_refuses_a_reservation_that_would_cross_it_and_writes_nothing(tmp_path: Path) -> None:
    notes = Notes()
    ledger = ledger_at(tmp_path, notes, cap_usd=1.0, alert_usd=1.0)
    held = [reserve(ledger, input_tokens=100_000) for _ in range(4)]   # $0.20 each
    assert ledger.totals().committed_usd == pytest.approx(0.8)
    before = len(rows(ledger))
    with pytest.raises(spend.CapRefused, match="could cost up to \\$0.3000"):
        reserve(ledger, input_tokens=150_000)                           # $0.30: would be $1.10
    assert len(rows(ledger)) == before and not ledger.stop_path.exists() and notes.sent == []
    ledger.settle(held[0], usage=spend.Usage(50_000, 0, 0))             # cost $0.10: frees $0.10
    assert ledger.totals().committed_usd == pytest.approx(0.7)
    reserve(ledger, input_tokens=150_000)                               # exactly the cap: allowed
    assert ledger.totals().committed_usd == pytest.approx(1.0)
    assert ledger.stop_path.exists() and len(notes.sent) == 1           # and the alert fired


def test_a_phase_budget_is_per_tool_and_phase(tmp_path: Path) -> None:
    ledger = ledger_at(tmp_path)
    reserve(ledger, input_tokens=100_000, phase="a", budget=0.5)
    reserve(ledger, input_tokens=100_000, phase="a", budget=0.5)
    with pytest.raises(spend.CapRefused, match="judge/a budget of \\$0.50"):
        reserve(ledger, input_tokens=100_000, phase="a", budget=0.5)
    reserve(ledger, input_tokens=100_000, phase="b", budget=0.5)            # another phase
    reserve(ledger, input_tokens=100_000, phase="a", tool="teacher_gen", budget=0.5)  # another tool
    assert ledger.totals().committed_usd == pytest.approx(0.8)


def test_the_alert_stops_every_paid_call_and_notifies_exactly_once(tmp_path: Path) -> None:
    notes = Notes()
    ledger = ledger_at(tmp_path, notes)                                 # $150 cap, $140 alert
    held = [reserve(ledger, output_tokens=1_000_000) for _ in range(23)]   # $6 each: $138
    assert notes.sent == [] and not ledger.stop_path.exists()
    held.append(reserve(ledger, output_tokens=1_000_000))               # $144: crosses $140
    assert ledger.stop_path.exists() and len(notes.sent) == 1
    message = notes.sent[0]
    assert len(message.splitlines()) <= 5 and "STOPPED at $144.00" in message
    assert "spend-ledger.STOP" in message
    with pytest.raises(spend.StopActive):
        reserve(ledger, output_tokens=10)
    other = spend.Ledger(ledger.path, notify_cmd="notify", notifier=notes)   # another tool's process
    with pytest.raises(spend.StopActive):
        other.reserve(tool="teacher_gen", phase="2b", model=MAX, region=SG, input_tokens=10, output_tokens=10)
    for res in held:                                                    # in-flight calls still settle
        ledger.settle(res, usage=spend.Usage(0, 0, 1_000_000))
    assert ledger.totals().spent_usd == pytest.approx(144.0)
    assert kinds(ledger).count("alert") == 1 and len(notes.sent) == 1

    # The operator removes STOP on purpose: calls resume under the hard cap, with no second notify.
    ledger.stop_path.unlink()
    reserve(ledger, output_tokens=1_000_000)                            # $150: the cap itself
    with pytest.raises(spend.CapRefused):
        reserve(ledger, output_tokens=1)
    assert len(notes.sent) == 1 and kinds(ledger).count("alert") == 1


def test_a_call_that_cost_more_than_its_reservation_stops_everything(tmp_path: Path) -> None:
    notes = Notes()
    ledger = ledger_at(tmp_path, notes)
    res = reserve(ledger, input_tokens=1000, output_tokens=10)
    settled = ledger.settle(res, usage=spend.Usage(1000, 0, 100_000))
    assert settled.stopped and "over-reservation" in settled.note
    assert ledger.stop_path.exists() and len(notes.sent) == 1
    assert "more than its" in notes.sent[0]
    with pytest.raises(spend.StopActive):
        reserve(ledger, input_tokens=1)


def test_an_over_reservation_stop_does_not_use_up_the_140_alert(tmp_path: Path) -> None:
    notes = Notes()
    ledger = ledger_at(tmp_path, notes)                                 # $150 cap, $140 alert
    res = reserve(ledger, input_tokens=1000, output_tokens=10)
    ledger.settle(res, usage=spend.Usage(1000, 0, 100_000))            # $0.60 against a $0.0021 hold
    assert ledger.stop_path.exists() and len(notes.sent) == 1 and "more than its" in notes.sent[0]
    ledger.stop_path.unlink()                                           # the operator resumes on purpose
    for _ in range(23):
        reserve(ledger, output_tokens=1_000_000)                        # $6 each: $138.60 committed
    assert not ledger.stop_path.exists() and len(notes.sent) == 1
    reserve(ledger, output_tokens=1_000_000)                            # $144.60: crosses $140
    assert ledger.stop_path.exists() and len(notes.sent) == 2 and "STOPPED at $144.60" in notes.sent[1]
    with pytest.raises(spend.StopActive):
        reserve(ledger, output_tokens=1)
    assert [r["note"].split(" ", 2)[1] for r in rows(ledger) if r["note"].startswith("alert")] == [
        "over-reservation", "threshold"]


class FlakyNotes(Notes):
    """A notifier whose first sends fail as scripted, then get through."""

    def __init__(self, results: list[str]) -> None:
        super().__init__()
        self.results = list(results)

    def __call__(self, command: str, message: str) -> str:
        self.sent.append(message)
        return self.results.pop(0) if self.results else "sent"


def test_a_failed_notify_is_sent_again_while_the_stop_stands(tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    notes = FlakyNotes(["failed (exit 1)", "failed (TimeoutExpired)"])
    ledger = ledger_at(tmp_path, notes, cap_usd=1.0, alert_usd=0.5)
    reserve(ledger, input_tokens=300_000)                   # $0.60: crosses the $0.50 alert, notify fails
    assert ledger.stop_path.exists() and len(notes.sent) == 1 and ledger.totals().notify_pending
    with pytest.raises(spend.StopActive, match=r"NOT notified \(notify failed \(TimeoutExpired\)\)"):
        reserve(ledger, input_tokens=1)                     # the second try fails too
    with pytest.raises(spend.StopActive) as info:
        ledger.check_open(tool="judge")                     # the third gets through
    assert "NOT notified" not in str(info.value) and len(notes.sent) == 3
    with pytest.raises(spend.StopActive) as info:
        reserve(ledger, input_tokens=1)                     # told: no more notifies
    assert "NOT notified" not in str(info.value) and len(notes.sent) == 3
    assert not ledger.totals().notify_pending and ledger.totals().alerted

    # A notify that never gets through: the tries stop at NOTIFY_ATTEMPTS, and every refusal,
    # preflight's included, says the operator was not told.
    monkeypatch.setenv("HOME", str(tmp_path / "h2"))
    never = FlakyNotes(["missing"] * 99)
    ledger = spend.Ledger(notify_cmd="notify", notifier=never, cap_usd=1.0, alert_usd=0.5)
    reserve(ledger, input_tokens=300_000)
    env = write_env(tmp_path, QWEN_PAYG_KEY=KEY)
    for _ in range(spend.NOTIFY_ATTEMPTS + 1):
        with pytest.raises(spend.StopActive, match="NOT notified"):
            spend.preflight(ledger=ledger, model=MAX, env={}, env_file=env)
    assert len(never.sent) == spend.NOTIFY_ATTEMPTS
    with pytest.raises(spend.StopActive, match="no more are made"):
        reserve(ledger, input_tokens=1)
    # The operator removes STOP on purpose: the $0.50 alert is recorded, so no new stop or notify.
    ledger.stop_path.unlink()
    reserve(ledger, input_tokens=1000)
    assert not ledger.stop_path.exists() and len(never.sent) == spend.NOTIFY_ATTEMPTS


def test_paid_calls_are_counted_only_in_the_one_ledger(tmp_path: Path, api: FakeAPI) -> None:
    # Another file would be a second, empty $150 with no STOP of its own.
    other = spend.Ledger(tmp_path / "data" / "spend-ledger.csv", notify_cmd=None)
    with pytest.raises(spend.LedgerError, match="counted only in ~/distill/spend-ledger.csv"):
        reserve(other, input_tokens=1)
    with pytest.raises(spend.LedgerError, match="counted only in"):
        other.check_open()
    with pytest.raises(spend.LedgerError, match="counted only in"):
        spend.PaygClient(creds(tmp_path, QWEN_PAYG_KEY=KEY), other, tool="teacher_gen", phase="2b", model=MAX)
    assert not other.path.exists() and api.requests == []
    # The cap and the alert can be lowered, never raised.
    for kw in ({"cap_usd": 300, "alert_usd": 290}, {"cap_usd": 10_000}, {"alert_usd": 145}):
        with pytest.raises(ValueError, match="cannot be raised"):
            spend.Ledger(**kw)
    ledger = spend.Ledger(notify_cmd=None)
    with pytest.raises(ValueError, match="cannot be raised"):
        ledger.cap_usd = 151
    assert ledger.path == ledger_path(tmp_path) and ledger.stop_path == tmp_path / "distill" / "spend-ledger.STOP"
    reserve(ledger, input_tokens=1)
    # No command line names another ledger.
    for argv in (["preflight", "--ledger", str(other.path)], ["status", "--ledger", str(other.path)]):
        with pytest.raises(SystemExit):
            spend.main(argv)


def test_a_reused_call_id_is_refused_and_a_crashed_calls_hold_stays(tmp_path: Path, api: FakeAPI) -> None:
    ledger = ledger_at(tmp_path)
    crashed = ledger.reserve(tool="teacher_gen", phase="2b", model=MAX, region=SG, input_tokens=0,
                             output_tokens=1_000_000, call_id="task-17")   # $6; its process died mid-call
    with pytest.raises(spend.LedgerError, match="already in the ledger"):
        ledger.reserve(tool="teacher_gen", phase="2b", model=MAX, region=SG, input_tokens=0,
                       output_tokens=1000, call_id="task-17")
    client = spend.PaygClient(creds(tmp_path, QWEN_PAYG_KEY=KEY), ledger, tool="teacher_gen", phase="2b",
                              model=MAX, opener=ToFake(api))
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "enable_thinking": False}
    with pytest.raises(spend.LedgerError, match="already in the ledger"):
        client.chat(body, call_id="task-17")
    assert api.requests == []
    assert ledger.totals().open == pytest.approx({"task-17": 6.0})          # the crashed call's hold stays
    ledger.settle(crashed, ok=False, maybe_billed=True, error_kind="interrupted")
    with pytest.raises(spend.LedgerError, match="already in the ledger"):   # nor is a settled id reused
        client.chat(body, call_id="task-17")
    assert client.chat(body).call_id != "task-17"                           # no id: a fresh one
    assert ledger.totals().spent_usd > 6.0 and ledger.totals().open == {}


def test_settling_charges_only_what_may_have_been_billed(tmp_path: Path) -> None:
    ledger = ledger_at(tmp_path)
    a, b, c, d = (reserve(ledger, input_tokens=10_000, output_tokens=1000) for _ in range(4))
    assert ledger.settle(a, ok=False, maybe_billed=True, error_kind="timeout").usd == pytest.approx(a.usd)
    assert ledger.settle(b, ok=False, maybe_billed=False, error_kind="rate_limit").usd == 0.0
    assert ledger.settle(c, ok=True).usd == pytest.approx(c.usd)                 # no usage block
    assert ledger.settle(d, usage=spend.Usage(5000, 0, 100)).usd == pytest.approx((5000 * 2 + 600) / 1e6)
    notes = [r["note"] for r in rows(ledger) if r["note"].startswith("settle")]
    assert notes == ["settle error:timeout charged-reservation", "settle error:rate_limit",
                     "settle ok usage-missing charged-reservation", "settle ok"]
    assert ledger.totals().open == {}
    with pytest.raises(spend.LedgerError, match="no open reservation"):
        ledger.settle(d, ok=True)


def test_only_the_judge_and_teacher_generation_may_spend(tmp_path: Path) -> None:
    ledger = ledger_at(tmp_path)
    for tool in ("daemon", "memory_extractor", "cron", ""):
        with pytest.raises(spend.SpendRefused, match="may not spend"):
            reserve(ledger, input_tokens=1, tool=tool)
    with pytest.raises(spend.SpendRefused, match="may not spend"):
        spend.PaygClient(creds(tmp_path, QWEN_PAYG_KEY=KEY), ledger, tool="daemon", phase="x", model=MAX)
    assert rows(ledger) == []


def test_a_ledger_with_other_columns_is_refused(tmp_path: Path) -> None:
    path = ledger_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("ts,phase,calls,input_tokens,output_tokens,est_usd,quota_units,quota_unit,model,note\n")
    ledger = spend.Ledger(path, notify_cmd=None)
    with pytest.raises(spend.LedgerError, match="columns"):
        reserve(ledger, input_tokens=1)
    path.write_text(",".join(spend.LEDGER_FIELDS) + "\nt,judge,p,m,c,0,0,0,lots,0,reserve\n")
    with pytest.raises(spend.LedgerError, match="not a number"):
        ledger.totals()


def test_a_busy_ledger_lock_times_out(tmp_path: Path) -> None:
    holder = ledger_at(tmp_path)
    waiter = ledger_at(tmp_path, lock_timeout=0.2)
    with holder.locked():
        t0 = time.monotonic()
        with pytest.raises(spend.LedgerBusy):
            reserve(waiter, input_tokens=1)
        assert time.monotonic() - t0 >= 0.2
    reserve(waiter, input_tokens=1)


WORKER = r"""
import importlib.util, sys
spec = importlib.util.spec_from_file_location("distill_spend", sys.argv[1])
m = importlib.util.module_from_spec(spec); sys.modules["distill_spend"] = m; spec.loader.exec_module(m)
ledger = m.Ledger(sys.argv[2], cap_usd=float(sys.argv[4]), alert_usd=float(sys.argv[4]), notify_cmd=None)
ok = 0
for _ in range(int(sys.argv[3])):
    try:
        r = ledger.reserve(tool=sys.argv[5], phase="p", model="qwen3.8-max", region="ap-southeast-1",
                           input_tokens=10_000, output_tokens=0)
    except m.SpendRefused:
        continue
    ledger.settle(r, usage=m.Usage(5_000, 0, 0))
    ok += 1
print(ok)
"""


def test_concurrent_tools_share_one_ledger_under_its_lock(tmp_path: Path) -> None:
    path = ledger_path(tmp_path)
    cap, n = 0.5, 30   # reserve $0.02, settle $0.01: about 50 calls fit
    procs = [subprocess.Popen([sys.executable, "-c", WORKER, str(_DIR / "spend.py"), str(path), str(n),
                               str(cap), tool], stdout=subprocess.PIPE, text=True)
             for tool in ("judge", "teacher_gen")]
    done: list[int] = []

    def thread_worker(tool: str) -> None:
        ledger = spend.Ledger(path, cap_usd=cap, alert_usd=cap, notify_cmd=None)
        ok = 0
        for _ in range(n):
            try:
                r = ledger.reserve(tool=tool, phase="p", model=MAX, region=SG, input_tokens=10_000, output_tokens=0)
            except spend.SpendRefused:
                continue
            ledger.settle(r, usage=spend.Usage(5_000, 0, 0))
            ok += 1
        done.append(ok)

    threads = [threading.Thread(target=thread_worker, args=(t,)) for t in ("judge", "teacher_gen")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    outs = [int(p.communicate(timeout=120)[0].strip()) for p in procs]
    ok_total = sum(outs) + sum(done)

    ledger = spend.Ledger(path, cap_usd=cap, alert_usd=cap, notify_cmd=None)
    all_rows = rows(ledger)
    # Replaying the rows in order reproduces every row's cumulative total: no writer appended on a
    # stale read, and the committed total never went above the cap.
    open_: dict[str, float] = {}
    spent = 0.0
    for r in all_rows:
        kind, usd = r["note"].split(" ", 1)[0], float(r["usd"])
        if kind == "reserve":
            open_[r["call_id"]] = usd
        elif kind == "settle":
            open_.pop(r["call_id"])
            spent += usd
        assert float(r["cumulative_usd"]) == pytest.approx(spent + sum(open_.values()), abs=1e-6)
        assert float(r["cumulative_usd"]) <= cap + 1e-9
    assert open_ == {}
    assert kinds(ledger).count("settle") == ok_total
    assert [r["note"].split(" ", 1)[1].startswith("threshold") for r in all_rows
            if r["note"].startswith("alert ")].count(True) == 1   # one $0.50 alert; the rest are renotify tries
    assert spent == pytest.approx(0.01 * ok_total)
    assert 0 < ok_total < 4 * n


# ── the notify command ───────────────────────────────────────────────────────


def test_the_notify_command_gets_five_lines_on_stdin_and_no_secrets(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QWEN_PAYG_KEY", KEY)
    monkeypatch.setenv("QWEN_API_KEY", PLAN_KEY)
    monkeypatch.setenv("QWEN_PAYG_BASE_URL", WS_URL)
    script = notify_script(tmp_path / "n")
    message = "\n".join(f"line {i}" for i in range(9))
    assert spend.run_notify(str(script), message) == "sent"
    assert (tmp_path / "n" / "notify-stdin.txt").read_text() == "\n".join(f"line {i}" for i in range(5)) + "\n"
    assert (tmp_path / "n" / "notify-argv.txt").read_text().split()[1:] == ["5"]
    env_dump = (tmp_path / "n" / "notify-env.txt").read_text()
    assert KEY not in env_dump and PLAN_KEY not in env_dump and WS_HOST not in env_dump
    assert spend.run_notify(str(tmp_path / "missing.sh"), "x") == "missing"


# ── the paid client against the fake API ─────────────────────────────────────


def test_a_paid_call_reserves_then_settles_from_its_usage(tmp_path: Path, api: FakeAPI) -> None:
    ledger = ledger_at(tmp_path)
    client = spend.PaygClient(creds(tmp_path, QWEN_PAYG_KEY=KEY), ledger, tool="teacher_gen", phase="2b",
                              model=MAX, opener=ToFake(api))
    api.steps.append({"body": chat_body("hello", usage={
        "prompt_tokens": 2000, "completion_tokens": 300, "total_tokens": 2800,
        "prompt_tokens_details": {"cached_tokens": 500}, "completion_tokens_details": {"reasoning_tokens": 500}})})
    payload = {"messages": [{"role": "user", "content": "word " * 2000}], "max_tokens": 600,
               "enable_thinking": True, "thinking_budget": 400}   # ~10K bytes: a bound above 2000 tokens
    r = client.chat(payload)
    assert (r.text, r.served_model, r.output_tokens, r.reasoning_tokens) == ("hello", MAX, 800, 500)
    assert r.usd == pytest.approx((1500 * 2 + 500 * 2 + 800 * 6) / 1e6)
    reserve_row, settle_row = rows(ledger)
    assert reserve_row["note"] == f"reserve {SG}" and reserve_row["output_tokens"] == "1000"
    assert (settle_row["input_tokens"], settle_row["cached_input_tokens"], settle_row["output_tokens"]) == (
        "2000", "500", "800")
    assert settle_row["note"] == "settle ok output-from-total"
    assert float(settle_row["cumulative_usd"]) == pytest.approx(r.usd)
    (req,) = api.posts()
    assert req["path"] == "/compatible-mode/v1/chat/completions"
    assert req["headers"]["Authorization"] == f"Bearer {KEY}"
    assert req["body"]["model"] == MAX and req["body"]["thinking_budget"] == 400


FAILURES = [
    ({"status": 429, "body": {"error": {"code": "Throttling", "message": "slow down"}},
      "headers": {"Retry-After": "7"}}, "rate_limit", False, 7.0),
    # A 5xx or 408 can come after the model ran (a gateway timeout on a long generation): charged.
    ({"status": 503, "body": {"error": {"message": "busy"}}}, "provider_error", True, None),
    ({"status": 504, "body": {"error": {"message": "gateway timeout"}}}, "provider_error", True, None),
    ({"status": 500, "raw": b"<html>oops</html>"}, "provider_error", True, None),
    ({"status": 408, "body": {"error": {"message": "request timeout"}}}, "timeout", True, None),
    ({"status": 401, "body": {"error": {"code": "invalid_api_key", "message": "bad key"}}}, "auth", False, None),
    ({"status": 400, "body": {"code": "Arrearage", "message": "account overdue"}}, "billing", False, None),
    ({"status": 400, "body": {"error": {"code": "invalid_parameter", "message": "no"}}}, "bad_request", False, None),
    ({"action": "drop"}, "unreachable", True, None),
    ({"action": "hang", "seconds": 1.0}, "timeout", True, None),
    ({"status": 200, "raw": b"not json"}, "provider_error", True, None),
    ({"status": 200, "body": {"model": MAX, "usage": USAGE}}, "provider_error", True, None),
]


@pytest.mark.parametrize("step, kind, billed, retry_after", FAILURES)
def test_a_failed_call_is_charged_only_when_it_may_have_been_billed(
        tmp_path: Path, api: FakeAPI, step: dict[str, Any], kind: str, billed: bool,
        retry_after: float | None) -> None:
    ledger = ledger_at(tmp_path)
    client = spend.PaygClient(creds(tmp_path, QWEN_PAYG_KEY=KEY), ledger, tool="judge", phase="2-judge",
                              model=MAX, timeout=0.3, opener=ToFake(api))
    api.steps.append(step)
    with pytest.raises(spend.PaidCallError) as info:
        client.chat({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100, "enable_thinking": False})
    err = info.value
    assert (err.kind, err.maybe_billed, err.retry_after) == (kind, billed, retry_after)
    reserved = float(rows(ledger)[0]["usd"])
    if "usage" in (step.get("body") or {}):   # a broken reply that still reported usage: billed by it
        assert err.usd == pytest.approx(spend.price_for(MAX, SG).usd(1200, 0, 80))
    else:
        assert err.usd == pytest.approx(reserved if billed else 0.0)
    assert ledger.totals().open == {} and ledger.totals().committed_usd == pytest.approx(err.usd)


def test_a_reply_without_usage_is_charged_its_reservation(tmp_path: Path, api: FakeAPI) -> None:
    ledger = ledger_at(tmp_path)
    client = spend.PaygClient(creds(tmp_path, QWEN_PAYG_KEY=KEY), ledger, tool="judge", phase="p",
                              model=MAX, opener=ToFake(api))
    api.steps.append({"body": chat_body("ok", usage=None)})
    r = client.chat({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100, "enable_thinking": False})
    assert r.usage_estimated and r.usd == pytest.approx(float(rows(ledger)[0]["usd"]))


def test_a_connection_refused_costs_nothing(tmp_path: Path) -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    class Closed:
        def open(self, request: Any, timeout: float | None = None) -> Any:
            request.full_url = f"http://127.0.0.1:{port}/x"
            return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout)

    ledger = ledger_at(tmp_path)
    client = spend.PaygClient(creds(tmp_path, QWEN_PAYG_KEY=KEY), ledger, tool="judge", phase="p",
                              model=MAX, opener=Closed())
    with pytest.raises(spend.PaidCallError) as info:
        client.chat({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "enable_thinking": False})
    assert (info.value.kind, info.value.maybe_billed, info.value.usd) == ("unreachable", False, 0.0)


def test_a_refused_call_sends_nothing(tmp_path: Path, api: FakeAPI) -> None:
    ledger = ledger_at(tmp_path, cap_usd=0.001, alert_usd=0.001)
    client = spend.PaygClient(creds(tmp_path, QWEN_PAYG_KEY=KEY), ledger, tool="judge", phase="p",
                              model=MAX, opener=ToFake(api))
    with pytest.raises(spend.CapRefused):
        client.chat({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1000, "enable_thinking": False})
    ledger.stop_path.write_text("stopped\n")
    with pytest.raises(spend.StopActive):
        client.chat({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "enable_thinking": False})
    with pytest.raises(spend.SpendRefused, match="priced for"):
        client.chat({"model": "qwen3.8-flash", "messages": [{"role": "user", "content": "x"}], "max_tokens": 1,
                     "enable_thinking": False})
    assert api.requests == [] and rows(ledger) == []


def test_the_default_opener_follows_no_redirect(api: FakeAPI) -> None:
    api.steps.append({"status": 302, "body": {}, "headers": {"Location": api.url + "/elsewhere"}})
    with pytest.raises(urllib.error.HTTPError) as info:
        spend.default_opener().open(urllib.request.Request(api.url + "/x", data=b"{}", method="POST"), timeout=5)
    assert info.value.code == 302
    assert [r["path"] for r in api.requests] == ["/x"]


# ── judge.py with the paid provider ─────────────────────────────────────────


def _paid_args(tmp_path: Path, work: Path, *extra: str, env: Path | None = None) -> list[str]:
    return ["--workdir", str(work), "run", "--provider", "max-payg", "--budget-usd", "5",
            "--notify-cmd", str(notify_script(tmp_path / "n")),
            "--env-file", str(env or write_env(tmp_path, QWEN_PAYG_KEY=KEY)), "--backoff-base", "0", *extra]


def test_judge_max_payg_end_to_end(tmp_path: Path, api: FakeAPI, capsys: pytest.CaptureFixture[str]) -> None:
    work = tmp_path / "phase2"
    write_queue(work, [make_item(i, line=i) for i in range(10, 13)] + [make_item(13, "fail")])
    api.steps += [{"status": 429, "body": {"error": {"message": "slow"}}}, {"action": "drop"},
                  {"body": chat_body(PASS_REPLY)}, {"body": chat_body(FAIL_REPLY)},
                  {"body": chat_body(PASS_REPLY)}, {"body": chat_body(FAIL_REPLY)}]
    code = judge.main(_paid_args(tmp_path, work, "--concurrency", "1"), opener=ToFake(api))
    assert code == judge.EXIT_DONE
    out = capsys.readouterr().out
    projection = json.loads(out[: out.index("\n}\n") + 2])
    assert projection["provider"] == "max-payg" and projection["phase"] == judge.PHASE
    assert projection["projected"]["items"] == 4 and projection["projected"]["usd_worst_case"] > 0
    assert "Singapore" in projection["endpoint"]

    results = judge.read_jsonl(work / judge.RESULTS_FILE)
    assert [r["status"] for r in results] == ["ok"] * 4
    assert {(r["provider"], r["judge_key"], r["model"], r["region"]) for r in results} == {
        ("max-payg", "payg", MAX, SG)}
    assert results[0]["attempts"] == 3   # a 429, a dropped connection, then the verdict
    ledger = spend.Ledger(ledger_path(tmp_path), notify_cmd=None)
    settles = [r for r in rows(ledger) if r["note"].startswith("settle")]
    assert [r["note"] for r in settles[:2]] == ["settle error:rate_limit",
                                                "settle error:unreachable charged-reservation"]
    assert {(r["tool"], r["phase"]) for r in settles} == {("judge", judge.PHASE)}
    assert sum(r["cost_usd"] for r in results) == pytest.approx(
        sum(float(r["usd"]) for r in settles[2:]), abs=1e-5)
    assert all(p["body"]["temperature"] == 0 and p["body"]["enable_thinking"] is False for p in api.posts())
    status = json.loads((work / judge.STATUS_FILE).read_text())
    assert status["state"] == "done" and "of $150.00" in status["budget"]


def test_the_judge_stops_at_the_alert_and_so_does_every_other_paid_tool(tmp_path: Path, api: FakeAPI) -> None:
    c = creds(tmp_path, QWEN_PAYG_KEY=KEY)
    notes = Notes()
    ledger = ledger_at(tmp_path, notes)
    client = judge.PaygJudgeClient(ledger=ledger, phase=judge.PHASE, budget_usd=10, creds=c, opener=ToFake(api))
    items = [make_item(i) for i in range(10, 16)]   # two-digit ids: every prompt is the same size
    cfg = judge.RunConfig(jitter=0.0)
    user, _ = judge.render_prompt(items[0], view="outcome")
    worst = client.worst_case_usd(judge.system_prompt("outcome"), user, cfg.max_tokens)
    actual = client.price.usd(USAGE["prompt_tokens"], 0, USAGE["completion_tokens"])
    # Two settled calls plus the third's reservation cross the alert; the third call still goes.
    ledger.cap_usd, ledger.alert_usd = 1.0, 2 * actual + worst - 1e-7
    run = judge.JudgeRun(items, client=client, results_path=tmp_path / judge.RESULTS_FILE,
                         status_path=tmp_path / judge.STATUS_FILE, config=cfg)
    outcome = asyncio.run(run.run())
    assert outcome.state == "budget_stop" and outcome.exit_code == judge.EXIT_BUDGET
    assert len(api.posts()) == 3 and outcome.judged == 3
    assert ledger.stop_path.exists() and len(notes.sent) == 1
    teacher = spend.PaygClient(c, spend.Ledger(ledger.path, notify_cmd=None), tool="teacher_gen",
                               phase="2b-teacher", model=MAX, opener=ToFake(api))
    with pytest.raises(spend.StopActive):
        teacher.chat({"messages": [{"role": "user", "content": "x"}], "max_tokens": 10, "enable_thinking": False})
    assert len(api.posts()) == 3 and len(notes.sent) == 1


def test_a_sub_budget_stop_leaves_the_rest_for_the_next_run(tmp_path: Path, api: FakeAPI) -> None:
    work = tmp_path / "phase2"
    items = [make_item(i) for i in range(10, 20)]   # two-digit ids: every prompt is the same size
    write_queue(work, items)
    user, _ = judge.render_prompt(items[0], view="outcome")
    body = judge.judge_payload(judge.system_prompt("outcome"), user, model=MAX, max_tokens=1024)
    price = spend.price_for(MAX, SG)
    est_in, est_out = spend.worst_case_tokens(body)
    worst = price.usd(est_in, 0, est_out)
    actual = price.usd(USAGE["prompt_tokens"], 0, USAGE["completion_tokens"])
    budget = worst + 2.5 * actual   # room for three calls: the fourth's reservation would cross it
    args = _paid_args(tmp_path, work, "--concurrency", "1")
    args[args.index("--budget-usd") + 1] = f"{budget:.6f}"
    code = judge.main(args, opener=ToFake(api))
    assert code == judge.EXIT_BUDGET
    status = json.loads((work / judge.STATUS_FILE).read_text())
    assert status["state"] == "budget_stop" and "judge/2-judge budget of" in status["reason"]
    assert len(api.posts()) == 3 and status["remaining"] == 7
    assert not spend.Ledger(ledger_path(tmp_path)).stop_path.exists()   # a sub-budget is not STOP


def test_the_blind_fail_side_check_has_its_own_budget(tmp_path: Path, api: FakeAPI) -> None:
    work = tmp_path / "phase2"
    write_queue(work, [make_item(i) for i in range(10, 12)] + [make_item(i, "fail") for i in range(12, 14)])
    ledger = ledger_at(tmp_path)
    # The outcome run has spent $6 of 2-judge: more than the blind check's own $5.
    held = ledger.reserve(tool="judge", phase=judge.PHASE, model=MAX, region=SG, input_tokens=0,
                          output_tokens=1_000_000)
    ledger.settle(held, usage=spend.Usage(0, 0, 1_000_000))
    api.default = {"body": chat_body(FAIL_REPLY)}
    assert judge.main(_paid_args(tmp_path, work, "--view", "blind", "--only", "fail"),
                      opener=ToFake(api)) == judge.EXIT_DONE   # --budget-usd 5
    assert len(api.posts()) == 2
    settles = [r for r in rows(ledger) if r["note"].startswith("settle")][1:]
    assert len(settles) == 2 and {r["phase"] for r in settles} == {judge.PHASE_BLIND}
    # The same $5 for the outcome side is refused: 2-judge has spent $6 of it.
    assert judge.main(_paid_args(tmp_path, work, "--only", "pass"), opener=ToFake(api)) == judge.EXIT_BUDGET
    assert len(api.posts()) == 2


def test_judge_estimate_prices_without_calling(tmp_path: Path, api: FakeAPI,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    work = tmp_path / "phase2"
    write_queue(work, [make_item(i) for i in range(10, 14)])
    assert judge.main(["--workdir", str(work), "estimate", "--provider", "max-payg",
                       "--env-file", str(tmp_path / "none.env")]) == judge.EXIT_DONE
    out = json.loads(capsys.readouterr().out)
    assert out["projected"]["items"] == 4 and "highest region" in out["price_basis"]
    assert out["ledger"]["committed_usd"] == 0 and out["ledger"]["stop"] is None
    assert api.requests == []


# ── judge.py with the local provider ────────────────────────────────────────


def _local_args(tmp_path: Path, work: Path, url: str, *extra: str) -> list[str]:
    return ["--workdir", str(work), "run", "--provider", "local", "--local-url", url,
            "--no-live-probe", "--backoff-base", "0", *extra]


def test_the_local_judge_is_free_counted_and_sends_no_key(tmp_path: Path, api: FakeAPI,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QWEN_PAYG_KEY", KEY)   # present in the environment, never used here
    work = tmp_path / "phase2"
    write_queue(work, [make_item(1, line=0), make_item(2, line=1)])
    ledger = spend.Ledger(ledger_path(tmp_path), notify_cmd=None)
    ledger.stop_path.parent.mkdir(parents=True)
    ledger.stop_path.write_text("stopped\n")   # STOP is for paid calls; the local fallback still runs
    api.default = {"body": chat_body(PASS_REPLY, model=LOCAL_ID)}
    assert judge.main(_local_args(tmp_path, work, api.url)) == judge.EXIT_DONE
    assert [r["path"] for r in api.requests] == ["/v1/models", "/v1/chat/completions", "/v1/chat/completions"]
    assert all("Authorization" not in r["headers"] for r in api.requests)
    assert all(r["body"]["model"] == LOCAL_ID for r in api.posts())
    results = judge.read_jsonl(work / judge.RESULTS_FILE)
    assert {(r["provider"], r["judge_key"], r["model"], r["served_model"]) for r in results} == {
        ("local", "none", LOCAL_ID, LOCAL_ID)}
    assert kinds(ledger) == ["free", "free"]
    assert ledger.totals().committed_usd == 0.0 and ledger.totals().free_calls == 2


@pytest.mark.parametrize("model_id, url, match", [
    ("qwen3.8-max", None, "a Max model is never the local judge"),
    ("other-model-9b", None, "does not match"),
    (LOCAL_ID, "http://192.0.2.10:8080", "loopback"),
    (LOCAL_ID, "http://judge.example.com:8080", "loopback"),
])
def test_the_local_judge_refuses_another_model_or_host(tmp_path: Path, api: FakeAPI, model_id: str,
                                                       url: str | None, match: str,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    work = tmp_path / "phase2"
    write_queue(work, [make_item(1)])
    api.model_id = model_id
    assert judge.main(_local_args(tmp_path, work, url or api.url)) == judge.EXIT_FATAL
    assert match in capsys.readouterr().err
    assert api.posts() == []


def test_a_model_swap_mid_run_stops_the_local_judge(tmp_path: Path, api: FakeAPI) -> None:
    work = tmp_path / "phase2"
    write_queue(work, [make_item(1), make_item(2)])
    api.steps.append({"body": chat_body(PASS_REPLY, model="other-model-9b")})
    assert judge.main(_local_args(tmp_path, work, api.url)) == judge.EXIT_FATAL
    assert json.loads((work / judge.STATUS_FILE).read_text())["state"] == "fatal"
    assert len(api.posts()) == 1


# ── verdict rows in the shape build.py reads ─────────────────────────────────


def test_verdict_rows_are_in_the_shape_build_reads(tmp_path: Path, api: FakeAPI) -> None:
    build = _load("distill_build", "build.py")
    items = [make_item(1, line=0), make_item(2, line=1), make_item(3, "fail")]
    api.steps += [{"body": chat_body(PASS_REPLY, model=LOCAL_ID)}, {"body": chat_body(FAIL_REPLY, model=LOCAL_ID)},
                  {"body": chat_body(FAIL_REPLY, model=LOCAL_ID)},
                  {"body": chat_body(PASS_REPLY)}, {"body": chat_body(FAIL_REPLY)}, {"body": chat_body(FAIL_REPLY)}]
    local, paid = tmp_path / "local", tmp_path / "paid"
    write_queue(local, items)
    write_queue(paid, items)
    assert judge.main(_local_args(tmp_path, local, api.url)) == judge.EXIT_DONE
    assert judge.main(_paid_args(tmp_path, paid, "--concurrency", "1"), opener=ToFake(api)) == judge.EXIT_DONE
    pass_key, fail_key = "golden_a.jsonl:1", "golden_a.jsonl:2"

    stats: collections.Counter[str] = collections.Counter()
    checks: dict[str, Any] = {}
    verdicts = build.load_verdicts([str(local / judge.RESULTS_FILE)], stats, checks)
    assert verdicts == {pass_key: "pass", fail_key: "fail"}
    assert checks == {pass_key: 1, fail_key: 2}
    assert stats["verdict_rows_unusable"] == 0 and stats["verdict_rows_not_ok"] == 0
    assert stats["verdict_rows_max_judge"] == 0 and stats["verdict_rows_fail_side"] == 1
    rows_local = judge.read_jsonl(local / judge.RESULTS_FILE)
    got = [build.verification_of({"verification": r["verification"]}) for r in rows_local[:2]]
    assert [(v.decision, v.basis) for v in got] == [(True, "judge-pass"), (False, "not_verified")]

    stats = collections.Counter()
    verdicts = build.load_verdicts([str(paid / judge.RESULTS_FILE)], stats, {})
    assert stats["verdict_rows_unusable"] == 0 and stats["verdict_rows_not_ok"] == 0
    if stats["verdict_rows_max_judge"]:
        # build.py's rule "the Max key never judges" (written for the Token Plan): a Max judge's
        # pass is not applied, its fail is. Applying PAYG Max verdicts is build.py's policy call.
        assert verdicts == {fail_key: "fail"}
    else:
        assert verdicts == {pass_key: "pass", fail_key: "fail"}
    rows_paid = judge.read_jsonl(paid / judge.RESULTS_FILE)
    assert all(r["verification"]["judge"]["provider"] == "max-payg" for r in rows_paid)


# ── run-all ───────────────────────────────────────────────────────────────


def _phase2_with_report(tmp_path: Path, decision: str) -> tuple[Path, Path, Path]:
    traj = tmp_path / "traj"
    traj.mkdir()
    index, lines = [], []
    for i, cat in enumerate(["coding"] * 6 + ["other"] * 3):
        trace = {"messages": make_item(i)["messages"], "tools": make_item(i)["tools"],
                 "_meta": {"model": "teacher", "tool_name": "bash", "timestamp": 1000.0 + i, "session_id": "s"}}
        lines.append(json.dumps(trace))
        index.append({"file": "golden_a.jsonl", "model": "teacher", "tool": "bash", "ts": 1000.0 + i,
                      "h_prompt_call": f"k{i}", "category": cat, "rowid": i + 1})
    (traj / "golden_a.jsonl").write_text("\n".join(lines) + "\n")
    idx = tmp_path / "index.jsonl"
    idx.write_text("".join(json.dumps(r) + "\n" for r in index))
    phase2 = tmp_path / "phase2"
    phase2.mkdir()
    judged = [p["id"] for p in judge.pass_pool(index, "teacher")][:2]
    (phase2 / judge.RESULTS_FILE).write_text("".join(json.dumps(
        {"id": i, "status": "ok", "view": "outcome", "rubric_version": judge.RUBRIC_VERSION, "verdict": "pass"})
        + "\n" for i in judged))
    (phase2 / judge.REPORT_FILE).write_text(json.dumps({
        "view": "outcome", "rubric_version": judge.RUBRIC_VERSION,
        "gate": {"decision": decision, "rejudge_priority": ["other", "coding"]}}))
    return phase2, idx, traj


def test_run_all_rejudges_the_worst_category_first_under_its_own_phase(tmp_path: Path, api: FakeAPI) -> None:
    phase2, idx, traj = _phase2_with_report(tmp_path, "re-judge")
    work = tmp_path / "phase2-rejudge"
    args = ["--workdir", str(work), "run-all", "--rejudge-from", str(phase2), "--index", str(idx),
            "--trajectories", str(traj), "--telemetry", str(tmp_path / "none.db"), "--model", "teacher",
            "--provider", "local", "--local-url", api.url, "--no-live-probe"]
    api.default = {"body": chat_body(PASS_REPLY, model=LOCAL_ID)}
    assert judge.main(args) == judge.EXIT_DONE
    results = judge.read_jsonl(work / judge.RESULTS_FILE)
    cats = [r["category"] for r in results]
    assert len(results) == 7 and cats == sorted(cats, key=["other", "coding"].index)
    assert {r["phase"] for r in results} == {judge.PHASE_ALL} and {r["view"] for r in results} == {"outcome"}
    assert judge.main(args) == judge.EXIT_DONE   # resumable: nothing left
    assert len(api.posts()) == 7


def test_run_all_does_nothing_when_the_gate_trusted_the_labels(tmp_path: Path, api: FakeAPI) -> None:
    phase2, idx, traj = _phase2_with_report(tmp_path, "trust-labels")
    work = tmp_path / "phase2-rejudge"
    assert judge.main(["--workdir", str(work), "run-all", "--rejudge-from", str(phase2), "--index", str(idx),
                       "--provider", "local", "--local-url", api.url, "--no-live-probe"]) == judge.EXIT_DONE
    assert not work.exists() and api.requests == []


# ── no key anywhere ──────────────────────────────────────────────────────────


def test_the_key_never_reaches_logs_output_files_or_argv(tmp_path: Path, api: FakeAPI,
                                                         caplog: pytest.LogCaptureFixture,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    caplog.set_level(logging.DEBUG)
    (tmp_path / "secrets").mkdir()
    env = write_env(tmp_path / "secrets", QWEN_PAYG_KEY=WS_KEY, QWEN_PAYG_BASE_URL=WS_URL)
    work = tmp_path / "phase2"
    write_queue(work, [make_item(i, line=i) for i in range(10, 13)])
    echo = {"error": {"code": "invalid_api_key", "message": f"Incorrect API key provided: {WS_KEY} for {WS_URL}"}}
    api.steps += [{"status": 429, "body": echo}, {"action": "drop"}, {"status": 200, "raw": b"<html>"},
                  {"body": chat_body(PASS_REPLY)}, {"status": 401, "body": echo}]
    argv = _paid_args(tmp_path, work, "--concurrency", "1", env=env)
    assert judge.main(argv, opener=ToFake(api)) == judge.EXIT_FATAL
    status = json.loads((work / judge.STATUS_FILE).read_text())
    assert status["state"] == "fatal" and "<key>" in status["reason"]
    # A preflight and an estimate on the same key.
    assert spend.main(["preflight", "--env-file", str(env),
                       "--notify-cmd", str(tmp_path / "n" / "notify.sh")]) == 0
    assert judge.main(["--workdir", str(work), "estimate", "--provider", "max-payg",
                       "--env-file", str(env)]) == judge.EXIT_DONE
    captured = capsys.readouterr()
    seen = [captured.out, captured.err, caplog.text, " ".join(argv)]
    for path in tmp_path.rglob("*"):
        if path.is_file() and path != env:
            seen.append(path.read_bytes().decode("utf-8", "replace"))
    for text in seen:
        assert WS_KEY not in text and WS_HOST not in text
    assert any(r["headers"].get("Authorization") == f"Bearer {WS_KEY}" for r in api.posts())


# ── judge_run.sh ──────────────────────────────────────────────────────────────


def _run_sh(tmp_path: Path, *args: str, key: str = KEY,
            extra_env: dict[str, str] | None = None) -> tuple[subprocess.CompletedProcess[str], str]:
    distill = tmp_path / "distill"
    notify_script(distill)
    env_path = write_env(tmp_path, name="prom.env", QWEN_PAYG_KEY=key)
    wrapper = tmp_path / "py-wrapper.sh"   # records every interpreter command line, then runs it
    wrapper.write_text(f'#!/bin/sh\necho "$@" >> "{tmp_path}/argv.log"\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "PY": str(wrapper),
           "DISTILL": str(distill), "PROMETHEUS_ENV_FILE": str(env_path), "LC_ALL": "C", **(extra_env or {})}
    cp = subprocess.run(["bash", str(_DIR / "judge_run.sh"), *args], env=env, capture_output=True,
                        text=True, timeout=180, check=False)
    argv_log = (tmp_path / "argv.log").read_text() if (tmp_path / "argv.log").exists() else ""
    return cp, argv_log


def test_judge_run_sh_is_a_dry_run_that_preflights_and_prints_the_projection(tmp_path: Path) -> None:
    cp, argv_log = _run_sh(tmp_path)
    assert cp.returncode == 0, cp.stderr
    assert "no queue" in cp.stdout and "nothing was sent" in cp.stdout
    write_queue(tmp_path / "distill" / "phase2", [make_item(i) for i in range(10, 13)])
    cp, argv_log = _run_sh(tmp_path, "--rejudge")
    assert cp.returncode == 0, cp.stderr
    assert '"usd_worst_case"' in cp.stdout and "pass --execute to run" in cp.stdout
    assert "run-all" in cp.stdout
    assert "estimate --provider max-payg" in argv_log and " run " not in argv_log
    assert "phase 2-judge-blind" in cp.stdout and "--budget-usd 5)" in cp.stdout
    assert not (tmp_path / "distill" / "phase2" / judge.RESULTS_FILE).exists()
    for text in (cp.stdout, cp.stderr, argv_log):
        assert KEY not in text


@pytest.mark.parametrize("setup, match", [
    ("plan-key", "subscription-plan key"),
    ("stop", "paid calls are stopped"),
    ("no-notify", "notify command"),
])
def test_judge_run_sh_preflight_refuses_before_anything_else(tmp_path: Path, setup: str, match: str) -> None:
    distill = tmp_path / "distill"
    write_queue(distill / "phase2", [make_item(10)])
    if setup == "stop":
        (distill / "spend-ledger.STOP").write_text("2026-10-31T00:00:00Z distill spend STOP: test\n")
    cp, argv_log = _run_sh(
        tmp_path, key=PLAN_KEY if setup == "plan-key" else KEY,
        extra_env={"NOTIFY": str(tmp_path / "missing-notify.sh")} if setup == "no-notify" else None)
    assert cp.returncode == 2
    assert match in cp.stderr
    assert "estimate" not in argv_log and "projected" not in cp.stdout
    for text in (cp.stdout, cp.stderr, argv_log):
        assert KEY not in text and PLAN_KEY not in text


def test_judge_run_sh_has_no_second_ledger(tmp_path: Path) -> None:
    data = tmp_path / "data"
    notify_script(data)
    write_queue(data / "phase2", [make_item(10)])
    cp, argv_log = _run_sh(tmp_path, extra_env={"LEDGER": str(data / "spend-ledger.csv")})
    assert cp.returncode == 2 and "fixed at ~/distill/spend-ledger.csv" in cp.stderr and argv_log == ""
    # DISTILL moves the workdirs and the notify command, not the ledger or its STOP file.
    stop = tmp_path / "distill" / "spend-ledger.STOP"
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.write_text("2026-10-31T00:00:00Z distill spend STOP: test\n")
    cp, argv_log = _run_sh(tmp_path, extra_env={"DISTILL": str(data)})
    assert cp.returncode == 2 and "paid calls are stopped" in cp.stderr
    assert "estimate" not in argv_log and not (data / "spend-ledger.csv").exists()


def test_judge_run_sh_gets_past_an_item_that_never_gets_a_verdict(tmp_path: Path) -> None:
    work = tmp_path / "distill" / "phase2"
    items = [make_item(i) for i in range(10, 14)] + [make_item(i, "fail") for i in range(14, 16)]
    write_queue(work, items, {"pass_pool": {"coding": 4}})
    # Every item has its verdict but pass-0013, which the API rejects every time.
    (work / judge.RESULTS_FILE).write_text("".join(json.dumps({
        "id": it["id"], "label": it["label"], "category": it["category"], "status": "ok", "view": "outcome",
        "rubric_version": judge.RUBRIC_VERSION, "verdict": it["label"],
        "rubric": {k: 2 for k in judge.RUBRIC_KEYS}}) + "\n" for it in items if it["id"] != "pass-0013"))
    # A stand-in for the paid `judge.py run`: the outcome run ends incomplete (6), the blind check done.
    fake = tmp_path / "fake-py.sh"
    fake.write_text(f"""#!/bin/sh
echo "$@" >> "{tmp_path}/argv.log"
case "$*" in
  *judge.py*" run "*"--view blind"*) exit 0 ;;
  *judge.py*" run "*) exit 6 ;;
esac
exec "{sys.executable}" "$@"
""")
    fake.chmod(0o755)
    cp, argv_log = _run_sh(tmp_path, "--execute", extra_env={"PY": str(fake)})
    assert cp.returncode == 6 and "stopped (exit 6)" in cp.stdout
    assert "--view blind" not in argv_log and "report --view outcome --allow-missing 0" in argv_log

    (tmp_path / "argv.log").unlink()
    cp, argv_log = _run_sh(tmp_path, "--execute", "--allow-missing", "1", extra_env={"PY": str(fake)})
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "--budget-usd 5 --view blind --only fail" in argv_log
    assert "report --view outcome --allow-missing 1" in argv_log and "gate: trust-labels" in cp.stdout
    report = json.loads((work / judge.REPORT_FILE).read_text())
    assert report["gate"]["decision"] == "trust-labels"


def test_spend_status_summarises_by_tool_and_phase(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ledger = ledger_at(tmp_path)
    ledger.settle(reserve(ledger, input_tokens=100_000, phase="2-judge"), usage=spend.Usage(50_000, 0, 0))
    reserve(ledger, input_tokens=100_000, tool="teacher_gen", phase="2b-teacher")
    ledger.record_free(tool="judge", phase="2-judge", model=LOCAL_ID)
    assert spend.main(["status"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["spent_usd"] == pytest.approx(0.1) and out["held_usd"] == pytest.approx(0.2)
    assert out["by_tool_phase"]["judge/2-judge"] == {"spent_usd": 0.1, "held_usd": 0.0, "paid_calls": 1,
                                                     "free_calls": 1}
    assert out["by_tool_phase"]["teacher_gen/2b-teacher"]["held_usd"] == pytest.approx(0.2)
