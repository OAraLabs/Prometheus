"""The Token Plan route for distillation batch work (the operator's informed override, 2026-10-31).

spend.py's ``TokenPlanClient`` and its seven guardrails, judge.py ``--provider token-plan``,
teacher_gen.py ``--teacher max-token-plan``, build.py's acceptance of their rows, judge_run.sh's
route choice and trial, and gen_runner.py's route switching and quota parking.

No real API and no real key: fake OpenAI-compatible servers on loopback stand in for Model
Studio (the client's checked https request is re-pointed at them by an injected opener). HOME is
under tmp_path, so the one ledger, its STOP file, the PAUSE file, the state file and the notify
command live there. Fixture keys are fake.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from tests.test_distill_build import feed
from tests.test_distill_gen_runner import Rig, _opt, alive
from tests.test_distill_spend import PASS_REPLY, chat_body, make_item, notify_script, write_queue
from tests.test_distill_teacher_ab import gym_builder
from tests.test_distill_teacher_gen import _load, _restore_state_env, tg  # noqa: F401 - fixtures by name
from tests import test_distill_teacher_gen_payg as payg
from tests.test_distill_teacher_gen_payg import FakeMax, ToFake, inputs, ledger_rows, taskset

#: the payg tests' fixtures, used here by name: HOME (the one ledger, STOP, PAUSE, the notify command)
#: under tmp_path with no ambient key (autouse), and no connection beyond loopback
_home = payg._home
no_outbound = payg.no_outbound

spend = _load("distill_spend", "spend.py")
judge = _load("distill_judge", "judge.py")
build = _load("distill_build", "build.py")

MAX = "qwen3.8-max"
PLAN_KEY = "sk-sp-" + "fixtureplankey0123456"
PLAN_KEY_2 = "sk-sp-" + "fixtureotherkey987654"
PAYG_KEY = "sk-" + "fixturepaygkey" + "0123456789"
TP_URL = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
REF = "operator Q6b 2026-10-30"
_DIR = Path(__file__).resolve().parent.parent / "scripts" / "distill"


# ── fakes and helpers ────────────────────────────────────────────────────────


class StepAPI:
    """Chat completions on loopback. Each POST pops the next step (``status``, ``body``, ``headers``,
    ``delay``); with none left, a plain OK reply. Records every request and the peak in flight."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.in_flight = 0
        self.peak = 0
        self.spans: list[tuple[float, float]] = []
        lock = threading.Lock()
        api = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                with lock:
                    api.requests.append({"headers": dict(self.headers), "body": body})
                    step = api.steps.pop(0) if api.steps else {}
                    api.in_flight += 1
                    api.peak = max(api.peak, api.in_flight)
                t0 = time.monotonic()
                try:
                    time.sleep(step.get("delay", 0.0))
                    payload = step.get("body", chat_body("ok"))
                    data = json.dumps(payload).encode()
                    self.send_response(step.get("status", 200))
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    for k, v in (step.get("headers") or {}).items():
                        self.send_header(k, v)
                    self.end_headers()
                    self.wfile.write(data)
                finally:
                    with lock:
                        api.in_flight -= 1
                        api.spans.append((t0, time.monotonic()))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def api() -> Any:
    a = StepAPI()
    yield a
    a.close()


class ToAPI:
    def __init__(self, api: StepAPI) -> None:
        self.base = api.url
        self.inner = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def open(self, request: urllib.request.Request, timeout: float | None = None) -> Any:
        assert urlsplit(request.full_url).hostname == "token-plan.ap-southeast-1.maas.aliyuncs.com"
        request.full_url = self.base + urlsplit(request.full_url).path
        return self.inner.open(request, timeout=timeout)


class Notes:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def __call__(self, command: str, message: str) -> str:
        self.sent.append(message)
        return "sent"


class Clock:
    def __init__(self, t: float = 1_793_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def env_file(home: Path, **values: str) -> Path:
    path = home / ".config" / "prometheus" / "env"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    path.chmod(0o600)
    return path


def body(text: str = "hello") -> dict[str, Any]:
    return {"model": MAX, "messages": [{"role": "user", "content": text}], "max_tokens": 64,
            "enable_thinking": False, "temperature": 0}


def err(status: int, code: str, message: str, rid: str = "req-fixture-1",
        headers: dict[str, str] | None = None) -> dict[str, Any]:
    return {"status": status, "body": {"error": {"code": code, "message": message}, "request_id": rid},
            "headers": headers or {}}


def make_client(ledger: Any, api: StepAPI, sleeps: list[float] | None = None, **kw: Any) -> Any:
    kw.setdefault("sleep", (sleeps if sleeps is not None else []).append)
    return spend.TokenPlanClient(spend.resolve_token_plan(), ledger, tool="judge", phase="2-judge", model=MAX,
                                 opener=ToAPI(api), **kw)


def tp_rows(home: Path) -> list[dict[str, str]]:
    return [r for r in ledger_rows(home) if r["note"].startswith("token-plan")]


@pytest.fixture
def plan(_home: Path) -> Path:
    """QWEN_API_KEY in the 0600 env file (no QWEN_PAYG_KEY anywhere)."""
    return env_file(_home, QWEN_API_KEY=PLAN_KEY)


# ── the key and its host ─────────────────────────────────────────────────────


def test_the_token_plan_key_and_host_are_checked_and_the_payg_key_wins(_home: Path) -> None:
    env_file(_home, QWEN_API_KEY=PLAN_KEY)
    c = spend.resolve_token_plan()
    assert (c.region, c.key_source, c.base_url) == ("ap-southeast-1", "env file", TP_URL)
    assert PLAN_KEY not in repr(c) and "Singapore" in c.describe()
    env_file(_home, QWEN_API_KEY=PLAN_KEY, QWEN_BASE_URL="https://token-plan.cn-beijing.maas.aliyuncs.com/"
             "compatible-mode/v1")
    assert spend.resolve_token_plan().region == "cn-beijing"
    for values, match in (
            ({}, "QWEN_API_KEY is not set"),
            ({"QWEN_API_KEY": PAYG_KEY}, "not a Token Plan key"),
            ({"QWEN_API_KEY": PLAN_KEY, "QWEN_BASE_URL": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"},
             "not a Token Plan host"),
            ({"QWEN_API_KEY": PLAN_KEY, "QWEN_BASE_URL": "http://token-plan.ap-southeast-1.maas.aliyuncs.com/"
                                                         "compatible-mode/v1"}, "https"),
            ({"QWEN_API_KEY": PLAN_KEY, "QWEN_BASE_URL": "https://token-plan.evil.example/compatible-mode/v1"},
             "not a Token Plan host")):
        env_file(_home, **values)
        with pytest.raises(spend.KeyRefused, match=match):
            spend.resolve_token_plan()
    # guardrail 6: QWEN_PAYG_KEY present (env file or environment) -> the Token Plan is not used
    env_file(_home, QWEN_API_KEY=PLAN_KEY, QWEN_PAYG_KEY=PAYG_KEY)
    with pytest.raises(spend.TokenPlanSuperseded):
        spend.resolve_token_plan()
    assert spend.which_key() == "payg"
    env_file(_home, QWEN_API_KEY=PLAN_KEY)
    assert spend.which_key() == "token-plan"
    assert spend.which_key({"QWEN_PAYG_KEY": PAYG_KEY}) == "payg"
    with pytest.raises(spend.TokenPlanSuperseded):
        spend.resolve_token_plan({"QWEN_PAYG_KEY": PAYG_KEY})
    env_file(_home)
    assert spend.which_key() == "none"


@pytest.mark.parametrize("status, code, message, kind", [
    (401, "InvalidApiKey", "Invalid API-key provided.", "stop"),
    (403, "AccessDenied.Unpurchased", "You have not purchased this service.", "stop"),
    (403, "", "", "stop"),
    (400, "Arrearage", "Access denied, please make sure your account is in good standing.", "stop"),
    (429, "Throttling", "Request blocked by risk control.", "stop"),
    (400, "InvalidParameter", "This request violates the usage policy of the plan.", "stop"),
    (200, "", "The subscription has been suspended.", "stop"),
    (400, "DataInspectionFailed", "Input data may contain inappropriate content.", "bad_request"),
    (400, "DataInspectionFailed", "Repeated violations: this account is suspended.", "stop"),
    (429, "Throttling.RateQuota", "Requests rate limit exceeded, please try again later.", "rate_limit"),
    (429, "", "Too many requests", "rate_limit"),
    (429, "Throttling.Quota", "Your token plan quota has been exhausted; it resets at 2026-11-01 00:00:00.",
     "quota"),
    # Model Studio's TPM/TPS limit is a rate limit, never an exhausted plan or a STOP ("billing" on a 429)
    (429, "Throttling.AllocationQuota", "Allocated quota exceeded, please increase your quota limit.",
     "rate_limit"),
    (429, "insufficient_quota", "You exceeded your current quota, please check your plan and billing details.",
     "rate_limit"),
    (429, "Throttling.AllocationQuota", "Free allocated quota exceeded.", "quota"),
    (429, "AllocationQuota.FreeTierOnly", "The free tier of the model has been exhausted.", "quota"),
    (400, "InvalidParameter", "The billing account is not active.", "stop"),
    # any 401/403 is a STOP, DataInspectionFailed included
    (403, "data_inspection_failed", "Input data may contain inappropriate content.", "stop"),
    (401, "DataInspectionFailed", "Output data may contain inappropriate content.", "stop"),
    (500, "InternalError", "oops", "provider_error"),
    (408, "", "timeout", "timeout"),
    (400, "InvalidParameter", "max_tokens is too large", "bad_request"),
])
def test_the_classifier(status: int, code: str, message: str, kind: str) -> None:
    assert spend.tp_classify(status, code, message) == kind


def test_reset_times_from_a_quota_reply() -> None:
    now = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc).timestamp()
    ts, how = spend.tp_reset_time("resets at 2026-11-01 00:00:00", None, now)
    assert ts == datetime(2026, 10, 31, 16, 0, tzinfo=timezone.utc).timestamp() and "UTC+8" in how
    ts, _ = spend.tp_reset_time("resets at 2026-11-01T00:00:00Z", None, now)
    assert ts == datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()
    assert spend.tp_reset_time("try again in 3 hours", None, now)[0] == now + 3 * 3600
    assert spend.tp_reset_time("quota exhausted", 7200, now)[0] == now + 7200
    assert spend.tp_reset_time("quota exhausted", 5, now)[0] is None          # a short Retry-After is a rate limit
    assert spend.tp_reset_time("resets at 2020-01-01 00:00:00", None, now)[0] is None   # in the past


# ── one call: $0, logged, the fingerprint ─────────────────────────────────────


def test_a_call_is_a_zero_dollar_ledger_row_with_its_tokens(plan: Path, _home: Path, api: StepAPI) -> None:
    ledger = spend.Ledger(notifier=Notes())
    client = make_client(ledger, api)
    r = client.chat(body())
    assert r.text == "ok" and r.usd == 0.0 and r.input_tokens == 1200
    (row,) = tp_rows(_home)
    assert (row["tool"], row["phase"], row["model"], row["usd"], row["cumulative_usd"]) == (
        "judge", "2-judge", MAX, "0.000000", "")
    assert row["note"] == "token-plan ok" and row["input_tokens"] == "1200" and row["output_tokens"] == "80"
    t = ledger.totals()
    assert (t.committed_usd, t.plan_calls, t.plan_ok) == (0.0, 1, 1)   # never under the PAYG $ cap
    assert api.requests[0]["headers"]["Authorization"] == "Bearer " + PLAN_KEY
    state_path = _home / "distill" / "token-plan-state.json"
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    state = json.loads(state_path.read_text())
    assert state["fingerprint"] and state["salt"] and PLAN_KEY not in state_path.read_text()
    # guardrail 5: batch tooling only
    for tool in ("daemon", "cron", "memory_extractor"):
        with pytest.raises(spend.SpendRefused):
            spend.TokenPlanClient(spend.resolve_token_plan(), ledger, tool=tool, phase="x", model=MAX)
        with pytest.raises(spend.SpendRefused):
            ledger.tp_record(tool=tool, phase="x", model=MAX, call_id="c", outcome="ok")
    with pytest.raises(spend.SpendRefused):
        spend.TokenPlanClient(spend.resolve_token_plan(), ledger, tool="judge", phase="x", model="qwen3.8-plus")


# ── guardrail 2: an account-level error stops everything ─────────────────────


@pytest.mark.parametrize("step", [
    err(401, "InvalidApiKey", "Invalid API-key provided."),
    err(403, "AccessDenied.Unpurchased", "Access to model denied. The plan was not purchased."),
    err(400, "Arrearage", "Access denied, please make sure your account is in good standing."),
    err(429, "Throttling", "Request rejected by risk control."),
    {"status": 200, "body": {"error": {"code": "Forbidden", "message": "subscription suspended"},
                             "request_id": "req-fixture-1"}},
    err(400, "DataInspectionFailed", "This account is suspended after repeated violations."),
])
def test_an_account_level_error_writes_stop_notifies_the_exact_error_and_never_retries(
        plan: Path, _home: Path, api: StepAPI, step: dict[str, Any]) -> None:
    notes = Notes()
    ledger = spend.Ledger(notifier=notes)
    client = make_client(ledger, api)
    api.steps.append(step)
    with pytest.raises(spend.TokenPlanStop):
        client.chat(body())
    assert len(api.requests) == 1                                       # no retry around it
    code, message = step["body"]["error"]["code"], step["body"]["error"]["message"]
    stop = (_home / "distill" / "spend-ledger.STOP").read_text()
    assert f"HTTP {step['status']}" in stop and code in stop and message in stop and "req-fixture-1" in stop
    (note,) = notes.sent
    assert "Token Plan batch use STOPPED" in note and f"HTTP {step['status']}" in note
    assert code in note and message in note and "req-fixture-1" in note and PLAN_KEY not in note
    assert len(note.splitlines()) <= 5
    # every later call refuses (no request), on either route
    with pytest.raises(spend.StopActive):
        client.chat(body())
    assert len(api.requests) == 1
    with pytest.raises(spend.StopActive):
        ledger.reserve(tool="judge", phase="p", model=MAX, region="ap-southeast-1", input_tokens=1,
                       output_tokens=1)
    with pytest.raises(spend.StopActive):
        spend.tokenplan_preflight(ledger=ledger, model=MAX)
    assert [r["note"].split(" ")[1] for r in tp_rows(_home)] == ["error:stop"]
    alerts = [r for r in ledger_rows(_home) if r["note"].startswith("alert token-plan-stop")]
    assert len(alerts) == 1 and "notify:sent" in alerts[0]["note"]
    assert json.loads((_home / "distill" / "token-plan-state.json").read_text())["last_errors"][-1]["code"] == code


def test_per_request_errors_are_not_a_stop(plan: Path, _home: Path, api: StepAPI) -> None:
    ledger = spend.Ledger(notifier=Notes())
    client = make_client(ledger, api)
    api.steps += [err(400, "DataInspectionFailed", "Input data may contain inappropriate content."),
                  err(500, "InternalError", "try later")]
    with pytest.raises(spend.PaidCallError) as e1:
        client.chat(body())
    with pytest.raises(spend.PaidCallError) as e2:
        client.chat(body())
    assert (e1.value.kind, e2.value.kind) == ("bad_request", "provider_error")
    assert not (_home / "distill" / "spend-ledger.STOP").exists()
    assert client.chat(body()).text == "ok"


def test_a_key_echoed_in_an_error_is_scrubbed_everywhere(plan: Path, _home: Path, api: StepAPI,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    notes = Notes()
    ledger = spend.Ledger(notifier=notes)
    client = make_client(ledger, api)
    api.steps.append(err(401, "InvalidApiKey", f"Incorrect API key provided: {PLAN_KEY}"))
    with pytest.raises(spend.TokenPlanStop) as exc:
        client.chat(body())
    assert spend.main(["tokenplan-status"]) == 0
    seen = [str(exc.value), *notes.sent, capsys.readouterr().out]
    seen += [p.read_text() for p in (_home / "distill").iterdir() if p.is_file()]
    assert all(PLAN_KEY not in s for s in seen) and "<key>" in notes.sent[0]


# ── guardrail 7: a key change ────────────────────────────────────────────────


def test_a_different_key_is_a_stop_at_start_or_mid_run(plan: Path, _home: Path, api: StepAPI,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    notes = Notes()
    ledger = spend.Ledger(notifier=notes)
    client = make_client(ledger, api)                                    # records the fingerprint
    assert client.chat(body()).text == "ok"
    time.sleep(0.01)
    env_file(_home, QWEN_API_KEY=PLAN_KEY_2)                             # mid-run: the env file changes
    with pytest.raises(spend.TokenPlanStop, match="changed mid-run"):
        client.chat(body())
    assert len(api.requests) == 1 and "key changed" in notes.sent[-1]
    (_home / "distill" / "spend-ledger.STOP").unlink()
    with pytest.raises(spend.TokenPlanStop, match="changed since its first use"):
        make_client(ledger, api)                                         # at a client's start
    assert "fingerprint mismatch" in notes.sent[-1] and PLAN_KEY_2 not in notes.sent[-1]
    assert (_home / "distill" / "spend-ledger.STOP").exists()
    # the operator accepts the new key on purpose; the STOP file stays until they remove it
    assert spend.main(["tokenplan-accept-key"]) == 0
    assert (_home / "distill" / "spend-ledger.STOP").exists()
    (_home / "distill" / "spend-ledger.STOP").unlink()
    assert make_client(ledger, api).chat(body()).text == "ok"
    assert all(PLAN_KEY_2 not in s for s in notes.sent + [capsys.readouterr().out])


# ── guardrail 6: the pay-per-token key appears ────────────────────────────────


def test_the_payg_key_appearing_stops_token_plan_use_at_the_next_call(plan: Path, _home: Path,
                                                                      api: StepAPI) -> None:
    client = make_client(spend.Ledger(notifier=Notes()), api)
    assert client.chat(body()).text == "ok"
    time.sleep(0.01)
    env_file(_home, QWEN_API_KEY=PLAN_KEY, QWEN_PAYG_KEY=PAYG_KEY)
    with pytest.raises(spend.TokenPlanSuperseded):
        client.chat(body())
    assert len(api.requests) == 1
    assert not (_home / "distill" / "spend-ledger.STOP").exists()        # a switch, not a stop
    assert spend.main(["tokenplan-preflight"]) == 3


# ── guardrail 1: backoff and adaptive concurrency ─────────────────────────────


def test_a_rate_limit_backs_off_honours_retry_after_and_a_burst_halves_concurrency(
        plan: Path, _home: Path, api: StepAPI) -> None:
    sleeps: list[float] = []
    client = make_client(spend.Ledger(notifier=Notes()), api, sleeps, rng=__import__("random").Random(1))
    assert client.limiter.limit == spend.TP_START_CONCURRENCY == 4
    rl = "Requests rate limit exceeded, please try again later."
    api.steps += [err(429, "Throttling.RateQuota", rl, headers={"Retry-After": "30"}),
                  err(429, "Throttling.RateQuota", rl), err(429, "Throttling.RateQuota", rl)]
    assert client.chat(body()).text == "ok"
    assert len(api.requests) == 4
    assert sleeps[0] >= 30 and 10 <= sleeps[1] <= 11 and 20 <= sleeps[2] <= 22   # 5 s doubling, 10% jitter
    assert client.limiter.limit == 2                                     # 3 in a minute: halved
    assert [r["note"].split(" ")[1] for r in tp_rows(_home)] == ["error:rate_limit"] * 3 + ["ok"]
    assert not (_home / "distill" / "spend-ledger.STOP").exists()
    limits = json.loads((_home / "distill" / "token-plan-state.json").read_text())["limits"]
    assert limits["judge/2-judge"]["limit"] == 2
    # out of retries: a rate_limit error for the caller's own backoff
    client = make_client(spend.Ledger(notifier=Notes()), api, retries_429=0)
    api.steps.append(err(429, "Throttling", rl))
    with pytest.raises(spend.PaidCallError) as exc:
        client.chat(body())
    assert exc.value.kind == "rate_limit" and exc.value.status == 429


def test_the_limiter_halves_on_a_burst_and_grows_back_slowly() -> None:
    now = [0.0]
    lim = spend.AdaptiveLimiter(4, 8, clock=lambda: now[0])
    assert not lim.on_rate_limit() and not lim.on_rate_limit()
    now[0] = 61.0                                                        # the first two fall out of the window
    assert not lim.on_rate_limit() and lim.limit == 4
    assert not lim.on_rate_limit() and lim.on_rate_limit() and lim.limit == 2
    for _ in range(spend.TP_GROW_AFTER_OK - 1):
        assert not lim.on_success()
    assert not lim.on_success()                                          # 50 clean, but within 5 min
    now[0] += spend.TP_GROW_MIN_INTERVAL_S
    assert lim.on_success() and lim.limit == 3                           # the streak held: +1 once
    assert not any(lim.on_success() for _ in range(spend.TP_GROW_AFTER_OK - 1))
    for _ in range(20):
        now[0] += spend.TP_GROW_MIN_INTERVAL_S
        for _ in range(spend.TP_GROW_AFTER_OK):
            lim.on_success()
    assert lim.limit == 8                                                # never above the max
    for _ in range(12):
        lim.on_rate_limit()
    assert lim.limit == 1                                                # never below one
    with pytest.raises(ValueError):
        spend.AdaptiveLimiter(4, 9)


def test_calls_in_flight_never_exceed_the_limit(plan: Path, _home: Path, api: StepAPI) -> None:
    client = make_client(spend.Ledger(notifier=Notes()), api, start_concurrency=2, max_concurrency=8)
    api.steps += [{"delay": 0.2, "body": chat_body("ok")} for _ in range(6)]
    threads = [threading.Thread(target=client.chat, args=(body(f"q{i}"),)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert len(api.requests) == 6 and api.peak == 2


# ── guardrail 3: quota exhausted ─────────────────────────────────────────────


def test_an_exhausted_quota_pauses_until_the_reset_notifies_once_then_one_probe(
        plan: Path, _home: Path, api: StepAPI) -> None:
    notes = Notes()
    reset = datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()
    clock = Clock(reset - 7200)
    ledger = spend.Ledger(notifier=notes, clock=clock)
    client = make_client(ledger, api)
    api.steps.append(err(429, "Throttling.Quota", "Token plan quota exhausted. Resets at 2026-11-01T00:00:00Z."))
    with pytest.raises(spend.QuotaPaused) as exc:
        client.chat(body())
    assert exc.value.until == reset
    pause_path = _home / "distill" / "token-plan.PAUSE"
    pause = json.loads(pause_path.read_text())
    assert pause["until"] == reset and "Token plan quota exhausted" in pause["reason"] and pause["notify"] == "sent"
    assert not (_home / "distill" / "spend-ledger.STOP").exists()        # a pause, never a STOP
    (note,) = notes.sent
    assert "quota exhausted" in note and "PAUSED" in note and "2026-11-01T00:00:00Z" in note
    # paused: refused locally, no request, no second notify
    for _ in range(3):
        with pytest.raises(spend.QuotaPaused):
            client.chat(body())
    assert len(api.requests) == 1 and len(notes.sent) == 1
    with pytest.raises(spend.QuotaPaused):
        spend.tokenplan_preflight(ledger=ledger, model=MAX)
    # the reset passes: one probe call; still exhausted -> paused again (1 h: no time given), no notify
    clock.t = reset + 1
    api.steps.append(err(429, "Throttling.Quota", "Token plan quota exhausted."))
    with pytest.raises(spend.QuotaPaused) as exc:
        client.chat(body())
    assert exc.value.until == clock.t + spend.TP_QUOTA_DEFAULT_PAUSE_S and len(notes.sent) == 1
    assert len(api.requests) == 2 and json.loads(pause_path.read_text())["probes"] == 1
    # the next reset: the probe succeeds and the pause is gone
    clock.t += spend.TP_QUOTA_DEFAULT_PAUSE_S + 1
    assert client.chat(body()).text == "ok" and not pause_path.exists()
    assert client.chat(body()).text == "ok"
    outcomes = [r["note"] for r in tp_rows(_home)]
    assert outcomes == ["token-plan error:quota http:429", "token-plan error:quota http:429 probe",
                        "token-plan ok probe", "token-plan ok"]
    assert len([r for r in ledger_rows(_home) if r["note"].startswith("alert token-plan-pause")]) == 1


def test_only_one_probe_goes_out_across_threads_and_processes(plan: Path, _home: Path, api: StepAPI) -> None:
    clock = Clock()
    ledger = spend.Ledger(notifier=Notes(), clock=clock)
    pause_path = _home / "distill" / "token-plan.PAUSE"
    pause_path.write_text(json.dumps({"until": clock.t - 1, "reason": "x", "notify": "sent",
                                      "probe": {"owner": "999:other", "claimed_at": clock.t - 10}}))
    client = make_client(ledger, api)
    with pytest.raises(spend.QuotaPaused, match="another process's probe"):
        client.chat(body())                                              # a fresh claim by another process
    assert not api.requests
    pause_path.write_text(json.dumps({"until": clock.t - 1, "reason": "x", "notify": "sent", "probe": None}))
    api.steps += [{"delay": 0.4, "body": chat_body("probe")}, {"body": chat_body("after")}]
    results: list[str] = []
    threads = [threading.Thread(target=lambda: results.append(client.chat(body()).text)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert sorted(results) == ["after", "probe"] and api.peak == 1       # the second waited for the probe
    assert not pause_path.exists()


# ── regressions: a STOP is final, written first, told once; waits apart from the request timeout ──


def _stop_lines(home: Path) -> list[str]:
    path = home / "distill" / "spend-ledger.STOP"
    return [ln for ln in path.read_text().splitlines() if "STOP: token-plan" in ln] if path.exists() else []


def test_threads_waiting_on_the_probe_never_send_after_it_wrote_a_stop(plan: Path, _home: Path,
                                                                        api: StepAPI) -> None:
    notes = Notes()
    clock = Clock()
    ledger = spend.Ledger(notifier=notes, clock=clock)
    (_home / "distill").mkdir(parents=True, exist_ok=True)
    (_home / "distill" / "token-plan.PAUSE").write_text(json.dumps(
        {"until": clock.t - 1, "reason": "x", "notify": "sent", "probe": None}))
    client = make_client(ledger, api)
    api.steps.append(dict(err(401, "InvalidApiKey", "Invalid API-key provided."), delay=0.4))
    api.steps += [{"body": chat_body("must not be sent")} for _ in range(4)]
    raised: list[str] = []

    def one() -> None:
        try:
            client.chat(body())
        except spend.SpendError as exc:
            raised.append(type(exc).__name__)

    threads = [threading.Thread(target=one) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert len(api.requests) == 1                                        # only the probe went out
    assert len(raised) == 4 and all(name in ("TokenPlanStop", "StopActive") for name in raised)
    assert len(_stop_lines(_home)) == 1 and len(notes.sent) == 1


def test_concurrent_account_errors_notify_once(plan: Path, _home: Path, api: StepAPI) -> None:
    notes = Notes()
    ledger = spend.Ledger(notifier=notes)
    client = make_client(ledger, api, start_concurrency=4)
    api.steps += [dict(err(403, "AccessDenied", "Access denied: the subscription is suspended."), delay=0.3)
                  for _ in range(4)]
    threads = [threading.Thread(target=lambda: pytest.raises(spend.StopActive, client.chat, body()))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert len(api.requests) == 4                                        # all four were in flight together
    assert len(notes.sent) == 1                                          # told once per STOP file
    assert len(_stop_lines(_home)) == 4                                  # each hit still adds its line
    assert len([r for r in tp_rows(_home) if "error:stop" in r["note"]]) == 4
    # the operator removes the STOP file: a later STOP is a new event, told again
    (_home / "distill" / "spend-ledger.STOP").unlink()
    api.steps.append(err(401, "InvalidApiKey", "Invalid API-key provided."))
    with pytest.raises(spend.TokenPlanStop):
        make_client(ledger, api).chat(body())
    assert len(notes.sent) == 2


def test_a_stop_is_written_and_told_even_when_its_ledger_row_fails(plan: Path, _home: Path, api: StepAPI,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    notes = Notes()
    ledger = spend.Ledger(notifier=notes)
    client = make_client(ledger, api)

    def busy(**_kw: Any) -> None:
        raise spend.LedgerBusy("the ledger lock stayed busy")

    monkeypatch.setattr(ledger, "tp_record", busy)
    api.steps.append(err(401, "InvalidApiKey", "Invalid API-key provided."))
    with pytest.raises(spend.TokenPlanStop):
        client.chat(body())
    assert len(_stop_lines(_home)) == 1 and len(notes.sent) == 1
    # the lock itself busy: the STOP line is still written (outside the lock)
    monkeypatch.setattr(ledger, "locked", lambda: (_ for _ in ()).throw(spend.LedgerBusy("busy")))
    assert ledger._tp_append_stop_line("x", "y") == ""
    with pytest.raises(spend.LedgerBusy):
        ledger.tp_stop(reason="again", exact="HTTP 403", tool="judge", phase="p", model=MAX)
    assert len(_stop_lines(_home)) == 3


class _CutBody:
    def read(self, *_a: Any) -> bytes:
        raise ConnectionResetError("connection reset while reading the error body")

    def close(self) -> None:
        pass


class _CutErrorOpener:
    def __init__(self, status: int) -> None:
        self.status = status
        self.calls = 0

    def open(self, request: urllib.request.Request, timeout: float | None = None) -> Any:
        import email.message
        import urllib.error

        self.calls += 1
        raise urllib.error.HTTPError(request.full_url, self.status, "x", email.message.Message(), _CutBody())


@pytest.mark.parametrize("status, exc", [(401, "TokenPlanStop"), (403, "TokenPlanStop")])
def test_an_unreadable_401_or_403_body_is_still_a_stop(plan: Path, _home: Path, status: int, exc: str) -> None:
    notes = Notes()
    opener = _CutErrorOpener(status)
    client = spend.TokenPlanClient(spend.resolve_token_plan(), spend.Ledger(notifier=notes), tool="judge",
                                   phase="2-judge", model=MAX, opener=opener, sleep=lambda _s: None)
    with pytest.raises(getattr(spend, exc)):
        client.chat(body())
    assert opener.calls == 1 and len(_stop_lines(_home)) == 1 and f"HTTP {status}" in notes.sent[0]
    assert [r["note"] for r in tp_rows(_home)] == [f"token-plan error:stop http:{status}"]


def test_an_unreadable_429_body_is_a_rate_limit(plan: Path, _home: Path) -> None:
    opener = _CutErrorOpener(429)
    client = spend.TokenPlanClient(spend.resolve_token_plan(), spend.Ledger(notifier=Notes()), tool="judge",
                                   phase="2-judge", model=MAX, opener=opener, sleep=lambda _s: None,
                                   retries_429=1)
    with pytest.raises(spend.PaidCallError) as e:
        client.chat(body())
    assert e.value.kind == "rate_limit" and opener.calls == 2
    assert not (_home / "distill" / "spend-ledger.STOP").exists()


def test_the_payg_key_supersedes_for_good(plan: Path, _home: Path, api: StepAPI) -> None:
    client = make_client(spend.Ledger(notifier=Notes()), api)
    assert client.chat(body()).text == "ok"
    time.sleep(0.01)
    env_file(_home, QWEN_API_KEY=PLAN_KEY, QWEN_PAYG_KEY=PAYG_KEY)
    for _ in range(3):                                                   # every later call, not just the next
        with pytest.raises(spend.TokenPlanSuperseded):
            client.chat(body())
    assert len(api.requests) == 1
    # a key added in a 0644 rewrite: refused while others can read the file, then seen once it is fixed
    env_file(_home, QWEN_API_KEY=PLAN_KEY)
    client = make_client(spend.Ledger(notifier=Notes()), api)
    time.sleep(0.01)
    path = env_file(_home, QWEN_API_KEY=PLAN_KEY, QWEN_PAYG_KEY=PAYG_KEY)
    path.chmod(0o644)
    with pytest.raises(spend.KeyRefused):
        client.chat(body())
    with pytest.raises(spend.KeyRefused):
        client.chat(body())
    path.chmod(0o600)
    with pytest.raises(spend.TokenPlanSuperseded):
        client.chat(body())
    assert len(api.requests) == 1


def test_slot_waits_and_backoff_do_not_use_the_request_timeout(plan: Path, _home: Path, api: StepAPI) -> None:
    # a request timeout of 1 s, the limit at 1, four callers of 0.4 s each: the last waits ~1.2 s for its slot
    client = make_client(spend.Ledger(notifier=Notes()), api, timeout=1.0, start_concurrency=1, max_concurrency=1)
    api.steps += [{"delay": 0.4, "body": chat_body("ok")} for _ in range(4)]
    results: list[str] = []
    threads = [threading.Thread(target=lambda: results.append(client.chat(body()).text)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert results == ["ok"] * 4 and api.peak == 1
    # 429 backoff (5 s, 10 s, ...) longer than the request timeout still retries
    sleeps: list[float] = []
    client = make_client(spend.Ledger(notifier=Notes()), api, sleeps, timeout=1.0)
    rl = "Requests rate limit exceeded, please try again later."
    api.steps += [err(429, "Throttling.RateQuota", rl), err(429, "Throttling.RateQuota", rl)]
    assert client.chat(body()).text == "ok" and len(sleeps) == 2
    # a wait budget still bounds the wait, as a timeout that sent nothing
    client = make_client(spend.Ledger(notifier=Notes()), api, start_concurrency=1, max_concurrency=1,
                         wait_budget=0.2)
    api.steps.append({"delay": 0.8, "body": chat_body("slow")})
    t = threading.Thread(target=client.chat, args=(body(),))
    t.start()
    time.sleep(0.1)
    n = len(api.requests)
    with pytest.raises(spend.PaidCallError) as e:
        client.chat(body())
    t.join(30)
    assert e.value.kind == "timeout" and not e.value.maybe_billed and len(api.requests) == n
    # close() ends a slot wait
    client = make_client(spend.Ledger(notifier=Notes()), api, start_concurrency=1, max_concurrency=1)
    api.steps.append({"delay": 1.0, "body": chat_body("slow")})
    t = threading.Thread(target=client.chat, args=(body(),))
    t.start()
    time.sleep(0.1)
    threading.Timer(0.2, client.close).start()
    with pytest.raises(spend.SpendRefused, match="closed"):
        client.chat(body())
    t.join(30)


def test_judge_stopped_by_a_signal_records_its_token_plan_calls_in_flight(
        plan: Path, _home: Path, api: StepAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = spend.Ledger(notifier=Notes())
    jc = judge.TokenPlanJudgeClient(ledger=ledger, phase="2-judge", model=MAX, opener=ToAPI(api))
    assert jc.paid is jc._tp
    api.steps.append({"delay": 1.5, "body": chat_body("late")})
    late: list[str] = []
    t = threading.Thread(target=lambda: late.append(jc._tp.chat(body()).text), daemon=True)
    t.start()
    deadline = time.monotonic() + 5
    while not api.requests and time.monotonic() < deadline:
        time.sleep(0.02)
    exits: list[int] = []

    class Exited(BaseException):  # os._exit never returns: nothing may catch it
        pass

    def fake_exit(code: int) -> None:
        exits.append(code)
        raise Exited

    monkeypatch.setattr(judge, "_hard_exit", fake_exit)
    monkeypatch.setattr(judge, "PAID_SIGNAL_WAIT_S", 0.1)
    import signal as _signal

    with pytest.raises(Exited):
        judge.paid_stop_on_signal(jc.paid, _signal.SIGTERM)
    assert exits == [128 + _signal.SIGTERM]
    assert [r["note"] for r in tp_rows(_home)] == ["token-plan error:interrupted"]
    with pytest.raises(spend.SpendRefused, match="closed"):                # no new call after the signal
        jc._tp.chat(body())
    t.join(10)
    assert late == ["late"]                                              # (here the process would be gone)
    # installed only for the paid client, and restored afterwards
    before = _signal.getsignal(_signal.SIGTERM)
    restore = judge.install_paid_signal_handlers(jc.paid)
    assert _signal.getsignal(_signal.SIGTERM) is not before
    restore()
    assert _signal.getsignal(_signal.SIGTERM) == before
    judge.install_paid_signal_handlers(None)()


# ── status and the CLI ───────────────────────────────────────────────────────


def test_tokenplan_status_and_preflight(plan: Path, _home: Path, api: StepAPI,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    notify_script(_home / "distill")
    assert spend.main(["which-key"]) == 0 and capsys.readouterr().out.strip() == "token-plan"
    assert spend.main(["tokenplan-preflight"]) == 0
    pre = json.loads(capsys.readouterr().out)
    assert pre["key_fingerprint"] == "recorded" and pre["concurrency"] == {"start": 4, "max": 8}
    ledger = spend.Ledger(notify_cmd=str(_home / "distill" / "notify.sh"))
    client = make_client(ledger, api)
    api.steps.append(err(500, "InternalError", "flaky"))
    with pytest.raises(spend.PaidCallError):
        client.chat(body())
    for _ in range(3):
        client.chat(body())
    assert spend.main(["tokenplan-status"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["route"] == "token-plan" and st["key"].startswith("present; matches")
    assert st["trial"]["calls"] == 4 and st["trial"]["of"] == 50 and not st["trial"]["done"]
    assert st["trial"]["clean"] is True and st["trial"]["errors"] == {"error:provider_error": 1}
    assert st["usage_7d"]["attempts"] == 4 and st["usage_7d"]["ok"] == 3
    assert st["usage_7d"]["input_tokens"] == 3 * 1200 and st["usage_7d"]["by_tool_phase"] == {"judge/2-judge": 4}
    assert "none: up to 100%" in st["budget"] and st["stop"] is None and st["pause"] is None
    assert st["last_errors"][-1]["code"] == "InternalError" and st["ramp"].startswith("none")
    # preflight exits: 2 STOP, 4 paused, 3 the PAYG key
    stop = _home / "distill" / "spend-ledger.STOP"
    stop.write_text("2026-10-31T00:00:00Z distill spend STOP: test\n")
    assert spend.main(["tokenplan-preflight"]) == 2
    stop.unlink()
    (_home / "distill" / "token-plan.PAUSE").write_text(json.dumps({"until": time.time() + 600, "notify": "sent"}))
    assert spend.main(["tokenplan-preflight"]) == 4
    (_home / "distill" / "token-plan.PAUSE").unlink()
    env_file(_home, QWEN_API_KEY=PLAN_KEY, QWEN_PAYG_KEY=PAYG_KEY)
    assert spend.main(["tokenplan-preflight"]) == 3
    assert spend.main(["which-key"]) == 0 and "payg" in capsys.readouterr().out
    assert PLAN_KEY not in capsys.readouterr().err


# ── judge.py --provider token-plan ───────────────────────────────────────────


def _tp_judge_args(home: Path, work: Path, *extra: str) -> list[str]:
    return ["--workdir", str(work), "run", "--provider", "token-plan", "--notify-cmd",
            str(notify_script(home / "n")), "--backoff-base", "0", "--no-live-probe", *extra]


def test_judge_on_the_token_plan_end_to_end(plan: Path, _home: Path, api: StepAPI,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    work = _home / "phase2"
    write_queue(work, [make_item(i, line=i) for i in range(10, 13)] + [make_item(13, "fail")])
    api.steps.append(err(429, "Throttling.RateQuota", "Requests rate limit exceeded"))
    api.steps += [{"body": chat_body(PASS_REPLY)} for _ in range(4)]
    assert judge.main(_tp_judge_args(_home, work), opener=ToAPI(api)) == judge.EXIT_DONE
    out = capsys.readouterr().out
    head = json.loads(out[: out.index("\n}\n") + 2])
    assert head["provider"] == "max-token-plan" and head["projected"]["usd_worst_case"] == 0
    results = judge.read_jsonl(work / judge.RESULTS_FILE)
    assert [r["status"] for r in results] == ["ok"] * 4
    assert {(r["provider"], r["judge_key"], r["model"], r["cost_usd"]) for r in results} == {
        ("max-token-plan", "token-plan", MAX, 0.0)}
    assert all(p["headers"]["Authorization"] == "Bearer " + PLAN_KEY for p in api.requests)
    notes = [r["note"] for r in tp_rows(_home)]
    assert notes.count("token-plan ok") == 4 and notes.count("token-plan error:rate_limit http:429") == 1
    assert not [r for r in ledger_rows(_home) if r["note"].startswith(("reserve", "settle"))]
    assert "Token Plan" in json.loads((work / judge.STATUS_FILE).read_text())["budget"]
    # no $ budget on this route; at most 8 workers
    fresh = _home / "phase2-b"
    write_queue(fresh, [make_item(20, line=20)])
    assert judge.main(_tp_judge_args(_home, fresh, "--budget-usd", "5")) == judge.EXIT_FATAL
    assert judge.main(_tp_judge_args(_home, fresh, "--concurrency", "9")) == judge.EXIT_FATAL
    assert len(api.requests) == 5


@pytest.mark.parametrize("step, code, state", [
    (err(403, "AccessDenied", "Access denied."), judge.EXIT_FATAL, "fatal"),
    (err(429, "Throttling.Quota", "Token plan quota exhausted."), judge.EXIT_PAUSED, "paused"),
])
def test_judge_stops_on_an_account_error_and_pauses_on_the_quota(plan: Path, _home: Path, api: StepAPI,
                                                                 step: dict, code: int, state: str) -> None:
    work = _home / "phase2"
    write_queue(work, [make_item(i, line=i) for i in range(10, 14)])
    api.steps.append(step)
    assert judge.main(_tp_judge_args(_home, work, "--concurrency", "1"), opener=ToAPI(api)) == code
    assert len(api.requests) == 1
    assert json.loads((work / judge.STATUS_FILE).read_text())["state"] == state
    assert (_home / "distill" / "spend-ledger.STOP").exists() == (state == "fatal")
    assert (_home / "distill" / "token-plan.PAUSE").exists() == (state == "paused")


def test_judge_refuses_the_token_plan_while_the_payg_key_is_present(_home: Path, api: StepAPI) -> None:
    env_file(_home, QWEN_API_KEY=PLAN_KEY, QWEN_PAYG_KEY=PAYG_KEY)
    work = _home / "phase2"
    write_queue(work, [make_item(10, line=10)])
    assert judge.main(_tp_judge_args(_home, work), opener=ToAPI(api)) == judge.EXIT_BUDGET
    assert not api.requests and "QWEN_PAYG_KEY" in json.loads((work / judge.STATUS_FILE).read_text())["reason"]


# ── build.py: the Token Plan judge and the Token Plan teacher's rows ─────────


def test_build_applies_token_plan_judge_verdicts(tmp_path: Path) -> None:
    rows = [{"source": {"file": "golden.jsonl", "line": i, "telemetry_rowid": i}, "status": "ok", "label": "pass",
             "verdict": "pass", "model": MAX, **extra}
            for i, extra in enumerate([{"provider": "max-token-plan", "judge_key": "token-plan"},
                                       {"provider": "max-token-plan", "judge_key": "payg"},
                                       {"provider": "max-payg", "judge_key": "payg"}])]
    path = tmp_path / "verdicts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    stats = __import__("collections").Counter()
    out = build.load_verdicts([str(path)], stats)
    assert out == {"golden.jsonl:1": "pass", "golden.jsonl:3": "pass"}
    assert (stats["verdict_rows_tokenplan_max_judge"], stats["verdict_rows_payg_max_judge"],
            stats["verdict_rows_max_judge"]) == (1, 1, 1)
    assert build.is_payg_judge({"provider": "max-token-plan", "judge_key": "token-plan"}, MAX)
    assert not build.is_payg_judge({"provider": "max-token-plan", "judge_key": "token-plan"}, "qwen3.8-27b")
    assert not build.is_payg_judge({"provider": "token-plan"}, MAX)


# ── teacher_gen.py --teacher max-token-plan ─────────────────────────────────


def _tp_args(tmp_path: Path, ts: Path, out: Path, *extra: str) -> list[str]:
    return ["run", "--teacher", "max-token-plan", "--tasks", str(ts), "--out", str(out),
            "--compaction-history-tokens", "150", *inputs(tmp_path), *extra]


def test_max_token_plan_teacher_end_to_end_and_build_only_under_the_allow(tmp_path, monkeypatch, no_outbound,
                                                                         _home):
    monkeypatch.setenv("QWEN_API_KEY", PLAN_KEY)
    ts, rows = taskset(tmp_path)
    out = tmp_path / "gen-max-derived"
    with FakeMax() as fake:
        rc = tg.main(_tp_args(tmp_path, ts, out, "--concurrency", "3", "--tp-start-concurrency", "2"),
                     payg_opener=ToFake(fake))
        posts = list(fake.requests)
    assert rc == tg.EXIT_DONE
    assert posts and all(p["headers"].get("Authorization") == "Bearer " + PLAN_KEY for p in posts)
    assert all(set(p["body"]) <= spend.PAYLOAD_KEYS for p in posts)
    led = ledger_rows(_home)
    assert not [r for r in led if r["note"].startswith(("reserve", "settle"))]
    plan_rows = [r for r in led if r["note"].startswith("token-plan")]
    assert len(plan_rows) == len(posts) and {r["usd"] for r in plan_rows} == {"0.000000"}
    assert {(r["tool"], r["phase"]) for r in plan_rows} == {("teacher_gen", "2b-teacher")}
    traces = out / f"{tg.TOKEN_PLAN_SOURCE}.traces.jsonl"
    got = [json.loads(x) for x in traces.read_text().splitlines()]
    assert {r["_meta"]["gen_task_id"] for r in got} == {r["id"] for r in rows}
    for r in got:
        m = r["_meta"]
        assert (m["source"], m["teacher"], m["model"], m["max_derived"], m["tags"]) == (
            "max-tokenplan-gym", "max-token-plan", MAX, True, ["max-derived"])
        j = m["verification"]["judge"]
        assert (j["kind"], j["provider"], j["judge_key"]) == ("self-judge", "max-token-plan", "token-plan")
    report = json.loads((out / f"{tg.TOKEN_PLAN_SOURCE}.report.json").read_text())
    assert report["spend"]["route"] == "max-token-plan" and report["spend"]["phase_token_plan_ok"] == len(posts)
    assert report["max_derived"] is True and report["private"] is True
    # build.py: excluded without the allow; admitted under it (the Token Plan judge verifies)
    b = gym_builder(tmp_path / "b-no", got, train_sources=build.DEFAULT_TRAIN_SOURCES)
    feed(b, traces, "extra")
    assert not b.examples and b.stats["excluded:max_payg_gym_needs_allow"] == len(got)
    b = gym_builder(tmp_path / "b-allow", got, train_sources=build.DEFAULT_TRAIN_SOURCES,
                    allow_max_derived=True, max_derived_approval=REF)
    feed(b, traces, "extra")
    assert len(b.examples) == len(got) and b.stats["verification:payg_max_judge"] == len(got)
    assert all(e.max_derived and e.source == "max-tokenplan-gym" for e in b.examples)


def test_max_token_plan_usage_errors_and_refusals(tmp_path, monkeypatch, no_outbound, capsys, _home):
    ts, _rows = taskset(tmp_path)
    tagged = tmp_path / "gen-max-derived"
    monkeypatch.setenv("QWEN_API_KEY", PLAN_KEY)
    for out, extra, why in ((tagged, ["--budget-usd", "5"], "no $ budget"),
                            (tagged, ["--concurrency", "9"], "--concurrency"),
                            (tmp_path / "gen", [], "max-derived tag")):
        assert tg.main(_tp_args(tmp_path, ts, out, *extra)) == tg.EXIT_USAGE, extra
        assert why in capsys.readouterr().err
    with FakeMax() as fake:
        def run() -> int:
            return tg.main(_tp_args(tmp_path, ts, tagged), payg_opener=ToFake(fake))

        monkeypatch.setenv("QWEN_PAYG_KEY", PAYG_KEY)                    # the PAYG key wins
        assert run() == tg.EXIT_PAYG_REFUSED and "QWEN_PAYG_KEY is present" in capsys.readouterr().err
        monkeypatch.delenv("QWEN_PAYG_KEY")
        monkeypatch.delenv("QWEN_API_KEY")
        assert run() == tg.EXIT_PAYG_REFUSED and "QWEN_API_KEY is not set" in capsys.readouterr().err
        monkeypatch.setenv("QWEN_API_KEY", PLAN_KEY)
        pause = _home / "distill" / "token-plan.PAUSE"
        pause.write_text(json.dumps({"until": time.time() + 600, "notify": "sent"}))
        assert run() == tg.EXIT_QUOTA_PAUSED and "quota pause" in capsys.readouterr().err
        pause.unlink()
        stop = _home / "distill" / "spend-ledger.STOP"
        stop.write_text("2026-10-31T00:00:00Z distill spend STOP: test\n")
        assert run() == tg.EXIT_PAYG_REFUSED
        stop.unlink()
        assert not fake.requests


def test_max_token_plan_account_error_mid_run_stops_and_notifies(tmp_path, monkeypatch, no_outbound, _home):
    monkeypatch.setenv("QWEN_API_KEY", PLAN_KEY)
    ts, _rows = taskset(tmp_path)
    out = tmp_path / "gen-max-derived"
    with FakeMax(errors={"agent": [403]}) as fake:
        rc = tg.main(_tp_args(tmp_path, ts, out, "--concurrency", "1"), payg_opener=ToFake(fake))
        n = len(fake.requests)
    assert rc == tg.EXIT_PAYG_REFUSED and n == 1
    assert "HTTP 403" in (_home / "distill" / "spend-ledger.STOP").read_text()
    assert "HTTP 403" in (_home / "notified.txt").read_text()


def test_payg_call_maps_the_token_plan_stops() -> None:
    for exc, kind in ((spend.TokenPlanStop("x"), "payg_fatal"), (spend.QuotaPaused("x", until=1.0), "quota_paused"),
                      (spend.TokenPlanSuperseded("x"), "superseded"), (spend.StopActive("x"), "spend_stop")):
        class Stub:
            def chat(self, payload: dict, **kw: Any) -> Any:
                raise exc  # noqa: B023
        with pytest.raises(tg.PaygStop) as got:
            tg.payg_call(Stub(), body())
        assert got.value.kind == kind


# ── judge_run.sh: the route and the trial ─────────────────────────────────────


def _run_sh(tmp_path: Path, *args: str, keys: dict[str, str],
            py: Path | None = None) -> tuple[subprocess.CompletedProcess[str], str]:
    distill = tmp_path / "distill"
    notify_script(distill)
    env_path = tmp_path / "prom.env"
    env_path.write_text("".join(f"{k}={v}\n" for k, v in keys.items()))
    env_path.chmod(0o600)
    wrapper = tmp_path / "py-wrapper.sh"
    wrapper.write_text(f'#!/bin/sh\necho "$@" >> "{tmp_path}/argv.log"\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "PY": str(py or wrapper),
           "DISTILL": str(distill), "PROMETHEUS_ENV_FILE": str(env_path), "LC_ALL": "C"}
    cp = subprocess.run(["bash", str(_DIR / "judge_run.sh"), *args], env=env, capture_output=True, text=True,
                        timeout=180, check=False)
    log = tmp_path / "argv.log"
    return cp, log.read_text() if log.exists() else ""


def test_judge_run_sh_picks_the_route_and_the_payg_key_wins(tmp_path: Path) -> None:
    write_queue(tmp_path / "distill" / "phase2", [make_item(i) for i in range(10, 13)])
    cp, argv = _run_sh(tmp_path, keys={"QWEN_API_KEY": PLAN_KEY})
    assert cp.returncode == 0, cp.stderr
    assert "route: Token Plan" in cp.stdout and "tokenplan-preflight" in argv
    assert "estimate --provider token-plan" in argv and " run " not in argv and "--budget-usd" not in argv
    assert "nothing was sent" in cp.stdout and PLAN_KEY not in cp.stdout + cp.stderr + argv
    (tmp_path / "argv.log").unlink()
    cp, argv = _run_sh(tmp_path, keys={"QWEN_API_KEY": PLAN_KEY, "QWEN_PAYG_KEY": PAYG_KEY})
    assert cp.returncode == 0, cp.stderr
    assert "route: pay-per-token" in cp.stdout and "estimate --provider max-payg" in argv
    cp, argv = _run_sh(tmp_path, "--provider", "token-plan", keys={"QWEN_API_KEY": PLAN_KEY,
                                                                   "QWEN_PAYG_KEY": PAYG_KEY})
    assert cp.returncode == 3 and "QWEN_PAYG_KEY is present" in cp.stderr
    cp, argv = _run_sh(tmp_path, keys={})
    assert cp.returncode == 2 and "no QWEN_PAYG_KEY or QWEN_API_KEY" in cp.stderr


def test_judge_run_sh_trial_is_the_first_50_judge_calls_then_the_status(tmp_path: Path) -> None:
    write_queue(tmp_path / "distill" / "phase2", [make_item(i) for i in range(10, 13)])
    fake = tmp_path / "fake-py.sh"
    fake.write_text(f"""#!/bin/sh
echo "$@" >> "{tmp_path}/argv.log"
case "$*" in
  *judge.py*" run "*) exit 6 ;;
esac
exec "{sys.executable}" "$@"
""")
    fake.chmod(0o755)
    cp, argv = _run_sh(tmp_path, "--execute", "--trial", keys={"QWEN_API_KEY": PLAN_KEY}, py=fake)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    (run_line,) = [x for x in argv.splitlines() if " run " in x]
    assert "--provider token-plan" in run_line and "--limit 50" in run_line and "--view outcome" in run_line
    assert "--budget-usd" not in run_line and "--view blind" not in argv and " report " not in argv
    assert "tokenplan-status" in argv and '"trial"' in cp.stdout


# ── gen_runner.py: the route, the switch, the quota pause ─────────────────────


def _progress(rig: Rig) -> dict[str, Any]:
    """progress.json, or {} until the runner has written it (or mid-replace)."""
    try:
        return rig.progress()
    except (OSError, ValueError):
        return {}


def _tp_rig(rig: Rig) -> Path:
    pause = rig.root / "token-plan.PAUSE"
    rig.conf.write_text(rig.conf.read_text() + f"GEN_TP_PAUSE={pause}\n")
    rig.envfile.write_text(f"QWEN_API_KEY={PLAN_KEY}\n")
    return pause


@pytest.fixture
def rig(tmp_path: Path):
    (tmp_path / "rig").mkdir()
    r = Rig(tmp_path / "rig")
    yield r
    r.close()


def test_gen_runner_uses_the_token_plan_until_the_payg_key_appears(rig: Rig) -> None:
    _tp_rig(rig)
    p = rig.start(until_s=60, env={"FAKE_JUDGE_S": "0.3"})
    rig.wait_for(lambda: (rig.state / "max-pid").exists(), what="the Max teacher")
    (judge_call,) = rig.calls("judge-calls.log")
    assert judge_call.startswith("--execute --provider token-plan --concurrency 8 |")
    paid = [c for c in rig.calls("tg-calls.log") if "--teacher" in c and _opt(c, "--teacher") != "local"][0]
    assert paid[:7] == ["run", "--teacher", "max-token-plan", "--concurrency", "8", "--tp-start-concurrency", "4"]
    assert "--budget-usd" not in paid and _opt(paid, "--out") == str(rig.runs / "max-tokenplan-max-derived")
    assert _opt(paid, "--sandbox-root").endswith("-max-token-plan")
    spend_calls = rig.calls("spend-calls.log")
    assert spend_calls and all(c[0] == "tokenplan-preflight" for c in spend_calls)
    rig.wait_for(lambda: _progress(rig).get("paid", {}).get("route") == "token-plan", what="the route")
    first = rig.pid("max")
    # guardrail 6: the pay-per-token key appears -> the Token Plan teacher stops, the branch starts over on it
    rig.envfile.write_text(f"QWEN_API_KEY={PLAN_KEY}\nQWEN_PAYG_KEY=sk-not-a-real-key-just-a-test-value\n")
    rig.wait_for(lambda: not alive(first), what="the Token Plan teacher's stop")
    rig.wait_for(lambda: len(rig.calls("judge-calls.log")) == 2, what="the judge on the new route")
    assert rig.calls("judge-calls.log")[1].startswith("--execute --provider payg --budget-usd 25")
    rig.wait_for(lambda: any(_opt(c, "--teacher") == "max-payg" for c in rig.calls("tg-calls.log")
                             if "--teacher" in c), what="the Max teacher on the pay-per-token key")
    assert any("QWEN_PAYG_KEY appeared" in n for n in rig.progress()["notes"])
    (rig.runs / "STOP").write_text("operator\n")
    assert p.wait(timeout=30) == 3


def test_gen_runner_parks_the_branch_during_the_quota_pause(rig: Rig) -> None:
    pause = _tp_rig(rig)
    rig.conf.write_text(rig.conf.read_text() + "GEN_TP_QUOTA_RECHECK_S=30\n")
    pause.write_text(json.dumps({"until": time.time() + 3.0, "notify": "sent"}))
    p = rig.start(until_s=60, env={"FAKE_JUDGE_S": "0.2", "FAKE_TG_MAX_RC": "7"})
    rig.wait_for(lambda: _progress(rig).get("paid", {}).get("state") == "quota", what="the branch parked")
    assert rig.calls("judge-calls.log") == []                           # nothing sent during the pause
    # the pause ends: the branch starts over (judge, then the teacher, which exits 7: parked again)
    rig.wait_for(lambda: len(rig.calls("judge-calls.log")) >= 1, timeout=20, what="the judge after the pause")
    rig.wait_for(lambda: any("Max teacher exit 7" in n for n in _progress(rig).get("notes", [])),
                 what="exit 7 parked")
    time.sleep(1.0)
    assert len(rig.calls("judge-calls.log")) == 1                       # no pause standing: waits, no spin
    assert rig.progress()["paid"]["state"] == "quota"
    (rig.runs / "STOP").write_text("operator\n")
    assert p.wait(timeout=30) == 3
