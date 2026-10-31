"""Shadow mirroring (engine/shadow.py): mirror primary rounds to a local model, act only on the primary.

What is pinned here, in the order the runbook asks for it:

* off (the default) changes nothing: no mirror is built, no shadow provider is ever constructed,
  and a round streams exactly as it did before the parameter existed;
* on, the shadow request runs beside the primary round and never holds it up;
* a shadow that fails, hangs or cannot write its row never reaches the primary;
* a shadow tool call is never executed, through the real ``run_loop``;
* the primary's error rate or latency turns mirroring off at runtime, and says so;
* the ``shadow:`` block is validated at load, and the shipped template validates.

No network and no model: providers are scripted, and the row directory is a tmp path.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from prometheus.config.template import load_template
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.fallback import stream_round_with_fallback
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolResultBlock, ToolUseBlock
from prometheus.engine.shadow import (
    ShadowConfigError,
    ShadowMirror,
    args_equivalent,
    build_shadow_mirror,
    classify_round,
    normalize_answer_text,
    reply_calls,
    resolve_shadow_config,
)
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
    ModelProvider,
)
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

REPO = Path(__file__).resolve().parents[1]
DAEMON = REPO / "src" / "prometheus" / "daemon.py"


# ── fixtures ───────────────────────────────────────────────────────────────────────────────


def _block(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    block: dict[str, Any] = {
        "enabled": True,
        "provider": "llama_cpp",
        "base_url": "http://127.0.0.1:8099",
        "model": "student-9b",
        "log_dir": str(tmp_path / "shadow"),
    }
    auto = overrides.pop("auto_disable", None)
    block.update(overrides)
    if auto is not None:
        block["auto_disable"] = auto
    return block


def _config(tmp_path: Path, **overrides: Any):
    return resolve_shadow_config({"shadow": _block(tmp_path, **overrides)})


def _text_events(text: str) -> list:
    msg = ConversationMessage(role="assistant", content=[TextBlock(text=text)])
    return [ApiTextDeltaEvent(text=text),
            ApiMessageCompleteEvent(message=msg, usage=UsageSnapshot(), stop_reason="stop")]


def _tool_events(name: str, args: dict[str, Any], tool_id: str = "t1") -> list:
    msg = ConversationMessage(role="assistant",
                              content=[ToolUseBlock(id=tool_id, name=name, input=args)])
    return [ApiMessageCompleteEvent(message=msg, usage=UsageSnapshot(), stop_reason="tool_calls")]


class ScriptedProvider(ModelProvider):
    """Replays scripted rounds; records every request. ``gate`` holds each reply until set."""

    def __init__(self, rounds: list, *, gate: asyncio.Event | None = None,
                 fail: BaseException | None = None) -> None:
        self.rounds = list(rounds)
        self.gate = gate
        self.fail = fail
        self.requests: list[ApiMessageRequest] = []
        self.finished = 0

    async def stream_message(self, request: ApiMessageRequest):
        self.requests.append(request)
        if self.gate is not None:
            await self.gate.wait()
        if self.fail is not None:
            raise self.fail
        events = self.rounds[(len(self.requests) - 1) % len(self.rounds)]
        for event in events:
            yield event
        self.finished += 1


def _request(model: str = "primary-model", **kw: Any) -> ApiMessageRequest:
    return ApiMessageRequest(
        model=model,
        messages=[ConversationMessage.from_user_text("fix the failing test")],
        system_prompt="SYSTEM",
        tools=[{"name": "bash", "description": "run", "input_schema": {"type": "object"}}],
        **kw,
    )


async def _stream(events: list):
    for event in events:
        yield event


async def _consume(agen) -> list:
    return [event async for event in agen]


def _rows(tmp_path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted((tmp_path / "shadow").glob("shadow-*.jsonl")):
        out += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return out


class FakeTelemetry:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict, str]] = []

    def record_signal_event(self, *, signal_type, payload, source_subsystem, **_):
        self.events.append((signal_type, payload, source_subsystem))
        return 1


# ── config: validated at load ──────────────────────────────────────────────────────────────


def test_an_absent_block_is_off_and_matches_the_shipped_template():
    absent = resolve_shadow_config({})
    shipped = resolve_shadow_config(load_template())
    assert absent.enabled is False
    assert absent == shipped, "the template documents exactly what absence means"


def test_the_shipped_template_builds_no_mirror_and_constructs_no_provider():
    built: list[Any] = []
    mirror = build_shadow_mirror(load_template(), provider_factory=lambda cfg: built.append(cfg))
    assert mirror is None
    assert built == [], "off must not even construct the shadow provider"


@pytest.mark.parametrize("block, needle", [
    ("yes", "expected a mapping"),
    ({"enabled": "yes"}, "shadow.enabled"),
    ({"sampel_rate": 0.5}, "unknown key"),
    ({"sample_rate": 1.5}, "shadow.sample_rate"),
    ({"sample_rate": True}, "shadow.sample_rate"),
    ({"max_concurrency": 0}, "shadow.max_concurrency"),
    ({"max_concurrency": 1.5}, "shadow.max_concurrency"),
    ({"timeout_seconds": 0}, "shadow.timeout_seconds"),
    ({"provider": "qwen"}, "local provider"),
    ({"provider": "anthropic"}, "local provider"),
    ({"base_url": "127.0.0.1:8099"}, "http(s) URL"),
    ({"log_dir": 7}, "shadow.log_dir"),
    ({"auto_disable": "on"}, "shadow.auto_disable"),
    ({"auto_disable": {"windw": 5}}, "unknown key"),
    ({"auto_disable": {"window": 0}}, "auto_disable.window"),
    ({"auto_disable": {"primary_error_rate": 0}}, "primary_error_rate"),
    ({"auto_disable": {"primary_error_rate": 2}}, "primary_error_rate"),
    ({"auto_disable": {"primary_latency_p95_ms": -1}}, "primary_latency_p95_ms"),
    ({"enabled": True, "base_url": "http://127.0.0.1:8099"}, "shadow.model is required"),
    ({"enabled": True, "model": "student-9b"}, "shadow.base_url is required"),
])
def test_a_bad_block_is_refused_with_the_key_named(block, needle):
    with pytest.raises(ShadowConfigError) as ei:
        resolve_shadow_config({"shadow": block})
    assert needle in str(ei.value)


def test_a_bad_block_is_refused_even_while_off():
    """Reported the day it is written, not the day someone switches it on."""
    with pytest.raises(ShadowConfigError):
        resolve_shadow_config({"shadow": {"enabled": False, "provider": "openai"}})


def test_the_shadow_may_not_be_the_primary_s_own_server(tmp_path):
    config = {
        "model": {"provider": "llama_cpp", "base_url": "http://127.0.0.1:8099/"},
        "shadow": _block(tmp_path),
    }
    with pytest.raises(ShadowConfigError, match="primary's own server"):
        resolve_shadow_config(config)
    # A hosted primary has no server of ours to collide with.
    config["model"] = {"provider": "qwen"}
    assert resolve_shadow_config(config).enabled


@pytest.mark.parametrize("model, shadow_url", [
    ({"provider": "llama_cpp"}, "http://127.0.0.1:8080"),                    # the default primary
    ({"provider": "llama_cpp", "base_url": "http://127.0.0.1:8080/"}, "http://127.0.0.1:8080/v1"),
    ({"provider": "llama_cpp", "base_url": "http://127.0.0.1:8092"}, "http://localhost:8092"),
    ({"provider": "llama_cpp", "base_url": "http://[::1]:8092"}, "http://LOCALHOST:8092/"),
    ({"provider": "llama_cpp", "base_url": "http://0.0.0.0:8092"}, "http://127.0.0.1:8092"),
    ({"provider": "llama_cpp", "base_url": "http://10.0.0.5"}, "http://10.0.0.5:80/v1"),
])
def test_the_primary_s_server_is_refused_however_its_url_is_written(tmp_path, model, shadow_url):
    config = {"model": model, "shadow": _block(tmp_path, base_url=shadow_url)}
    with pytest.raises(ShadowConfigError, match="primary's own server"):
        resolve_shadow_config(config)


def test_another_port_on_the_same_host_is_another_server(tmp_path):
    config = {"model": {"provider": "llama_cpp", "base_url": "http://127.0.0.1:8092"},
              "shadow": _block(tmp_path, base_url="http://localhost:8093")}
    assert resolve_shadow_config(config).enabled


@pytest.mark.parametrize("fallback, shadow_url, refused", [
    ({"provider": "llama_cpp", "base_url": "http://localhost:8099"}, "http://127.0.0.1:8099", True),
    ({"enabled": True}, "http://127.0.0.1:8080", True),     # no server named: llama.cpp's default
    ({"enabled": False}, "http://127.0.0.1:8080", False),   # no fallback, nothing to collide with
    ({"provider": "openai"}, "http://127.0.0.1:8080", False),  # a hosted fallback
])
def test_the_fallback_s_server_is_refused(tmp_path, fallback, shadow_url, refused):
    config = {"model": {"provider": "qwen", "fallback": fallback},
              "shadow": _block(tmp_path, base_url=shadow_url)}
    if refused:
        with pytest.raises(ShadowConfigError, match="fallback's server"):
            resolve_shadow_config(config)
    else:
        assert resolve_shadow_config(config).enabled


def test_a_backends_entry_s_server_is_refused(tmp_path):
    """A session can be switched to any backend (and a fallback can name one)."""
    config = {
        "model": {"provider": "qwen", "fallback": {"backend": "4090"}},
        "backends": {4090: {"provider": "llama_cpp", "base_url": "http://10.0.0.5:8080"}},
        "shadow": _block(tmp_path, base_url="http://10.0.0.5:8080/v1"),
    }
    with pytest.raises(ShadowConfigError, match="backends.4090"):
        resolve_shadow_config(config)
    config["shadow"] = _block(tmp_path, base_url="http://10.0.0.5:8081")
    assert resolve_shadow_config(config).enabled


@pytest.mark.parametrize("provider, url", [
    ("openai_compatible", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    ("vllm", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    ("llama_cpp", "https://inference.example.com"),
    ("lm_studio", "http://8.8.8.8:1234"),
    ("ollama", "http://134744072:11434"),          # 8.8.8.8, spelt the way inet_aton reads it
    ("llama_cpp", "http://[2001:4860:4860::8888]:8080"),
    ("vllm", "http://gpu-box.example.com:8000"),   # a public name, even for a LAN box
])
def test_a_hosted_endpoint_is_refused_under_every_local_provider_label(tmp_path, provider, url):
    """The provider label proves nothing: what is sent where is the base_url's call."""
    built: list[Any] = []
    block = _block(tmp_path, provider=provider, base_url=url)
    with pytest.raises(ShadowConfigError, match="not on this machine or its own network"):
        resolve_shadow_config({"shadow": block})
    assert build_shadow_mirror({"shadow": block}, provider_factory=built.append) is None
    assert built == [], "no provider is even constructed for a hosted endpoint"
    # Refused while off too: reported the day it is written.
    with pytest.raises(ShadowConfigError):
        resolve_shadow_config({"shadow": {**block, "enabled": False}})


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8099", "http://localhost:8099", "http://[::1]:8099",
    "http://192.168.1.20:8080", "http://10.0.0.5:8080/v1", "http://172.16.0.9:8080",
    "http://100.64.0.7:8080",                      # tailnet
    "http://[fd00::5]:8080",                       # IPv6 unique-local
    "http://gpu-box:8080", "http://gpu-box.lan:8080", "http://gpu-box.local:8080",
    "http://gpu-box.tail1234.ts.net:8080", "http://gpu-box.home.arpa:8080",
])
def test_a_local_endpoint_is_accepted(tmp_path, url):
    assert resolve_shadow_config({"shadow": _block(tmp_path, base_url=url)}).base_url


def test_log_dir_defaults_under_the_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path / "data"))
    block = _block(tmp_path)
    del block["log_dir"]
    assert resolve_shadow_config({"shadow": block}).log_dir == tmp_path / "data" / "shadow"


def test_a_bad_block_logs_an_error_and_leaves_mirroring_off(caplog):
    with caplog.at_level("ERROR", logger="prometheus.engine.shadow"):
        assert build_shadow_mirror({"shadow": {"enabled": True, "sample_rate": 9}}) is None
    assert any("sample_rate" in r.getMessage() and "OFF" in r.getMessage() for r in caplog.records)


def test_a_valid_block_builds_a_mirror_on_a_local_provider(tmp_path):
    mirror = build_shadow_mirror({"shadow": _block(tmp_path)})
    assert isinstance(mirror, ShadowMirror) and mirror.enabled
    assert type(mirror.provider).__name__ == "LlamaCppProvider"
    assert mirror.config.max_concurrency == 1 and mirror.config.window == 20


def test_an_unbuildable_provider_is_no_mirror_not_a_crash(tmp_path):
    def boom(cfg):
        raise RuntimeError("cannot build")

    assert build_shadow_mirror({"shadow": _block(tmp_path)}, provider_factory=boom) is None


# ── off changes nothing ────────────────────────────────────────────────────────────────────


class FakeEnvelope:
    def __init__(self, events: list) -> None:
        self.events = events
        self.attempts: list[tuple[Any, Any]] = []

    def stream(self, *, provider, request, **_):
        self.attempts.append((provider, request))
        return _stream(self.events)


async def _round(envelope, *, shadow=None, request=None):
    request = request or _request()
    return await _consume(stream_round_with_fallback(
        envelope=envelope, provider="PRIMARY", model=request.model,
        build_request=lambda m: request, target=None, enabled=False,
        window_for=lambda m: (0, False), estimate_tokens=lambda: 0, on_degrade=lambda d: None,
        operation="loop_round", round_index=0, session_id="telegram:1", shadow=shadow,
    ))


async def test_no_shadow_streams_the_round_exactly_as_before():
    events = _text_events("hello")
    env = FakeEnvelope(events)
    seen = await _round(env)
    assert len(seen) == len(events) and all(a is b for a, b in zip(seen, events))
    assert [p for p, _ in env.attempts] == ["PRIMARY"]


async def test_a_mirror_that_is_off_hands_back_the_very_same_stream(tmp_path):
    shadow = ScriptedProvider([_text_events("x")])
    mirror = ShadowMirror(_config(tmp_path), shadow)
    mirror.disable("test")
    stream = _stream(_text_events("hello"))
    assert mirror.mirror(stream, _request()) is stream
    await _consume(stream)
    await mirror.drain()
    assert shadow.requests == []


def _registry(*tools: BaseTool) -> ToolRegistry:
    reg = ToolRegistry()
    for tool in tools:
        reg.register(tool)
    return reg


class _Args(BaseModel):
    text: str = "hi"


def _recording_tool(name: str, calls: list[str]) -> BaseTool:
    class _Tool(BaseTool):
        input_model = _Args
        description = f"{name} tool"

        async def execute(self, arguments, context):  # noqa: ANN001
            calls.append(name)
            return ToolResult(output=f"{name} ran")

        def is_read_only(self, arguments) -> bool:  # noqa: ANN001
            return True

    _Tool.name = name
    return _Tool()


async def _run_turn(context: LoopContext, text: str = "go") -> list:
    out = []
    async for event, _usage in run_loop(context, [ConversationMessage.from_user_text(text)]):
        out.append(event)
    return out


def _turn_shape(events: list) -> list[tuple[str, str]]:
    return [(type(e).__name__, getattr(e, "text", "") or getattr(getattr(e, "message", None), "text", ""))
            for e in events]


async def test_the_primary_path_is_identical_with_and_without_a_mirror(tmp_path):
    """Same requests to the primary, same events out of the loop, whether or not a shadow listens."""
    rounds = [_tool_events("echo", {"text": "hi"}), _text_events("done")]

    def context(provider, shadow):
        return LoopContext(provider=provider, model="primary-model", system_prompt="BASE",
                           max_tokens=256, tool_registry=_registry(_recording_tool("echo", [])),
                           session_id="telegram:1", shadow=shadow)

    bare = ScriptedProvider(rounds)
    bare_events = await _run_turn(context(bare, None))

    mirrored = ScriptedProvider(rounds)
    shadow = ScriptedProvider([_text_events("something else entirely")])
    # Two slots: a scripted primary never yields to the loop, so with one the shadow of round 1
    # would still hold it when round 2 starts (and round 2 would be skipped, correctly).
    mirror = ShadowMirror(_config(tmp_path, max_concurrency=2), shadow)
    mirrored_events = await _run_turn(context(mirrored, mirror))
    await mirror.drain()

    assert mirrored.requests == bare.requests
    assert _turn_shape(mirrored_events) == _turn_shape(bare_events)
    assert len(shadow.requests) == 2, "both rounds were mirrored"


# ── on: mirrors without blocking ───────────────────────────────────────────────────────────


async def test_the_primary_round_finishes_while_the_shadow_is_still_working(tmp_path):
    gate = asyncio.Event()
    shadow = ScriptedProvider([_tool_events("bash", {"command": "pytest -x"})], gate=gate)
    mirror = ShadowMirror(_config(tmp_path), shadow)
    events = _tool_events("bash", {"command": "pytest -x"})

    seen = await _consume(mirror.mirror(_stream(events), _request(), session_id="telegram:1",
                                        round_index=3))

    assert all(a is b for a, b in zip(seen, events)) and len(seen) == len(events)
    assert shadow.finished == 0 and mirror.snapshot()["in_flight"] == 1, (
        "the round ended while the shadow was still waiting: nothing awaited it")
    assert _rows(tmp_path) == []

    gate.set()
    await mirror.drain()
    [row] = _rows(tmp_path)
    assert row["session_id"] == "telegram:1" and row["round_index"] == 3
    assert row["primary"]["tools"] == row["shadow"]["tools"] == ["bash"]
    assert row["same_tool"] is True and row["args_equivalent"] is True
    assert row["category"] == "coding"


async def test_the_shadow_gets_the_same_request_under_its_own_model_name(tmp_path):
    shadow = ScriptedProvider([_text_events("ok")])
    mirror = ShadowMirror(_config(tmp_path), shadow)
    request = _request()
    await _consume(mirror.mirror(_stream(_text_events("ok")), request))
    await mirror.drain()
    [sent] = shadow.requests
    assert sent.model == "student-9b"
    assert sent.messages == request.messages and sent.messages is not request.messages
    assert (sent.system_prompt, sent.tools, sent.max_tokens, sent.tool_choice) == (
        request.system_prompt, request.tools, request.max_tokens, request.tool_choice)


async def test_a_row_carries_the_comparison(tmp_path):
    shadow = ScriptedProvider([_text_events("The answer is 42.")])
    mirror = ShadowMirror(_config(tmp_path), shadow)
    await _consume(mirror.mirror(_stream(_text_events("the answer is 42")), _request(),
                                 session_id="telegram:1"))
    await mirror.drain()
    [row] = _rows(tmp_path)
    assert row["schema"] == "shadow-row/1" and len(row["request_id"]) == 32
    assert row["primary"]["model"] == "primary-model" and row["shadow"]["model"] == "student-9b"
    assert row["comparable"] is True and row["same_tool"] is True
    assert row["args_equivalent"] is None, "no tool round, nothing to compare"
    assert row["answer_match"] is True, "normalized match ignores case and the full stop"
    assert row["primary"]["latency_ms"] >= 0 and row["shadow"]["error"] is None


async def test_the_row_text_goes_through_the_capture_scrubber(tmp_path):
    token = "hf_" + "a1" * 17
    shadow = ScriptedProvider([_tool_events("bash", {"command": f"export HF_TOKEN={token}"})])
    mirror = ShadowMirror(_config(tmp_path), shadow)
    await _consume(mirror.mirror(_stream(_text_events(f"use {token} for that")), _request()))
    await mirror.drain()
    raw = (tmp_path / "shadow").glob("shadow-*.jsonl").__next__().read_text(encoding="utf-8")
    assert token not in raw
    assert "<redacted>" in raw


async def test_rows_are_owner_only(tmp_path):
    mirror = ShadowMirror(_config(tmp_path), ScriptedProvider([_text_events("ok")]))
    await _consume(mirror.mirror(_stream(_text_events("ok")), _request()))
    await mirror.drain()
    [path] = (tmp_path / "shadow").glob("shadow-*.jsonl")
    assert path.stat().st_mode & 0o077 == 0


# ── a shadow failure never reaches the primary ─────────────────────────────────────────────


async def test_a_failing_shadow_is_a_row_not_an_exception(tmp_path):
    shadow = ScriptedProvider([], fail=RuntimeError("connection refused"))
    mirror = ShadowMirror(_config(tmp_path), shadow)
    events = _text_events("primary answer")
    seen = await _consume(mirror.mirror(_stream(events), _request()))
    await mirror.drain()
    assert all(a is b for a, b in zip(seen, events))
    [row] = _rows(tmp_path)
    assert "RuntimeError" in row["shadow"]["error"] and row["comparable"] is False
    assert row["primary"]["error"] is None and mirror.enabled


async def test_a_hanging_shadow_times_out_into_its_row(tmp_path):
    shadow = ScriptedProvider([_text_events("late")], gate=asyncio.Event())  # never released
    mirror = ShadowMirror(_config(tmp_path, timeout_seconds=0.05), shadow)
    await _consume(mirror.mirror(_stream(_text_events("on time")), _request()))
    await mirror.drain()
    [row] = _rows(tmp_path)
    assert row["shadow"]["error"].startswith("timeout")


async def test_an_unwritable_row_directory_costs_the_primary_nothing(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    mirror = ShadowMirror(_config(tmp_path, log_dir=str(blocker / "shadow")),
                          ScriptedProvider([_text_events("ok")]))
    events = _text_events("primary")
    seen = await _consume(mirror.mirror(_stream(events), _request()))
    await mirror.drain()
    assert all(a is b for a, b in zip(seen, events))
    assert mirror.counters["write_errors"] == 1


async def test_a_primary_error_is_re_raised_unchanged_and_recorded(tmp_path):
    boom = ConnectionError("primary down")

    async def failing():
        yield ApiTextDeltaEvent(text="par")
        raise boom

    mirror = ShadowMirror(_config(tmp_path), ScriptedProvider([_text_events("ok")]))
    with pytest.raises(ConnectionError) as ei:
        await _consume(mirror.mirror(failing(), _request()))
    assert ei.value is boom
    await mirror.drain()
    [row] = _rows(tmp_path)
    assert row["primary"]["error"].startswith("ConnectionError") and row["comparable"] is False


async def test_a_stopped_turn_drops_its_shadow_request_and_writes_no_row(tmp_path):
    shadow = ScriptedProvider([_text_events("x")], gate=asyncio.Event())
    mirror = ShadowMirror(_config(tmp_path), shadow)
    agen = mirror.mirror(_stream(_text_events("partial answer")), _request())
    await agen.__anext__()
    await agen.aclose()  # the consumer stopped (a Stop, an overflow guard)
    await mirror.drain()
    assert _rows(tmp_path) == []
    assert mirror.snapshot()["in_flight"] == 0
    assert mirror.enabled, "a stop says nothing about the primary's health"


# ── shadow tool calls are never executed ───────────────────────────────────────────────────


async def test_a_shadow_tool_call_never_runs_through_the_real_loop(tmp_path):
    ran: list[str] = []
    registry = _registry(_recording_tool("echo", ran), _recording_tool("danger", ran))
    primary = ScriptedProvider([_tool_events("echo", {"text": "hi"}), _text_events("done")])
    shadow = ScriptedProvider([_tool_events("danger", {"text": "rm"}, tool_id="s1")])
    mirror = ShadowMirror(_config(tmp_path, max_concurrency=2), shadow)
    context = LoopContext(provider=primary, model="primary-model", system_prompt="BASE",
                          max_tokens=256, tool_registry=registry, session_id="telegram:1",
                          shadow=mirror)

    await _run_turn(context)
    await mirror.drain()

    assert ran == ["echo"], "only the primary's call ran; the shadow's 'danger' never did"
    assert len(shadow.requests) == 2
    rows = _rows(tmp_path)
    assert [r["shadow"]["tools"] for r in rows] == [["danger"], ["danger"]]
    assert [r["primary"]["tools"] for r in rows] == [["echo"], []]
    assert [r["same_tool"] for r in rows] == [False, False]


# ── sampling, skips ────────────────────────────────────────────────────────────────────────


async def test_sample_rate_zero_mirrors_nothing(tmp_path):
    shadow = ScriptedProvider([_text_events("x")])
    mirror = ShadowMirror(_config(tmp_path, sample_rate=0.0), shadow)
    for _ in range(3):
        await _consume(mirror.mirror(_stream(_text_events("ok")), _request()))
    await mirror.drain()
    assert shadow.requests == [] and mirror.counters["skipped_sample"] == 3


async def test_a_busy_shadow_skips_the_round_instead_of_queueing(tmp_path):
    gate = asyncio.Event()
    shadow = ScriptedProvider([_text_events("x")], gate=gate)
    mirror = ShadowMirror(_config(tmp_path), shadow)
    await _consume(mirror.mirror(_stream(_text_events("one")), _request()))
    await _consume(mirror.mirror(_stream(_text_events("two")), _request()))
    assert mirror.counters["mirrored"] == 1 and mirror.counters["skipped_busy"] == 1
    gate.set()
    await mirror.drain()
    assert len(shadow.requests) == 1 and len(_rows(tmp_path)) == 1


@pytest.mark.parametrize("request_kw, counter", [
    ({"tool_choice": {"tool": "bash"}}, "skipped_forced"),
    ({"tool_choice": "required"}, "skipped_forced"),
    ({"model": "student-9b"}, "skipped_same_model"),
])
async def test_rounds_that_cannot_be_compared_are_not_mirrored(tmp_path, request_kw, counter):
    shadow = ScriptedProvider([_text_events("x")])
    mirror = ShadowMirror(_config(tmp_path), shadow)
    model = request_kw.pop("model", "primary-model")
    await _consume(mirror.mirror(_stream(_text_events("ok")), _request(model, **request_kw)))
    await mirror.drain()
    assert shadow.requests == [] and mirror.counters[counter] == 1


# ── auto-disable ───────────────────────────────────────────────────────────────────────────


async def _fail_round(mirror: ShadowMirror) -> None:
    async def failing():
        raise ConnectionError("primary down")
        yield  # pragma: no cover

    with pytest.raises(ConnectionError):
        await _consume(mirror.mirror(failing(), _request()))


async def test_a_primary_error_rate_past_the_threshold_turns_mirroring_off(tmp_path):
    telemetry = FakeTelemetry()
    notes: list[str] = []

    async def notify(text: str) -> None:
        notes.append(text)

    gate = asyncio.Event()
    shadow = ScriptedProvider([_text_events("x")], gate=gate)
    mirror = ShadowMirror(
        _config(tmp_path, max_concurrency=4,
                auto_disable={"window": 4, "primary_error_rate": 0.5}),
        shadow, telemetry=telemetry, notify=notify)

    await _consume(mirror.mirror(_stream(_text_events("ok")), _request()))
    for _ in range(3):
        await _fail_round(mirror)
    assert not mirror.enabled
    assert "error rate 75%" in (mirror.disabled_reason or "")
    await mirror.drain()

    assert mirror.snapshot()["in_flight"] == 0, "in-flight shadow requests are cancelled"
    assert shadow.finished == 0
    assert [e[0] for e in telemetry.events] == ["shadow_auto_disabled"]
    assert len(notes) == 1 and "turned itself off" in notes[0]

    stream = _stream(_text_events("later"))
    assert mirror.mirror(stream, _request()) is stream, "off for the rest of the process"


async def test_a_slow_primary_turns_mirroring_off(tmp_path):
    now = [0.0]

    def clock() -> float:
        return now[0]

    async def slow_round():
        now[0] += 0.5  # every primary round takes 500 ms
        for event in _text_events("ok"):
            yield event

    mirror = ShadowMirror(
        _config(tmp_path, sample_rate=0.0,
                auto_disable={"window": 3, "primary_latency_p95_ms": 400}),
        ScriptedProvider([_text_events("x")]), clock=clock)
    for _ in range(2):
        await _consume(mirror.mirror(slow_round(), _request()))
        assert mirror.enabled, "not judged before the window is full"
    await _consume(mirror.mirror(slow_round(), _request()))
    assert not mirror.enabled
    assert "p95 latency (time to its first event) 500 ms" in (mirror.disabled_reason or "")
    await mirror.drain()


async def test_the_consumer_s_time_between_events_is_not_the_primary_s(tmp_path):
    """A primary that streams at once, read by a slow consumer (a websocket broadcast per delta):
    the reader's time is not the model's, for auto-disable or in the row."""
    now = [0.0]
    mirror = ShadowMirror(
        _config(tmp_path, auto_disable={"window": 1, "primary_latency_p95_ms": 400}),
        ScriptedProvider([_text_events("x")]), clock=lambda: now[0])
    events = [ApiTextDeltaEvent(text="a"), ApiTextDeltaEvent(text="b"), *_text_events("ab")]
    async for _event in mirror.mirror(_stream(events), _request()):
        now[0] += 10.0  # ten seconds of consumer per event
    await mirror.drain()
    assert mirror.enabled, "a slow reader is not a slow primary"
    [row] = _rows(tmp_path)
    assert row["primary"]["latency_ms"] == 0.0 and row["primary"]["first_event_ms"] == 0.0


async def test_a_long_reply_is_not_a_slow_primary(tmp_path):
    """Auto-disable judges the wait for the first event; the whole generation is in the row."""
    now = [0.0]

    async def long_reply():
        now[0] += 0.1                       # 100 ms to the first token
        yield ApiTextDeltaEvent(text="def main():")
        now[0] += 200.0                     # then a 200 s file write at a hosted model's pace
        for event in _text_events("def main(): ..."):
            yield event

    mirror = ShadowMirror(_config(tmp_path, auto_disable={"window": 1}),
                          ScriptedProvider([_text_events("x")]), clock=lambda: now[0])
    await _consume(mirror.mirror(long_reply(), _request()))
    await mirror.drain()
    assert mirror.enabled, "200 s of output at 120 s p95 tripped auto-disable"
    [row] = _rows(tmp_path)
    assert row["primary"]["first_event_ms"] == pytest.approx(100.0)
    assert row["primary"]["latency_ms"] == pytest.approx(200_100.0)
    assert row["shadow"]["first_event_ms"] is not None


def _http_error(status: int, body: dict[str, Any]) -> Exception:
    import httpx

    req = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
    resp = httpx.Response(status, text=json.dumps(body), request=req)
    return httpx.HTTPStatusError(f"error '{status}'", request=req, response=resp)


OVER_WINDOW = {"error": {
    "code": 400, "type": "exceed_context_size_error",
    "message": "request (33094 tokens) exceeds the available context size (32768 tokens)",
    "n_prompt_tokens": 33094, "n_ctx": 32768}}


@pytest.mark.parametrize("status, body, counted", [
    (400, OVER_WINDOW, False),                                   # the conversation's size
    (500, {"error": {"message": "internal error"}}, True),        # the primary failing
])
async def test_a_context_overflow_refusal_is_not_a_primary_failure(tmp_path, status, body, counted):
    exc = _http_error(status, body)

    async def refusing():
        raise exc
        yield  # pragma: no cover

    mirror = ShadowMirror(_config(tmp_path, auto_disable={"window": 1}),
                          ScriptedProvider([_text_events("x")]))
    with pytest.raises(type(exc)):
        await _consume(mirror.mirror(refusing(), _request()))
    await mirror.drain()
    assert mirror.enabled is not counted
    if not counted:
        # Not a health signal, but still a round that could not be compared: its row says so.
        [row] = _rows(tmp_path)
        assert row["primary"]["error"].startswith("HTTPStatusError") and row["comparable"] is False


async def test_a_healthy_primary_keeps_mirroring_on(tmp_path):
    mirror = ShadowMirror(
        _config(tmp_path, sample_rate=0.0, auto_disable={"window": 4, "primary_error_rate": 0.5}),
        ScriptedProvider([_text_events("x")]))
    for i in range(8):
        if i % 4 == 0:
            await _fail_round(mirror)  # 25% errors: not above 50%
        else:
            await _consume(mirror.mirror(_stream(_text_events("ok")), _request()))
    assert mirror.enabled


async def test_auto_disable_goes_through_the_signal_bus_once_wired(tmp_path):
    class Bus:
        def __init__(self) -> None:
            self.kinds: list[str] = []

        async def emit(self, signal) -> None:
            self.kinds.append(signal.kind)

    telemetry = FakeTelemetry()
    mirror = ShadowMirror(_config(tmp_path, sample_rate=0.0, auto_disable={"window": 1}),
                          ScriptedProvider([]), telemetry=telemetry)
    mirror.signal_bus = Bus()
    await _fail_round(mirror)
    await mirror.drain()
    assert mirror.signal_bus.kinds == ["shadow_auto_disabled"]
    assert telemetry.events == [], "one durable write, not two"


async def test_a_broken_notify_hook_is_logged_not_raised(tmp_path):
    def notify(text: str) -> None:
        raise RuntimeError("telegram down")

    mirror = ShadowMirror(_config(tmp_path, sample_rate=0.0, auto_disable={"window": 1}),
                          ScriptedProvider([]), notify=notify)
    await _fail_round(mirror)
    await mirror.drain()
    assert not mirror.enabled


# ── reading replies ────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("a, b, same", [
    ({"path": "a.py", "line": 3}, {"line": 3, "path": "a.py"}, True),       # key order
    ({"n": 2}, {"n": 2.0}, True),                                           # number spelling
    ({"flag": True}, {"flag": 1}, False),                                   # true is not 1
    ({"command": "ls -la\n"}, {"command": "ls -la"}, True),                 # trailing whitespace
    ({"command": "  ls -la"}, {"command": "ls -la"}, True),                 # on a command line
    ({"path": "a.py", "limit": None}, {"path": "a.py"}, True),              # null == absent
    ({"items": [1, 2]}, {"items": [2, 1]}, False),                          # order matters in lists
    ({"command": "ls"}, {"command": "ls -la"}, False),
    # Whitespace anywhere else is the argument: a re-indent is not a no-op edit, nor YAML its
    # unindented self, nor a file without its final newline the same file.
    ({"path": "a.py", "old_string": "    pass", "new_string": "        pass"},
     {"path": "a.py", "old_string": "pass", "new_string": "pass"}, False),
    ({"path": "c.yaml", "content": "  key: 1\n"}, {"path": "c.yaml", "content": "key: 1"}, False),
    ({"path": "a.py", "content": "x = 1\n"}, {"path": "a.py", "content": "x = 1"}, False),
    ({"args": {"command": "ls\n"}}, {"args": {"command": "ls"}}, False),   # top level only
])
def test_args_equivalence(a, b, same):
    assert args_equivalent(a, b) is same


READ_FILE = {"name": "read_file", "description": "read", "input_schema": {
    "type": "object", "properties": {
        "path": {"type": "string"},
        "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "raw": {"type": "boolean"},
    }}}


def test_text_emitted_values_are_typed_by_the_round_s_schemas():
    """``<parameter=limit>50`` is the integer 50 when the offered schema says so, as the loop's own
    extraction (which passes its registry) reads it; text otherwise."""
    xml = ConversationMessage(role="assistant", content=[TextBlock(
        text="<function=read_file><parameter=path>a.py</parameter><parameter=limit>50</parameter>"
             "<parameter=raw>true</parameter></function>")])
    [call], from_text = reply_calls(xml, [READ_FILE])
    assert from_text and call.arguments == {"path": "a.py", "limit": 50, "raw": True}
    assert args_equivalent({"path": "a.py", "limit": 50, "raw": True}, call.arguments)
    # A string parameter that happens to look like a number stays a string.
    num_path = ConversationMessage(role="assistant", content=[TextBlock(
        text="<function=read_file><parameter=path>50</parameter></function>")])
    assert reply_calls(num_path, [READ_FILE])[0][0].arguments == {"path": "50"}
    # A tool the round did not offer is still the reply's call (untyped), never filtered out.
    other = ConversationMessage(role="assistant", content=[TextBlock(
        text="<function=made_up><parameter=n>3</parameter></function>")])
    assert [(c.name, c.arguments) for c in reply_calls(other, [READ_FILE])[0]] == [
        ("made_up", {"n": "3"})]


async def test_a_text_call_agrees_with_the_same_structured_call(tmp_path):
    """The primary (hosted, structured) and the shadow (text) make the same typed call."""
    shadow = ScriptedProvider([_text_events(
        "<tool_call><function=read_file><parameter=path>a.py</parameter>"
        "<parameter=limit>50</parameter></function></tool_call>")])
    mirror = ShadowMirror(_config(tmp_path), shadow)
    request = ApiMessageRequest(model="primary-model", tools=[READ_FILE],
                                messages=[ConversationMessage.from_user_text("read a.py")])
    await _consume(mirror.mirror(_stream(_tool_events("read_file", {"path": "a.py", "limit": 50})),
                                 request, session_id="telegram:1"))
    await mirror.drain()
    [row] = _rows(tmp_path)
    assert row["shadow"]["calls_from_text"] is True and row["shadow"]["args"] == [
        {"path": "a.py", "limit": 50}]
    assert row["same_tool"] is True and row["args_equivalent"] is True


def test_text_emitted_calls_are_read_only_from_tool_call_markup():
    xml = ConversationMessage(role="assistant", content=[TextBlock(
        text="<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>")])
    calls, from_text = reply_calls(xml)
    assert [c.name for c in calls] == ["bash"] and from_text
    prose = ConversationMessage(role="assistant", content=[TextBlock(
        text='Send it like this: {"name": "bash", "arguments": {"command": "ls"}}')])
    assert reply_calls(prose) == ([], False)


def test_answer_normalization():
    assert normalize_answer_text("  The Answer\n is 42. ") == "the answer is 42"
    assert normalize_answer_text("<think>hmm</think>Yes") == "yes"
    assert normalize_answer_text(None) == ""


@pytest.mark.parametrize("calls, session, kw, history, expected", [
    ([], "telegram:1", {"suppress_tools": True}, None, "chat-mode"),
    ([], "telegram:1", {"tool_choice": "none"}, None, "chat-mode"),
    ([("bash", {"command": "systemctl --user status prometheus"})], "telegram:1", {}, None,
     "sentinel/daemon"),
    ([("cron_create", {})], "telegram:1", {}, None, "sentinel/daemon"),
    ([("read_file", {"path": "a.py"})], None, {}, None, "sentinel/daemon"),
    ([("wiki_search", {"q": "x"})], "telegram:1", {}, None, "memory/wiki"),
    ([("read_file", {"path": "~/wiki/notes.md"})], "telegram:1", {}, None, "memory/wiki"),
    ([("edit_file", {"path": "a.py"})], "telegram:1", {}, None, "coding"),
    ([("bash", {"command": "pytest -x"})], "telegram:1", {}, None, "coding"),
    ([("web_search", {"q": "x"})], "telegram:1", {}, None, "other"),
    ([], "telegram:1", {}, None, "other"),
    ([], "telegram:1", {}, ("edit_file", {"path": "a.py"}), "coding"),       # earlier in the turn
])
def test_round_categories(calls, session, kw, history, expected):
    from prometheus.engine.shadow import Call

    messages = [ConversationMessage.from_user_text("do the thing")]
    if history:
        messages += [
            ConversationMessage(role="assistant",
                                content=[ToolUseBlock(id="h1", name=history[0], input=history[1])]),
            ConversationMessage(role="user",
                                content=[ToolResultBlock(tool_use_id="h1", content="ok")]),
        ]
    request = ApiMessageRequest(model="m", messages=messages, tools=[{"name": "x"}], **kw)
    assert classify_round(request, [Call(n, a) for n, a in calls], session) == expected


def test_a_turn_opened_by_the_runtime_is_sentinel_work():
    from prometheus.engine.shadow import Call

    request = ApiMessageRequest(model="m", tools=[{"name": "x"}], messages=[
        ConversationMessage.from_injected("A worker you started finished.",
                                          provenance="task_supervisor"),
    ])
    assert classify_round(request, [Call("edit_file", {})], "telegram:1") == "sentinel/daemon"


# ── wiring ─────────────────────────────────────────────────────────────────────────────────


def test_the_daemon_hands_one_mirror_to_both_loops():
    tree = ast.parse(DAEMON.read_text(encoding="utf-8"))
    passed: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("AgentLoop", "LoopContext"):
            for kw in node.keywords:
                if kw.arg == "shadow" and isinstance(kw.value, ast.Name):
                    passed[node.func.id] = kw.value.id  # type: ignore[attr-defined]
    assert passed == {"AgentLoop": "shadow_mirror", "LoopContext": "shadow_mirror"}


def test_loop_context_shadow_defaults_off():
    import dataclasses

    fields = {f.name: f for f in dataclasses.fields(LoopContext)}
    assert fields["shadow"].default is None
