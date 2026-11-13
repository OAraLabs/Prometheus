"""Pause everything (Piece 17, PR 2): the doors the second review of #919 found still open, and its lows.

Held here:

  * a STREAMED side call (session titles, conversation summaries) reaches no model while paused: the envelope's
    ``stream()`` refuses as ``call()`` does, before the provider is asked; a pause there is not a provider error
    (no provider-error row, no fallback considered), and a summary pass stops without counting a failure;
  * the curator's tick while paused is skipped, not failed (no failure count, no back-off, no run);
  * ``POST /api/cron/{name}/run`` answers 423 and runs nothing; ``POST /api/code``'s 423 tells the operator who
    paused and why, as the other doors do;
  * a model-written shell is told the pause file too, and a child passes on the file it was told (a grandchild
    reads the same file as its parent).
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from prometheus.engine import pause


@pytest.fixture
def paused(tmp_path):
    state = pause.PauseState(tmp_path / "pause.json")
    pause.install(state)
    state.pause(by="owner", reason="private words")
    yield state
    pause.install(None)


class _Provider:
    def __init__(self) -> None:
        self.called: list[str] = []

    async def stream_message(self, request):
        self.called.append("model")
        if False:
            yield None


def _request():
    from prometheus.providers.base import ApiMessageRequest

    return ApiMessageRequest(model="m", messages=[], system_prompt="s", max_tokens=16)


# ── streamed side calls ──────────────────────────────────────────────────────

def test_a_streamed_side_call_while_paused_reaches_no_model(paused):
    from prometheus.learning.llm_envelope import LLMCallEnvelope

    provider = _Provider()
    env = LLMCallEnvelope("session_titles", telemetry=None)

    async def go():
        async for _ in env.stream(provider=provider, request=_request(), operation="generate"):
            pass

    with pytest.raises(pause.Paused):
        asyncio.run(go())
    assert provider.called == []


def test_a_session_title_is_not_asked_for_while_paused(paused):
    from prometheus.engine.session_titles import generate_title

    provider = _Provider()
    assert asyncio.run(generate_title(provider, "m", "hello", "hi there")) is None
    assert provider.called == []


def test_a_summary_while_paused_asks_nothing_and_counts_no_failure(paused):
    from prometheus.memory.lcm_summarize import LCMSummarizer

    provider = _Provider()
    summarizer = LCMSummarizer(provider, max_retries=2)
    with pytest.raises(pause.Paused):
        asyncio.run(summarizer._call_model("summarize this"))
    assert provider.called == [] and summarizer._consecutive_failures == 0


def test_a_compaction_pass_while_paused_stops_and_marks_nothing(paused, tmp_path):
    from prometheus.memory.lcm_compaction import LCMCompactor
    from prometheus.memory.lcm_conversation_store import LCMConversationStore
    from prometheus.memory.lcm_summarize import LCMSummarizer
    from prometheus.memory.lcm_summary_store import LCMSummaryStore
    from prometheus.memory.lcm_types import CompactionConfig, MessagePart

    db = tmp_path / "lcm.db"
    conv, sums = LCMConversationStore(db_path=db), LCMSummaryStore(db_path=db)
    for i in range(10):
        conv.insert_message(MessagePart(role="user", content=f"message {i}", session_id="s1", turn_index=i,
                                        token_count=5))
    provider = _Provider()
    summarizer = LCMSummarizer(provider)
    compactor = LCMCompactor(conv, sums, summarizer, CompactionConfig(fresh_tail_count=2, compaction_batch_size=3))
    result = asyncio.run(compactor.compact("s1"))
    assert result.summaries_created == 0 and result.messages_compacted == 0
    assert len(conv.get_uncompacted_messages("s1")) == 10
    assert provider.called == [] and summarizer._consecutive_failures == 0


def test_a_pause_in_a_rounds_stream_is_not_a_provider_error():
    from prometheus.engine.agent_loop import _on_stream_error

    async def stream():
        raise pause.Paused(pause.PauseInfo(since=1.0, by="owner"))
        yield None

    errors: list[int] = []

    async def go():
        async for _ in _on_stream_error(stream(), lambda: errors.append(1)):
            pass

    with pytest.raises(pause.Paused):
        asyncio.run(go())
    assert errors == []


def test_a_pause_in_a_rounds_stream_is_not_considered_for_the_fallback(caplog):
    from prometheus.engine.fallback import FallbackTarget, stream_round_with_fallback

    asked: list[str] = []

    class _Envelope:
        def stream(self, *, provider, request, **_):
            asked.append(provider)

            async def gen():
                raise pause.Paused(pause.PauseInfo(since=1.0, by="owner"))
                yield None

            return gen()

    async def go():
        async for _ in stream_round_with_fallback(
                envelope=_Envelope(), provider="CLOUD", model="big", build_request=lambda m: m,
                target=FallbackTarget(model="small", provider_name="llama_cpp", provider="LOCAL",
                                      is_local_backend=True),
                enabled=True, window_for=lambda m: (32_768, True), estimate_tokens=lambda: 100,
                on_degrade=lambda d: None, operation="loop_round", round_index=0, session_id="s1"):
            pass

    caplog.set_level(logging.INFO, logger="prometheus.engine.fallback")
    with pytest.raises(pause.Paused):
        asyncio.run(go())
    assert asked == ["CLOUD"]
    assert "classified" not in caplog.text and "DECLINED" not in caplog.text


def test_a_pause_meeting_the_cascade_is_not_a_compaction_failure(tmp_path):
    """Round 2 of #919's review: a pause landing while the last batch's summary was in flight let the batches finish
    and then met the cascade, which caught only the circuit breaker: compact() raised and the session logged
    "LCM compaction FAILED" with a silent-failure row."""
    from prometheus.memory.lcm_compaction import LCMCompactor
    from prometheus.memory.lcm_conversation_store import LCMConversationStore
    from prometheus.memory.lcm_summary_store import LCMSummaryStore
    from prometheus.memory.lcm_types import CompactionConfig, MessagePart

    class _Summarizer:
        async def summarize_messages(self, messages, *, session_id=None):
            return f"summary of {len(messages)} messages"

        async def summarize_summaries(self, nodes, *, session_id=None):
            raise pause.Paused(pause.PauseInfo(since=1.0, by="owner"))

    db = tmp_path / "lcm.db"
    conv, sums = LCMConversationStore(db_path=db), LCMSummaryStore(db_path=db)
    for i in range(20):
        conv.insert_message(MessagePart(role="user", content=f"message {i}", session_id="s1", turn_index=i,
                                        token_count=5))
    compactor = LCMCompactor(conv, sums, _Summarizer(), CompactionConfig(fresh_tail_count=2, compaction_batch_size=3))
    result = asyncio.run(compactor.compact("s1"))                  # must not raise
    assert result.summaries_created == 6 and len(sums.get_leaf_summaries("s1")) == 6


def test_a_prompt_hook_that_lets_calls_through_on_failure_blocks_while_paused(paused, tmp_path):
    """Round 2 of #919's review: a hook's model call met the pause as an ordinary failure, so a hook with
    block_on_failure: false let a pre_tool_use call run without its verdict (a worker child finishes its round
    with hooks loaded). While paused no verdict can be asked: the call is blocked (fail closed)."""
    from prometheus.hooks.events import HookEvent
    from prometheus.hooks.executor import HookExecutionContext, HookExecutor
    from prometheus.hooks.registry import HookRegistry
    from prometheus.hooks.schemas import PromptHookDefinition

    provider = _Provider()
    registry = HookRegistry()
    registry.add(HookEvent.PRE_TOOL_USE, PromptHookDefinition(prompt="Safe? $ARGUMENTS", block_on_failure=False))
    executor = HookExecutor(registry, HookExecutionContext(cwd=tmp_path, provider=provider, default_model="stub"))
    result = asyncio.run(executor.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "bash"}))
    [one] = result.results
    assert one.success is False and one.blocked is True and "paused" in one.reason
    assert "private words" not in one.reason and provider.called == []


# ── the curator ──────────────────────────────────────────────────────────────

def test_a_curator_tick_while_everything_is_paused_is_skipped_not_failed(paused, tmp_path, monkeypatch):
    from prometheus.learning.curator import Curator
    from prometheus.learning.skill_state import SkillStateStore

    auto = tmp_path / "skills" / "auto"
    auto.mkdir(parents=True)
    c = Curator(_Provider(), auto_dir=auto, reports_dir=tmp_path / "curator",
                state_store=SkillStateStore(tmp_path / "_state.json"), interval_seconds=60, telemetry=None)
    outcomes: list[str] = []
    monkeypatch.setattr(c, "_record_loop_outcome", lambda outcome, duration_ms: outcomes.append(outcome))

    async def _no_run(**kw):
        raise AssertionError("the curator ran while everything was paused")

    monkeypatch.setattr(c, "run_once", _no_run)
    monkeypatch.setattr(c, "_initial_delay_seconds", lambda now: 0.0)

    def _one_tick(failures):
        c._running = False
        return 0.0

    monkeypatch.setattr(c, "_compute_sleep", _one_tick)
    c._running = True
    asyncio.run(c._loop())
    assert outcomes == ["skipped"] and c._consecutive_failures == 0


# ── the routes ───────────────────────────────────────────────────────────────

def test_running_a_cron_job_now_while_paused_is_423_and_runs_nothing(paused, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from prometheus.gateway import cron_scheduler
    from tests.support.device_world import World

    async def _never(job):
        raise AssertionError("a cron job ran while paused")

    monkeypatch.setattr(cron_scheduler, "execute_job", _never)
    w = World(tmp_path)
    pause.install(paused)                                     # World builds the app; keep this state installed
    r = w.call("op", "POST", "/api/cron/nightly/run")
    assert r.status_code == 423, r.text
    assert r.json()["error"] == "paused"


def test_the_coding_door_tells_the_operator_who_paused_and_why(paused, tmp_path):
    pytest.importorskip("fastapi")
    from tests.support.device_world import World

    w = World(tmp_path)
    pause.install(paused)
    r = w.call("op", "POST", "/api/code", json={"description": "fix it", "repo": "/tmp/x"})
    assert r.status_code == 423, r.text
    assert r.json()["reason"] == "private words" and r.json()["by"] == "owner"


# ── what a child is told ─────────────────────────────────────────────────────

def test_a_model_written_shell_is_told_the_pause_file(paused):
    from prometheus.security.shell_floor import model_shell_env

    assert model_shell_env()[pause.CHILD_ENV] == str(paused.path)


def test_a_child_passes_on_the_file_it_was_told(tmp_path, monkeypatch):
    pause.install(None)
    told = tmp_path / "elsewhere" / "pause.json"
    monkeypatch.setenv(pause.CHILD_ENV, str(told))
    monkeypatch.setattr(pause, "pause_path", lambda: tmp_path / "default" / "pause.json")
    assert pause.child_env_entry() == {pause.CHILD_ENV: str(told)}
