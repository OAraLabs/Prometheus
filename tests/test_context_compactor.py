"""ContextCompactor — side-effect tests per SPRINT-CONTEXT-COMPACTOR.md.

The load-bearing assertion: lcm.db rows are IDENTICAL before/after
compaction (full-table dump comparison — raw file bytes are WAL-unstable
even without writes, so dump-identical is the honest equivalent of the
spec's byte-identical). Recorded/canned providers only — no live model.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prometheus.context.compactor import (
    MIN_SPAN_MESSAGES,
    SUMMARY_MARKER_PREFIX,
    ContextCompactor,
)
from prometheus.context.token_estimation import estimate_tokens
from prometheus.engine.agent_loop import AgentLoop
from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
)
from prometheus.engine.session import SessionManager
from prometheus.engine.usage import UsageSnapshot
from prometheus.memory.lcm_engine import LCMEngine
from prometheus.memory.lcm_types import CompactionConfig
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiTextDeltaEvent,
)
from prometheus.telemetry.tracker import ToolCallTelemetry

SUMMARY_TEXT = (
    "The user configured the deploy pipeline; files: deploy.sh, config.yaml; "
    "decision: pin python 3.12; pending: firewall rule for the staging box."
)


class _SummarizerProvider:
    """Streams a canned summary; counts calls and records the last request."""

    def __init__(self, text: str = SUMMARY_TEXT) -> None:
        self._text = text
        self.calls = 0
        self.last_request = None

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        self.last_request = request
        yield ApiTextDeltaEvent(text=self._text)


def _history(n_turns: int = 12, filler: int = 400) -> list[ConversationMessage]:
    """A long plain conversation: n_turns user/assistant pairs."""
    msgs: list[ConversationMessage] = []
    for i in range(n_turns):
        msgs.append(ConversationMessage.from_user_text(
            f"User question {i}: " + "x" * filler))
        msgs.append(ConversationMessage(
            role="assistant",
            content=[TextBlock(text=f"Answer {i}: " + "y" * filler)],
        ))
    return msgs


def _compactor(provider=None, telemetry=None, **kw) -> ContextCompactor:
    defaults = dict(
        provider=provider or _SummarizerProvider(),
        model="test-model",
        effective_limit=3000,
        reserve_tokens=500,
        threshold_pct=0.4,   # threshold = 1000 estimated tokens
        protect_recent_turns=3,
        telemetry=telemetry,
    )
    defaults.update(kw)
    return ContextCompactor(**defaults)


def _dump_lcm(db_path: Path) -> list:
    conn = sqlite3.connect(db_path)
    try:
        msgs = conn.execute(
            "SELECT * FROM lcm_messages ORDER BY rowid").fetchall()
        sums = conn.execute(
            "SELECT * FROM lcm_summaries ORDER BY rowid").fetchall()
    finally:
        conn.close()
    return [msgs, sums]


# ---------------------------------------------------------------------------
# The load-bearing test: compaction touches the prompt, never the storage
# ---------------------------------------------------------------------------

async def test_compacts_below_threshold_and_lcm_rows_identical(tmp_path):
    telemetry = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    lcm = LCMEngine(MagicMock(), config=CompactionConfig(),
                    db_path=tmp_path / "lcm.db")
    sm = SessionManager()
    sm.lcm_engine = lcm
    session = sm.get_or_create("s1")
    history = _history()
    session.messages.extend(history)
    session.persist_loop_result(0)  # everything lands in lcm.db

    before_rows = _dump_lcm(tmp_path / "lcm.db")
    assert before_rows[0], "fixture must actually persist rows"

    provider = _SummarizerProvider()
    compactor = _compactor(provider, telemetry)
    tokens_before = compactor.estimate_total("", session.messages)
    assert tokens_before > compactor._threshold_tokens()

    out = await compactor.apply(session.messages, session_id="s1")

    # The assembled estimate drops below the threshold.
    tokens_after = compactor.estimate_total("", out)
    assert tokens_after < tokens_before
    assert tokens_after <= compactor._threshold_tokens()

    # The summary message is present, marked, and provenance-tagged.
    assert out[0].text.startswith(SUMMARY_MARKER_PREFIX)
    assert SUMMARY_TEXT in out[0].text
    assert out[0].provenance == "compactor"
    assert out[0].is_trusted is True
    assert out[0].role == "user"

    # STORAGE UNTOUCHED — the load-bearing assertion.
    assert _dump_lcm(tmp_path / "lcm.db") == before_rows

    # Input never mutated: same length, same objects; the render view shares
    # the un-compacted tail by identity.
    assert len(session.messages) == len(history)
    assert all(a is b for a, b in zip(session.messages, history))
    span_len = len(history) - (len(out) - 1)
    assert out[1:] == history[span_len:]
    assert all(a is b for a, b in zip(out[1:], history[span_len:]))

    # Telemetry records the compaction (spans, tokens, duration).
    conn = sqlite3.connect(tmp_path / "telemetry.db")
    rows = conn.execute(
        "SELECT payload FROM signal_events WHERE signal_type='context_compaction'"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    import json
    payload = json.loads(rows[0][0])
    assert payload["span_messages"] == span_len
    assert payload["tokens_before"] == tokens_before
    assert payload["tokens_after"] == tokens_after
    assert payload["cache_hit"] is False


# ---------------------------------------------------------------------------
# Selection rules
# ---------------------------------------------------------------------------

async def test_protected_tail_never_compacted():
    history = _history()
    compactor = _compactor()
    out = await compactor.apply(history, session_id="s")

    # The last protect_recent_turns user messages (and everything after the
    # earliest of them) survive verbatim, by identity.
    boundary = compactor._protected_boundary(history)
    assert history[boundary:] == out[-(len(history) - boundary):]
    for msg in history[boundary:]:
        assert msg in out


async def test_trust_tagged_injection_is_a_barrier_and_never_summarized():
    history = _history()
    injected = ConversationMessage.from_injected(
        "managed-task context: job 42 still running",
        provenance="task_supervisor",
    )
    history.insert(4, injected)

    provider = _SummarizerProvider()
    compactor = _compactor(provider)
    out = await compactor.apply(history, session_id="s")

    # Span stopped at the barrier: only messages 0-3 were summarized.
    assert out[0].text.startswith(SUMMARY_MARKER_PREFIX)
    assert out[1] is injected  # present verbatim, in order

    # And the injected content never reached the summarizer's input.
    sent_prompt = provider.last_request.messages[0].text
    assert "managed-task context" not in sent_prompt


async def test_under_threshold_is_identity_with_zero_calls(tmp_path):
    telemetry = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    provider = _SummarizerProvider()
    compactor = _compactor(provider, telemetry)
    short = _history(n_turns=2)

    out = await compactor.apply(short, session_id="s")
    assert out is short  # the very same list object
    assert provider.calls == 0
    conn = sqlite3.connect(tmp_path / "telemetry.db")
    assert conn.execute("SELECT COUNT(*) FROM signal_events").fetchone()[0] == 0
    conn.close()


async def test_span_below_minimum_is_left_alone():
    # Over threshold but the compactable prefix is too small to be worth a
    # call: a barrier right at the start.
    history = _history()
    history.insert(0, ConversationMessage.from_injected(
        "cron digest blob " + "z" * 400, provenance="cron"))
    provider = _SummarizerProvider()
    compactor = _compactor(provider)
    out = await compactor.apply(history, session_id="s")
    assert out is history
    assert provider.calls == 0
    assert MIN_SPAN_MESSAGES > 0  # the constant the rule keys on


# ---------------------------------------------------------------------------
# Failure path — loud, explicit fallback
# ---------------------------------------------------------------------------

async def test_failure_path_is_loud_and_falls_back(tmp_path):
    telemetry = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    provider = _SummarizerProvider(text="   ")  # garbage: whitespace only
    compactor = _compactor(provider, telemetry)
    history = _history()

    out = await compactor.apply(history, session_id="s1")
    assert out is history  # explicit fallback: unmodified prompt

    conn = sqlite3.connect(tmp_path / "telemetry.db")
    failed = conn.execute(
        "SELECT payload FROM signal_events"
        " WHERE signal_type='context_compaction_failed'").fetchall()
    silent = conn.execute(
        "SELECT subsystem, operation FROM silent_failures").fetchall()
    conn.close()
    assert len(failed) == 1
    assert ("context_compactor", "summarize_span") in silent


async def test_summary_not_smaller_than_span_is_rejected(tmp_path):
    telemetry = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    provider = _SummarizerProvider(text="w" * 50_000)  # "summary" bigger than span
    compactor = _compactor(provider, telemetry)
    history = _history()

    out = await compactor.apply(history, session_id="s1")
    assert out is history
    conn = sqlite3.connect(tmp_path / "telemetry.db")
    n = conn.execute(
        "SELECT COUNT(*) FROM signal_events"
        " WHERE signal_type='context_compaction_failed'").fetchone()[0]
    conn.close()
    assert n == 1


# ---------------------------------------------------------------------------
# Idempotence — second assembly pays zero model calls (telemetry side effect)
# ---------------------------------------------------------------------------

async def test_second_assembly_makes_zero_additional_calls(tmp_path):
    telemetry = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    provider = _SummarizerProvider()
    compactor = _compactor(provider, telemetry)
    history = _history()

    first = await compactor.apply(history, session_id="s1")
    second = await compactor.apply(history, session_id="s1")

    assert provider.calls == 1
    assert [m.text for m in second] == [m.text for m in first]

    conn = sqlite3.connect(tmp_path / "telemetry.db")
    runs = conn.execute(
        "SELECT COUNT(*) FROM subsystem_runs"
        " WHERE subsystem='context_compactor'").fetchone()[0]
    compactions = conn.execute(
        "SELECT COUNT(*) FROM signal_events"
        " WHERE signal_type='context_compaction'").fetchone()[0]
    conn.close()
    assert runs == 1          # exactly ONE summarization call ever happened
    assert compactions == 2   # both assemblies recorded (second is cache_hit)


# ---------------------------------------------------------------------------
# Config gate — disabled by default
# ---------------------------------------------------------------------------

def test_from_config_default_off_and_overrides():
    assert ContextCompactor.from_config(
        None, provider=MagicMock(), model="m") is None
    assert ContextCompactor.from_config(
        {}, provider=MagicMock(), model="m") is None
    assert ContextCompactor.from_config(
        {"compaction": {"enabled": False}},
        provider=MagicMock(), model="m") is None

    enabled = ContextCompactor.from_config(
        {
            "compaction": {"enabled": True, "threshold_pct": 0.6},
            "context": {
                "effective_limit": 72000,
                "model_overrides": {"small-model": {"effective_limit": 9000}},
            },
        },
        provider=MagicMock(), model="small-model",
    )
    assert enabled is not None
    assert enabled._effective_limit == 9000   # per-model n_ctx override applied
    assert enabled._threshold_pct == 0.6


# ---------------------------------------------------------------------------
# Loop wiring — the provider receives the compacted render view
# ---------------------------------------------------------------------------

class _CapturingLoopProvider:
    """Main-loop provider: returns one plain answer, captures the request."""

    def __init__(self) -> None:
        self.requests = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        msg = ConversationMessage(
            role="assistant", content=[TextBlock(text="ok")])
        yield ApiMessageCompleteEvent(
            message=msg,
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


async def test_run_loop_sends_compacted_render_view():
    loop_provider = _CapturingLoopProvider()
    compactor = _compactor()  # its own summarizer provider
    agent = AgentLoop(
        provider=loop_provider, model="test-model", compactor=compactor)

    history = _history()
    result = await agent.run_async(
        system_prompt="sys", messages=history, session_id="s-loop")
    assert result.text == "ok"

    sent = loop_provider.requests[0].messages
    assert sent[0].text.startswith(SUMMARY_MARKER_PREFIX)
    assert not any("User question 0" in m.text for m in sent)  # span gone
    # And the session's own list was never mutated by the render view.
    assert history[0].text.startswith("User question 0")


async def test_run_loop_without_compactor_sends_full_history():
    loop_provider = _CapturingLoopProvider()
    agent = AgentLoop(provider=loop_provider, model="test-model")
    history = _history()
    await agent.run_async(system_prompt="sys", messages=history,
                          session_id="s-loop")
    sent = loop_provider.requests[0].messages
    assert any("User question 0" in m.text for m in sent)
    assert not any(m.text.startswith(SUMMARY_MARKER_PREFIX) for m in sent)
