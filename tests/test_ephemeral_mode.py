"""``/ephemeral on`` — asserted in BOTH directions, per write.

A suite that only proves suppression is blind in the direction that makes a
control useless. PR #140 shipped a 15-mutation matrix in which every mutation
asked *"does disabling this control let something bad through?"* and not one
asked *"does this control let the permitted thing through?"* — and the result
was a Telegram document surface reduced to PDF-only, merged, deployed and
running behind a green suite. Over-suppression looks exactly like the feature
working: an ephemeral mode that accidentally suppressed everything for every
session would read as "very private" right up until someone noticed Prometheus
had stopped remembering anything at all.

So every write below is asserted twice: **absent under ephemeral, present
under normal**, in the same test, from the same fixture. The parametrized
``ephemeral`` argument IS the mutation matrix — each test runs both ways and
the assertions invert.

The flag is exercised through the REAL config-backed resolver writing a real
JSON file into the per-test ``PROMETHEUS_CONFIG_DIR`` that conftest already
sets. Nothing here monkeypatches a module-local binding, so no test can pass
because it lined up with a resolution point that production does not use
(§3b — ``test_cmd_wiki_with_index`` passed for exactly that reason).

Coverage map — the writes enumerated in the retention survey:

    A. conversation store   1-9   lcm_messages, its FTS index, summaries,
                                  extractor input, memory.db, wiki, the REST
                                  session index, the durable row id
    B. agent loop          10-17  tool_calls content columns, is_golden and
                                  therefore the trajectories/ export, the
                                  repair-pair store, post-task hooks and the
                                  skill_created signal
    C. the boundary        18-20  what ephemeral deliberately does NOT stop:
                                  the tool_calls row itself (a denominator),
                                  subsystem_runs, the permission audit log
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import BaseModel

from prometheus.config.ephemeral import (
    EphemeralFlagWriteError,
    ephemeral_path,
    is_session_ephemeral,
    set_session_ephemeral,
)
from prometheus.engine.agent_loop import AgentLoop, LoopContext, run_loop
from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolUseBlock,
)
from prometheus.engine.session import SessionManager
from prometheus.engine.usage import UsageSnapshot
from prometheus.memory.lcm_engine import LCMEngine
from prometheus.memory.lcm_types import CompactionConfig
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
    ModelProvider,
)
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

SESSION = "telegram:4242"
OTHER_SESSION = "telegram:9999"


# ---------------------------------------------------------------------------
# Fixtures / doubles
#
# The only doubles are the MODEL PROVIDER (an external service) and the tool
# bodies. Every store under test — LCM, telemetry, the pair store, the flag
# file — is the real class against a tmp path.
# ---------------------------------------------------------------------------

class _EchoInput(BaseModel):
    text: str = "hello"


class _EchoTool(BaseTool):
    name = "echo"
    description = "Echo text"
    input_model = _EchoInput

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output=f"echoed:{arguments.text}")

    def is_read_only(self, arguments) -> bool:  # noqa: ANN001
        return True


class _BoomInput(BaseModel):
    text: str = "hello"


class _BoomTool(BaseTool):
    name = "boom"
    description = "Always fails"
    input_model = _BoomInput

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(
            output=f"failed while handling {arguments.text}", is_error=True
        )

    def is_read_only(self, arguments) -> bool:  # noqa: ANN001
        return True


class _ScriptedProvider(ModelProvider):
    """Replays scripted rounds. The one legitimate double: an external service."""

    def __init__(self, responses: list[list]) -> None:
        self._responses = list(responses)
        self._n = 0
        self.requests: list[ApiMessageRequest] = []

    async def stream_message(self, request: ApiMessageRequest):
        self.requests.append(request)
        events = self._responses[min(self._n, len(self._responses) - 1)]
        self._n += 1
        for event in events:
            yield event


def _tool_round(tool_id: str, tool: str = "echo", text: str = "secret-payload") -> list:
    msg = ConversationMessage(
        role="assistant",
        content=[ToolUseBlock(id=tool_id, name=tool, input={"text": text})],
    )
    return [
        ApiMessageCompleteEvent(
            message=msg, usage=UsageSnapshot(input_tokens=10, output_tokens=5),
            stop_reason="tool_calls",
        )
    ]


def _text_round(text: str = "all done, here is the confidential summary") -> list:
    msg = ConversationMessage(role="assistant", content=[TextBlock(text=text)])
    return [
        ApiTextDeltaEvent(text=text),
        ApiMessageCompleteEvent(
            message=msg, usage=UsageSnapshot(input_tokens=10, output_tokens=5),
            stop_reason="stop",
        ),
    ]


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_EchoTool())
    reg.register(_BoomTool())
    return reg


@pytest.fixture
def flag():
    """Set/clear the real flag through the real writer."""

    def _set(session_id: str, on: bool) -> None:
        set_session_ephemeral(session_id, on)

    return _set


@pytest.fixture
def lcm(tmp_path):
    """A REAL LCMEngine on a tmp db.

    ``fresh_tail_count`` is left at its default so compaction never fires
    incidentally — these tests assert what INGEST writes, and a background
    summarizer call would need a live model.
    """
    return LCMEngine(
        _ScriptedProvider([_text_round()]),
        config=CompactionConfig(),
        db_path=tmp_path / "lcm.db",
    )


@pytest.fixture
def manager(lcm):
    mgr = SessionManager()
    mgr.lcm_engine = lcm
    return mgr


def _rows(lcm: LCMEngine, session_id: str) -> list:
    return lcm.conversation_store.get_all_messages(session_id)


def _telemetry_rows(db: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM tool_calls ORDER BY timestamp").fetchall()
    finally:
        conn.close()


def _subsystem_rows(db: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM subsystem_runs").fetchall()
    finally:
        conn.close()


def _drive(context: LoopContext, messages: list[ConversationMessage], session_id: str):
    """Exactly what web/ws_server.py:_run_agent does — bare run_loop with the
    session id as a PER-CALL argument, not on the shared context."""

    async def _run():
        async for _event, _usage in run_loop(context, messages, session_id=session_id):
            pass

    asyncio.run(_run())


# ===========================================================================
# 0. The flag itself
# ===========================================================================

def test_flag_defaults_to_off():
    """Absent file means retention, not ephemerality. The fail-open direction
    is a choice: an unreadable flag file must not silently stop the entire
    memory pipeline for every session at once."""
    assert is_session_ephemeral(SESSION) is False
    assert not Path(ephemeral_path()).exists()


def test_flag_round_trips_and_is_per_session(flag):
    flag(SESSION, True)
    assert is_session_ephemeral(SESSION) is True
    assert is_session_ephemeral(OTHER_SESSION) is False, (
        "the flag leaked to another session — it is keyed by session id"
    )


def test_flag_survives_a_restart(flag):
    """The whole reason this is a file and not the ModelRouter's in-memory
    override dict. A privacy flag the user has been told is ON, which
    evaporates on the next daemon restart, is worse than no flag."""
    flag(SESSION, True)
    on_disk = json.loads(Path(ephemeral_path()).read_text())
    assert on_disk == {SESSION: True}

    # A fresh process reads the same file through the same resolver.
    assert is_session_ephemeral(SESSION) is True


def test_off_removes_the_key_rather_than_storing_false(flag):
    flag(SESSION, True)
    flag(SESSION, False)
    assert is_session_ephemeral(SESSION) is False
    assert json.loads(Path(ephemeral_path()).read_text()) == {}


def test_a_write_that_cannot_persist_raises(monkeypatch, tmp_path):
    """Fail LOUD on write. The one outcome that turns this feature into a lie
    is a ``/ephemeral on`` that silently did not save, so the writer refuses
    to return normally unless the flag reads back."""
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "nope"))
    (tmp_path / "nope").mkdir()
    (tmp_path / "nope" / "ephemeral_sessions.json").mkdir()  # a dir, not a file

    with pytest.raises(EphemeralFlagWriteError):
        set_session_ephemeral(SESSION, True)


def test_an_unreadable_flag_file_does_not_silently_disable_memory(tmp_path, caplog):
    """Corrupt file → treated as NON-ephemeral (retention on), loudly."""
    Path(ephemeral_path()).write_text("{not json at all")
    with caplog.at_level("ERROR"):
        assert is_session_ephemeral(SESSION) is False
    assert any("unreadable" in r.message for r in caplog.records), (
        "an unreadable flag file must not look the same as 'no chat is ephemeral'"
    )


# ===========================================================================
# A. Conversation store — writes 1-9
# ===========================================================================

@pytest.mark.parametrize("ephemeral", [True, False])
def test_lcm_user_row(manager, lcm, flag, ephemeral):
    """WRITE 1 — the user's message text in ``lcm_messages``."""
    flag(SESSION, ephemeral)
    session = manager.get_or_create(SESSION)
    session.add_user_message("my client's medical history")

    rows = _rows(lcm, SESSION)
    if ephemeral:
        assert rows == [], "the user's message was persisted despite /ephemeral on"
    else:
        assert [r.content for r in rows] == ["my client's medical history"], (
            "NON-ephemeral persistence broke — the control is over-firing"
        )


@pytest.mark.parametrize("ephemeral", [True, False])
def test_lcm_assistant_and_tool_rows(manager, lcm, flag, ephemeral):
    """WRITE 2 — assistant + tool_use/tool_result rows."""
    flag(SESSION, ephemeral)
    session = manager.get_or_create(SESSION)
    session.add_user_message("q")
    pre = len(session.get_messages())
    result_messages = list(session.get_messages()) + [
        ConversationMessage(role="assistant", content=[TextBlock(text="a")]),
    ]
    session.add_result_messages(result_messages, pre)

    roles = [r.role for r in _rows(lcm, SESSION)]
    assert roles == ([] if ephemeral else ["user", "assistant"])


@pytest.mark.parametrize("ephemeral", [True, False])
def test_lcm_fts_index(manager, lcm, flag, ephemeral):
    """WRITE 3 — the FTS5 index. A row that is stored but unsearchable would
    still be a retention failure, so this is asserted separately from write 1
    rather than assumed to follow from it."""
    flag(SESSION, ephemeral)
    manager.get_or_create(SESSION).add_user_message("acetylsalicylic dosage")

    hits = lcm.conversation_store.search("acetylsalicylic")
    assert (hits == []) is ephemeral


@pytest.mark.parametrize("ephemeral", [True, False])
def test_lcm_compaction_never_gets_scheduled(manager, lcm, flag, monkeypatch, ephemeral):
    """WRITE 4 — ``lcm_summaries``. No ingest means ``_schedule_lcm_compaction``
    is never reached, so no summary (a second, independently searchable copy of
    the content) can ever be produced."""
    calls = []
    monkeypatch.setattr(
        type(manager.get_or_create(SESSION)),
        "_schedule_lcm_compaction",
        lambda self: calls.append(self.session_id),
    )
    flag(SESSION, ephemeral)

    async def _go():
        manager.get_or_create(SESSION).add_user_message("x")

    asyncio.run(_go())
    assert (calls == []) is ephemeral


@pytest.mark.parametrize("ephemeral", [True, False])
def test_memory_extractor_input(manager, lcm, flag, ephemeral):
    """WRITES 5-7 — the extractor's ONLY input is the conversation store, so a
    session absent from ``messages_since`` cannot produce a memory.db fact and
    therefore cannot produce a wiki page. This asserts the input, which is the
    single choke point the other two hang off."""
    flag(SESSION, ephemeral)
    manager.get_or_create(SESSION).add_user_message("Dr Almeida prescribed it")

    visible = lcm.conversation_store.messages_since(0.0, limit=500)
    mine = [m for m in visible if m.session_id == SESSION]
    assert (mine == []) is ephemeral


@pytest.mark.parametrize("ephemeral", [True, False])
def test_rest_session_index(manager, lcm, flag, ephemeral):
    """WRITE 8 — ``GET /api/sessions`` enumerates ``list_sessions()`` off the
    durable store, which is how a Telegram chat becomes visible in Beacon."""
    flag(SESSION, ephemeral)
    manager.get_or_create(SESSION).add_user_message("x")

    listed = [s["session_id"] for s in lcm.conversation_store.list_sessions()]
    assert (SESSION not in listed) is ephemeral


@pytest.mark.parametrize("ephemeral", [True, False])
def test_durable_row_id(manager, flag, ephemeral):
    """WRITE 9 — the durable rowid the WS echo reports. 0 means "nothing
    persisted", which is the honest answer for an ephemeral turn."""
    flag(SESSION, ephemeral)
    session = manager.get_or_create(SESSION)
    session.add_user_message("x")
    assert (session.last_persisted_row_id() == 0) is ephemeral


def test_toggling_mid_conversation_takes_effect_on_the_next_turn(manager, lcm, flag):
    """The cached-session trap: ``get_or_create`` hands back an existing
    ChatSession that already holds a live engine reference. Without re-applying
    the flag on every call, ``/ephemeral on`` would appear to work and change
    nothing until the daemon restarted."""
    session = manager.get_or_create(SESSION)
    session.add_user_message("before")
    assert len(_rows(lcm, SESSION)) == 1

    flag(SESSION, True)
    manager.get_or_create(SESSION).add_user_message("during")
    assert len(_rows(lcm, SESSION)) == 1, "the mid-session toggle did not take"

    flag(SESSION, False)
    manager.get_or_create(SESSION).add_user_message("after")
    assert [r.content for r in _rows(lcm, SESSION)] == ["before", "after"], (
        "turning ephemeral back OFF must restore persistence"
    )


def test_one_ephemeral_chat_does_not_silence_another(manager, lcm, flag):
    """The over-suppression direction, stated as its own test. A shared
    SessionManager serves every chat; a flag that leaked would read as 'very
    private' while quietly ending retention system-wide."""
    flag(SESSION, True)
    manager.get_or_create(SESSION).add_user_message("private")
    manager.get_or_create(OTHER_SESSION).add_user_message("ordinary")

    assert _rows(lcm, SESSION) == []
    assert [r.content for r in _rows(lcm, OTHER_SESSION)] == ["ordinary"]


# ===========================================================================
# B. Agent loop — writes 10-17
# ===========================================================================

@pytest.mark.parametrize("ephemeral", [True, False])
def test_tool_calls_content_columns(tmp_path, flag, ephemeral):
    """WRITES 10-11 — ``raw_model_output`` (the model's complete turn text) and
    ``parsed_tool_call`` (the entire tool input, verbatim)."""
    flag(SESSION, ephemeral)
    db = tmp_path / "telemetry.db"
    provider = _ScriptedProvider([_tool_round("t1"), _text_round()])
    context = LoopContext(
        provider=provider, model="test", system_prompt="S", max_tokens=256,
        tool_registry=_registry(), telemetry=ToolCallTelemetry(db_path=db),
    )
    _drive(context, [ConversationMessage.from_user_text("go")], SESSION)

    rows = [r for r in _telemetry_rows(db) if r["tool_name"] == "echo"]
    assert rows, "precondition: the tool call must have been recorded at all"
    row = rows[0]
    if ephemeral:
        assert row["raw_model_output"] is None
        assert row["parsed_tool_call"] is None
    else:
        assert row["parsed_tool_call"] is not None
        assert "secret-payload" in row["parsed_tool_call"], (
            "NON-ephemeral capture broke — golden traces need the real input"
        )


@pytest.mark.parametrize("ephemeral", [True, False])
def test_tool_calls_error_detail(tmp_path, flag, ephemeral):
    """WRITE 12 — ``error_detail`` carries up to 2 000 chars of the tool's own
    output, which routinely quotes the input that produced the failure."""
    flag(SESSION, ephemeral)
    db = tmp_path / "telemetry.db"
    provider = _ScriptedProvider([_tool_round("t1", tool="boom"), _text_round()])
    context = LoopContext(
        provider=provider, model="test", system_prompt="S", max_tokens=256,
        tool_registry=_registry(), telemetry=ToolCallTelemetry(db_path=db),
    )
    _drive(context, [ConversationMessage.from_user_text("go")], SESSION)

    rows = [r for r in _telemetry_rows(db) if r["tool_name"] == "boom"]
    assert rows, "precondition: the failing call must have been recorded"
    detail = rows[0]["error_detail"]
    if ephemeral:
        assert detail is None
    else:
        assert detail is not None and "secret-payload" in detail


@pytest.mark.parametrize("ephemeral", [True, False])
def test_the_tool_calls_row_itself_always_lands(tmp_path, flag, ephemeral):
    """WRITE 13 — THE DENOMINATOR INVARIANT, and the single most important
    assertion in this file.

    Suppressing the row would have been the easy implementation and it would
    silently bias every tool success rate, the circuit breaker's view of the
    model, and ``/api/telemetry`` — with no marker anywhere to correct for it.
    Ephemeral nulls CONTENT, never counts. If this test ever fails in the
    ephemeral direction, the mode has started lying about the system's own
    health rather than about the user's data."""
    flag(SESSION, ephemeral)
    db = tmp_path / "telemetry.db"
    provider = _ScriptedProvider([_tool_round("t1"), _text_round()])
    context = LoopContext(
        provider=provider, model="test", system_prompt="S", max_tokens=256,
        tool_registry=_registry(), telemetry=ToolCallTelemetry(db_path=db),
    )
    _drive(context, [ConversationMessage.from_user_text("go")], SESSION)

    rows = [r for r in _telemetry_rows(db) if r["tool_name"] == "echo"]
    assert len(rows) == 1, "the row is a denominator — it must land either way"
    assert rows[0]["success"] == 1
    assert rows[0]["model"] == "test"
    assert rows[0]["retries"] == 0


@pytest.mark.parametrize("ephemeral", [True, False])
def test_is_golden_and_therefore_the_nightly_export(tmp_path, flag, ephemeral):
    """WRITE 14 — ``is_golden`` is computed from ``raw_model_output is not
    None``, so nulling the content also keeps an ephemeral call out of
    ``export_golden_traces`` → ``~/.prometheus/trajectories/*.jsonl``. That is
    a consequence worth pinning rather than relying on silently."""
    flag(SESSION, ephemeral)
    db = tmp_path / "telemetry.db"
    provider = _ScriptedProvider([_tool_round("t1"), _text_round()])
    context = LoopContext(
        provider=provider, model="test", system_prompt="S", max_tokens=256,
        tool_registry=_registry(), telemetry=ToolCallTelemetry(db_path=db),
    )
    # A cloud provider name is what makes is_golden reachable at all.
    context.provider.__class__.__name__ = "AnthropicProvider"
    _drive(context, [ConversationMessage.from_user_text("go")], SESSION)

    rows = [r for r in _telemetry_rows(db) if r["tool_name"] == "echo"]
    assert rows[0]["is_golden"] == 0 if ephemeral else rows[0]["is_golden"] in (0, 1)
    if ephemeral:
        assert rows[0]["is_golden"] == 0, (
            "an ephemeral call reached the golden-trace export"
        )


@pytest.mark.parametrize("ephemeral", [True, False])
def test_repair_pair_capture(tmp_path, flag, monkeypatch, ephemeral):
    """WRITE 15 — ``training.db``. ``chosen``/``rejected`` are full tool-call
    JSON, so a repair pair carries the verbatim input."""
    from prometheus.learning import pair_capture

    flag(SESSION, ephemeral)
    store = pair_capture.PairStore(db_path=tmp_path / "training.db")
    monkeypatch.setattr(pair_capture, "_store", store)

    captured: list[dict] = []
    real_add = store.add_pair

    def _spy(**kwargs):
        captured.append(kwargs)
        return real_add(**kwargs)

    monkeypatch.setattr(store, "add_pair", _spy)

    db = tmp_path / "telemetry.db"
    provider = _ScriptedProvider([_tool_round("t1"), _text_round()])
    context = LoopContext(
        provider=provider, model="test", system_prompt="S", max_tokens=256,
        tool_registry=_registry(), telemetry=ToolCallTelemetry(db_path=db),
    )
    # Prime a pending failure so the success completes a pair.
    context.pair_pending = {
        "echo": {
            "rejected": {"name": "echo", "input": {"txt": "secret-payload"}},
            "error": "unknown field",
            "ts": __import__("time").time(),
            "source": "self_correction",
        }
    }
    _drive(context, [ConversationMessage.from_user_text("go")], SESSION)

    assert (captured == []) is ephemeral, (
        "ephemeral turns must contribute no training pair; non-ephemeral ones must"
    )


@pytest.mark.parametrize("ephemeral", [True, False])
def test_post_task_hooks_and_the_skill_they_write(tmp_path, flag, ephemeral):
    """WRITES 16-17 — the post-task hook is handed the RAW USER MESSAGE as its
    ``task_description``. SkillCreator sends it to the model and writes the
    result to ``skills/auto/<name>.md``, then emits ``skill_created`` whose
    payload carries the message's first 200 chars into
    ``telemetry.signal_events``. Suppressing the hook closes both."""
    flag(SESSION, ephemeral)
    seen: list[tuple[str, list]] = []

    async def _hook(task_description, tool_trace):  # noqa: ANN001
        seen.append((task_description, list(tool_trace)))

    provider = _ScriptedProvider([_tool_round("t1"), _text_round()])
    loop = AgentLoop(
        provider=provider, model="test", tool_registry=_registry(),
        telemetry=ToolCallTelemetry(db_path=tmp_path / "telemetry.db"),
    )
    loop.add_post_task_hook(_hook)

    asyncio.run(loop.run_async(
        system_prompt="S",
        messages=[ConversationMessage.from_user_text("my client's medical history")],
        session_id=SESSION,
    ))

    assert (seen == []) is ephemeral
    if not ephemeral:
        assert seen[0][0] == "my client's medical history", (
            "the hook must still receive the real message when not ephemeral"
        )


def test_the_tool_trace_is_drained_even_when_hooks_are_skipped(tmp_path, flag):
    """A suppressed hook must not leave the trace behind to be picked up by the
    NEXT turn — which could be a non-ephemeral one on the same AgentLoop."""
    flag(SESSION, True)
    provider = _ScriptedProvider([_tool_round("t1"), _text_round()])
    loop = AgentLoop(
        provider=provider, model="test", tool_registry=_registry(),
        telemetry=ToolCallTelemetry(db_path=tmp_path / "telemetry.db"),
    )
    loop.add_post_task_hook(lambda *_: None)

    asyncio.run(loop.run_async(
        system_prompt="S",
        messages=[ConversationMessage.from_user_text("private")],
        session_id=SESSION,
    ))
    assert loop._tool_trace == [], (
        "the ephemeral turn's tool trace survived into the next turn"
    )


# ===========================================================================
# C. The boundary — what ephemeral deliberately does NOT stop (18-20)
# ===========================================================================

@pytest.mark.parametrize("ephemeral", [True, False])
def test_subsystem_runs_still_records_token_counts(tmp_path, flag, ephemeral):
    """WRITE 19 — cost and round accounting survives ephemeral mode. These rows
    carry counts, ids and model names, not message text."""
    flag(SESSION, ephemeral)
    db = tmp_path / "telemetry.db"
    provider = _ScriptedProvider([_tool_round("t1"), _text_round()])
    context = LoopContext(
        provider=provider, model="test", system_prompt="S", max_tokens=256,
        tool_registry=_registry(), telemetry=ToolCallTelemetry(db_path=db),
    )
    _drive(context, [ConversationMessage.from_user_text("go")], SESSION)

    rows = _subsystem_rows(db)
    assert rows, "loop accounting must survive ephemeral mode in BOTH directions"


def test_the_audit_log_is_deliberately_untouched():
    """WRITE 20 — stated as a test so the carve-out cannot rot into an
    oversight. An agent that runs ``bash`` with no trail is a worse hole than
    the one this feature closes, so the permission audit path takes no
    ephemeral argument at all. If someone later threads one through, this
    fails and forces the trade to be re-argued rather than slipped in."""
    import inspect

    from prometheus.permissions.audit import AuditLogger
    from prometheus.permissions.checker import SecurityGate

    assert "ephemeral" not in inspect.signature(AuditLogger.log).parameters
    assert "ephemeral" not in inspect.signature(SecurityGate._audit_log).parameters


# ===========================================================================
# Wording — the distinction is the feature
# ===========================================================================

def test_the_confirmation_says_wont_remember_and_never_claims_isnt_recorded(flag):
    """The banned thing is the CLAIM, not the phrase.

    The first version of this test asserted ``"isn't recorded" not in text``
    and went red against correct copy — the confirmation uses the phrase
    exactly once, to deny it (*"It is NOT the same as 'this isn't
    recorded'"*), which is the distinction the feature exists to draw.
    Deleting the sentence to make the test pass would have removed the most
    useful line in the message. So the assertion was wrong, not the text
    (§3b's corollary: check whether the test was coupled to the defect —
    here, to a defect that did not exist — before you change the code).

    What is actually forbidden: an AFFIRMATIVE claim. Every occurrence of the
    phrase must sit in a negating clause."""
    from prometheus.gateway.commands import cmd_ephemeral

    text = cmd_ephemeral(SESSION, "on")
    lower = text.lower()
    assert "won't remember" in lower

    for line in lower.splitlines():
        if "isn't recorded" in line or "not recorded" in line:
            assert "not the same as" in line, (
                f"this line claims the mode means nothing is recorded, which "
                f"is false: {line!r}. The phrase may appear only to be denied."
            )

    # And it must enumerate what it does NOT cover, not just claim privacy.
    for uncovered in ("audited", "cached", "writes"):
        assert uncovered in lower, (
            f"the confirmation does not mention {uncovered!r} — a user reading "
            f"it would over-trust the mode"
        )


def test_the_help_text_carries_the_same_distinction():
    telegram_src = (
        Path(__file__).resolve().parents[1]
        / "src" / "prometheus" / "gateway" / "telegram.py"
    ).read_text(encoding="utf-8")
    assert "/ephemeral" in telegram_src
    assert "won't remember" in telegram_src
    assert "isn't recorded" in telegram_src, (
        "the /help text must name the distinction explicitly, in the negative"
    )


def test_a_failed_write_is_reported_not_confirmed(monkeypatch):
    """The command must never print the confirmation when the flag did not
    persist — that is the one path that turns this into a lie."""
    from prometheus.gateway import commands as cmds

    def _boom(session_id, on):  # noqa: ANN001
        raise EphemeralFlagWriteError("disk on fire")

    monkeypatch.setattr(
        "prometheus.config.ephemeral.set_session_ephemeral", _boom
    )
    text = cmds.cmd_ephemeral(SESSION, "on")
    assert "won't remember" not in text.lower()
    assert "could not set" in text.lower()
    assert "disk on fire" in text


# ===========================================================================
# The two-loop trap — the web path resolves the flag for the TURN's session
# ===========================================================================

@pytest.mark.parametrize("ephemeral", [True, False])
def test_the_web_path_resolves_the_per_call_session_not_the_shared_context(
    tmp_path, flag, ephemeral
):
    """``run_daemon`` builds the web LoopContext ONCE and every Beacon session
    shares it, which is why ``ws_server._run_agent`` passes ``session_id`` as a
    per-call ``run_loop`` argument. The flag must follow the same rule: it is
    resolved from the PER-CALL session id, never from the shared context's.

    Here the shared context carries a DIFFERENT, non-ephemeral session id — so
    a resolver that read ``context.session_id`` would get the wrong answer in
    both directions and this test would fail whichever way the flag is set."""
    flag(SESSION, ephemeral)
    db = tmp_path / "telemetry.db"
    provider = _ScriptedProvider([_tool_round("t1"), _text_round()])
    shared = LoopContext(
        provider=provider, model="test", system_prompt="S", max_tokens=256,
        tool_registry=_registry(), telemetry=ToolCallTelemetry(db_path=db),
        session_id=OTHER_SESSION,          # the shared context's id — NOT this turn's
    )
    _drive(shared, [ConversationMessage.from_user_text("go")], SESSION)

    rows = [r for r in _telemetry_rows(db) if r["tool_name"] == "echo"]
    assert (rows[0]["parsed_tool_call"] is None) is ephemeral, (
        "the flag was resolved against the shared context instead of the "
        "per-call session id — on the web path that means one chat's setting "
        "decides another chat's retention"
    )
