"""``memory.extraction.enabled: false`` means nothing learns memory from chats.

Off means nothing RUNS, not just nothing saved: no extractor is built (building
one already writes a cursor row to ``memory.db``), no ``memory_extractor`` task
starts, the LCM pre-compaction flush has no extractor to call, and the wiki
compile that runs after each extraction never fires. The probe is the real
``memory.db`` and wiki tree, and it must read zero; the same chat with the
switch on is the positive control that proves the probe can see a write.

Beside the background path, the switch also takes the model's own writers off
the table: the ``memory`` tool (MEMORY.md / USER.md) is not registered, and
``wiki_query`` stops filing the question asked into ``wiki/queries/``. The
idle-time insight writer, which works from extracted facts, stays off too.
``/note`` and the Beacon memory editor are a person typing, and stay.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from prometheus.memory.extractor import MemoryExtractor
from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_types import MessagePart
from prometheus.memory.learning_switch import (
    extraction_enabled,
    insight_synthesis_enabled,
    start_memory_extraction,
)
from prometheus.memory.store import MemoryStore
from prometheus.memory.wiki_compiler import WikiCompiler

MARKER = "SWITCHPROBE_FAVOURITE_COLOUR"
OFF = {"memory": {"extraction": {"enabled": False}}}
ON = {"memory": {"extraction": {"enabled": True}}}


# ── reading the key ──────────────────────────────────────────────────


@pytest.mark.parametrize("config", [{}, {"memory": None}, {"memory": {"extraction": {}}}, ON])
def test_extraction_is_on_unless_switched_off(config):
    assert extraction_enabled(config) is True


@pytest.mark.parametrize("value", [False, "false", "off", "no", "0", 0])
def test_every_spelling_of_off_is_off(value):
    """A privacy switch must not read a quoted ``"false"`` as on."""
    assert extraction_enabled({"memory": {"extraction": {"enabled": value}}}) is False


# ── the background path, probed on the real stores ───────────────────


class _Flush:
    """Stands in for the LCM engine: records what was wired into its pre-compaction flush."""

    def __init__(self) -> None:
        self.extractor: object | None = None

    def set_memory_extractor(self, extractor: object) -> None:
        self.extractor = extractor


def _counts(memory_db: Path, wiki_root: Path) -> tuple[int, int, int]:
    conn = sqlite3.connect(memory_db)
    try:
        facts = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        cursors = conn.execute("SELECT COUNT(*) FROM extractor_cursors").fetchone()[0]
    finally:
        conn.close()
    pages = sum(1 for p in wiki_root.rglob("*") if p.is_file()) if wiki_root.exists() else 0
    return facts, cursors, pages


async def _chat_with_switch(config: dict, tmp_path: Path, monkeypatch) -> dict:
    from prometheus.tools.builtin import lcm_grep

    conv = LCMConversationStore(db_path=tmp_path / "lcm.db")
    # The production path: the extractor finds conversations through the wired LCM engine.
    monkeypatch.setattr(lcm_grep, "_engine", SimpleNamespace(conversation_store=conv), raising=False)

    async def mines_the_marker(self, prompt: str) -> str:
        if MARKER not in prompt:
            return ""
        return ('{"entity_type": "person", "entity_name": "user",'
                f' "fact": "likes {MARKER}", "confidence": 0.9}}')

    monkeypatch.setattr(MemoryExtractor, "_call_model", mines_the_marker)

    memory_db = tmp_path / "memory.db"
    wiki_root = tmp_path / "wiki"
    store = MemoryStore(db_path=memory_db)
    compiler = WikiCompiler(store=store, wiki_root=wiki_root)
    before = _counts(memory_db, wiki_root)
    flush = _Flush()

    extractor, task = start_memory_extraction(
        config, store=store, wiki_compiler=compiler, provider=object(), model="m",
        telemetry=None, lcm_engine=flush,
    )
    # A chat arrives while the daemon runs.
    conv.add_message("web:s1", MessagePart(
        role="user", content=f"My favourite colour is {MARKER}.", session_id="web:s1",
        turn_index=0, provenance="user", is_trusted=True,
    ))
    running = [t.get_name() for t in asyncio.all_tasks()]
    for _ in range(100):  # the first pass runs as soon as the task is scheduled
        await asyncio.sleep(0.01)
        if _counts(memory_db, wiki_root)[0]:
            break
    if task is not None:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    return {
        "extractor": extractor, "task": task, "running": running, "flush": flush,
        "before": before, "after": _counts(memory_db, wiki_root),
    }


@pytest.mark.asyncio
async def test_switched_off_nothing_is_built_started_or_written(tmp_path, monkeypatch):
    seen = await _chat_with_switch(OFF, tmp_path, monkeypatch)

    assert seen["extractor"] is None and seen["task"] is None
    assert "memory_extractor" not in seen["running"]
    assert seen["flush"].extractor is None, "the pre-compaction flush must have nothing to run"
    facts, cursors, pages = seen["after"]
    assert (facts, cursors, pages) == seen["before"] == (0, 0, 0), (
        "facts / extractor cursor rows / wiki files written with extraction off")


@pytest.mark.asyncio
async def test_switched_on_the_same_chat_is_learned(tmp_path, monkeypatch):
    """The positive control: without it a probe that reads zero proves nothing."""
    seen = await _chat_with_switch(ON, tmp_path, monkeypatch)

    assert seen["extractor"] is not None
    assert "memory_extractor" in seen["running"]
    assert seen["flush"].extractor is seen["extractor"]
    facts, _cursors, pages = seen["after"]
    assert facts >= 1, "the chat's fact was not stored"
    assert pages >= 1, "the wiki compile after extraction wrote nothing"


# ── the insight writer ───────────────────────────────────────────────


def test_the_insight_writer_follows_the_switch():
    """It turns extracted facts into wiki insight pages, so it is learning too."""
    assert insight_synthesis_enabled(ON, {"synthesis_enabled": True}) is True
    assert insight_synthesis_enabled(OFF, {"synthesis_enabled": True}) is False
    assert insight_synthesis_enabled(ON, {"synthesis_enabled": False}) is False


# ── the model's own writers ──────────────────────────────────────────


def _tool_names(**kwargs) -> set[str]:
    from prometheus.__main__ import create_tool_registry

    return {t.name for t in create_tool_registry({}, **kwargs).list_tools()}


def test_the_memory_tool_is_not_offered_when_off():
    assert "memory" in _tool_names()
    assert "memory" not in _tool_names(memory_extraction=False)


def test_wiki_query_files_nothing_back_when_off():
    from prometheus.__main__ import create_tool_registry

    for switch in (True, False):
        registry = create_tool_registry({}, memory_extraction=switch)
        tool = next(t for t in registry.list_tools() if t.name == "wiki_query")
        assert tool.files_back is switch


# ── doctor ───────────────────────────────────────────────────────────


def test_doctor_reports_the_switch():
    from prometheus.cli.doctor import check_memory_extraction

    on, off = check_memory_extraction(ON), check_memory_extraction(OFF)
    assert on.status == off.status == "ok"
    assert on.message.startswith("on")
    assert off.message.startswith("off")
    assert "memory.extraction.enabled" in off.message
