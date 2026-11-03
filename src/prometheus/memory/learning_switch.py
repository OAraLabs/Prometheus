"""``memory.extraction.enabled``: whether memory is learned from chats.

On (the default), the daemon mines conversations for facts every 30 minutes and
before a conversation is compacted, compiles what it finds into the wiki, and
the model has the ``memory`` tool. Off, none of that is built or started, so a
deployment that answers for other people can run without keeping what they said.

Off covers the paths that turn chat content into memory without a person asking:

* the fact extractor (its periodic sweep, and the LCM pre-compaction flush it is
  wired into), and the wiki compile that runs after each extraction;
* the idle-time insight writer, which works from the extracted facts;
* the model's ``memory`` tool (MEMORY.md / USER.md) and ``wiki_query`` filing the
  question asked into ``wiki/queries/``.

Recall (reading stored facts) is ``memory.recall.enabled``, a separate switch.
``/note`` and the Beacon memory editor are a person writing, and are not covered.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)

_OFF_SPELLINGS = frozenset({"false", "off", "no", "0"})


def extraction_enabled(config: dict[str, Any] | None) -> bool:
    """True unless ``memory.extraction.enabled`` is set off.

    A quoted ``"false"`` counts as off: ``bool("false")`` is True, and a privacy
    switch must not read a spelling of off as on.
    """
    section = (((config or {}).get("memory") or {}).get("extraction")) or {}
    value = section.get("enabled", True)
    if isinstance(value, str):
        return value.strip().lower() not in _OFF_SPELLINGS
    return bool(value)


def insight_synthesis_enabled(config: dict[str, Any] | None, sentinel_config: dict[str, Any]) -> bool:
    """Whether the idle-time insight writer may run: its own key, and extraction on."""
    return extraction_enabled(config) and bool(sentinel_config.get("synthesis_enabled", True))


def start_memory_extraction(
    config: dict[str, Any] | None,
    *,
    store: Any,
    wiki_compiler: Any,
    provider: Any,
    model: str,
    telemetry: Any,
    lcm_engine: Any,
) -> tuple[Any, asyncio.Task | None]:
    """Build the fact extractor and start its loop, or do neither when switched off.

    Returns ``(extractor, task)``, both None when off. The check comes before the
    extractor is BUILT, not just before its task starts: construction already
    writes a cursor row to ``memory.db``. Needs a running event loop when on.
    """
    if not extraction_enabled(config):
        log.info("Memory extraction off (memory.extraction.enabled: false): no facts or "
                 "wiki pages are learned from chats")
        return None, None

    from prometheus.memory.extractor import MemoryExtractor

    extractor = MemoryExtractor(
        store=store,
        provider=provider,
        model=model,
        post_extract_callback=wiki_compiler.compile,
        telemetry=telemetry,
    )

    # Wire extractor into LCM for pre-compaction flush
    if lcm_engine is not None:
        lcm_engine.set_memory_extractor(extractor)
        log.info("Memory extractor wired to LCM pre-compaction flush")

    # SUNRISE: name the task so it's visible in asyncio.all_tasks() and logs.
    task = asyncio.create_task(extractor.run_forever(), name="memory_extractor")
    log.info("Memory extractor started (task=memory_extractor)")
    return extractor, task
