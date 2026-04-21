"""LCM Engine -- the top-level orchestrator.

Provides the single interface that the agent loop uses to ingest messages,
assemble context, and trigger compaction.  All internal stores, the compactor,
assembler, and summarizer are wired up automatically.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from prometheus.config.paths import get_config_dir, get_data_dir
from prometheus.context.token_estimation import estimate_tokens
from prometheus.memory.lcm_assembler import LCMAssembler
from prometheus.memory.lcm_compaction import LCMCompactor
from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_summarize import LCMCircuitBreakerOpen, LCMSummarizer
from prometheus.memory.lcm_summary_store import LCMSummaryStore
from prometheus.memory.lcm_types import (
    AssemblyResult,
    CompactionConfig,
    CompactionResult,
    LCMStats,
    MessagePart,
)
from prometheus.providers.base import ModelProvider

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config file path
# ---------------------------------------------------------------------------

_PROMETHEUS_YAML = Path(__file__).resolve().parents[4] / "config" / "prometheus.yaml"


def _load_config_from_yaml() -> CompactionConfig:
    """Read compaction-related settings from ``prometheus.yaml``.

    Falls back to :class:`CompactionConfig` defaults if the file is missing
    or the relevant keys are absent.
    """
    cfg = CompactionConfig()
    yaml_path = _PROMETHEUS_YAML
    if not yaml_path.is_file():
        logger.debug("prometheus.yaml not found at %s -- using defaults", yaml_path)
        return cfg

    try:
        with open(yaml_path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        ctx = data.get("context", {})
        if "effective_limit" in ctx:
            cfg.context_threshold = int(ctx["effective_limit"])
        if "fresh_tail_count" in ctx:
            cfg.fresh_tail_count = int(ctx["fresh_tail_count"])
        # These may be added to prometheus.yaml in the future; honour them if
        # present, otherwise keep dataclass defaults.
        if "compaction_batch_size" in ctx:
            cfg.compaction_batch_size = int(ctx["compaction_batch_size"])
        if "max_summary_depth" in ctx:
            cfg.max_summary_depth = int(ctx["max_summary_depth"])
        if "summary_model" in ctx:
            cfg.summary_model = str(ctx["summary_model"])
    except Exception:
        logger.warning("Failed to parse prometheus.yaml -- using default CompactionConfig", exc_info=True)

    return cfg


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class LCMEngine:
    """Top-level orchestrator for the Lossless Context Management system.

    Usage::

        engine = LCMEngine(provider)
        msg_id = await engine.ingest(session, "user", "Hello!")
        ctx = engine.assemble(session, token_budget=16000)
        result = await engine.maybe_compact(session)
        engine.close()

    Or as a context manager::

        with LCMEngine(provider) as engine:
            ...
    """

    def __init__(
        self,
        provider: ModelProvider,
        *,
        config: CompactionConfig | None = None,
        db_path: Path | None = None,
    ) -> None:
        self._config = config or _load_config_from_yaml()
        self._db_path = db_path or (get_data_dir() / "lcm.db")

        # Internal stores.
        self._conv_store = LCMConversationStore(self._db_path)
        self._sum_store = LCMSummaryStore(self._db_path)

        # Summarizer and sub-engines.
        self._summarizer = LCMSummarizer(
            provider,
            model=self._config.summary_model,
        )
        self._compactor = LCMCompactor(
            self._conv_store,
            self._sum_store,
            self._summarizer,
            self._config,
        )
        self._assembler = LCMAssembler(
            self._conv_store,
            self._sum_store,
            self._config,
        )

        # Stats tracking.
        self._total_compactions: int = 0
        self._last_compaction_at: float | None = None

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    async def ingest(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        turn_index: int = 0,
    ) -> str:
        """Persist a new message and return its ID.

        Args:
            session_id: Conversation session identifier.
            role: Message role (``"user"``, ``"assistant"``, ``"system"``).
            content: The message text.
            turn_index: Turn counter within the session.

        Returns:
            The generated message ID.
        """
        msg = MessagePart(
            role=role,
            content=content,
            session_id=session_id,
            turn_index=turn_index,
            token_count=estimate_tokens(content),
        )
        self._conv_store.add_message(session_id, msg)
        return msg.message_id

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def assemble(self, session_id: str, token_budget: int) -> AssemblyResult:
        """Build the context window for the LLM.

        Args:
            session_id: Conversation session identifier.
            token_budget: Maximum token count for the assembled context.

        Returns:
            An :class:`AssemblyResult` with summaries and fresh messages.
        """
        return self._assembler.assemble(session_id, token_budget)

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    async def compact(self, session_id: str) -> CompactionResult:
        """Force a compaction pass regardless of thresholds.

        Calls :meth:`pre_compaction_flush` before compaction to give the
        memory extractor a chance to persist important facts.

        Returns:
            A :class:`CompactionResult` with statistics.
        """
        await self.pre_compaction_flush(session_id)
        result = await self._compactor.compact(session_id)
        self._total_compactions += 1
        self._last_compaction_at = time.time()
        return result

    async def maybe_compact(self, session_id: str) -> CompactionResult | None:
        """Compact only if the uncompacted message count exceeds the threshold.

        Returns:
            A :class:`CompactionResult` if compaction ran, otherwise ``None``.
        """
        if not self._compactor.should_compact(session_id):
            return None
        return await self.compact(session_id)

    def set_memory_extractor(self, extractor: object) -> None:
        """Register a :class:`MemoryExtractor` for pre-compaction flush.

        When set, :meth:`pre_compaction_flush` will call
        ``extractor.run_once(session_id)`` before compaction begins,
        ensuring important facts are persisted to long-term memory
        before messages are compressed.
        """
        self._memory_extractor = extractor

    async def pre_compaction_flush(self, session_id: str) -> None:
        """Flush the memory extractor before compaction begins.

        If a memory extractor has been registered via
        :meth:`set_memory_extractor`, runs one extraction pass against the
        current session so that important facts from about-to-be-compacted
        messages are persisted to long-term memory.

        Does nothing if no extractor is registered.
        """
        extractor = getattr(self, "_memory_extractor", None)
        if extractor is None:
            return
        try:
            persisted, _extracted_facts = await extractor.run_once(session_id=session_id)
            if persisted:
                logger.info(
                    "Pre-compaction flush: persisted %d memories for session %s",
                    persisted,
                    session_id,
                )
        except Exception:
            logger.warning(
                "Pre-compaction memory flush failed for session %s",
                session_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self, session_id: str) -> LCMStats:
        """Return runtime statistics for the given session.

        Args:
            session_id: Conversation session identifier.

        Returns:
            An :class:`LCMStats` snapshot.
        """
        return LCMStats(
            total_messages=self._conv_store.count_all(session_id),
            total_summaries=self._sum_store.count_all(session_id),
            max_depth=self._sum_store.get_max_depth(session_id),
            total_compactions=self._total_compactions,
            last_compaction_at=self._last_compaction_at,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close underlying database connections."""
        self._conv_store.close()
        self._sum_store.close()

    def __enter__(self) -> LCMEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
