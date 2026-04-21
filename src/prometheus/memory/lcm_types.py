"""Data types for Lossless Context Management (LCM).

Defines the core dataclasses used by the LCM DAG-based compression system
for conversation compaction and context assembly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class MessagePart:
    """A chunk of a conversation message."""

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: uuid4().hex)
    session_id: str = ""
    turn_index: int = 0
    token_count: int = 0


@dataclass
class SummaryNode:
    """A node in the summary DAG.

    Leaf nodes have not yet been summarised further (is_leaf=True).
    Depth 0 nodes are direct summaries of raw messages; higher depths
    summarise earlier summaries.
    """

    id: str = field(default_factory=lambda: uuid4().hex)
    parent_ids: list[str] = field(default_factory=list)
    source_message_ids: list[str] = field(default_factory=list)
    summary_text: str = ""
    depth: int = 0
    token_count: int = 0
    created_at: float = field(default_factory=time.time)
    is_leaf: bool = True


@dataclass
class CompactionConfig:
    """Tunable settings for the compaction engine."""

    context_threshold: int = 18_000
    """Token budget before compaction is triggered."""

    fresh_tail_count: int = 32
    """Number of most-recent messages to keep uncompacted."""

    summary_model: str = "default"
    """Model identifier used for summarisation calls."""

    max_summary_depth: int = 5
    """Maximum DAG depth before refusing further compaction."""

    compaction_batch_size: int = 10
    """Number of messages to compact in a single pass."""


@dataclass
class AssemblyResult:
    """Result of assembling context from the LCM stores."""

    summaries: list[SummaryNode] = field(default_factory=list)
    fresh_messages: list[MessagePart] = field(default_factory=list)
    total_tokens: int = 0
    compression_ratio: float = 1.0


@dataclass
class CompactionResult:
    """Result of a single compaction pass."""

    summaries_created: int = 0
    messages_compacted: int = 0
    new_depth: int = 0
    tokens_saved: int = 0


@dataclass
class LCMStats:
    """Runtime statistics for the LCM system."""

    total_messages: int = 0
    total_summaries: int = 0
    max_depth: int = 0
    total_compactions: int = 0
    last_compaction_at: float | None = None
