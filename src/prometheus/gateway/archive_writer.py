"""Archive writer — fire-and-forget event logging to JSONL.

Source: Novel code for Prometheus Sprint 6 (inspired by OpenClaw archive_bridge).
Writes events to a local JSONL file for auditing and replay.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prometheus.config.paths import get_data_dir

logger = logging.getLogger(__name__)


def get_archive_path() -> Path:
    """Return the default archive file path."""
    return get_data_dir() / "archive.jsonl"


class ArchiveWriter:
    """Append-only JSONL event writer."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else get_archive_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def archive_event(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Write a single event to the archive (fire-and-forget)."""
        entry = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data or {},
        }
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.error("Failed to write archive event: %s", exc)

    def read_events(
        self,
        *,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Read recent events from the archive."""
        if not self.path.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event_type and entry.get("type") != event_type:
                continue
            entries.append(entry)
        return entries[-limit:]
