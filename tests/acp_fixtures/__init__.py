"""Fixtures for the ACP core tests — EXTRACTED, never hand-written.

Two sources, both the other side's real code:

1. The pinned schema itself (``prometheus/acp/schema/schema.json``, tag
   ``schema-v2.0.0-beta.4``): required-field skeletons and const discriminators
   are read from the schema at test time, so a schema bump re-derives them.

2. The TCK's conforming v2 fixture agent (acp-tck, Apache-2.0,
   ``tests/fixtures/agents/v2/_base.py`` @ commit 3ea94133): the wire exchanges
   it produces are captured by DRIVING the real fixture over stdio and
   recording its responses — see ``capture_tck_exchanges`` below. The captured
   lines are stored verbatim in ``tck_exchanges.json``; nothing is edited.

Regenerate the capture with:
    python tests/acp_fixtures/capture.py <path-to-acp-tck-checkout>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE.parent.parent / "src" / "prometheus" / "acp" / "schema" / "schema.json"


def pinned_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text())


def schema_def(name: str) -> dict[str, Any]:
    return pinned_schema()["$defs"][name]


def tck_exchanges() -> list[dict[str, Any]]:
    """Verbatim wire lines captured from the TCK's conforming v2 agent."""
    return json.loads((HERE / "tck_exchanges.json").read_text())