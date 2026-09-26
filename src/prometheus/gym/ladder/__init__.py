"""Model ladder — which task classes each model size can handle, measured.

A frozen suite of eight task classes (``gym/ladder/v1/``) run through the
gym's real pipeline against one served model at a time. Every run is a
verdict — a deterministic predicate, an acceptance command, or a pinned local
judge that is never the model under test — and is recorded in telemetry.db
for Instinct's routing to learn from.

Entry point: ``uv run python scripts/ladder_run.py --help``.
Definition, judge pinning and field mapping: ``docs/MODEL-LADDER.md``.
"""

from prometheus.gym.ladder.suite import CLASS_IDS, LadderSuite, LadderTask, load_suite

__all__ = ["CLASS_IDS", "LadderSuite", "LadderTask", "load_suite"]
