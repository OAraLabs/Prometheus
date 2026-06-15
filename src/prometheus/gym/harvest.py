"""Pair-harvest mode for gym runs (tool-calling closeout follow-up #6 — the
realistic 1,000-pair path).

The gym drives REAL ``run_loop`` turns through the real adapter. The
repair-inducing variants (notably ``adapter_unwrap``) make the model's
dict-wrapped / fuzzily-named calls trip the adapter's repair path — and every
such repair is a training pair the flywheel wants. But the loop only captures
when ``pair_capture`` has a configured store, and the gym process does not
configure one (the daemon does, in its own process). This module wires capture
into the gym process, pointed at a SEPARATE ``gym-training.db`` so induced
pairs never mingle with the live daemon's organic ``training.db``.

No agent-loop change is needed: ``run_loop`` lazily initializes
``context.pair_pending`` and every ``capture_pair`` call already guards on
``get_store() is not None`` — so configuring the singleton here is sufficient
for both the immediate adapter-repair captures and the pending-stash ones.
"""

from __future__ import annotations

from prometheus.config.paths import get_data_dir
from prometheus.learning import pair_capture


def default_harvest_db() -> str:
    """Sibling of the live ``training.db`` — ``~/.prometheus/data/gym-training.db``."""
    return str(get_data_dir() / "gym-training.db")


def configure_harvest(db_path: str | None = None) -> str:
    """Point the pair-capture singleton at the gym harvest DB and return the path.

    cloud_golden stays OFF (gym runs the local model only). Pairs are stamped
    ``pair_source=gym_harvest`` (series-2 provenance decision) so harvest pairs
    are separable from organic/cloud at training time; the original repair type
    survives in ``meta["origin_source"]``. Idempotent — safe to call once
    before an experiment.
    """
    path = db_path or default_harvest_db()
    pair_capture.configure(
        {
            "capture_enabled": True,
            "cloud_golden_capture": False,
            "db_path": path,
            "source_override": "gym_harvest",
        }
    )
    return path


def pair_total() -> int:
    """Total pairs currently in the configured store (0 if capture isn't on)."""
    store = pair_capture.get_store()
    if store is None:
        return 0
    try:
        return int(store.stats().get("total", 0))
    except Exception:
        return 0
