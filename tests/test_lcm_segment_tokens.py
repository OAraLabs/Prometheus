"""THE INVARIANT: the reported split is the total's own arithmetic, not a second estimate.

A segmented context bar is only honest if its segments add up to the number the
bar is drawn inside. If a client had to derive the split by re-estimating the
summary and fresh lists separately, the two figures could disagree with
``total_tokens`` — and nothing on screen would say which to believe.

They cannot disagree here, because the assembler has always computed
``fresh_tokens`` and ``summary_tokens`` and then thrown them away on the line
``total_tokens = fresh_tokens + summary_tokens``. This change stops discarding
them. The identity below is therefore structural, and the test exists to keep
it that way.

The unmeasured branches report ``null``, not ``0`` — see
test_unassembled_reports_null_not_zero for why that distinction is the whole
point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.memory.lcm_types import AssemblyResult, MessagePart, SummaryNode  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

LOCAL_MODEL = "Qwen3.8-27B-UD-Q4_K_XL.gguf"
CONFIG = {
    "model": {"model": LOCAL_MODEL, "provider": "llama_cpp"},
    "context": {"effective_limit": 72000},
}


# ── the identity, at the source ───────────────────────────────────────────

def test_assembler_split_sums_to_the_total(tmp_path):
    """Drive the real assembler and assert the halves reconstruct the whole."""
    from prometheus.memory.lcm_assembler import LCMAssembler

    # Real stores would need an engine + provider; the arithmetic under test is
    # the assembler's own, so drive it against doubles returning known counts.
    class _Conv:
        def get_uncompacted_messages(self, sid):
            return [MessagePart(role="user", content="x" * 400, token_count=100),
                    MessagePart(role="assistant", content="y" * 400, token_count=150)]
        def count_all(self, sid): return 2
        def get_all_messages(self, sid): return self.get_uncompacted_messages(sid)

    class _Sum:
        def get_leaf_summaries(self, sid):
            return [SummaryNode(summary_text="s", depth=0, token_count=70),
                    SummaryNode(summary_text="s2", depth=0, token_count=30)]

    @dataclass
    class _Cfg:
        fresh_tail_count: int = 32

    a = LCMAssembler.__new__(LCMAssembler)
    a._conv_store, a._sum_store, a._config = _Conv(), _Sum(), _Cfg()

    r = a.assemble("s", token_budget=10_000)
    assert r.fresh_tokens == 250
    assert r.summary_tokens == 100
    assert r.fresh_tokens + r.summary_tokens == r.total_tokens, (
        "the split must reconstruct the total exactly — a segmented bar drawn "
        "from these would otherwise not fill the measured portion"
    )


def test_a_summary_dropped_for_budget_is_excluded_from_BOTH(tmp_path):
    """A summary that does not fit is not in the list and not in the tokens.

    The split has to describe what was actually assembled, not what was
    considered — otherwise the bar's segments exceed the context that exists.
    """
    from prometheus.memory.lcm_assembler import LCMAssembler

    class _Conv:
        def get_uncompacted_messages(self, sid):
            return [MessagePart(role="user", content="x", token_count=100)]
        def count_all(self, sid): return 1
        def get_all_messages(self, sid): return self.get_uncompacted_messages(sid)

    class _Sum:
        def get_leaf_summaries(self, sid):
            return [SummaryNode(summary_text="fits", token_count=40),
                    SummaryNode(summary_text="does not", token_count=9_000)]

    @dataclass
    class _Cfg:
        fresh_tail_count: int = 32

    a = LCMAssembler.__new__(LCMAssembler)
    a._conv_store, a._sum_store, a._config = _Conv(), _Sum(), _Cfg()

    r = a.assemble("s", token_budget=200)
    assert len(r.summaries) == 1
    assert r.summary_tokens == 40, "dropped summary leaked into the token split"
    assert r.fresh_tokens + r.summary_tokens == r.total_tokens


def test_default_construction_reports_no_split():
    """An AssemblyResult built without the split reports 0/0, not a fake one.

    Consumers must render that as unsegmented rather than as 'all fresh'.
    """
    r = AssemblyResult(total_tokens=500)
    assert r.fresh_tokens == 0 and r.summary_tokens == 0


# ── the wire ──────────────────────────────────────────────────────────────

class _Engine:
    def __init__(self, result): self._r = result
    def assemble(self, session_id, token_budget): return self._r


def _get(engine, **kw):
    client = TestClient(create_app(
        CONFIG, lcm_engine=engine, detected_context_size=32768,
        local_model=LOCAL_MODEL, **kw,
    ))
    return client.get("/api/lcm/desktop:1").json()


def test_route_reports_the_split_and_it_sums():
    body = _get(_Engine(AssemblyResult(
        summaries=[SummaryNode()], fresh_messages=[MessagePart(role="u", content="c")],
        total_tokens=350, fresh_tokens=250, summary_tokens=100,
        compression_ratio=2.0,
    )))
    assert body["assembled"] is True
    assert body["fresh_tokens"] == 250
    assert body["summary_tokens"] == 100
    assert body["fresh_tokens"] + body["summary_tokens"] == body["total_tokens"]
    assert body["limit"] == 32768 and body["limit_source"] == "detected"


def test_unassembled_reports_null_not_zero():
    """0/0 would draw an empty two-segment bar — a measured breakdown of nothing.

    null says there is no split to draw, which is what is true when the daemon
    never ran an assembly.
    """
    body = _get(None)  # no engine wired
    assert body["assembled"] is False
    assert body["fresh_tokens"] is None
    assert body["summary_tokens"] is None


def test_unknown_window_also_reports_null_split():
    """No resolvable limit => assembly is skipped => no split."""
    client = TestClient(create_app({}, lcm_engine=_Engine(AssemblyResult())))
    body = client.get("/api/lcm/desktop:1").json()
    assert body["limit"] is None and body["assembled"] is False
    assert body["fresh_tokens"] is None and body["summary_tokens"] is None


def test_segments_never_exceed_the_limit_they_are_drawn_in():
    """Control: a client can safely draw segments against `limit`."""
    body = _get(_Engine(AssemblyResult(
        total_tokens=350, fresh_tokens=250, summary_tokens=100)))
    assert body["fresh_tokens"] + body["summary_tokens"] <= body["limit"]


# ── the boundary guards its own invariant ─────────────────────────────────

def test_a_result_without_the_fields_still_reports_the_total():
    """A duck-typed result missing the split must lose ONLY the split.

    The first version of this change read `result.fresh_tokens` directly, so a
    stand-in without the attribute raised into the route's except and collapsed
    the entire response to the unmeasured shape — total, limit and all. Eight
    existing tests caught it.
    """
    @dataclass
    class _Older:
        total_tokens: int = 350
        compression_ratio: float = 2.0
        fresh_messages: list = field(default_factory=list)
        summaries: list = field(default_factory=list)

    body = _get(_Engine(_Older()))
    assert body["assembled"] is True, "lost the whole reading over a missing field"
    assert body["total_tokens"] == 350
    assert body["limit"] == 32768
    assert body["fresh_tokens"] is None and body["summary_tokens"] is None


def test_a_split_that_does_not_add_up_is_refused():
    """0/0 against a non-zero total would draw as 'all free space'.

    AssemblyResult defaults the split to 0/0 to keep positional construction
    valid, so this is the shape a pre-split assembler produces. Publishing it
    would be a confident picture of a context that is actually full.
    """
    body = _get(_Engine(AssemblyResult(total_tokens=350)))  # 0 + 0 != 350
    assert body["total_tokens"] == 350
    assert body["fresh_tokens"] is None, "published a split that does not reconstruct the total"
    assert body["summary_tokens"] is None


def test_a_coherent_zero_split_IS_published():
    """Control: an empty context legitimately splits 0/0 against a 0 total.

    Without this, 'refuse when it does not add up' could be satisfied by
    refusing every zero, and a genuinely empty session would lose its split.
    """
    body = _get(_Engine(AssemblyResult(total_tokens=0, fresh_tokens=0, summary_tokens=0)))
    assert body["fresh_tokens"] == 0 and body["summary_tokens"] == 0
