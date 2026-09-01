"""The context budget /api/lcm reports is the one it ASSEMBLES against, and
both are the budget the loop enforces.

The bug this pins: /api/lcm carried a literal 24000 while the live config said
72000 and llama.cpp reported n_ctx=32768. The reported denominator was wrong,
which was visible; the ``token_budget=`` argument was wrong too, which was not
— the endpoint assembled a real context against a window that does not exist,
so every number it returned (total_tokens, fresh_count, summary_count,
compression_ratio) described a parallel session. Utilisation of 9888/32768
(30%) rendered as 41%.

So the assertions here are deliberately paired: it is not enough that the
reported ``limit`` is right. The value handed to ``assemble()`` is captured and
asserted equal to it, because a fix that corrected only the label would leave
the damaging half in place and still look correct on the panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from prometheus.context.budget import TokenBudget, resolve_effective_limit
from prometheus.web.server import create_app

LOCAL_MODEL = "Qwen3.8-27B-UD-Q4_K_XL.gguf"
SERVER_N_CTX = 32768
CONFIG_LIMIT = 72000


# ---------------------------------------------------------------------------
# The resolver itself — one implementation, five distinguishable answers
# ---------------------------------------------------------------------------

CTX = {
    "effective_limit": CONFIG_LIMIT,
    "cloud_default_limit": 1_000_000,
    "model_overrides": {"claude-sonnet-4-5": {"effective_limit": 200_000}},
}


def test_detected_window_beats_a_stale_config():
    """The incident, in one assertion: the server's number wins."""
    limit, source = resolve_effective_limit(
        CTX, model=LOCAL_MODEL, local_model=LOCAL_MODEL, detected_limit=SERVER_N_CTX
    )
    assert (limit, source) == (SERVER_N_CTX, "detected")


def test_explicit_override_beats_detection():
    limit, source = resolve_effective_limit(
        CTX,
        model="claude-sonnet-4-5",
        local_model=LOCAL_MODEL,
        detected_limit=SERVER_N_CTX,
    )
    assert (limit, source) == (200_000, "model_override")


def test_cloud_session_does_not_inherit_the_local_gguf_window():
    limit, source = resolve_effective_limit(
        CTX, model="some-cloud-model", local_model=LOCAL_MODEL,
        detected_limit=SERVER_N_CTX,
    )
    assert (limit, source) == (1_000_000, "cloud_default")


def test_config_wins_only_when_the_backend_said_nothing():
    """detected_limit=None is the unreachable-backend case."""
    limit, source = resolve_effective_limit(
        CTX, model=LOCAL_MODEL, local_model=LOCAL_MODEL, detected_limit=None
    )
    assert (limit, source) == (CONFIG_LIMIT, "config")


@pytest.mark.parametrize(
    "ctx", [{}, {"effective_limit": None}, {"effective_limit": 0},
            {"effective_limit": "lots"}],
    ids=["empty", "null", "zero", "non-numeric"],
)
def test_unresolvable_is_unknown_not_a_plausible_number(ctx):
    """No detection and no usable config = "unknown".

    This is the branch that must never invent a value: a fabricated
    denominator is indistinguishable from a real one on the panel, which is
    how 24000 survived.
    """
    assert resolve_effective_limit(ctx, model=LOCAL_MODEL) == (None, "unknown")


def test_token_budget_still_carries_a_number_for_legacy_callers():
    """from_loaded_config must keep working where "unknown" has no meaning."""
    budget = TokenBudget.from_loaded_config({"context": CTX}, model=LOCAL_MODEL,
                                            local_model=LOCAL_MODEL,
                                            detected_limit=SERVER_N_CTX)
    assert budget.effective_limit == SERVER_N_CTX
    assert TokenBudget.from_loaded_config({}).effective_limit == 24000


# ---------------------------------------------------------------------------
# The route — reported limit and assembled budget are the same number
# ---------------------------------------------------------------------------

@dataclass
class _FakeResult:
    total_tokens: int = 9888
    compression_ratio: float = 0.42
    fresh_messages: list = field(default_factory=lambda: [1, 2, 3])
    summaries: list = field(default_factory=lambda: [1])


class _SpyEngine:
    """Records the budget it was assembled against."""

    def __init__(self) -> None:
        self.budgets: list[int] = []

    def assemble(self, session_id: str, token_budget: int) -> _FakeResult:
        self.budgets.append(token_budget)
        return _FakeResult()


def _client(ctx: dict[str, Any], engine: Any, **kw: Any) -> TestClient:
    config = {"model": {"model": LOCAL_MODEL, "provider": "llama_cpp"}, "context": ctx}
    return TestClient(create_app(config, lcm_engine=engine, **kw))


def test_lcm_assembles_against_the_detected_window():
    engine = _SpyEngine()
    client = _client(CTX, engine, detected_context_size=SERVER_N_CTX,
                     local_model=LOCAL_MODEL)
    body = client.get("/api/lcm/desktop:1").json()

    assert body["limit"] == SERVER_N_CTX
    assert body["limit_source"] == "detected"
    assert body["assembled"] is True
    # The half that was invisible: the budget the assembly actually ran with.
    assert engine.budgets == [SERVER_N_CTX], (
        "assemble() must receive the resolved window, not a literal — a fix "
        "that only corrects the reported denominator leaves the real damage"
    )
    assert body["total_tokens"] == 9888


def test_lcm_falls_back_to_config_when_the_backend_was_unreachable():
    engine = _SpyEngine()
    client = _client(CTX, engine, detected_context_size=None, local_model=LOCAL_MODEL)
    body = client.get("/api/lcm/desktop:1").json()
    assert (body["limit"], body["limit_source"]) == (CONFIG_LIMIT, "config")
    assert engine.budgets == [CONFIG_LIMIT]


def test_lcm_reports_unknown_rather_than_inventing_a_denominator():
    """No window resolvable → no assembly, and a null limit on the wire."""
    engine = _SpyEngine()
    client = _client({}, engine)
    body = client.get("/api/lcm/desktop:1").json()

    assert body["limit"] is None
    assert body["limit_source"] == "unknown"
    assert body["assembled"] is False
    assert body["reason"] == "context_limit_unknown"
    assert engine.budgets == [], "must not assemble against a guessed budget"


def test_lcm_marks_unwired_and_failed_states_as_unmeasured():
    """Zeros survive for shape compatibility; `assembled` says they aren't
    readings. Otherwise "measured zero" and "never measured" look identical."""
    unwired = _client(CTX, None, detected_context_size=SERVER_N_CTX,
                      local_model=LOCAL_MODEL).get("/api/lcm/s").json()
    assert unwired["assembled"] is False and unwired["reason"] == "lcm_engine_unwired"
    # ...and the limit is still honest even with no engine.
    assert unwired["limit"] == SERVER_N_CTX

    class _Boom:
        def assemble(self, *a, **k):
            raise RuntimeError("no leaf summaries")

    failed = _client(CTX, _Boom(), detected_context_size=SERVER_N_CTX,
                     local_model=LOCAL_MODEL).get("/api/lcm/s").json()
    assert failed["assembled"] is False and failed["reason"] == "assemble_failed"


# ---------------------------------------------------------------------------
# Status and lcm cannot disagree
# ---------------------------------------------------------------------------

def test_status_and_lcm_report_the_same_window():
    engine = _SpyEngine()
    client = _client(CTX, engine, detected_context_size=SERVER_N_CTX,
                     local_model=LOCAL_MODEL)
    status = client.get("/api/status").json()["context"]
    lcm = client.get("/api/lcm/desktop:1").json()

    assert status["limit"] == lcm["limit"] == SERVER_N_CTX
    assert status["source"] == lcm["limit_source"] == "detected"


def test_status_shows_the_inputs_that_produced_the_budget():
    """All three numbers on one surface — the incident was three disagreeing
    values with nothing that displayed them together."""
    client = _client(CTX, _SpyEngine(), detected_context_size=SERVER_N_CTX,
                     local_model=LOCAL_MODEL)
    ctx = client.get("/api/status").json()["context"]
    assert ctx["detected_n_ctx"] == SERVER_N_CTX
    assert ctx["configured_limit"] == CONFIG_LIMIT
    assert ctx["local_model"] == LOCAL_MODEL
    assert ctx["limit"] == SERVER_N_CTX


def test_status_context_block_survives_a_bare_app():
    """create_app({}) must still answer, honestly, without a backend."""
    ctx = TestClient(create_app({})).get("/api/status").json()["context"]
    assert ctx["limit"] is None and ctx["source"] == "unknown"
    assert ctx["detected_n_ctx"] is None
