"""One LoopContext serves every session — so no run may write to it.

Audit item 6, second half. ``run_daemon`` builds the web ``LoopContext``
ONCE and hands the same object to every Beacon, REST and WebSocket turn.
Most of what it holds is a service bundle that SHOULD be shared. Six fields
are not, and each was written mid-run on the shared instance:

* ``provider`` / ``adapter`` / ``model`` / ``backend`` — the ROUTER RACE.
  ``/claude`` in one chat swapped all four for every concurrent session,
  because turn locks are per-session and this object is not.
* ``system_prompt`` — the IDENTITY-LINE LEAK. Rewritten in place to name the
  routed model, so the next session's prompt claimed to be the previous
  session's model, compounding on each turn.
* ``pair_pending`` — a rejected call in one session could pair with a
  matching success in ANOTHER and be filed as training data.

And one write went THROUGH the context to a shared object: the tier bump
set ``context.adapter.tier``, the only write to ``.tier`` outside the
adapter's constructor, with nothing anywhere to put it back.

⚠ EVERY ASSERTION HERE IS ON THE CALLER'S OBJECT OR ON THE WIRE, never on
the run's own view. Reading back the field a run wrote is how the old
coverage agreed with the old bug: ``test_the_prompt_identity_line_follows_
the_serving_model`` asserted ``ctx.system_prompt`` and passed for a year on
a rewrite the outgoing request never carried.
"""

from __future__ import annotations

import asyncio
import ast
from pathlib import Path

import pytest

from prometheus.engine.agent_loop import (
    LoopContext,
    SharedLoopContextMutation,
    run_loop,
)
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ModelProvider,
)

REPO = Path(__file__).resolve().parent.parent


class _Recorder(ModelProvider):
    """Records every request it is asked to serve, then answers plainly."""

    def __init__(self, name: str, *, delay: float = 0.0) -> None:
        self.name = name
        self._delay = delay
        self._suppress_thinking = True
        self.requests: list[ApiMessageRequest] = []

    async def stream_message(self, request: ApiMessageRequest):
        self.requests.append(request)
        if self._delay:
            await asyncio.sleep(self._delay)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant", content=[TextBlock(text=f"served by {self.name}")]
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


class _Decision:
    def __init__(self, provider, model_name, provider_name, *, adapter=None,
                 backend=None):
        self.provider = provider
        self.adapter = adapter
        self.model_name = model_name
        self.provider_name = provider_name
        self.backend = backend
        self.reason = "user override"


class _PerSessionRouter:
    """Routes each session to its own provider — the override case."""

    def __init__(self, table: dict[str, _Decision]) -> None:
        self._table = table

    def route(self, _text, context=None):
        sid = (context or {}).get("session_id")
        return self._table[sid]


def _base_context(**kw) -> LoopContext:
    defaults = dict(
        provider=_Recorder("boot"),
        model="boot-model",
        system_prompt="- Model: boot-model (provider: llama_cpp)",
        max_tokens=64,
    )
    defaults.update(kw)
    return LoopContext(**defaults)


async def _drive(context, text, session_id):
    out = []
    async for event, _usage in run_loop(
        context, [ConversationMessage.from_user_text(text)], session_id=session_id
    ):
        out.append(event)
    return out


# ── the router race ─────────────────────────────────────────────────

class _GatingCompactor:
    """Parks ONE session inside the loop, between the router's write and the
    read that picks the provider to stream from.

    ⚠ THE WHOLE POINT OF THIS CLASS. A race test that merely runs two turns
    with ``asyncio.gather`` does NOT catch this bug and quietly passes with
    the fix reverted — measured. The provider is read out of the context and
    passed into ``stream_round_with_fallback`` in one go, so an ordinary
    interleave never lands between the write and the read.

    ``context.compactor.apply`` is awaited AFTER the router swap and BEFORE
    that read, so parking session A here and letting session B run its whole
    turn to completion puts the interleave exactly where the race lives.
    """

    def __init__(self, park_model: str) -> None:
        # Keyed on the MODEL, not the session: the compactor is handed
        # `context.session_id` (empty for every web turn) rather than the
        # per-call one, so the model the router just chose is the only thing
        # here that distinguishes the two turns.
        self._park = park_model
        self.released = asyncio.Event()
        self.parked = asyncio.Event()

    async def apply(self, messages, *, model="", **_kw):
        if model == self._park:
            self.parked.set()
            await self.released.wait()
        return messages

    def limit_for(self, _model, backend=None):
        return 0

    def estimate_total(self, *_a, **_kw):
        return 0


def test_two_sessions_routed_differently_do_not_serve_each_other():
    """THE RACE, made deterministic.

    Session A routes to provider A, then parks inside the loop. Session B
    runs start to finish — routing to provider B, which under the bug wrote
    ``context.provider`` on the object A is still using. A then wakes and
    picks the provider to stream from. It must be A's.
    """
    a_provider, b_provider = _Recorder("A"), _Recorder("B")
    compactor = _GatingCompactor("model-a")
    context = _base_context(
        compactor=compactor,
        model_router=_PerSessionRouter({
            "sess-a": _Decision(a_provider, "model-a", "prov-a"),
            "sess-b": _Decision(b_provider, "model-b", "prov-b"),
        }),
    )

    async def go():
        a = asyncio.create_task(_drive(context, "question a", "sess-a"))
        await asyncio.wait_for(compactor.parked.wait(), timeout=5)
        # A is parked mid-turn, after its route. B runs to completion.
        await _drive(context, "question b", "sess-b")
        compactor.released.set()
        await asyncio.wait_for(a, timeout=5)

    asyncio.run(go())

    assert len(b_provider.requests) == 1, "session B was not served by its own provider"
    assert len(a_provider.requests) == 1, (
        "session A's turn was served by session B's provider — B's route() "
        "wrote provider/model onto the context A was mid-turn on"
    )
    assert a_provider.requests[0].model == "model-a"
    assert b_provider.requests[0].model == "model-b"


def test_the_parked_session_also_keeps_its_own_identity_line():
    """Same interleave, the system_prompt half: A must not be told it is B."""
    a_provider, b_provider = _Recorder("A"), _Recorder("B")
    compactor = _GatingCompactor("model-a")
    context = _base_context(
        compactor=compactor,
        model_router=_PerSessionRouter({
            "sess-a": _Decision(a_provider, "model-a", "prov-a"),
            "sess-b": _Decision(b_provider, "model-b", "prov-b"),
        }),
    )

    async def go():
        a = asyncio.create_task(_drive(context, "what model is this?", "sess-a"))
        await asyncio.wait_for(compactor.parked.wait(), timeout=5)
        await _drive(context, "and you?", "sess-b")
        compactor.released.set()
        await asyncio.wait_for(a, timeout=5)

    asyncio.run(go())
    sent = a_provider.requests[0].system_prompt or ""
    assert "model-a" in sent and "model-b" not in sent, (
        f"session A was handed session B's identity line: {sent!r}"
    )


def test_a_routed_turn_leaves_the_callers_context_untouched():
    """Every per-run field, checked against the values it was built with."""
    routed = _Recorder("routed")
    context = _base_context(
        model_router=_PerSessionRouter({
            "s": _Decision(routed, "routed-model", "routed-prov", backend="4090"),
        })
    )
    before = {f: getattr(context, f) for f in LoopContext.PER_RUN_FIELDS}

    asyncio.run(_drive(context, "hi", "s"))

    assert routed.requests, "the router never fired — the test proves nothing"
    after = {f: getattr(context, f) for f in LoopContext.PER_RUN_FIELDS}
    changed = {k: (before[k], after[k]) for k in before if before[k] is not after[k]}
    assert not changed, f"the shared context was mutated: {changed}"


def test_the_identity_line_reaches_the_routed_provider():
    """The other half: the run's OWN view must still be rewritten, or the
    fix would have closed the leak by removing the feature."""
    routed = _Recorder("routed")
    context = _base_context(
        model_router=_PerSessionRouter({
            "s": _Decision(routed, "claude-x", "anthropic"),
        })
    )
    asyncio.run(_drive(context, "what model is this?", "s"))
    sent = routed.requests[0].system_prompt or ""
    assert "claude-x" in sent, f"the routed model was told it was something else: {sent!r}"
    assert "boot-model" not in sent


# ── the pair stash ──────────────────────────────────────────────────

def test_a_stash_made_during_a_run_does_not_escape_to_the_next():
    from prometheus.engine.agent_loop import _stash_pending_pair

    context = _base_context(pair_pending={})
    run_view = context.for_run()
    _stash_pending_pair(
        run_view, "echo", rejected_name="echo", rejected_input={"a": 1},
        error="bad", source="self_correction",
    )
    assert run_view.pair_pending, "the stash did not land on the run's own dict"
    assert context.pair_pending == {}, (
        "the stash escaped to the shared context — another session's success "
        "would complete it and file a repair pair that never happened"
    )


def test_a_caller_supplied_stash_is_still_honoured():
    """Emptying rather than copying would silently drop a seeded stash."""
    context = _base_context(pair_pending={"echo": {"error": "seeded"}})
    assert context.for_run().pair_pending == {"echo": {"error": "seeded"}}


# ── the seal ────────────────────────────────────────────────────────

def test_a_sealed_context_refuses_every_per_run_field():
    context = _base_context().seal()
    for name in sorted(LoopContext.PER_RUN_FIELDS):
        with pytest.raises(SharedLoopContextMutation):
            setattr(context, name, object())


def test_a_sealed_context_still_allows_everything_else():
    """The seal is a scalpel, not a freeze — config fields stay writable so
    sealing cannot break an unrelated caller."""
    context = _base_context().seal()
    context.max_turns = 7
    assert context.max_turns == 7


def test_the_copy_of_a_sealed_context_is_writable():
    context = _base_context().seal()
    run_view = context.for_run()
    assert not run_view.sealed
    run_view.model = "routed"
    assert context.model == "boot-model"


def test_the_service_bundle_is_shared_by_reference_not_copied():
    """A deep copy would give each turn its own telemetry and its own
    verifier, which is the opposite mistake."""
    sentinel_registry, sentinel_telemetry = object(), object()
    context = _base_context(
        tool_registry=sentinel_registry, telemetry=sentinel_telemetry,
    )
    run_view = context.for_run()
    assert run_view.tool_registry is sentinel_registry
    assert run_view.telemetry is sentinel_telemetry


def test_a_sealed_context_survives_a_real_turn():
    """The seal must not be a landmine: an ordinary routed turn through the
    real run_loop has to complete against a SEALED context."""
    routed = _Recorder("routed")
    context = _base_context(
        model_router=_PerSessionRouter({"s": _Decision(routed, "m", "p")}),
    ).seal()
    asyncio.run(_drive(context, "hi", "s"))
    assert routed.requests, "the sealed turn did not reach a provider"


# ── the daemon actually seals ───────────────────────────────────────

def test_the_daemon_seals_the_context_it_shares():
    """The guard is worth nothing if the one shared instance is unsealed.
    AST rather than a substring so a mention in a comment cannot satisfy it."""
    tree = ast.parse((REPO / "src" / "prometheus" / "daemon.py").read_text(
        encoding="utf-8"))
    sealed = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "seal"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "loop_context"
    ]
    assert sealed, (
        "daemon.py no longer seals the LoopContext it shares with every web "
        "session. Without it a per-run write leaks silently again instead of "
        "raising."
    )


def test_run_loop_copies_before_it_reads_anything():
    """`context = context.for_run()` must be the FIRST statement in the
    body. A read that drifts above it silently reads the shared object."""
    tree = ast.parse((REPO / "src" / "prometheus" / "engine" / "agent_loop.py")
                     .read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_loop"
    )
    body = [n for n in fn.body if not (isinstance(n, ast.Expr)
                                       and isinstance(n.value, ast.Constant))]
    first = body[0]
    assert isinstance(first, ast.Assign), ast.dump(first)[:200]
    assert isinstance(first.value, ast.Call), "first statement is not a call"
    assert getattr(first.value.func, "attr", None) == "for_run", (
        "run_loop no longer takes its per-run copy first — everything after "
        "the drifted line reads the context every session shares"
    )


# ── the user-visible surface this fixes ─────────────────────────────

def test_api_models_vision_answers_from_the_BOOT_provider():
    """`_primary_supports_vision` promises "the boot provider's detected
    vision capability". Before the copy, any `/claude` turn replaced the
    provider on the shared context and /api/models started answering for
    whatever last routed."""
    from prometheus.providers.registry import provider_supports_vision

    class _Vision:
        supports_vision = True

    class _NoVision:
        supports_vision = False

    boot = _Recorder("boot")
    boot.supports_vision = True
    routed = _Recorder("routed")
    routed.supports_vision = False

    context = _base_context(
        provider=boot,
        model_router=_PerSessionRouter({"s": _Decision(routed, "m", "p")}),
    )
    assert provider_supports_vision(context.provider) is True
    asyncio.run(_drive(context, "hi", "s"))
    assert provider_supports_vision(context.provider) is True, (
        "/api/models would now report the ROUTED provider's vision flag"
    )
