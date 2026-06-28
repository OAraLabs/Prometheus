"""
test_force_search.py — red spec for per-call tool_choice (force-search).

═══════════════════════════════════════════════════════════════════════════════
CC HANDOFF BRIEF — read this before touching code.
═══════════════════════════════════════════════════════════════════════════════

WHAT THIS IS
    A failing test suite that DEFINES "done" for force-search. Workflow is TDD:
    implement the feature + wire the fixtures below until this whole file goes
    green, without regressing the existing suite (2990 passing) or
    test_per_message_mode.py (8 passing). No --no-verify.

WHAT'S ALREADY ON main (PR #72, 3a506fe) — build ON this, don't re-derive:
    - Per-message `mode` (agent|chat) threaded as a per-call `run_loop` param
      through: WS send_message / REST send_chat -> dispatch_user_message
      -> _handle_send_message -> _run_agent -> run_loop.  NEVER on the shared
      loop_context.
    - mode="chat" -> empty tool_schema + per-call `suppress_tools` flag on
      ApiMessageRequest. LlamaCppProvider honors suppress_tools by DROPPING its
      boot-time full-registry GBNF grammar. Selection is per-call; the shared
      provider's cached grammar is never mutated.

THE DESIGN THIS SPEC ASSUMES (one lever, four states):
    Generalize the per-call `suppress_tools: bool` into a per-call tool_choice
    union on ApiMessageRequest. Vocabulary mirrors the cloud API so the same
    field drives both tiers:

        auto         -> boot grammar as-is        (today's default agent path)
        none         -> grammar dropped           (today's suppress_tools=True)
        required     -> grammar constrained to require a tool_use (any tool)
        {"tool": X}  -> grammar constrained to require tool_use of X

    `mode` stays as sugar that resolves into this: agent->auto, chat->none.
    Force-search is just required / {"tool": "<search tool name>"} carried on
    the message. If product later prefers keeping suppress_tools + a separate
    force field instead, change ONLY the make_request() seam in conftest — the
    assertions here are behavioral and survive either representation.

    Per-tier translation the provider must implement:
        LlamaCppProvider (local): tool_choice -> GBNF grammar SELECTION
            (immutable, cached, per-call — same discipline as suppress_tools).
        Cloud provider:           tool_choice -> native API tool_choice param,
            with synthetic tool_use prefill as the gated fallback ONLY for
            providers lacking native `required` support (capability-checked,
            never the default path).

★ THE ONE CRUX — DO THIS FIRST, then report which case you found ★
    Inspect the boot-time GBNF root that the registry generates.
      • If it's a top-level alternation (root := text | tool_use):
            `required` == drop the `text` branch. Afternoon-sized. Clean inverse
            of suppress_tools (which dropped the whole grammar).
      • If text and tool_use are interleaved / tool_use is nested under a wrapper
        that also admits prose:
            `required` needs a root REWRITE emitting tool_use-only. Sprint-sized.
    Either way, TestLocalGrammarSelection asserts the *behavioral* outcome
    (directive=required yields a grammar that admits tool_use and rejects bare
    prose), so it is valid regardless of how you build the grammar internally.

FIXTURE CONTRACT — implement these in conftest.py (or reuse existing ones from
test_per_message_mode.py where they already exist; prefer reuse):

    spy_provider
        A LlamaCppProvider (or faithful double) that:
          - records every per-call ApiMessageRequest it receives  -> .requests
            (list) and .last_request
          - exposes the grammar it SELECTED for the last call      -> .last_grammar
          - exposes the immutable boot grammar                     -> .boot_grammar
          - is DIRECTIVE-COMPLIANT when asked to produce a turn: returns a
            tool_use block when the effective directive requires/permits one and
            the canned script calls for it, and a plain-text block when tools are
            disallowed. (This lets daemon-level turn-shape asserts be
            deterministic without real inference.)
          - lets a test queue canned model output via .script([...])

    cloud_spy_provider
        Same recording surface (.last_request, .requests), but for the cloud
        path: exposes the native tool_choice it would send -> .last_tool_choice,
        and .supports_required (bool) to exercise the synthetic-prefill fallback.

    make_request(*, mode=None, tool_choice=None, tools=None, text="hi")
        Builds the INBOUND payload for a turn (the dict a WS/REST client sends).
        This is the single seam that encodes the field representation. If the
        design keeps suppress_tools+force-field, map here.

    run_turn(provider, payload)  -> list[content_block]
        Drives ONE turn through _run_agent / run_loop with `provider` injected,
        returns the assistant response content blocks. Abstracts WS-vs-REST.

    rest_client / ws_client
        Edge clients hitting real send_chat / send_message, for the
        validation + threading tests that must exercise the HTTP/WS boundary
        (status codes, error frames).

    SEARCH_TOOL  (str)
        The real registered name of the search tool, for the force-specific case.

HOW TO RUN
    pytest tests/test_force_search.py -q
    Full gate before merge:  pytest -q   (expect prior 2990 + these, 0 failed)
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio

import pytest

# Conftest-provided helpers used as module globals (the fixtures spy_provider,
# cloud_spy_provider, rest_client, ws_client are injected as params instead).
from tests.conftest import SEARCH_TOOL, make_request, run_turn, run_turn_async


# Reference vocabulary for the per-call tool_choice union. Source these from the
# real module once it exists (e.g. `from prometheus.api.tool_choice import ...`);
# kept inline here so the spec reads standalone.
AUTO = "auto"
NONE = "none"
REQUIRED = "required"


def _content_types(blocks):
    return [b["type"] for b in blocks]


def _tool_uses(blocks):
    return [b for b in blocks if b["type"] == "tool_use"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. MERGE GATE — the non-negotiable. Generalizing suppress_tools -> tool_choice
#    must not move the default or the shipped chat behavior by a single byte.
# ─────────────────────────────────────────────────────────────────────────────
class TestMergeGate:

    def test_absent_resolves_to_auto_and_offers_tools(self, spy_provider):
        """No mode / no tool_choice == today's agent path: tools offered."""
        payload = make_request(text="solve it")  # neither field set
        run_turn(spy_provider, payload)
        req = spy_provider.last_request
        assert req.tool_choice == AUTO
        assert req.tool_schema, "auto must still offer the full tool_schema"
        assert spy_provider.last_grammar is spy_provider.boot_grammar, (
            "auto must select the unmodified boot grammar"
        )

    def test_auto_is_byte_identical_to_pre_feature_agent_call(self, spy_provider):
        """The ApiMessageRequest emitted under auto matches the pre-feature
        agent request field-for-field (modulo the new tool_choice=auto tag)."""
        payload = make_request(mode="agent", text="solve it")
        run_turn(spy_provider, payload)
        req = spy_provider.last_request
        assert req.tool_choice == AUTO
        assert req.suppress_tools is False  # legacy semantics preserved
        assert req.tool_schema  # unchanged

    def test_chat_mode_still_zero_tools(self, spy_provider):
        """mode=chat must remain exactly today's behavior: empty tool_schema,
        grammar dropped. Proves the rename didn't regress Piece 2."""
        payload = make_request(mode="chat", text="just chat")
        blocks = run_turn(spy_provider, payload)
        req = spy_provider.last_request
        assert req.tool_choice == NONE
        assert req.tool_schema == []         # no prompt/payload tools
        assert spy_provider.last_grammar is None  # grammar dropped
        assert _content_types(blocks) == ["text"]  # single plain reply, no tool round


# ─────────────────────────────────────────────────────────────────────────────
# 2. REQUIRED — force a tool_use of ANY tool.
# ─────────────────────────────────────────────────────────────────────────────
class TestRequired:

    def test_required_constrains_grammar_local(self, spy_provider):
        payload = make_request(tool_choice=REQUIRED, text="look it up")
        run_turn(spy_provider, payload)
        g = spy_provider.last_grammar
        assert g is not None, "required must NOT drop the grammar"
        assert g is not spy_provider.boot_grammar, "required must select a constrained grammar"

    def test_required_emits_a_tool_use(self, spy_provider):
        """Behavioral: a required turn produces a tool round (>=1 tool_use),
        never a bare-text reply."""
        spy_provider.script(["<any tool call>"])
        payload = make_request(tool_choice=REQUIRED, text="look it up")
        blocks = run_turn(spy_provider, payload)
        assert _tool_uses(blocks), "required turn must contain a tool_use"

    def test_required_maps_to_native_tool_choice_cloud(self, cloud_spy_provider):
        payload = make_request(tool_choice=REQUIRED, text="look it up")
        run_turn(cloud_spy_provider, payload)
        assert cloud_spy_provider.last_tool_choice == REQUIRED

    def test_required_uses_synthetic_prefill_only_when_unsupported(self, cloud_spy_provider):
        """Synthetic tool_use prefill is the GATED fallback, not the default."""
        cloud_spy_provider.supports_required = True
        run_turn(cloud_spy_provider, make_request(tool_choice=REQUIRED))
        assert cloud_spy_provider.last_request.synthetic_prefill is None

        cloud_spy_provider.supports_required = False
        run_turn(cloud_spy_provider, make_request(tool_choice=REQUIRED))
        assert cloud_spy_provider.last_request.synthetic_prefill is not None


# ─────────────────────────────────────────────────────────────────────────────
# 3. FORCE SPECIFIC TOOL — {"tool": X}. Only X is reachable.
# ─────────────────────────────────────────────────────────────────────────────
class TestForceSpecificTool:

    def test_specific_tool_is_the_only_one_emitted(self, spy_provider):
        spy_provider.script(["<call SEARCH_TOOL>"])
        payload = make_request(tool_choice={"tool": SEARCH_TOOL}, text="search this")
        blocks = run_turn(spy_provider, payload)
        calls = _tool_uses(blocks)
        assert calls, "forced-tool turn must contain a tool_use"
        assert all(c["name"] == SEARCH_TOOL for c in calls)

    def test_other_tools_structurally_unreachable_local(self, spy_provider):
        """The selected grammar must fix the tool name — a different tool is not
        a permissible production, not merely discouraged."""
        run_turn(spy_provider, make_request(tool_choice={"tool": SEARCH_TOOL}))
        g = spy_provider.last_grammar
        assert g is not None and g is not spy_provider.boot_grammar
        assert spy_provider.grammar_admits_tool(g, SEARCH_TOOL) is True
        assert spy_provider.grammar_admits_tool(g, "__some_other_tool__") is False

    def test_specific_tool_maps_to_native_cloud(self, cloud_spy_provider):
        run_turn(cloud_spy_provider, make_request(tool_choice={"tool": SEARCH_TOOL}))
        assert cloud_spy_provider.last_tool_choice == {"tool": SEARCH_TOOL}


# ─────────────────────────────────────────────────────────────────────────────
# 4. NO-CROSSTALK — the test that proves the DESIGN (per-call selection), not
#    just behavior. Mirrors the Piece 2 shared-context proof.
# ─────────────────────────────────────────────────────────────────────────────
class TestConcurrencyNoCrosstalk:

    @pytest.mark.asyncio
    async def test_concurrent_auto_and_required_do_not_interfere(self, spy_provider):
        """Two turns on the SAME shared LlamaCppProvider, concurrently: one auto,
        one required. Each call selects its own grammar; the shared boot grammar
        is unmutated afterward."""
        boot_before = spy_provider.boot_grammar

        auto_payload = make_request(mode="agent", text="A")
        req_payload = make_request(tool_choice=REQUIRED, text="B")

        results = await asyncio.gather(
            run_turn_async(spy_provider, auto_payload),
            run_turn_async(spy_provider, req_payload),
        )

        per_call = {r.tool_choice: r for r in spy_provider.requests}
        assert per_call[AUTO].selected_grammar is boot_before
        assert per_call[REQUIRED].selected_grammar is not boot_before
        assert per_call[REQUIRED].selected_grammar is not None
        # shared state untouched
        assert spy_provider.boot_grammar is boot_before
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_tool_choice_never_lands_on_shared_loop_context(self, spy_provider):
        """tool_choice is a per-call run_loop param, like mode. It must never be
        written onto the shared loop_context."""
        await run_turn_async(spy_provider, make_request(tool_choice=REQUIRED))
        assert not hasattr(spy_provider.shared_loop_context, "tool_choice"), (
            "tool_choice leaked onto shared loop_context"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. VALIDATION — mirror malformed-mode handling: 400 (REST) / error frame (WS).
# ─────────────────────────────────────────────────────────────────────────────
class TestValidation:

    @pytest.mark.asyncio
    async def test_force_nonexistent_tool_rejected_rest(self, rest_client):
        resp = await rest_client.send_chat(make_request(tool_choice={"tool": "__nope__"}))
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_malformed_tool_choice_rejected_rest(self, rest_client):
        resp = await rest_client.send_chat(make_request(tool_choice="garbage"))
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_force_nonexistent_tool_error_frame_ws(self, ws_client):
        frame = await ws_client.send_message(make_request(tool_choice={"tool": "__nope__"}))
        assert frame["type"] == "error"

    @pytest.mark.asyncio
    async def test_absent_tool_choice_never_errors(self, rest_client):
        """Absent must default to agent/auto, never 400 — same as absent mode."""
        resp = await rest_client.send_chat(make_request(text="hi"))
        assert resp.status == 200


# ─────────────────────────────────────────────────────────────────────────────
# 6. THREADING — tool_choice survives the full chain onto the ApiMessageRequest.
# ─────────────────────────────────────────────────────────────────────────────
class TestThreading:

    def test_tool_choice_reaches_api_request_via_full_chain(self, spy_provider):
        """send_chat/send_message -> dispatch_user_message -> _handle_send_message
        -> _run_agent -> run_loop -> ApiMessageRequest.tool_choice."""
        run_turn(spy_provider, make_request(tool_choice=REQUIRED, text="x"))
        assert spy_provider.last_request.tool_choice == REQUIRED

    def test_mode_sugar_resolves_to_tool_choice(self, spy_provider):
        """mode is sugar: agent->auto, chat->none, resolved before the provider."""
        run_turn(spy_provider, make_request(mode="agent"))
        assert spy_provider.last_request.tool_choice == AUTO
        run_turn(spy_provider, make_request(mode="chat"))
        assert spy_provider.last_request.tool_choice == NONE
