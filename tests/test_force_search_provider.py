"""Real LlamaCppProvider GBNF grammar SELECTION for force-search (IGNITION Piece 3).

Drives the ACTUAL provider + the ACTUAL StructuredOutputEnforcer (boot grammar
and derived grammars both come from the real generate path — no hand-written
grammar, no doubles, nothing to register). Proves:

  * auto/absent  -> the UNTOUCHED boot grammar — asserted by IDENTITY (`is`),
                    not equality: the dormant path never regenerates.
  * none/suppress-> grammar dropped (chat parity).
  * required     -> enforcer-GENERATED tool-call-only root
                    (require_tool_use=True) — the draft's string-replace hack
                    is dead; equality with the enforcer's own output is the proof.
  * {tool:X}     -> single-alternative grammar admitting ONLY X — X-rejects-Y
                    asserted both by grammar_admits_tool and structurally.
                    (Draft #73 degraded this to plain `required`; that
                    degradation is the inverted assertion below.)
  * fail-loud    -> required/{tool} without a wired grammar source raises;
                    an unmapped tool_choice value raises (never degrades).
"""

from __future__ import annotations

import pytest

from prometheus.adapter.enforcer import StructuredOutputEnforcer
from prometheus.engine.messages import ConversationMessage
from prometheus.providers.base import ApiMessageRequest
from prometheus.providers.llama_cpp import LlamaCppProvider

SCHEMAS = [
    {"name": "web_search", "description": "search the web",
     "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}},
    {"name": "read_file", "description": "read a file",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
]


def _provider(*, wire_source: bool = True) -> tuple[LlamaCppProvider, StructuredOutputEnforcer]:
    enforcer = StructuredOutputEnforcer()
    boot = enforcer.generate_grammar(SCHEMAS)
    p = LlamaCppProvider(grammar=boot)
    if wire_source:
        p.set_grammar_source(enforcer, SCHEMAS)  # mirrors the daemon boot wiring
    return p, enforcer


def _req(**kw) -> ApiMessageRequest:
    return ApiMessageRequest(model="m", messages=[ConversationMessage.from_user_text("hi")], tools=[], **kw)


def test_auto_is_the_boot_grammar_object_identity():
    p, _ = _provider()
    payload = p._build_request_payload(_req(tool_choice="auto"))
    # Identity, not equality: the dormant path must not regenerate or re-derive.
    assert payload["grammar"] is p._grammar
    # Absent tool_choice (default) — same object again.
    assert p._build_request_payload(_req())["grammar"] is p._grammar


def test_none_and_suppress_drop_grammar():
    p, _ = _provider()
    assert "grammar" not in p._build_request_payload(_req(tool_choice="none"))
    assert "grammar" not in p._build_request_payload(_req(suppress_tools=True))


def test_required_comes_from_the_generate_path():
    p, enforcer = _provider()
    g = p._build_request_payload(_req(tool_choice="required"))["grammar"]
    # The proof the string-replace hack is dead: byte-equal to the enforcer's
    # own require_tool_use output for the same schemas.
    assert g == enforcer.generate_grammar(SCHEMAS, require_tool_use=True)
    assert "root ::= tool-call\n" in g and "| prose" not in g
    # Both tools remain admissible under required (forced SOME tool, not one).
    assert StructuredOutputEnforcer.grammar_admits_tool(g, "web_search")
    assert StructuredOutputEnforcer.grammar_admits_tool(g, "read_file")
    # Boot grammar untouched by derivation.
    assert p._build_request_payload(_req(tool_choice="auto"))["grammar"] is p._grammar


def test_specific_tool_admits_only_that_tool():
    # THE INVERTED DEGRADATION: draft #73 asserted {"tool": X} -> plain required
    # ("either way prose is gone"). Now X is locked: the grammar admits X and
    # REJECTS Y, and matches the enforcer's only_tool output exactly.
    p, enforcer = _provider()
    g = p._build_request_payload(_req(tool_choice={"tool": "web_search"}))["grammar"]
    assert g == enforcer.generate_grammar(SCHEMAS, require_tool_use=True, only_tool="web_search")
    assert "root ::= tool-call\n" in g and "| prose" not in g
    assert StructuredOutputEnforcer.grammar_admits_tool(g, "web_search")
    assert not StructuredOutputEnforcer.grammar_admits_tool(g, "read_file"), (
        "the X-only grammar must REJECT a call to tool Y"
    )


def test_derived_grammars_are_cached_by_name():
    p, _ = _provider()
    g1 = p._build_request_payload(_req(tool_choice={"tool": "web_search"}))["grammar"]
    g2 = p._build_request_payload(_req(tool_choice={"tool": "web_search"}))["grammar"]
    assert g1 is g2  # cache hit — same object, keyed "tool:web_search"
    r1 = p._build_request_payload(_req(tool_choice="required"))["grammar"]
    r2 = p._build_request_payload(_req(tool_choice="required"))["grammar"]
    assert r1 is r2


def test_forced_turn_without_grammar_source_fails_loud():
    p, _ = _provider(wire_source=False)
    with pytest.raises(RuntimeError, match="set_grammar_source"):
        p._build_request_payload(_req(tool_choice="required"))
    # auto still works without the source (boot grammar only) — dormant path safe.
    assert p._build_request_payload(_req(tool_choice="auto"))["grammar"] is p._grammar


def test_unknown_forced_tool_raises_via_the_enforcer():
    # Assert-not-revalidate: the provider does no name checking; the enforcer's
    # own only_tool raise is the (loud) backstop for a name that slipped past ingress.
    p, _ = _provider()
    with pytest.raises(ValueError):
        p._build_request_payload(_req(tool_choice={"tool": "not_registered"}))


def test_unmapped_value_raises_never_degrades():
    p, _ = _provider()
    with pytest.raises(ValueError, match="unmapped tool_choice"):
        p._build_request_payload(_req(tool_choice="bogus"))
