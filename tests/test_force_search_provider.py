"""Real LlamaCppProvider GBNF grammar SELECTION for force-search (live tier — beyond the
spy double in test_force_search.py). Proves the per-call tool_choice picks the right grammar
on the actual provider, immutably (boot grammar never mutated), with auto/none byte-identical
to pre-force-search behavior."""

from __future__ import annotations

from prometheus.engine.messages import ConversationMessage
from prometheus.providers.base import ApiMessageRequest
from prometheus.providers.llama_cpp import LlamaCppProvider

# A minimal boot grammar with the real top-level shape the enforcer emits.
_BOOT = 'root ::= tool-call | prose\ntool-call ::= "{" "web_search" "}"\nprose ::= [^{] anychar*\n'


def _req(**kw) -> ApiMessageRequest:
    return ApiMessageRequest(model="m", messages=[ConversationMessage.from_user_text("hi")], tools=[], **kw)


def test_auto_keeps_boot_grammar_byte_identical():
    p = LlamaCppProvider(grammar=_BOOT)
    payload = p._build_request_payload(_req(tool_choice="auto"))
    assert payload["grammar"] == _BOOT


def test_none_and_suppress_drop_grammar():
    p = LlamaCppProvider(grammar=_BOOT)
    assert "grammar" not in p._build_request_payload(_req(tool_choice="none"))
    # legacy suppress_tools path still drops too (Piece 2 back-compat)
    assert "grammar" not in p._build_request_payload(_req(suppress_tools=True))


def test_required_drops_prose_branch_so_a_tool_is_forced():
    p = LlamaCppProvider(grammar=_BOOT)
    payload = p._build_request_payload(_req(tool_choice="required"))
    g = payload["grammar"]
    assert g is not None
    assert "| prose" not in g, "required must drop the prose alternative"
    assert "root ::= tool-call\n" in g, "root must require a tool-call"
    # the boot grammar is never mutated by selection
    assert p._grammar == _BOOT


def test_specific_tool_local_falls_back_to_required():
    # Local tier: {"tool": X} forces a tool (prose dropped); the specific-tool constraint is
    # carried natively by the cloud path (follow-up for local). Either way prose is gone.
    p = LlamaCppProvider(grammar=_BOOT)
    g = p._build_request_payload(_req(tool_choice={"tool": "web_search"}))["grammar"]
    assert "| prose" not in g
