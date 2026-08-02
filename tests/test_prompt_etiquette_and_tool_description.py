"""Two strings that only matter if they actually reach the model.

Both changes here are text — a prompt section and a tool description — and text
changes are the easiest kind to write and never wire. [[Standing-Principles]] §1
is a list of exactly that: a `tool_result_max` truncator never wired, a
grammar-enforcement path never applied, `MEMORY.md` empty for six weeks because
`registry.register(MemoryTool())` was never called.

So neither test asserts on the source constant. The etiquette test asserts on
the **assembled prompt**, and the description test asserts on the **schema the
registry hands a provider** — the bytes a model would actually receive.
"""

from __future__ import annotations

from prometheus.context.prompt_assembler import (
    _MEMORY_ETIQUETTE,
    build_runtime_system_prompt,
)
from prometheus.tools.base import ToolRegistry
from prometheus.tools.builtin.file_write import FileWriteTool


# ---------------------------------------------------------------------------
# (a) memory etiquette — present, and ORDERED before the facts it governs
# ---------------------------------------------------------------------------

_MEMORY_MARKER = "ETIQUETTE-TEST-MEMORY-FACT"


def _assembled(tmp_path) -> str:
    return build_runtime_system_prompt(
        cwd=str(tmp_path),
        config={},
        memory_content=f"- {_MEMORY_MARKER}\n",
    )


def test_memory_etiquette_reaches_the_assembled_prompt(tmp_path):
    """The etiquette must be in the real output, not merely defined."""
    prompt = _assembled(tmp_path)
    assert _MEMORY_ETIQUETTE in prompt, (
        "_MEMORY_ETIQUETTE is defined but never reaches the assembled prompt — "
        "the string is written but not wired."
    )


def test_memory_etiquette_precedes_the_facts_it_governs(tmp_path):
    """Ordering is the whole point: instructions before the data they qualify."""
    prompt = _assembled(tmp_path)
    assert _MEMORY_MARKER in prompt, "memory_content did not reach the prompt at all"

    etiquette_at = prompt.index(_MEMORY_ETIQUETTE)
    facts_at = prompt.index(_MEMORY_MARKER)
    assert etiquette_at < facts_at, (
        "the etiquette appears AFTER the memory facts — a model reading in order "
        "sees the data before the instruction telling it how to use the data"
    )


def test_memory_etiquette_sits_under_the_memory_header(tmp_path):
    """It must be inside the # Memory section, not floating elsewhere."""
    prompt = _assembled(tmp_path)
    header_at = prompt.index("# Memory")
    etiquette_at = prompt.index(_MEMORY_ETIQUETTE)
    assert header_at < etiquette_at, (
        "the etiquette appears before its own '# Memory' header"
    )
    between = prompt[header_at + len("# Memory") : etiquette_at]
    assert between.strip() == "", (
        f"unexpected content between the '# Memory' header and the etiquette: "
        f"{between!r}"
    )


def test_no_memory_section_when_there_is_no_memory(tmp_path):
    """No facts, no section — the etiquette must not appear on its own."""
    prompt = build_runtime_system_prompt(
        cwd=str(tmp_path), config={}, memory_content=""
    )
    if "# Memory" not in prompt:
        assert _MEMORY_ETIQUETTE not in prompt, (
            "the etiquette leaked into a prompt that has no memory section"
        )


# ---------------------------------------------------------------------------
# (c) file_write description — what the REGISTRY exposes, not the source
# ---------------------------------------------------------------------------


def _registered_write_file_schemas() -> tuple[dict, dict]:
    """Register the real tool and return (anthropic_schema, openai_schema)."""
    registry = ToolRegistry()
    registry.register(FileWriteTool())

    api = next(s for s in registry.to_api_schema() if s["name"] == "write_file")
    openai = next(
        s
        for s in registry.to_openai_schemas()
        if s["function"]["name"] == "write_file"
    )
    return api, openai


def test_description_reaches_both_provider_schemas():
    """Both wire formats must carry the same description the tool declares."""
    api, openai = _registered_write_file_schemas()
    declared = FileWriteTool().description

    assert api["description"] == declared, (
        "the Anthropic-format schema does not expose the declared description"
    )
    assert openai["function"]["description"] == declared, (
        "the OpenAI/llama.cpp-format schema does not expose the declared "
        "description — this is the path the local model actually sees"
    )


def test_description_states_the_convert_via_bash_route():
    """The substantive change: binary formats are redirected, not refused.

    This is a routing instruction to the model. Asserting the text is here is
    the honest limit of a unit test — whether it changes behaviour is a
    measurement question, not an assertion (see the PR notes).
    """
    api, _ = _registered_write_file_schemas()
    desc = api["description"]

    assert "do not refuse" in desc.lower(), (
        "the description no longer tells the model not to refuse binary formats"
    )
    assert "bash" in desc.lower(), (
        "the description no longer names the conversion route (bash)"
    )
    for fmt in (".pdf", ".docx", ".xlsx"):
        assert fmt in desc, f"the description no longer names {fmt}"


def test_description_still_states_the_primary_behaviour():
    """Widening must not lose the original contract: create or overwrite text."""
    api, _ = _registered_write_file_schemas()
    desc = api["description"].lower()
    assert "overwrite" in desc
    assert "utf-8" in desc or "text" in desc
