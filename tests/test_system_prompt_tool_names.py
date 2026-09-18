"""The system prompt may only name tools that are actually registered.

WHY THIS EXISTS. `_format_documents_section()` told the model to save drafted plans with
``file_write``. The registered tool is ``write_file`` (``tools/builtin/file_write.py:36`` sets
``name = "write_file"`` — the MODULE is file_write, the TOOL is write_file, and the prompt took the
module name). There is no alias mechanism, so the instruction named a tool that does not exist.

It failed silently and completely: nothing validates prompt prose against the registry, a model
asking for an unknown tool just gets a rejection mid-turn, and the surface it was pointing at —
``<documents_root>/loops/`` — held two files in thirty-five days. The instruction that was supposed
to populate the tree could never have worked.

The guard targets the class, not the instance: a snake_case token that is NOT registered but whose
WORD SET matches a registered tool is a transposition, which is the mistake that actually happens
when a module name and a tool name differ in order.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re

import pytest


def _registered_tool_names() -> set[str]:
    """Every tool name the builtin package actually registers."""
    import prometheus.tools.builtin as pkg
    from prometheus.tools.base import BaseTool

    names: set[str] = set()
    for mod_info in pkgutil.iter_modules(pkg.__path__):
        try:
            mod = importlib.import_module(f"prometheus.tools.builtin.{mod_info.name}")
        except Exception:
            continue  # an optional dependency missing is not this test's business
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, BaseTool) and obj is not BaseTool:
                name = getattr(obj, "name", None)
                if isinstance(name, str) and name:
                    names.add(name)
    return names


@pytest.fixture(scope="module")
def tool_names() -> set[str]:
    names = _registered_tool_names()
    # Fail loud rather than vacuously pass: an empty registry would make every assertion below
    # trivially true, which is exactly the "test that cannot fail" this codebase keeps finding.
    assert len(names) > 20, f"tool enumeration collapsed ({len(names)} found) — the guard is blind"
    assert "write_file" in names, "write_file missing from the registry; the fixture is wrong"
    return names


def _prompt_text() -> str:
    from prometheus.context.system_prompt import build_system_prompt

    return build_system_prompt()


def test_documents_section_names_a_real_tool(tool_names: set[str]) -> None:
    """The instruction that points at documents/loops must name a registered tool."""
    from prometheus.context.system_prompt import _format_documents_section

    section = _format_documents_section()
    named = [t for t in re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", section) if t in tool_names]
    assert named, (
        "the documents-library section names no registered tool at all — it tells the model to "
        f"save a plan with something that does not exist. Section text:\n{section}"
    )


def test_prompt_contains_no_transposed_tool_name(tool_names: set[str]) -> None:
    """No snake_case token is a word-order variant of a real tool.

    This is the assertion that fails on the original defect: `file_write` is unregistered, and its
    word set {file, write} equals registered `write_file`'s. A plain "is it in the registry" check
    would flag half the prose in the prompt; matching on word SET flags only the mistake that
    actually happens.
    """
    by_words: dict[frozenset[str], str] = {frozenset(n.split("_")): n for n in tool_names}
    offenders: list[tuple[str, str]] = []
    for token in set(re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", _prompt_text())):
        if token in tool_names:
            continue
        real = by_words.get(frozenset(token.split("_")))
        if real and real != token:
            offenders.append((token, real))

    assert not offenders, "the prompt names tools that do not exist: " + ", ".join(
        f"{bad!r} (did you mean {good!r}?)" for bad, good in sorted(offenders)
    )
