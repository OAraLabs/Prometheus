"""Every tool's ``example_call`` validates against that tool's own input model.

``example_call`` is not part of any schema the model is sent. Its one reader is
``_build_structured_error`` (adapter/validator.py), which appends
``Example: {...}`` — taken from the FIRST registered tool that declares one —
to the error text a model gets back after a failed call: the moment it is
most likely to copy what it is shown.

read_file, write_file and edit_file declared ``file_path`` / ``old_string`` /
``new_string``, none of which exist in their models (``path``, ``old_str``,
``new_str``). That stayed latent only because bash registers first in the
default registry and its example is right; any registry without bash in
front would have taught those names on every failed call. The earlier
checks (grep's in #134, the vault tools') each covered one tool by hand,
which is how three wrong examples in one directory went unnoticed.

So this walks every registry that serves tools to a model and checks every
example two ways. Each key must be a schema property: the models keep
pydantic's default ``extra="ignore"``, so a misspelled OPTIONAL key would
validate and still teach a name that does nothing. And ``model_validate``
must succeed: a key-subset check passes an example that omits a required
field or gives a wrong type.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from prometheus.adapter.validator import Strictness, ToolCallValidator
from prometheus.tools.base import BaseTool, ToolRegistry
from prometheus.tools.builtin.file_edit import FileEditTool
from prometheus.tools.builtin.file_read import FileReadTool
from prometheus.tools.builtin.file_write import FileWriteTool

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture
def served_tools(tmp_path, monkeypatch) -> list[tuple[str, BaseTool]]:
    """Every tool a model can be served, labelled by the registry serving it."""
    from prometheus.__main__ import create_tool_registry
    from prometheus.coding.sandbox import ProcessSandbox
    from prometheus.coding.tools import build_coding_registry
    from prometheus.symbiote.github_search import GitHubClient, GitHubSearchTool

    # The default build writes under ~ (it creates ~/.prometheus/skills).
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    box = tmp_path / "box"
    box.mkdir()
    # web_discover is served by the default registry only when an Exa key is
    # configured; a fake one makes the walk reach it.
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")

    tools = [("default", t) for t in create_tool_registry({}).list_tools()]
    tools += [
        ("coding", t)
        for t in build_coding_registry(ProcessSandbox(root=box)).list_tools()
    ]
    # create_tool_registry registers github_search too, but through
    # try_register, which logs and moves on when the import fails. Built
    # directly here so its example is checked either way.
    tools.append(
        ("symbiote", GitHubSearchTool(client=GitHubClient.from_config(None)))
    )
    return tools


def _example_problems(tool: BaseTool) -> list[str]:
    example = tool.example_call
    props = set(tool.input_model.model_json_schema().get("properties", {}))
    problems = [
        f"key {key!r} is not a property of {tool.input_model.__name__} "
        f"(properties: {sorted(props)})"
        for key in sorted(set(example) - props)
    ]
    try:
        tool.input_model.model_validate(example)
    except ValidationError as exc:
        problems.append(f"model_validate failed: {exc.errors(include_url=False)}")
    return problems


def test_every_example_call_validates_against_its_input_model(served_tools):
    checked: list[str] = []
    offenders: list[str] = []
    for source, tool in served_tools:
        if tool.example_call is None:
            continue
        checked.append(tool.name)
        offenders += [
            f"{source}:{tool.name} ({type(tool).__name__}): {problem}"
            for problem in _example_problems(tool)
        ]

    assert checked, "no served tool declares an example_call: the walk checked nothing"
    assert not offenders, "\n".join(offenders)


def _classes_declaring_an_example() -> set[tuple[str, str]]:
    """(module, class) for every class under src/prometheus that assigns a
    non-None ``example_call`` in its body."""
    declared: set[tuple[str, str]] = set()
    for path in sorted((SRC / "prometheus").rglob("*.py")):
        parts = path.relative_to(SRC).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module = ".".join(parts)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    names = [stmt.target.id]
                else:
                    continue
                if "example_call" not in names or stmt.value is None:
                    continue
                if isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
                    continue
                declared.add((module, node.name))
    return declared


def test_the_walk_reaches_every_class_that_declares_an_example(served_tools):
    """The check above is only as complete as its walk. A tool that declares
    an example but is served from a registry not walked here would never be
    checked — so every declaring class must be reached."""
    walked = {
        (cls.__module__, cls.__name__)
        for _, tool in served_tools
        for cls in type(tool).__mro__
    }
    declared = _classes_declaring_an_example()

    assert declared, "the source scan found no example_call declarations"
    unreached = sorted(declared - walked)
    assert not unreached, (
        f"these classes declare an example_call that no walked registry serves: "
        f"{unreached}. Add the registry that serves them to served_tools."
    )


def test_the_error_example_teaches_real_names_without_bash_in_front():
    """The latent path: no bash, so the structured error's ``Example:`` comes
    from a file tool."""
    for first in (FileReadTool(), FileWriteTool(), FileEditTool()):
        registry = ToolRegistry()
        registry.register(first)
        result = ToolCallValidator(strictness=Strictness.MEDIUM).validate(
            "nonexistent", {}, registry,
        )

        line = next(
            ln for ln in result.error.splitlines() if ln.startswith("Example: ")
        )
        example = json.loads(line.removeprefix("Example: "))
        assert example["name"] == first.name
        first.input_model.model_validate(example["arguments"])
        props = set(first.input_model.model_json_schema()["properties"])
        assert set(example["arguments"]) <= props, first.name
