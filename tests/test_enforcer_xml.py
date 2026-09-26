"""The full-tier reader reads Qwen's XML tool calls (WP-X.28, PR 1).

THE DEFECT
----------
A model whose chat template teaches Qwen3-Coder XML — Qwen3.5 / Qwen3.8, and
Bonsai 2, the same checkpoint — keeps writing it at adapter tier ``full``,
where the tools are withheld from the request and the server parses nothing.
The enforcer read JSON only, so the call reached the markup stripper unparsed
and was deleted: a reply that was only the call became a PARSE DISAGREEMENT
retry, and a call after prose vanished while the prose became the answer. The
Bonsai tier sweep (611 runs) ended 21 tier-full runs in that disagreement and
scored ``full`` 25 points under ``light``.

THE FORMAT is the one the production server's own template renders. The
example below is the template's own, checked here against the ``/props``
payload recorded in the ``tool_calls`` golden, so the fixture cannot drift
from what the model was taught.

THE MUTATION: remove Strategy 0 from ``extract_tool_calls`` and every reading
test here goes red; the JSON regression tests stay green either way, which is
what proves the new strategy changed nothing else.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest
from pydantic import BaseModel

from prometheus.adapter import ModelAdapter
from prometheus.adapter.enforcer import StructuredOutputEnforcer
from prometheus.adapter.formatter import QwenFormatter, strip_tool_call_markup
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiTextDeltaEvent,
    ModelProvider,
)
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult

REPO = Path(__file__).resolve().parents[1]
TOOL_CALLS_GOLDEN = REPO / "tests" / "fixtures" / "parity" / "tool_calls.trace.json"

# Verbatim from the recorded template's instruction block (see the first test).
EXAMPLE = (
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)
MULTILINE_VALUE = "This is the value for the second parameter\nthat can span\nmultiple lines"


def _call(name: str, **params: str) -> str:
    """Render one call exactly the way the template renders a history call."""
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def _recorded_chat_template() -> str:
    trace = json.loads(TOOL_CALLS_GOLDEN.read_text())
    for exchange in trace["exchanges"]:
        if exchange.get("path") == "/props":
            return json.loads(exchange["body"])["chat_template"]
    raise AssertionError("the tool_calls golden records no /props exchange")


def _blocks(calls: list[ToolUseBlock]) -> list[tuple[str, dict]]:
    return [(c.name, c.input) for c in calls]


# ---------------------------------------------------------------------------
# The fixture is the template's own
# ---------------------------------------------------------------------------

def test_the_example_is_the_recorded_templates_own():
    """The template source holds the example as a Jinja string literal, so its
    newlines are the two characters backslash-n there."""
    template = _recorded_chat_template()
    assert EXAMPLE.replace("\n", "\\n") in template
    # And the template renders history calls with the same shape the reader
    # expects: a newline after the opening tags and one before the closers.
    assert "'<parameter=' + args_name + '>\\n'" in template
    assert "'\\n</parameter>\\n'" in template
    assert "'</function>\\n</tool_call>'" in template


# ---------------------------------------------------------------------------
# Reading (no registry: every value is text)
# ---------------------------------------------------------------------------

class TestReadsTheTemplatesFormat:
    def test_the_templates_own_example(self):
        calls = StructuredOutputEnforcer().extract_tool_calls(EXAMPLE)
        assert _blocks(calls) == [(
            "example_function_name",
            {"example_parameter_1": "value_1", "example_parameter_2": MULTILINE_VALUE},
        )]
        assert calls[0].id.startswith("toolu_")

    def test_two_calls_in_order(self):
        text = _call("read_file", path="a.txt") + "\n" + _call("read_file", path="b.txt")
        assert _blocks(StructuredOutputEnforcer().extract_tool_calls(text)) == [
            ("read_file", {"path": "a.txt"}),
            ("read_file", {"path": "b.txt"}),
        ]

    def test_prose_before_the_call(self):
        text = "Let me read that file first.\n\n" + _call("read_file", path="notes.md")
        assert _blocks(StructuredOutputEnforcer().extract_tool_calls(text)) == [
            ("read_file", {"path": "notes.md"}),
        ]

    def test_a_call_with_no_parameters(self):
        text = "<tool_call>\n<function=sessions_list>\n</function>\n</tool_call>"
        assert _blocks(StructuredOutputEnforcer().extract_tool_calls(text)) == [
            ("sessions_list", {}),
        ]

    def test_a_truncated_envelope_still_reads(self):
        # max_tokens cut the reply inside the value: no closers at all.
        text = "<tool_call>\n<function=bash>\n<parameter=command>\nls -la /ho"
        assert _blocks(StructuredOutputEnforcer().extract_tool_calls(text)) == [
            ("bash", {"command": "ls -la /ho"}),
        ]

    def test_angle_brackets_inside_a_value_survive(self):
        command = 'echo "<b>bold</b>" > out.html && cat < in.txt'
        text = _call("bash", command=command)
        assert _blocks(StructuredOutputEnforcer().extract_tool_calls(text)) == [
            ("bash", {"command": command}),
        ]

    def test_a_value_that_ends_in_a_newline_keeps_it(self):
        # The template adds one newline before </parameter>; the reader strips
        # exactly one, so content ending in a newline is not shortened.
        from prometheus.adapter.enforcer import parse_xml_tool_calls

        text = "<function=write>\n<parameter=content>\nline one\nline two\n\n</parameter>\n</function>"
        assert parse_xml_tool_calls(text)[0].input == {"content": "line one\nline two\n"}

    def test_a_bare_function_block_without_the_envelope(self):
        text = "<function=read_file>\n<parameter=path>\nREADME.md\n</parameter>\n</function>"
        assert _blocks(StructuredOutputEnforcer().extract_tool_calls(text)) == [
            ("read_file", {"path": "README.md"}),
        ]

    def test_the_same_call_twice_is_one_call(self):
        text = _call("read_file", path="a.txt") + _call("read_file", path="a.txt")
        assert len(StructuredOutputEnforcer().extract_tool_calls(text)) == 1

    def test_no_function_tag_means_no_xml_call(self):
        from prometheus.adapter.enforcer import parse_xml_tool_calls

        assert parse_xml_tool_calls("<tool_call>\nnot a call\n</tool_call>") == []
        assert parse_xml_tool_calls("") == []


# ---------------------------------------------------------------------------
# Reading with a registry: values typed by the schema, names filtered
# ---------------------------------------------------------------------------

class _TypedArgs(BaseModel):
    path: str
    count: int
    tags: list[str] = []
    dry_run: bool = False
    limit: int | None = None


class _TypedTool(BaseTool):
    name = "typed_tool"
    description = "a tool with one parameter of each kind"
    input_model = _TypedArgs

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        return ToolResult(output="ok")


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_TypedTool())
    return reg


class TestReadsWithASchema:
    def test_values_are_typed_the_way_the_template_renders_them(self, registry):
        # A string parameter stays text even when it looks like a number (the
        # file literally named 42); everything else is the JSON the template
        # would have rendered for it.
        text = _call("typed_tool", path="42", count="3", tags='["a", "b"]',
                     dry_run="true", limit="7")
        calls = StructuredOutputEnforcer().extract_tool_calls(text, registry)
        assert _blocks(calls) == [(
            "typed_tool",
            {"path": "42", "count": 3, "tags": ["a", "b"], "dry_run": True, "limit": 7},
        )]

    def test_text_that_is_not_json_for_a_typed_parameter_stays_text(self, registry):
        # The reader does not invent a value; the validator reports it.
        text = _call("typed_tool", path="x", count="three")
        calls = StructuredOutputEnforcer().extract_tool_calls(text, registry)
        assert calls[0].input == {"path": "x", "count": "three"}

    def test_an_unknown_tool_is_filtered_like_a_json_one(self, registry):
        text = _call("not_registered", path="x")
        assert StructuredOutputEnforcer().extract_tool_calls(text, registry) == []

    def test_a_registry_without_schemas_keeps_every_value_as_text(self):
        class _Duck:
            def get(self, name):  # noqa: ANN001
                return object()

        text = _call("typed_tool", count="3")
        calls = StructuredOutputEnforcer().extract_tool_calls(text, _Duck())
        assert calls[0].input == {"count": "3"}


# ---------------------------------------------------------------------------
# The JSON strategies are untouched
# ---------------------------------------------------------------------------

JSON_SHAPES = {
    "fenced json": '```json\n{"name": "bash", "arguments": {"command": "ls"}}\n```',
    "generic fence": '```\n{"name": "bash", "arguments": {"command": "ls"}}\n```',
    "own line": 'Sure.\n{"name": "bash", "arguments": {"command": "ls"}}\nDone.',
    "json in the envelope": '<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>',
    "bare object": '{"name": "bash", "arguments": {"command": "ls"}}',
}


@pytest.mark.parametrize("label", list(JSON_SHAPES))
def test_json_shapes_read_exactly_as_before(label):
    calls = StructuredOutputEnforcer().extract_tool_calls(JSON_SHAPES[label])
    assert _blocks(calls) == [("bash", {"command": "ls"})]


@pytest.mark.parametrize("label", list(JSON_SHAPES))
def test_json_shapes_carry_no_xml_call(label):
    from prometheus.adapter.enforcer import parse_xml_tool_calls

    assert parse_xml_tool_calls(JSON_SHAPES[label]) == []


# ---------------------------------------------------------------------------
# The formatter reads the same format through the same helper
# ---------------------------------------------------------------------------

class TestQwenFormatterReadsXml:
    def test_reads_the_templates_example(self):
        calls = QwenFormatter().parse_tool_calls(EXAMPLE)
        assert _blocks(calls) == [(
            "example_function_name",
            {"example_parameter_1": "value_1", "example_parameter_2": MULTILINE_VALUE},
        )]

    def test_still_reads_json(self):
        calls = QwenFormatter().parse_tool_calls(JSON_SHAPES["bare object"])
        assert _blocks(calls) == [("bash", {"command": "ls"})]


# ---------------------------------------------------------------------------
# The one-shot stripper leaves no call markup in the residual text
# ---------------------------------------------------------------------------

class TestStripLeavesNoXml:
    def test_the_envelope_strips_to_nothing(self):
        assert strip_tool_call_markup(EXAMPLE) == ""

    def test_a_bare_function_block_is_removed_and_prose_kept(self):
        text = "On it. <function=read_file>\n<parameter=path>\na\n</parameter>\n</function> Reading now."
        assert strip_tool_call_markup(text) == "On it.  Reading now."

    def test_a_truncated_bare_block_is_removed_to_the_end(self):
        assert strip_tool_call_markup("Sure. <function=bash>\n<parameter=command>\nls") == "Sure. "

    def test_text_without_calls_is_unchanged(self):
        text = "a < b and b > c, <not a tag>, done"
        assert strip_tool_call_markup(text) == text


# ---------------------------------------------------------------------------
# Through the loop: a tier-full XML reply executes the tool
# ---------------------------------------------------------------------------

RECOVERED = "The file named 42 says: forty-two."


class _RecordingTool(BaseTool):
    name = "note_read"
    description = "reads a note"
    input_model = _TypedArgs

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        self.calls.append(arguments.model_dump())
        return ToolResult(output="forty-two")


class _XmlThenAnswerProvider(ModelProvider):
    """Turn 1: the call, as Qwen writes it. Turn 2: the answer."""

    def __init__(self) -> None:
        self.calls = 0
        self.saw_tool_result = False
        self.saw_feedback = False

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        for m in request.messages:
            if any(isinstance(b, ToolResultBlock) for b in m.content):
                self.saw_tool_result = True
            if "could not be parsed" in (getattr(m, "text", "") or ""):
                self.saw_feedback = True
        text = _call("note_read", path="42", count="3") if self.calls == 1 else RECOVERED
        yield ApiTextDeltaEvent(text=text)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text=text)]),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


def test_a_tier_full_xml_reply_executes_the_tool(caplog):
    tool = _RecordingTool()
    reg = ToolRegistry()
    reg.register(tool)
    provider = _XmlThenAnswerProvider()
    ctx = LoopContext(
        provider=provider,
        model="test",
        system_prompt="- Model: test (provider: test)",
        max_tokens=256,
        # The daemon's own tier-full adapter, as create_adapter builds it.
        adapter=ModelAdapter(formatter=QwenFormatter(), strictness="MEDIUM", tier="full"),
        tool_registry=reg,
        max_turns=4,
    )
    messages = [ConversationMessage.from_user_text("what does the file named 42 say?")]

    async def _run() -> str:
        last = ""
        async for event, _usage in run_loop(ctx, messages):
            if type(event).__name__ == "AssistantTurnComplete":
                last = event.message.text or ""
        return last

    with caplog.at_level(logging.WARNING):
        final = asyncio.run(_run())

    assert tool.calls == [{"path": "42", "count": 3, "tags": [], "dry_run": False, "limit": None}], (
        "the XML call was not executed — the reader missed it and the stripper "
        "deleted it, which is the defect"
    )
    assert provider.saw_tool_result, "the tool's result never went back to the model"
    assert not provider.saw_feedback, "the loop retried a parse disagreement instead of running the call"
    assert not [r for r in caplog.records if "PARSE DISAGREEMENT" in r.getMessage()]
    assert final == RECOVERED


# ---------------------------------------------------------------------------
# Both formats in one reply: document order, the JSON reader kept off the XML
# ---------------------------------------------------------------------------

# coding_run's exchange 12 as recorded on 2026-09-25 (the production 27B, told
# to write JSON and trained to write XML, wrote two DIFFERENT calls in one
# reply): the first reader kept only the XML one and dropped the JSON one.
EXCHANGE_12 = (
    '```json\n{"name": "code_view", "arguments": {"path": '
    '"/tmp/prometheus-parity/home/.prometheus/coding/cparity01-1790383216/calc.py"}}\n```\n'
    "<tool_call>\n<function=code_view>\n<parameter=path>\n"
    "/tmp/prometheus-parity/home/.prometheus/coding/cparity01-1790383216/test_calc.py\n"
    "</parameter>\n</function>\n</tool_call>"
)
CALC = "/tmp/prometheus-parity/home/.prometheus/coding/cparity01-1790383216/calc.py"
TEST_CALC = "/tmp/prometheus-parity/home/.prometheus/coding/cparity01-1790383216/test_calc.py"


class TestDocumentOrder:
    def test_exchange_12_yields_both_calls_in_the_order_written(self):
        assert _blocks(StructuredOutputEnforcer().extract_tool_calls(EXCHANGE_12)) == [
            ("code_view", {"path": CALC}),
            ("code_view", {"path": TEST_CALC}),
        ]

    def test_xml_before_json_keeps_that_order(self):
        text = _call("read_file", path="first.txt") + '\n{"name": "read_file", "arguments": {"path": "second.txt"}}'
        assert _blocks(StructuredOutputEnforcer().extract_tool_calls(text)) == [
            ("read_file", {"path": "first.txt"}),
            ("read_file", {"path": "second.txt"}),
        ]

    def test_the_same_call_in_both_formats_is_one_call(self):
        # The old recording's shape: the JSON call, then the XML rendering of it.
        text = ('{"name": "code_run", "arguments": {"command": "python test_calc.py"}}\n\n'
                + _call("code_run", command="python test_calc.py"))
        assert _blocks(StructuredOutputEnforcer().extract_tool_calls(text)) == [
            ("code_run", {"command": "python test_calc.py"}),
        ]

    def test_a_json_object_inside_a_parameter_value_is_the_value_not_a_call(self):
        payload = '{"name": "bash", "arguments": {"command": "rm -rf /"}}'
        text = _call("write_file", path="notes.json", content=payload)
        calls = StructuredOutputEnforcer().extract_tool_calls(text)
        assert _blocks(calls) == [("write_file", {"path": "notes.json", "content": payload})]

    def test_a_fenced_json_object_inside_a_parameter_value_is_the_value(self):
        payload = '```json\n{"name": "bash", "arguments": {"command": "ls"}}\n```'
        text = "Saving the snippet.\n" + _call("write_file", path="snippet.md", content=payload)
        assert _blocks(StructuredOutputEnforcer().extract_tool_calls(text)) == [
            ("write_file", {"path": "snippet.md", "content": payload}),
        ]

    def test_a_truncated_xml_call_still_blanks_its_text(self):
        # No closers at all: the JSON inside the cut value is still the value.
        text = '<function=write_file>\n<parameter=content>\n{"name": "bash", "arguments": {"command": "ls"'
        calls = StructuredOutputEnforcer().extract_tool_calls(text)
        assert [c.name for c in calls] == ["write_file"]


# A reply with no "<function=" takes exactly the path it always took: the
# same strategies, in the same order, with the same precedence between them.
NO_XML_SHAPES = {
    "two fenced calls, in order": (
        '```json\n{"name": "read_file", "arguments": {"path": "a"}}\n```\nthen\n'
        '```json\n{"name": "read_file", "arguments": {"path": "b"}}\n```',
        [("read_file", {"path": "a"}), ("read_file", {"path": "b"})]),
    "a fenced call wins over a bare line elsewhere (strategy precedence)": (
        '{"name": "read_file", "arguments": {"path": "line"}}\n'
        '```json\n{"name": "read_file", "arguments": {"path": "fenced"}}\n```',
        [("read_file", {"path": "fenced"})]),
    "two calls on their own lines": (
        'Sure.\n{"name": "read_file", "arguments": {"path": "a"}}\n'
        '{"name": "read_file", "arguments": {"path": "b"}}\nDone.',
        [("read_file", {"path": "a"}), ("read_file", {"path": "b"})]),
    "objects in prose (the greedy last resort)": (
        'Call {"name": "read_file", "arguments": {"path": "a"}} and then '
        '{"name": "read_file", "arguments": {"path": "b"}} please',
        [("read_file", {"path": "a"}), ("read_file", {"path": "b"})]),
    "the same call twice is one": (
        '{"name": "read_file", "arguments": {"path": "a"}}\n{"name": "read_file", "arguments": {"path": "a"}}',
        [("read_file", {"path": "a"})]),
    "a truncated bare object is no call (main finds no closing brace either)": (
        '{"name": "read_file", "arguments": {"path": "a"', []),
    "no call at all": ("Just prose, with a {brace} and <tool_call> mentioned.", []),
}


@pytest.mark.parametrize("label", list(NO_XML_SHAPES))
def test_no_xml_replies_read_exactly_as_before(label):
    text, expected = NO_XML_SHAPES[label]
    assert _blocks(StructuredOutputEnforcer().extract_tool_calls(text)) == expected
