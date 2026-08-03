"""The dispatch content gate — both directions.

WHY BOTH DIRECTIONS. A sanitizer exercised only on garbage is indistinguishable
from one that rejects everything (Standing Principles §2c: a control suite that
only tests refusals is blind by construction). Over-rejection here is not a
cosmetic problem — this gate sits on the live dispatch path, so a pattern that
is too eager is its own outage. Half the cases below therefore assert that
legitimate strings containing angle brackets and pipes are ADMITTED, and the
canonical one is real: ``test_real_bash_regex_is_admitted`` is a bash call that
actually ran and succeeded on 2026-07-24, and it is exactly what a
single-marker threshold would have broken.

The corrupt fixtures are verbatim values from the corpus — not invented shapes.
Gemma emits half-formed tokens with the pipe on either side, so a conventional
``<|name|>`` pattern would have matched none of the single-piped ones.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from prometheus.adapter import ModelAdapter
from prometheus.engine.agent_loop import LoopContext, _execute_tool_call
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

from prometheus.adapter.markup_guard import (
    MARKUP_RE,
    REJECT_AT,
    describe,
    find_markup,
    rejection_message,
    scan_arguments,
)


# Verbatim from the corpus (training.db / telemetry.db parsed_tool_call).
CORRUPT_WIKI_COMPILE = '{"}}<tool_call|><|tool_response>call:[]}'
CORRUPT_GREP_ROOT = (
    '{"@tmp/prometheus-gym/src<|"|>}}<tool_call|><|tool_response>"\n'
    "<|channel>thought\n<channel|><|tool_call>call:1.2345678901234567e+00}"
)
CORRUPT_TASK_PROMPT = (
    '{"<|"|>Audit" open security advisories."}}<tool_call|><|tool_response>\n'
    '<channel|><|tool_call>call:"<|thought|><|thought|>'
)
CORRUPT_MINIMAL = '<tool_call|><|tool_response>'  # the observed floor: 2 markers

# The real bash command from 2026-07-24 that SUCCEEDED. Contains `|>` inside a
# Python regex. A threshold of 1 would have rejected this working call.
REAL_BASH_REGEX = (
    "python3 - <<'PY'\nimport re\nseen=[]\n"
    r"for t in re.findall(r'<([a-zA-Z0-9_:-]+)(?:\s|>)', text):"
    "\n    seen.append(t)\nPY"
)


class TestRejects:
    """Markup is caught — the direction the gate exists for."""

    @pytest.mark.parametrize(
        "value",
        [CORRUPT_WIKI_COMPILE, CORRUPT_GREP_ROOT, CORRUPT_TASK_PROMPT, CORRUPT_MINIMAL],
        ids=["wiki_compile", "grep_root", "task_prompt", "minimal_floor"],
    )
    def test_corpus_values_are_rejected(self, value):
        assert scan_arguments({"arg": value}), (
            "a verbatim corrupt value from the corpus must be rejected"
        )

    def test_single_piped_tokens_match(self):
        """The shapes a conventional <|name|> pattern would miss entirely."""
        for token in ("<|tool_response>", "<tool_call|>", "<|tool_call>",
                      "<channel|>", "<|channel>"):
            assert find_markup(token), f"{token!r} must be recognised"

    def test_double_piped_tokens_match(self):
        for token in ("<|thought|>", '<|"|>'):
            assert find_markup(token), f"{token!r} must be recognised"

    def test_nested_argument_is_not_missed(self):
        found = scan_arguments({"outer": {"inner": [CORRUPT_WIKI_COMPILE]}})
        assert "outer.inner[0]" in found

    def test_reports_the_offending_path_and_markers(self):
        found = scan_arguments({"entity_name": CORRUPT_WIKI_COMPILE})
        assert list(found) == ["entity_name"]
        assert "<tool_call|>" in found["entity_name"]


class TestAdmits:
    """Legitimate strings survive — the direction that keeps this from being an outage."""

    def test_real_bash_regex_is_admitted(self):
        """THE CASE THAT SET THE THRESHOLD.

        A real bash call from 2026-07-24, success=1, whose command carries a
        Python regex containing ``|>``. One marker. Rejecting it would have
        broken working traffic to catch nothing.
        """
        assert find_markup(REAL_BASH_REGEX), "sanity: it does contain one marker"
        assert len(find_markup(REAL_BASH_REGEX)) < REJECT_AT
        assert not scan_arguments({"command": REAL_BASH_REGEX})

    def test_searching_for_the_tokens_still_works(self):
        """Investigating this very bug must not be blocked by the fix for it."""
        assert not scan_arguments({"pattern": "<|tool_call|>"})

    @pytest.mark.parametrize(
        "value",
        [
            "cat file.txt | grep foo > out.txt",          # shell pipe + redirect
            "if a < b and c > d: pass",                    # comparisons
            "List<String> items = new ArrayList<>();",     # generics + diamond
            "<div class='x'>hello</div>",                  # HTML
            "dict[str, Any] | None",                       # PEP 604 union
            "items.iter().map(|x| x > 0).collect()",       # Rust closure
            "SELECT * FROM t WHERE a <> b",                # SQL not-equal
            "printf '%s\\n' <<<'here string'",             # bash here-string
            "a|b|c<d>e",                                   # pipes AND angles, no adjacency
            "",                                            # empty
        ],
        ids=["shell_pipe", "comparison", "generics", "html", "pep604",
             "rust_closure", "sql_ne", "here_string", "mixed", "empty"],
    )
    def test_ordinary_strings_are_admitted(self, value):
        assert not scan_arguments({"arg": value}), f"{value!r} must not be rejected"

    def test_non_string_arguments_are_ignored(self):
        assert not scan_arguments({"n": 5, "flag": True, "nothing": None})

    def test_two_unrelated_single_marker_args_are_admitted(self):
        """The threshold is per string leaf, not per call.

        Two legitimate strings that each happen to contain one marker are two
        legitimate strings — summing them into one 'corrupt call' would
        re-introduce over-rejection through the back door.
        """
        assert not scan_arguments({"a": "<|tool_call|>", "b": "x <foo|> y"})


class TestMessages:
    def test_rejection_message_names_argument_and_markers(self):
        msg = rejection_message("wiki_compile", scan_arguments(
            {"entity_name": CORRUPT_WIKI_COMPILE}))
        assert "wiki_compile" in msg
        assert "entity_name" in msg
        assert "<tool_call|>" in msg
        assert "Re-issue" in msg, "feedback must say what to do, not just what failed"

    def test_describe_is_single_line(self):
        text = describe(scan_arguments({"root": CORRUPT_GREP_ROOT}))
        assert "\n" not in text and "root" in text


class TestPatternShape:
    def test_threshold_is_the_measured_value(self):
        """Pinned: 2 scored 0 false positives on 2,591 real calls; 1 did not."""
        assert REJECT_AT == 2

    def test_pattern_requires_pipe_adjacent_to_a_bracket(self):
        assert MARKUP_RE.search("<|x>")
        assert MARKUP_RE.search("<x|>")
        assert not MARKUP_RE.search("<x|y>"), (
            "a pipe in the middle is ordinary text (e.g. a regex alternation)"
        )

    def test_span_bound_prevents_splicing_distant_brackets(self):
        """A '<' and a '|>' far apart must not be spliced into one marker."""
        assert not MARKUP_RE.search("<" + "z" * 200 + "|>")


# ---------------------------------------------------------------------------
# The gate at the real dispatch site
# ---------------------------------------------------------------------------


class _EchoInput(BaseModel):
    text: str


class _EchoTool(BaseTool):
    """A tool whose argument is a plain `str` — i.e. one pydantic cannot protect.

    That is the whole point: the corrupt values ARE valid strings, so type
    validation admits them. This tool executes and reports what it received, so
    a test can prove the markup never reached it.
    """

    name = "echo_tool"
    description = "echoes its argument"
    input_model = _EchoInput

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def execute(self, arguments, context):  # noqa: ANN001
        self.seen.append(arguments.text)
        return ToolResult(output=f"echo={arguments.text}")


class _Telemetry:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def record(self, **kw):
        self.rows.append(kw)


class TestDispatchGate:
    def _ctx(self, tool, telemetry=None):
        reg = ToolRegistry()
        reg.register(tool)
        return LoopContext(
            provider=None, model="m", system_prompt="", max_tokens=64,
            tool_registry=reg, adapter=ModelAdapter(tier=ModelAdapter.TIER_LIGHT),
            telemetry=telemetry,
        )

    def test_markup_never_reaches_the_tool(self):
        """The claim that matters: it used to execute, and now it does not."""
        tool = _EchoTool()
        block = asyncio.run(_execute_tool_call(
            self._ctx(tool), "echo_tool", "t1", {"text": CORRUPT_WIKI_COMPILE}
        ))
        assert block.is_error
        assert tool.seen == [], "the corrupt value must never reach execute()"

    def test_rejection_feedback_is_specific(self):
        block = asyncio.run(_execute_tool_call(
            self._ctx(_EchoTool()), "echo_tool", "t1", {"text": CORRUPT_WIKI_COMPILE}
        ))
        assert "echo_tool" in block.content
        assert "text" in block.content
        assert "Re-issue" in block.content

    def test_legitimate_call_still_executes(self):
        """The other direction, at the dispatch site rather than the matcher."""
        tool = _EchoTool()
        block = asyncio.run(_execute_tool_call(
            self._ctx(tool), "echo_tool", "t1", {"text": REAL_BASH_REGEX}
        ))
        assert not block.is_error
        assert tool.seen == [REAL_BASH_REGEX]

    def test_rejection_records_telemetry(self):
        """Per the brief: the rate must be measurable, not assumed."""
        tel = _Telemetry()
        asyncio.run(_execute_tool_call(
            self._ctx(_EchoTool(), tel), "echo_tool", "t1",
            {"text": CORRUPT_WIKI_COMPILE},
        ))
        rows = [r for r in tel.rows if r.get("error_type") == "template_markup"]
        assert len(rows) == 1
        assert rows[0]["success"] is False
        assert rows[0]["tool_name"] == "echo_tool"
        assert "text" in rows[0]["error_detail"]

    def test_clean_call_records_no_markup_row(self):
        tel = _Telemetry()
        asyncio.run(_execute_tool_call(
            self._ctx(_EchoTool(), tel), "echo_tool", "t1", {"text": "ordinary"}
        ))
        assert not [r for r in tel.rows if r.get("error_type") == "template_markup"]
