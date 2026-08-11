"""The truncation-notice CONTRACT: what survives truncation is true and actionable.

Selector survey, 2026-08-11 (audits/20260811T222239Z-selector-survey.md,
target 2): the truncation stack's WIRING was tested — H1 goes through the
real ``_execute_tool_call`` — and its NOTICES were not, and all three were
dead in live use:

* the default strategy's head-only cut beheaded every tool's own
  tail-positioned notice (vault_read's continue offset, wiki_query's budget
  marker, web_fetch's cut notice) — the model lost the content AND the text
  saying how to get it;
* the default trailer reported the size of the payload it was handed as if
  it were the artifact ("truncated at 12041 tokens" for an 18,000-token
  page whose upstream cap had already cut it);
* the turn-budget notice prescribed ``lcm_expand``, which expands LCM
  summary nodes and cannot recover a tool result truncated before injection
  — it was never stored.

A notice is part of the tool result the model reasons over. These tests hold
every notice to two properties: TRUE (numbers describe what this layer
actually received and kept) and ACTIONABLE (any prescribed recovery must be
one that works). Both directions per §2c: truncation preserves what matters
AND under-budget results pass through untouched.
"""

from __future__ import annotations

import re

from unittest.mock import MagicMock

from prometheus.context.truncation import ToolResultTruncator
from prometheus.engine.agent_loop import LoopContext, _apply_cross_result_budget
from prometheus.engine.messages import ToolResultBlock, ToolUseBlock


# ---------------------------------------------------------------------------
# Default strategy — tail survival and trailer truth
# ---------------------------------------------------------------------------

_TAIL_NOTICE = "[my-tool notice: 60000 chars remain — continue with offset=16000]"


def _big_output(total: int = 100_000) -> str:
    body = "".join(f"line-{i:05d}\n" for i in range(total // 11))
    return body + _TAIL_NOTICE


class TestDefaultStrategyContract:
    def test_a_tools_own_tail_notice_survives(self):
        """The beheading, pinned. Tools put their state at the END of a
        result; a head-only default cut every one of them off. vault_read
        worked around it for itself (#154, head-positioned notices) — this
        makes the layer safe for the tools that haven't."""
        out = ToolResultTruncator(4000).truncate("some_tool", _big_output())
        assert _TAIL_NOTICE in out, "the tail window no longer preserves a tool's own notice"

    def test_the_head_is_still_the_head(self):
        """Tail preservation must not cost the head — H1's original semantics
        (lead content survives) stay intact."""
        src = _big_output()
        out = ToolResultTruncator(4000).truncate("some_tool", src)
        assert out.startswith(src[:1000])

    def test_trailer_numbers_are_true(self):
        """The trailer states what THIS layer received and kept, as exact
        figures — never a number that reads as the artifact's size."""
        src = _big_output()
        out = ToolResultTruncator(4000).truncate("some_tool", src)
        m = re.search(
            r"kept the first (\d+) and last (\d+) chars of the (\d+) received",
            out,
        )
        assert m, f"trailer missing or reworded: {out[-300:]!r}"
        head, tail, received = (int(g) for g in m.groups())
        assert received == len(src), "'received' must be the true input size"
        assert out.startswith(src[:head]), "'first N' must be the actual head kept"
        assert src.endswith(out.split("\n[truncated by the per-result budget")[0][-tail:]), (
            "'last N' must be the actual tail kept"
        )

    def test_the_gap_is_stated_not_silent(self):
        src = _big_output()
        out = ToolResultTruncator(4000).truncate("some_tool", src)
        m = re.search(r"\[\.\.\. (\d+) chars omitted \.\.\.\]", out)
        assert m, "the omitted span must be marked in place"
        head_m = re.search(r"kept the first (\d+) and last (\d+)", out)
        head, tail = int(head_m.group(1)), int(head_m.group(2))
        assert int(m.group(1)) == len(src) - head - tail

    def test_the_budget_is_still_respected(self):
        """Tail-keep must not blow the budget it exists to enforce: kept
        content stays within the char budget; only the markers ride on top."""
        out = ToolResultTruncator(4000).truncate("some_tool", _big_output())
        assert len(out) <= 4000 * 4 + 600, "kept content exceeds the budget"


# ---------------------------------------------------------------------------
# Actionability — no notice prescribes a recovery that cannot work
# ---------------------------------------------------------------------------

def _make_context(budget: int) -> LoopContext:
    ctx = MagicMock(spec=LoopContext)
    ctx.tool_results_turn_budget = budget
    ctx.tool_registry = None
    return ctx


class TestNoticesAreActionable:
    def test_no_strategy_notice_names_lcm_expand(self):
        """lcm_expand expands LCM summary nodes; a truncated tool result was
        never stored, so naming it is advice that cannot be followed."""
        t = ToolResultTruncator(10)
        for tool in ("bash", "read_file", "grep", "anything_else"):
            out = t.truncate(tool, "x\n" * 2_000)
            assert "lcm_expand" not in out, f"{tool} notice prescribes lcm_expand"

    def test_turn_budget_notice_is_truthful_and_actionable(self):
        """The cross-result budget's notice must say the content is gone and
        point at the recovery that works — not at lcm_expand, which cannot
        recover it, nor a bare 're-read', which returns the same head."""
        ctx = _make_context(budget=100)
        tcs = [ToolUseBlock(name="big", id="id_big", input={})]
        results = [ToolResultBlock(tool_use_id="id_big", content="y" * 5_000)]
        out = _apply_cross_result_budget(ctx, tcs, results)
        trimmed = out[0].content
        assert "[truncated" in trimmed
        assert "lcm_expand" not in trimmed
        assert "not retained" in trimmed
        assert "re-run the tool" in trimmed


# ---------------------------------------------------------------------------
# Negative direction — under budget, every strategy is the identity
# ---------------------------------------------------------------------------

class TestUnderBudgetIdentity:
    def test_every_strategy_passes_small_output_untouched(self):
        t = ToolResultTruncator(4000)
        small = "short result\n" + _TAIL_NOTICE
        for tool in ("bash", "read_file", "grep", "vault_read", "unknown"):
            assert t.truncate(tool, small) == small


# ---------------------------------------------------------------------------
# The strategy table is pinned to the real registry
# ---------------------------------------------------------------------------

def test_strategy_table_names_exist_in_the_real_registry():
    """The table is name-keyed; a tool rename would silently demote it to the
    default strategy with no error anywhere. Pin every key to the registry
    the daemon actually builds — the same real-entry-point standard as the
    vault registration tests."""
    from prometheus.__main__ import create_tool_registry

    names = {t.name for t in create_tool_registry({}).list_tools()}
    strategy_names = set(ToolResultTruncator._STRATEGIES)
    assert strategy_names, "strategy table is empty — dispatch refactor broke it"
    missing = strategy_names - names
    assert not missing, (
        f"ToolResultTruncator has strategies for tools that do not exist in "
        f"the registry: {sorted(missing)} — a rename silently demotes the "
        f"tool to head-only default truncation"
    )
    # The oracle can tell a fake from a real name — otherwise the assertion
    # above could pass vacuously against a broken registry read.
    assert "definitely_not_a_registered_tool" not in names
    assert "bash" in names


def test_the_strategies_dispatch_to_the_intended_methods():
    """Registered-but-wrong is its own failure mode: the bash key must map to
    the tail-keeping strategy, not merely to any callable."""
    t = ToolResultTruncator(10)
    lines = "\n".join(f"line {i}" for i in range(200))
    bash_out = t.truncate("bash", lines)
    assert "line 199" in bash_out and "line 0" not in bash_out  # tail strategy
    file_out = t.truncate("read_file", lines)
    assert "line 0" in file_out and "line 199" in file_out      # head+tail
