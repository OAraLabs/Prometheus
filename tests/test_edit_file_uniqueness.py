"""edit_file must enforce the uniqueness it advertises.

The schema served to the model says, of ``replace_all``: "By default ONLY THE FIRST occurrence of
old_str is replaced, and the edit fails if old_str is not unique." The second half was fiction —
the code called ``str.replace(old, new, 1)`` and returned ``Updated {path}``, so an ambiguous edit
silently changed the wrong occurrence and reported success.

That description was added on purpose, by a piece of work whose own test calls
``edit_file.replace_all=False`` "a correctness question, not merely an efficiency one" — so the
ADVERTISEMENT was made accurate-sounding while the behaviour stayed wrong. This file pins both
halves together: the promise must be in the schema the model reads, AND the code must keep it.

The caller that made it urgent: an agent ticking its own plan checklist. Plan steps share phrasing
constantly, so first-match-wins ticks the wrong box with nothing to notice.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from prometheus.tools.builtin.file_edit import FileEditTool, FileEditToolInput
from prometheus.tools.base import ToolExecutionContext


def _run(tmp_path: Path, text: str, **kw):
    f = tmp_path / "plan.md"
    f.write_text(text, encoding="utf-8")
    tool = FileEditTool()
    args = FileEditToolInput(path=str(f), **kw)
    result = asyncio.run(tool.execute(args, ToolExecutionContext(cwd=tmp_path)))
    return result, f.read_text(encoding="utf-8")


AMBIGUOUS = "- [ ] Add smoke tests\n- [ ] Ship it\n- [ ] Add smoke tests\n"


def test_the_promise_is_actually_in_the_schema_the_model_reads() -> None:
    """The control. If this description is ever reworded away, the pin below is meaningless."""
    schema = FileEditTool().to_api_schema()
    desc = schema["input_schema"]["properties"]["replace_all"]["description"]
    assert "not unique" in desc, f"the uniqueness promise left the schema: {desc!r}"


def test_an_ambiguous_edit_fails_and_changes_NOTHING(tmp_path: Path) -> None:
    """The bug. Two identical steps; ticking one must not silently tick the other."""
    result, after = _run(tmp_path, AMBIGUOUS, old_str="- [ ] Add smoke tests", new_str="- [x] Add smoke tests")
    assert result.is_error, "an ambiguous edit reported success"
    assert "2 MATCHES" in result.output and "ambiguous" in result.output
    # The file is the whole point: a refusal that already wrote is not a refusal.
    assert after == AMBIGUOUS, "the file was modified despite the error"


def test_replace_all_is_the_documented_way_through(tmp_path: Path) -> None:
    result, after = _run(
        tmp_path, AMBIGUOUS, old_str="- [ ] Add smoke tests", new_str="- [x] Add smoke tests", replace_all=True
    )
    assert not result.is_error
    assert after.count("- [x] Add smoke tests") == 2


def test_a_unique_edit_still_works(tmp_path: Path) -> None:
    """The positive control — the guard must not have made the tool useless."""
    result, after = _run(tmp_path, AMBIGUOUS, old_str="- [ ] Ship it", new_str="- [x] Ship it")
    assert not result.is_error
    assert "- [x] Ship it" in after
    assert after.count("- [ ] Add smoke tests") == 2, "an unrelated duplicate was touched"


def test_no_match_still_fails_loudly(tmp_path: Path) -> None:
    result, after = _run(tmp_path, AMBIGUOUS, old_str="- [ ] Not present", new_str="x")
    assert result.is_error and "not found" in result.output
    assert after == AMBIGUOUS
