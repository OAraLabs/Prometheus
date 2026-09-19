"""The plan-filename convention: the prompt must speak the wire's language.

Beacon's board joins a story to its plan file on a `<story_id>-` filename prefix. The model is
never shown the words "story_id" — the ONLY place a story's key reaches it is the dispatch
message, which the daemon stamps as `_Task ID: BC-4_`. A convention written as "<story_id>-"
would ask the model to equate two labels it has never been told are the same thing.

WHAT THESE TESTS DO AND DO NOT PROVE. They pin the prompt's wording against the wire and pin the
prompt's own EXAMPLE against Beacon's resolution rule. That is worth having — it catches drift on
either side. It is NOT evidence that the model obeys the instruction; only re-issuing a real
request against the deployed daemon and reading what lands on disk shows that.
"""

from __future__ import annotations

import re
from pathlib import Path

from prometheus.context.system_prompt import (
    _TASK_ID_LABEL,
    _format_documents_section,
    build_system_prompt,
)

_SERVER = Path(__file__).resolve().parents[1] / "src" / "prometheus" / "web" / "server.py"

# The dispatch route's own f-string, e.g.  parts.append(f"_Task ID: {story['story_id']}_")
_DISPATCH_STAMP = re.compile(r"""parts\.append\(\s*f?["']_(?P<label>[^:"']+):\s*\{""")


def _wire_label() -> str:
    """The label the daemon actually stamps on a dispatched story, read from the source."""
    source = _SERVER.read_text(encoding="utf-8")
    matches = _DISPATCH_STAMP.findall(source)
    # Fail loud rather than vacuously: if the dispatch line is reworded or moved, this test must
    # break and be re-pointed, not silently start asserting nothing.
    assert matches, (
        "could not find the dispatch label in web/server.py — the regex no longer matches the "
        "line that stamps a story's id onto its chat message. Re-point it; do not delete it."
    )
    assert len(set(matches)) == 1, f"more than one dispatch label found: {sorted(set(matches))}"
    return matches[0].strip()


def test_prompt_uses_the_label_the_wire_actually_sends() -> None:
    """The prompt's constant and the dispatch stamp are the same string."""
    assert _wire_label() == _TASK_ID_LABEL


def test_the_convention_reaches_the_assembled_prompt() -> None:
    prompt = build_system_prompt()
    assert _TASK_ID_LABEL in prompt, "the dispatch label never reaches the model"
    # The shape itself, not just the label.
    assert "BC-4-<slug>.md" in prompt
    assert "invisible" in prompt, "the prompt must say what happens if the name is wrong"


# ── Beacon's rule, transcribed. Pins the EXAMPLE, not the model. ─────────────────────────────
# Mirrors @shared/plan.ts planFileFor after the hardening in beacon-desktop#159:
# case-insensitive `<id>-` prefix, and the slug may not open with an all-digit segment.
_PLAN_EXT = re.compile(r"\.md$", re.IGNORECASE)
_SLUG_OPENS_NUMERIC = re.compile(r"^\d+(?:-|$)")


def _beacon_claims(filename: str, story_id: str) -> bool:
    if not _PLAN_EXT.search(filename):
        return False
    prefix = f"{story_id.lower()}-"
    lowered = filename.lower()
    if not lowered.startswith(prefix):
        return False
    slug = _PLAN_EXT.sub("", lowered[len(prefix) :])
    return not _SLUG_OPENS_NUMERIC.match(slug)


def test_the_prompts_own_example_resolves_under_beacons_rule() -> None:
    assert _beacon_claims("BC-4-sprint-plan.md", "BC-4")
    assert "BC-4-sprint-plan.md" in build_system_prompt(), "the worked example must be present"


def test_the_prompts_two_cautions_are_the_real_hazards() -> None:
    """Both warnings name a way the join genuinely fails, not an imagined one."""
    # Caution 2: a numeric-opening slug reads as a LONGER task id.
    assert not _beacon_claims("BC-4-001-draft.md", "BC-4")
    assert _beacon_claims("BC-4-001-draft.md", "BC-4-001")
    # Caution 1 is about preserving the id's spelling. Case itself is forgiven by the resolver —
    # the caution earns its place because a *different* id is not.
    assert _beacon_claims("bc-4-plan.md", "BC-4")
    assert not _beacon_claims("BC-5-plan.md", "BC-4")


# ── the agent ticks its own steps (2026-09-19) ───────────────────────────────────────────────
# Beacon's TAG_RE, transcribed: a trailing "[Tag]" is stripped into its own field, and setTaskDone
# rewrites ONLY the first "[ ]" cell — so a "[by:agent]" suffix survives a human tick byte-for-byte.
_TAG_RE = re.compile(r"\s*\[([^\[\]]+)\]\s*$")
# Beacon's TASK_RE (src/shared/loop.ts), transcribed alongside it.
_BEACON_TASK_RE = re.compile(r"^(\s*)([-*+])\s+\[([ xX])\]\s?(.*)$")


def test_the_prompt_tells_the_agent_to_tick_and_how() -> None:
    prompt = build_system_prompt()
    assert "TICK YOUR OWN STEPS" in prompt
    # The REGISTERED tool name, never the module name file_edit — that transposition is what made
    # the documents tree unreachable for weeks.
    assert "edit_file" in prompt and "file_edit" not in prompt
    assert "[by:agent]" in prompt, "the provenance marker must be named, not implied"


def test_the_prompt_forbids_the_failure_the_probe_actually_found() -> None:
    """A live dispatch ticked a step it could not do AND reworded it to match what it did.

    That is the failure mode — not a lazy tick, but moving the acceptance criteria and then
    meeting them. Both halves of the instruction are pinned because either alone permits it.
    """
    prompt = build_system_prompt()
    assert "never rewrite the step's wording" in prompt
    assert "leave it unticked" in prompt
    assert "never rewrite the whole file with write_file" in prompt


def test_the_prompts_tick_example_parses_under_beacons_rules() -> None:
    """The worked example must survive Beacon's parser, tag slot included."""
    line = "- [x] Wire the reader [by:agent]"
    assert line in build_system_prompt(), "the example must be present verbatim"
    m = _BEACON_TASK_RE.match(line)
    assert m is not None and m.group(3).lower() == "x", "the example is not a ticked task line"
    text = m.group(4)
    tag = _TAG_RE.search(text)
    assert tag is not None and tag.group(1) == "by:agent", "the provenance tag does not land in the tag slot"
    assert text[: tag.start()].strip() == "Wire the reader", "the step text is polluted by the tag"


def test_documents_root_override_is_honoured() -> None:
    """The prompt resolves the root the way /api/documents does when the caller has the config.

    get_documents_dir() reads ONLY PROMETHEUS_DOCUMENTS_DIR; /api/documents honours the config key
    documents.root. They agree only while no deployment sets the key — and a deployment that set it
    would send the model to a directory the Board cannot read, silently.
    """
    section = _format_documents_section(Path("/srv/custom-docs"))
    assert "/srv/custom-docs" in section
    assert "/srv/custom-docs/loops/" in section
    assert str(Path.home() / ".prometheus" / "documents") not in section


def test_the_override_is_actually_WIRED_not_merely_supported() -> None:
    """The parameter reaching the leaf is not the same fact as the callers passing it.

    The first version of this file asserted only the leaf function, and severing the argument in
    build_system_prompt left every test green — the test could not reach the change it was meant
    to protect. Both callers are exercised here.
    """
    # 1. the static builder forwards it
    assert "/srv/custom-docs/loops/" in build_system_prompt(documents_root=Path("/srv/custom-docs"))

    # 2. the runtime assembler reads config["documents"]["root"] and forwards THAT — this is the
    #    hop that closes the split with /api/documents.
    from prometheus.context.prompt_assembler import build_runtime_system_prompt

    runtime = build_runtime_system_prompt(
        cwd="/tmp", config={"documents": {"root": "/srv/custom-docs"}}
    )
    assert "/srv/custom-docs/loops/" in runtime

    # ...and with no key set, it falls back rather than emitting a broken path.
    plain = build_runtime_system_prompt(cwd="/tmp", config={})
    assert "/srv/custom-docs" not in plain
    assert "/loops/" in plain


def test_the_prompt_says_how_to_tick_when_two_steps_read_the_same() -> None:
    """The gap the uniqueness fix opens, and the reason it is closed HERE.

    edit_file now refuses an ambiguous old_str — correct, and the whole point. But its
    refusal advises "or set replace_all to change every occurrence", which is generic
    advice that is ACTIVELY WRONG for this caller: on a plan, replace_all ticks every
    matching step, including ones the agent has not done. The caller and the refusal
    ship together, so the guidance that overrides that advice has to ship with them.
    """
    prompt = build_system_prompt()
    assert "TWO STEPS READ THE SAME" in prompt, "the duplicate-step case is unaddressed"
    assert "ABOVE OR BELOW" in prompt, "no disambiguation method is given"
    assert "NEVER reach for replace_all on a plan" in prompt, (
        "the prompt does not countermand the refusal's own replace_all suggestion — "
        "an agent following the error message literally would tick a step it never did"
    )


def test_the_refusal_message_is_what_the_prompt_is_countermanding() -> None:
    """A control: if edit_file stops suggesting replace_all, the warning above is
    answering a message nobody sends, and should be re-read rather than left to rot."""
    from prometheus.tools.builtin.file_edit import FileEditTool, FileEditToolInput
    from prometheus.tools.base import ToolExecutionContext
    import asyncio, tempfile, pathlib

    tmp = pathlib.Path(tempfile.mkdtemp())
    f = tmp / "plan.md"
    f.write_text("- [ ] Same\n- [ ] Same\n", encoding="utf-8")
    res = asyncio.run(
        FileEditTool().execute(
            FileEditToolInput(path=str(f), old_str="- [ ] Same", new_str="- [x] Same"),
            ToolExecutionContext(cwd=tmp),
        )
    )
    assert res.is_error and "replace_all" in res.output, (
        "the refusal no longer suggests replace_all — the prompt's countermand may be stale"
    )
