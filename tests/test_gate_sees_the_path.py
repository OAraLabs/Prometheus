"""The SecurityGate must actually receive the path a tool is about to touch.

THE DEFECT
----------
``agent_loop`` built the gate's subject as ``tool_input.get("file_path")``.
**No registered tool declares ``file_path``** — they declare ``path``,
``output_path``, ``image_path``, ``destination``, ``save_to``. So the gate got
``None`` on every call and both ``_check_denied_path`` and
``_within_workspace`` (each guarded by ``if file_path``) were skipped.
``denied_paths`` and ``workspace_root`` were inert for every file tool from
the initial commit, 2026-04-20, until 2026-08-13 — four months, live.

WHY EVERY EXISTING TEST PASSED
-------------------------------
They all constructed a gate and called ``gate.evaluate(..., file_path=p)``
**by hand**, supplying the argument production never supplies. A 7-of-7
verification passed against this an hour before the defect was found: true
about the gate's logic, and about a world that does not exist.

So every test in this file goes through ``_execute_tool_call`` — the real
dispatch path, with the tool's real parameter names — and
:func:`test_no_test_hands_the_gate_a_file_path_by_hand` makes the old pattern
structurally impossible to reintroduce.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from prometheus.engine.agent_loop import LoopContext, _execute_tool_call
from prometheus.permissions.checker import SecurityGate
from prometheus.permissions.tool_paths import (
    PATH_PARAM_EXEMPT,
    TOOL_PATH_PARAM,
    gate_path_for,
)

HOME = Path.home()


def _ctx(tmp_path: Path, prompted: list, *, roots=None, cwd=None):
    """A loop context wired to a REAL gate, with a prompt that says YES.

    The prompt answering yes is the point: if a DENY ever consults it, the
    write lands and the test fails loudly rather than passing on a refusal
    that came from somewhere else.
    """
    from prometheus.__main__ import create_tool_registry

    # The denied path is a TMP directory, never the operator's real ~/.gnupg.
    # A mutation that reinstates the defect genuinely performs the write, and
    # the first version of this file created a file inside the real ~/.gnupg
    # while proving the mutation was caught. A test that damages the thing it
    # protects is not a test worth running.
    denied_dir = tmp_path / "denied"
    denied_dir.mkdir(exist_ok=True)
    gate = SecurityGate(
        denied_paths=["/etc", str(denied_dir)],
        workspace_root=roots if roots is not None else [str(tmp_path / "ws")],
    )

    async def prompt(tool_name, reason):
        prompted.append(reason)
        return True

    return LoopContext(
        provider=None, model="t", system_prompt="", max_tokens=512,
        tool_registry=create_tool_registry({}, gate),
        permission_checker=gate, permission_prompt=prompt,
        cwd=str(cwd) if cwd else None,
    )


def _write(ctx, path: str, tool: str = "write_file"):
    args = {"write_file": {"path": path, "content": "x\n"},
            "read_file": {"path": path},
            "edit_file": {"path": path, "old_str": "x", "new_str": "y"}}[tool]
    return asyncio.run(_execute_tool_call(ctx, tool, "t1", args))


# ── BREACH: a denied path is refused, through the real dispatch path ────────

def test_a_denied_path_is_refused(tmp_path):
    prompted: list = []
    ctx = _ctx(tmp_path, prompted)
    target = str(tmp_path / "denied" / "should-not-exist")
    result = _write(ctx, target)
    assert result.is_error, "the write was NOT refused — the gate never saw the path"
    assert not Path(target).exists(), f"{target} was created despite denied_paths"
    assert not prompted, (
        f"a DENY consulted the approval prompt ({prompted}) — a denial that "
        f"degrades to a prompt is not a denial"
    )


def test_a_denied_path_is_refused_for_reads_too(tmp_path):
    """Reads were never gated because the gate never got a path. denied_paths
    runs BEFORE the read-only allowance, so wiring the path is all it took."""
    prompted: list = []
    ctx = _ctx(tmp_path, prompted)
    result = _write(ctx, "/etc/hostname", tool="read_file")
    assert result.is_error, "reading a denied path was allowed"
    assert not prompted


def test_a_write_outside_the_workspace_prompts(tmp_path):
    prompted: list = []
    ctx = _ctx(tmp_path, prompted)
    outside = tmp_path / "elsewhere" / "f.txt"
    outside.parent.mkdir(parents=True)
    _write(ctx, str(outside))
    assert prompted, "a write outside every workspace root did not prompt"


# ── ADMISSION: legitimate work still proceeds, unprompted ──────────────────

def test_a_write_inside_the_workspace_proceeds_without_a_prompt(tmp_path):
    """The direction that decides whether this control survives contact. A
    boundary that prompts on ordinary work is one the operator turns off."""
    prompted: list = []
    ctx = _ctx(tmp_path, prompted)
    (tmp_path / "ws").mkdir()
    target = tmp_path / "ws" / "ok.txt"
    result = _write(ctx, str(target))
    assert not result.is_error, f"legitimate write refused: {result.content}"
    assert target.read_text() == "x\n"
    assert not prompted, f"an in-workspace write prompted: {prompted}"


@pytest.mark.parametrize("root_idx", [0, 1, 2])
def test_every_configured_root_admits_writes(tmp_path, root_idx):
    """Multi-root: each root must actually work. One root silently winning
    would look identical from the breach side."""
    roots = [str(tmp_path / f"r{i}") for i in range(3)]
    for r in roots:
        Path(r).mkdir()
    prompted: list = []
    ctx = _ctx(tmp_path, prompted, roots=roots)
    target = Path(roots[root_idx]) / "f.txt"
    result = _write(ctx, str(target))
    assert not result.is_error and not prompted, f"root {root_idx} did not admit"


# ── UNKNOWN never falls through to allowed ─────────────────────────────────

def test_a_relative_path_prompts_rather_than_resolving_against_cwd(tmp_path):
    """A relative target is UNKNOWN, not "allowed" and not resolved against
    the process's cwd — that is the control-whose-target-is-chosen-by-cwd
    defect, on the input side where it would decide whether a write happens.
    """
    prompted: list = []
    # The prompt answers YES, so the write PROCEEDS — point the loop's cwd at
    # tmp or the relative target lands in the repo working tree. (monkeypatch
    # .chdir does NOT help: LoopContext captures cwd at construction.)
    ctx = _ctx(tmp_path, prompted, cwd=tmp_path)
    _write(ctx, "subdir/notes.md")
    assert prompted, "a relative path did not prompt — it fell through"
    # Assert the REASON, not a substring: written as `"relative" in reason`
    # first, and mutation M3 (resolve against cwd anyway) still passed —
    # because the word appeared inside the RESOLVED PATH in the
    # outside-workspace message. A guard whose evidence can appear in its own
    # subject is not evidence (§3b).
    assert "the gate resolves no path against the working directory" in prompted[0], (
        f"prompted for the wrong reason: {prompted[0]!r}"
    )


def test_an_unmapped_tool_with_a_path_argument_is_loud():
    """An unrecognised file tool must be UNKNOWN, not exempt. Silence is what
    cost four months."""
    path, unknown = gate_path_for("some_new_tool", {"output_path": "/tmp/x"})
    assert path is None
    assert unknown and "unrecognised" in unknown


def test_a_tool_with_no_path_argument_is_not_made_unknown():
    """The loud fallback must not fire for ordinary tools, or every bash call
    starts prompting."""
    assert gate_path_for("bash", {"command": "ls"}) == (None, None)
    assert gate_path_for("web_search", {"query": "x"}) == (None, None)


# ── The mapping matches the REGISTRY, not a name pattern ───────────────────

def test_every_registered_tool_with_a_path_is_mapped_or_exempt():
    """Build-time guard. Grepping parameter NAMES for '*path*' finds nine
    tools and misses download_file.destination and youtube_transcript.save_to
    — matching a name instead of reading the schema is the original defect one
    level up. This reads the schema."""
    from prometheus.__main__ import create_tool_registry

    signal = re.compile(
        r"\b(path|file|directory|dir|save|destination|written to|write to)\b",
        re.I)
    unaccounted = []
    for schema in create_tool_registry({}, None).list_schemas():
        name = schema["name"]
        if name in PATH_PARAM_EXEMPT:
            continue
        props = (schema.get("input_schema") or schema.get("parameters") or {}
                 ).get("properties", {})
        for pname, pspec in props.items():
            if pname in ("cwd", "root", "watch_dir"):
                continue  # directory params — deliberately out of scope
            blob = f"{pname} {pspec.get('description', '')}"
            if signal.search(blob) and TOOL_PATH_PARAM.get(name) != pname:
                if "path" in pname.lower() or "destination" in pname.lower() \
                        or "save" in pname.lower():
                    unaccounted.append(f"{name}.{pname}")
    assert not unaccounted, (
        "these tool parameters look like filesystem paths and are neither "
        "mapped in TOOL_PATH_PARAM nor exempt in PATH_PARAM_EXEMPT — the gate "
        f"would not see them:\n  " + "\n  ".join(sorted(set(unaccounted)))
    )


def test_no_mapped_parameter_is_missing_from_its_tool():
    """The mirror direction: a mapping naming a parameter the tool does not
    declare is exactly the original bug, and would read as configured."""
    from prometheus.__main__ import create_tool_registry

    by_name = {s["name"]: s for s in create_tool_registry({}, None).list_schemas()}
    for tool, param in TOOL_PATH_PARAM.items():
        schema = by_name.get(tool)
        assert schema, f"TOOL_PATH_PARAM names {tool!r}, which is not registered"
        props = (schema.get("input_schema") or schema.get("parameters") or {}
                 ).get("properties", {})
        assert param in props, (
            f"TOOL_PATH_PARAM says {tool}.{param}, but the tool declares "
            f"{sorted(props)} — this is the original defect verbatim"
        )


# ── The exemptions carry their own guards ──────────────────────────────────

def test_vault_read_confines_itself(tmp_path):
    """An exemption without its own guard is a hole. vault_read is exempt
    because its `path` is vault-relative; it must still refuse to escape."""
    from prometheus.__main__ import create_tool_registry
    from prometheus.tools.base import ToolExecutionContext

    tool = create_tool_registry({}, None).get("vault_read")
    result = asyncio.run(tool.execute(
        tool.input_model(path="../../../../etc/passwd"),
        ToolExecutionContext(cwd=str(tmp_path)),
    ))
    assert result.is_error, "vault_read escaped its root via ../"


def test_todo_write_confines_itself(tmp_path):
    from prometheus.__main__ import create_tool_registry
    from prometheus.tools.base import ToolExecutionContext

    tool = create_tool_registry({}, None).get("todo_write")
    work = tmp_path / "work"
    work.mkdir()
    escape = tmp_path / "escaped-todo.md"      # hermetic: outside `work`,
    # inside tmp_path. An earlier version aimed at a FIXED /tmp path, so a
    # mutated run left the file behind and the RESTORED run then failed — a
    # test with global state cannot be used in a mutation matrix.
    result = asyncio.run(tool.execute(
        tool.input_model(item="x", path=f"../{escape.name}"),
        ToolExecutionContext(cwd=str(work)),
    ))
    assert result.is_error, "todo_write did not refuse a ../ escape"
    assert not escape.exists(), "todo_write escaped its working directory"


# ── The pattern that hid this must be structurally impossible ──────────────

#: Tests allowed to call ``gate.evaluate(file_path=...)`` directly, with the
#: reason. A shrinking debt list, enforced in BOTH directions like
#: KNOWN_UNREAD: an unlisted offender fails (new drift), and a listed file
#: that no longer does it fails too (stale entry).
#:
#: These were never wrong to EXIST — unit-testing the gate's logic is fine.
#: What was wrong is that they were the ONLY coverage, so the gate's logic was
#: proven while the caller that feeds it was not. This file is the other half.
_DIRECT_EVALUATE_TESTS: dict[str, str] = {
    "test_permissions.py":
        "unit tests of the gate's decision table; the wiring they assume is "
        "now covered here through _execute_tool_call",
    "test_approval_queue.py":
        "asserts the queue's handling of an APPROVE decision, not the path "
        "extraction that produces one",
}


def test_direct_gate_evaluate_calls_are_registered():
    """THE guard. Every test of this control used to call
    ``gate.evaluate(..., file_path=p)`` directly — supplying the argument the
    caller never supplied. That is why four months of green meant nothing.

    ``file_path=None`` is not the pattern (it asserts the absence of a path,
    which is honest), so only a real value counts.
    """
    tests_dir = Path(__file__).resolve().parent
    pattern = re.compile(r"\.evaluate\s*\([^)]*\bfile_path\s*=\s*(?!None)", re.S)
    found = set()
    for py in tests_dir.rglob("test_*.py"):
        if py.resolve() == Path(__file__).resolve():
            continue
        if pattern.search(py.read_text(encoding="utf-8", errors="replace")):
            found.add(py.name)
    new = sorted(found - set(_DIRECT_EVALUATE_TESTS))
    assert not new, (
        "these tests call gate.evaluate(file_path=<value>) by hand — the exact "
        "pattern that hid a four-month-live hole, because it simulates a "
        "caller that did not exist. Drive the gate through _execute_tool_call, "
        "or register the file with a reason:\n  " + "\n  ".join(new)
    )
    stale = sorted(set(_DIRECT_EVALUATE_TESTS) - found)
    assert not stale, (
        "registered file no longer calls evaluate(file_path=...) — remove it "
        "from _DIRECT_EVALUATE_TESTS:\n  " + "\n  ".join(stale)
    )


def test_this_file_reaches_the_gate_through_the_real_dispatch_path():
    """The registry above is only safe because this file exists. If these
    tests ever stop driving _execute_tool_call, the debt list becomes an
    allowlist over nothing."""
    text = Path(__file__).read_text(encoding="utf-8")
    assert "_execute_tool_call(" in text
    assert text.count("_execute_tool_call(") >= 2
