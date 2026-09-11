"""Writes outside the workspace prompt — ALL of them, not four tool names.

THE DEFECT
----------
`permissions/checker.py` gated the workspace boundary on::

    _APPROVE_TOOLS = frozenset({"write_file", "edit_file"})
    ...
    if tool_name in _APPROVE_TOOLS:
        if file_path and not self._within_workspace(file_path, ...):
            return PermissionDecision.approve(...)

The gate's own docstring promised that writes outside the workspace prompt for
approval. Four registered tools write to arbitrary caller-supplied paths and
were not on that list:

    notebook_edit       path          rewrites any .ipynb
    download_file       destination   writes a remote body to any local file
    tts                 output_path   writes an audio file anywhere
    youtube_transcript  save_to       writes a transcript anywhere

The path was reaching the gate correctly the whole time — `tool_paths.py`
already resolves each tool's own destination parameter, and did so before this
change. Only the DECISION was keyed on a name. The gate knew the target and
declined to rule on it.

THE FIFTH ENUMERATION
---------------------
`path_schema.py` records three earlier controls in this subsystem that decided
something by matching a parameter NAME and got it wrong; `tool_paths.py`
records the fourth. This was the fifth, one level up: not "which parameter is a
path" but "which tool writes".

So the boundary is now a property of the WRITE, declared per PARAMETER. Per
parameter and not per tool, because neither the tool nor the call is a fine
enough unit — two paths are READ by tools that are not read-only:

    video_generate.image_path   the source image it animates
    task_create.watch_dir       a directory a file_watch task watches

Keying on the tool, or on `not is_read_only`, prompts for both. That is the
over-refusing direction, and a boundary that cries wolf is one people route
around — so it is worth the extra precision.

THE MEASUREMENT THAT SETTLED IT
-------------------------------
The naive repair — "the boundary applies whenever the call carries a path and
is not read-only" — is a genuine property rather than a list of names, and it
is still WRONG. Measured: **23 test failures** across cron, denied-paths and
the grant-floor invariant, from two causes that are the same cause:

  * `gateway/cron_scheduler.py` passes `file_path=cwd` on a **bash** call. That
    path is the job's WORKING DIRECTORY — context for `denied_paths`, not a
    file the command writes. bash is not read-only, so the naive predicate
    treated every cron job's cwd as a write target.
  * `tests/test_denied_paths_absence.py` passes `read_file` with a path and
    does not pass `is_read_only=True`. The target is a READ, and the default
    for that argument is False, so the naive predicate called it a write.

A path on the call says only that a path is INVOLVED. It does not say the tool
writes there, and the difference is not recoverable at runtime without
guessing. So the boundary rules on a DECLARED write, and the declaration is
enforced at build time by
`test_every_production_call_site_declares_whether_its_path_is_written`.

A separate over-refusal probe is kept below for the same reason: a predicate
that returns True for everything passes every "does it prompt" test ever
written.

WHY THESE TESTS GO THROUGH _execute_tool_call
---------------------------------------------
Same reason as `test_gate_sees_the_path.py`, whose docstring records it: every
test of the ORIGINAL defect passed because it handed the gate a resolved target
by hand, supplying an argument production never supplied. A gate test that does
not come through dispatch is a test about a world that does not exist. These
build a real registry and a real gate and drive `_execute_tool_call`.

(Phrased without the literal call shape on purpose — the meta-guard in
`test_gate_sees_the_path.py` scans this directory as TEXT, so writing that
pattern even in prose registers as a violation. It fired on this docstring
while the file was being written, which is the guard working.)
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import LoopContext, _execute_tool_call
from prometheus.permissions import checker as checker_mod
from prometheus.permissions.checker import SecurityGate
from prometheus.permissions.path_schema import (
    PATH_ACCESS_KEY,
    declared_path_access,
    declared_path_params,
)
from prometheus.permissions.tool_paths import (
    PATH_PARAM_EXEMPT,
    TOOL_PATH_PARAM,
    gate_path_is_write,
)


def _ctx(tmp_path: Path, prompted: list):
    """A real registry and a real gate, with a prompt that answers YES.

    Answering yes matters: if the boundary fails to fire, nothing is recorded
    in `prompted` and the assertion fails — rather than the call being blocked
    for some unrelated reason and the test passing by accident.
    """
    from prometheus.__main__ import create_tool_registry

    gate = SecurityGate(workspace_root=[str(tmp_path / "ws")])

    async def prompt(tool_name, reason):
        prompted.append(reason)
        return True

    (tmp_path / "ws").mkdir(exist_ok=True)
    return LoopContext(
        provider=None, model="t", system_prompt="", max_tokens=512,
        tool_registry=create_tool_registry({}, gate),
        permission_checker=gate, permission_prompt=prompt,
        cwd=str(tmp_path / "ws"),
    )


# The four tools that crossed the boundary unprompted, with an argument set
# that targets an absolute path OUTSIDE the workspace. Each uses that tool's
# OWN destination parameter name — the whole point.
def _outside_write_calls(tmp_path: Path) -> dict[str, dict]:
    outside = tmp_path / "outside"
    outside.mkdir(exist_ok=True)
    return {
        "notebook_edit": {
            "path": str(outside / "nb.ipynb"),
            "cell_index": 0, "new_source": "print(1)",
        },
        "download_file": {
            "url": "https://example.invalid/x.bin",
            "destination": str(outside / "x.bin"),
        },
        "tts": {"text": "hello", "output_path": str(outside / "a.wav")},
        "youtube_transcript": {
            "url": "https://example.invalid/watch?v=x",
            "save_to": str(outside / "t.txt"),
        },
    }


@pytest.mark.parametrize("tool_name", sorted(_outside_write_calls(Path("/tmp"))))
def test_a_write_outside_the_workspace_prompts(tool_name, tmp_path):
    """The boundary must fire for every tool that writes, not four names."""
    prompted: list = []
    ctx = _ctx(tmp_path, prompted)
    args = _outside_write_calls(tmp_path)[tool_name]

    result = asyncio.run(_execute_tool_call(ctx, tool_name, "t1", args))

    # A fixture whose argument NAMES are wrong fails pydantic validation
    # before dispatch ever reaches the gate, and then "nothing prompted" looks
    # exactly like a missed boundary. It did, while this file was being
    # written: `cell_id` instead of `cell_index`. Say which failure it is.
    content = str(getattr(result, "content", ""))
    assert "Invalid input for" not in content, (
        f"{tool_name} never reached the security gate — its arguments failed "
        f"validation first, so this test says nothing about the boundary:\n"
        f"{content[:400]}"
    )

    assert prompted, (
        f"{tool_name} wrote outside the workspace WITHOUT prompting. Its "
        f"destination parameter is "
        f"{TOOL_PATH_PARAM.get(tool_name)!r} and the target was outside "
        f"{tmp_path / 'ws'}."
    )
    assert any("outside workspace" in reason for reason in prompted), (
        f"{tool_name} prompted, but not for the workspace boundary: {prompted}"
    )


def test_a_write_inside_the_workspace_does_not_prompt(tmp_path):
    """The boundary must not fire inside the workspace.

    A fix that prompts for everything would pass every test above and be
    useless — the control's credibility is what stops people routing around it.
    """
    prompted: list = []
    ctx = _ctx(tmp_path, prompted)
    target = str(tmp_path / "ws" / "nb.ipynb")

    asyncio.run(_execute_tool_call(
        ctx, "notebook_edit", "t1",
        {"path": target, "cell_index": 0, "new_source": "print(1)"},
    ))

    assert not any("outside workspace" in r for r in prompted), (
        f"a write INSIDE the workspace prompted anyway: {prompted}"
    )


# ── The precision that per-parameter declaration buys ───────────────────────

@pytest.mark.parametrize(
    "tool_name,param",
    [("video_generate", "image_path"), ("task_create", "watch_dir"),
     ("grep", "root"), ("glob", "root"), ("read_file", "path")],
)
def test_paths_that_are_only_read_do_not_trip_the_write_boundary(tool_name, param):
    """These are READ by tools that are mostly not read-only.

    `video_generate` and `task_create` both return False from `is_read_only`,
    so a boundary keyed on the tool — or on `not is_read_only` — would prompt
    for a source image and a watched directory. Keying on the PARAMETER does
    not, which is why the declaration lives next to the field.
    """
    schema = _schema_for(tool_name)
    assert declared_path_access(schema).get(param) == "read", (
        f"{tool_name}.{param} is no longer declared read-only in its schema"
    )
    assert gate_path_is_write(tool_name, schema=schema) is False, (
        f"{tool_name}.{param} is being treated as a write target"
    )


def _schema_for(tool_name: str):
    import importlib

    modules = {
        "notebook_edit": "notebook_edit", "download_file": "download_file",
        "tts": "tts", "youtube_transcript": "youtube_transcript",
        "grep": "grep", "glob": "glob", "video_generate": "video_generate",
        "task_create": "task_create", "read_file": "file_read",
        "write_file": "file_write", "edit_file": "file_edit",
        "image_generate": "image_generate",
    }
    mod = importlib.import_module(f"prometheus.tools.builtin.{modules[tool_name]}")
    models = [
        obj for _, obj in inspect.getmembers(mod, inspect.isclass)
        if issubclass(obj, BaseModel) and obj is not BaseModel
        and obj.__module__ == mod.__name__
    ]
    assert models, f"no argument model found for {tool_name}"
    return models[0].model_json_schema()


# ── Structural guards: the name list must stay dead ─────────────────────────

def test_the_tool_name_list_is_not_consulted():
    """`_APPROVE_TOOLS` is retired. Nothing may read it again.

    It is kept, empty, only as the anchor for the note explaining why. If a
    future change repopulates it, or keys a decision on it, that is the fifth
    enumeration coming back.
    """
    assert checker_mod._APPROVE_TOOLS == frozenset(), (
        "_APPROVE_TOOLS has been repopulated — the workspace boundary is a "
        "property of the write, not a list of tool names"
    )
    source = Path(checker_mod.__file__).read_text(encoding="utf-8")
    uses = [
        line.strip() for line in source.splitlines()
        if "_APPROVE_TOOLS" in line
        and not line.lstrip().startswith("#")
        and "_APPROVE_TOOLS: frozenset" not in line
    ]
    assert not uses, f"_APPROVE_TOOLS is being read again: {uses}"


def test_every_declared_path_param_declares_its_access():
    """No path field may leave access unstated.

    Unstated resolves to WRITE, which prompts rather than passes — the correct
    direction — but a tool whose author never considered the question is
    exactly the state that produced this defect. Make them say it.
    """
    unstated: list[str] = []
    for tool_name in sorted(TOOL_PATH_PARAM):
        schema = _schema_for(tool_name) if tool_name in _KNOWN else None
        if schema is None:
            continue
        props = schema.get("properties", {})
        for param in declared_path_params(schema):
            spec = props.get(param) or {}
            if spec.get(PATH_ACCESS_KEY) not in ("read", "write"):
                unstated.append(f"{tool_name}.{param}")
    assert not unstated, (
        "these path parameters do not declare whether the tool reads or "
        f"writes them: {unstated}. Use PATH_FIELD_READ / PATH_FIELD_WRITE "
        f"(or the DIR_ equivalents) so the boundary is answered, not guessed."
    )


_KNOWN = {
    "notebook_edit", "download_file", "tts", "youtube_transcript", "grep",
    "glob", "video_generate", "task_create", "read_file", "write_file",
    "edit_file", "image_generate",
}


def test_the_mapped_tools_are_all_accounted_for():
    """Every tool in TOOL_PATH_PARAM is classified by this file.

    Without this, adding a tool to TOOL_PATH_PARAM would silently escape the
    access audit above — the test would simply skip it, which is the quiet
    failure mode this whole area keeps producing.
    """
    unaccounted = sorted(set(TOOL_PATH_PARAM) - _KNOWN - set(PATH_PARAM_EXEMPT))
    assert not unaccounted, (
        f"these mapped tools are not covered by this file's access audit: "
        f"{unaccounted}. Add them to _KNOWN (and give their path params an "
        f"explicit access declaration)."
    )


# ── Mutation check ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("tool_name", sorted(_outside_write_calls(Path("/tmp"))))
def test_the_old_name_list_would_have_let_these_through(tool_name):
    """Replay the removed predicate and prove each tool really did escape it.

    If this passes trivially — because a tool stopped writing, or stopped
    being registered — the corresponding boundary test above proves nothing.
    """
    old_approve_tools = frozenset({"write_file", "edit_file"})
    assert tool_name not in old_approve_tools, (
        f"{tool_name} IS in the old name list, so it was never escaping the "
        f"boundary — this test's premise is stale"
    )
    schema = _schema_for(tool_name)
    assert gate_path_is_write(tool_name, schema=schema) is True, (
        f"{tool_name} is no longer classified as writing to its mapped path; "
        f"the boundary test above is no longer testing a write"
    )


# ── The build-time half of the fail-closed property ─────────────────────────
#
# At runtime an undeclared `path_is_write` cannot be resolved without guessing,
# and guessing is what this subsystem keeps getting wrong. So it is resolved
# here instead: a production call site that hands the gate a path and does not
# say whether it is written fails the build.
#
# Each registered site passes a path that is CONTEXT, not a write target —
# the same distinction `video_generate.image_path` and `task_create.watch_dir`
# make one level down. An exemption without a stated reason is a hole, so each
# carries one.
_CONTEXT_PATH_CALL_SITES: dict[str, str] = {
    "gateway/cron_scheduler.py":
        "`file_path=cwd` is the job's WORKING DIRECTORY, not a file the "
        "command writes. It is passed so denied_paths can rule on where the "
        "job runs; the workspace boundary would be ruling on the wrong thing.",
    "web/references.py":
        "read_file with is_read_only=True — a read, and it says so.",
    "documents/service.py":
        "the Documents service confines paths to its own root before calling "
        "the gate, and passes is_read_only for the read case; the boundary "
        "here would duplicate a confinement that already happened.",
}


def test_every_production_call_site_declares_whether_its_path_is_written():
    """No `file_path=` into the gate from src/ without a stated access.

    This is what makes "undeclared does not trip the boundary" a safe runtime
    default rather than a fail-open one: an undeclared WRITE is a broken build.

    Scanned with `ast`, not with a regex over the file text. The sibling guard
    in `test_gate_sees_the_path.py` matches raw text and therefore fires on
    PROSE — it flagged this file's own docstring, and a regex version of this
    test flagged `tool_paths.py`'s. Only real call nodes count here.
    """
    src = Path(__file__).resolve().parents[1] / "src" / "prometheus"
    undeclared: list[str] = []

    for py in sorted(src.rglob("*.py")):
        rel = str(py.relative_to(src)).replace("\\", "/")
        if rel in _CONTEXT_PATH_CALL_SITES or py.name == "checker.py":
            continue  # checker.py DEFINES evaluate(); it is not a caller
        text = py.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        try:
            tree = ast.parse(text)
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "evaluate"):
                continue
            kwargs = {kw.arg for kw in node.keywords if kw.arg}
            if "file_path" not in kwargs or "path_is_write" in kwargs:
                continue
            # `file_path=None` asserts the absence of a path, which is honest.
            fp = next(kw.value for kw in node.keywords if kw.arg == "file_path")
            if isinstance(fp, ast.Constant) and fp.value is None:
                continue
            # Line-local opt-out, so exempting one call does not exempt the
            # whole file — `agent_loop.py` holds both the real dispatch call
            # and the legacy-gate shim, and only the second may skip this.
            window = "\n".join(
                lines[max(0, node.lineno - 10):node.lineno]
            )
            if "gate-path-context:" in window:
                continue
            undeclared.append(f"{rel}:{node.lineno}")

    assert not undeclared, (
        "these call sites hand the SecurityGate a path without declaring "
        "whether it is WRITTEN, so the workspace boundary cannot rule on "
        "them:\n  " + "\n  ".join(undeclared) + "\n\n"
        "Pass path_is_write=<bool> (gate_path_is_write() computes it from the "
        "tool's schema), or register the file in _CONTEXT_PATH_CALL_SITES with "
        "the reason its path is context rather than a target."
    )


def test_the_registered_context_sites_still_exist():
    """A registry entry for a file that no longer calls the gate is rot.

    It would silently exempt whatever that path grows into next.
    """
    src = Path(__file__).resolve().parents[1] / "src" / "prometheus"
    missing = [
        rel for rel in _CONTEXT_PATH_CALL_SITES
        if not (src / rel).exists()
        or ".evaluate(" not in (src / rel).read_text(encoding="utf-8", errors="replace")
    ]
    assert not missing, (
        f"registered as context-path call sites but no longer call the gate: "
        f"{missing}. Remove the entry rather than leaving a standing exemption."
    )


# ── The over-refusal direction ──────────────────────────────────────────────

def test_a_context_path_on_a_non_read_only_call_does_not_prompt():
    """The two shapes that broke the naive predicate, pinned directly.

    Both hand the gate a real path on a call that is NOT read-only, and
    neither is a write target:

      * a bash call carrying the job's working directory (cron)
      * a read_file call whose caller did not pass is_read_only

    A predicate of "carries a path and is not read-only" prompts for both —
    that cost 23 failures across cron, denied-paths and the grant-floor
    invariant. A predicate of `True` prompts for everything and would pass
    every other test in this file. This one fails for both.
    """
    gate = SecurityGate(workspace_root=["/nonexistent-workspace-root"])

    cron_like = gate.evaluate(
        "bash", command="echo hi", file_path="/var/tmp/some-job-cwd",
        origin="system",
    )
    assert cron_like.action != "APPROVE", (
        "a bash call carrying its working directory was sent to the workspace "
        f"boundary; got {cron_like.action} ({cron_like.reason!r}). cwd is "
        "context for denied_paths, not a write target."
    )

    read_like = gate.evaluate("read_file", file_path="/etc/hostname")
    assert read_like.action != "APPROVE" or "outside workspace" not in (
        read_like.reason or ""
    ), (
        "an undeclared read tripped the WRITE boundary: "
        f"{read_like.action} ({read_like.reason!r})"
    )
