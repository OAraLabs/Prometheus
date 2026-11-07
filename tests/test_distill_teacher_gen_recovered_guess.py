"""teacher_gen.py round 2 long-context: a failed guessed read the teacher recovered from within the next
two calls (the next call lists the guessed file's directory, the one after reads a file there) is not a
tool error (operator 2026-11-06, option a). Everything else still rejects. Synthetic fixtures; a fake
loopback teacher; no network."""

from __future__ import annotations

import posixpath
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from tests.test_distill_teacher_gen import FakeTeacher, _load, scripted, settings_for
from tests import test_distill_teacher_gen_r2 as r2
from tests.test_distill_teacher_gen_r2 import _lc_task, _process, _settings

tg = _load("distill_teacher_gen", "teacher_gen.py")
no_outbound = r2.no_outbound  # the fixture: any connection off loopback fails the test

FILES = {"docs/a.md": "# Alpha\n" + "alpha text\n" * 30, "docs/b.md": "# Bravo\ndef frobnicate_widgets():\n",
         "notes/c.md": "# Charlie\n"}
#: the two seeded context reads are the fake teacher's steps 0 and 1: the task's own turn starts at 2
SEEDED = ["echo", "echo"]
GUESS = ("call", "read_file", {"path": "docs/guessed_name.md"})
GLOB = ("call", "glob", {"pattern": "docs/*.md"})
READ_B = ("call", "read_file", {"path": "docs/b.md"})
READ_A = ("call", "read_file", {"path": "docs/a.md"})
ANSWER = ("text", "docs/b.md defines frobnicate_widgets; docs/a.md is a page of alpha text.")


def _lc(tmp_path: Path) -> Any:
    t = _lc_task(tmp_path, dict(FILES), None)
    t.context_reads = [{"path": "docs/a.md", "offset": 0, "limit": 200}, {"path": "docs/b.md", "offset": 0, "limit": 200}]
    return t


def _run(tmp_path: Path, plan: list[Any], task: Any = None, **kw: Any) -> tuple[str, list[dict[str, Any]],
                                                                               list[str], dict[str, Any]]:
    t = task if task is not None else _lc(tmp_path)
    with FakeTeacher(scripted({"Summarize": plan})) as fake:
        return _process(t, kw.pop("settings", None) or _settings(tmp_path, fake.url, **kw))


# ── end to end ─────────────────────────────────────────────────────────────────


def test_a_recovered_guess_is_kept_recorded_and_stays_in_the_later_rows_context(tmp_path: Path,
                                                                                no_outbound: Any) -> None:
    status, rows, reasons, extra = _run(tmp_path, SEEDED + [GUESS, GLOB, READ_B, ANSWER])
    assert status == "kept", reasons
    rec = extra["verification"]
    assert rec["recovered_guess"] == [{"call_index": 0, "tool": "read_file", "path": "docs/guessed_name.md",
                                       "listing_index": 1, "read_index": 2}]
    assert rec["tool_errors"] == 1 and rec["tool_success"] is True  # the raw count stays honest
    assert all(r["_meta"]["source"] == tg.SOURCE_TAG for r in rows)  # strict rows, not relaxed
    assert all(r["_meta"]["verification"]["recovered_guess"] for r in rows)
    # the failed call is never a target; the rows after it carry it (and its error) in their context
    targets = [str(r["messages"][-1].get("tool_calls") or "") for r in rows]
    assert not any("guessed_name" in t for t in targets)
    later = [r for r in rows if any("guessed_name" in str(m.get("tool_calls") or "") for m in r["messages"][:-1])]
    assert later and all(r["_meta"]["error_recovery"] for r in later)
    assert any(m.get("role") == "tool" and m.get("is_error") for m in later[0]["messages"])


def test_a_guess_without_a_recovery_is_rejected(tmp_path: Path, no_outbound: Any) -> None:
    status, _rows, reasons, extra = _run(tmp_path, SEEDED + [GUESS, READ_B, ANSWER])
    assert status == "rejected" and "tool_error" in reasons
    assert "recovered_guess" not in extra["verification"]


def test_a_recovery_that_lists_another_directory_is_rejected(tmp_path: Path, no_outbound: Any) -> None:
    plan = SEEDED + [GUESS, ("call", "glob", {"pattern": "notes/*.md"}), READ_B, ANSWER]
    status, _rows, reasons, extra = _run(tmp_path, plan)
    assert status == "rejected" and "tool_error" in reasons
    assert "recovered_guess" not in extra["verification"]


def test_the_whole_recovery_must_be_the_next_two_calls(tmp_path: Path, no_outbound: Any) -> None:
    # the listing at +2 (one call between the guess and the listing): too late
    plan = SEEDED + [GUESS, READ_A, GLOB, READ_B, ANSWER]
    status, _rows, reasons, extra = _run(tmp_path, plan)
    assert status == "rejected" and "tool_error" in reasons
    assert "recovered_guess" not in extra["verification"]
    # the listing at +3: too late
    plan = SEEDED + [GUESS, READ_A, ("call", "read_file", {"path": "docs/a.md", "offset": 5}), GLOB, READ_B, ANSWER]
    status, _rows, reasons, extra = _run(tmp_path / "b", plan)
    assert status == "rejected" and "tool_error" in reasons
    assert "recovered_guess" not in extra["verification"]
    # the listing at +1 but the read at +3 (another call between them): too late
    plan = SEEDED + [GUESS, GLOB, ("call", "bash", {"command": "echo looking"}), READ_B, ANSWER]
    status, _rows, reasons, extra = _run(tmp_path / "c", plan)
    assert status == "rejected" and "tool_error" in reasons
    assert "recovered_guess" not in extra["verification"]


def test_a_non_read_tool_error_still_rejects_even_beside_a_recovered_guess(tmp_path: Path,
                                                                            no_outbound: Any) -> None:
    # a failed grep (no match, exit 1) is not a guessed read: the row still has a natural error
    plan = SEEDED + [GUESS, GLOB, READ_B, ("call", "bash", {"command": "grep -n zzz_nowhere docs/a.md"}), ANSWER]
    status, _rows, reasons, extra = _run(tmp_path, plan)
    assert status == "rejected" and "tool_error" in reasons
    assert len(extra["verification"]["recovered_guess"]) == 1  # recorded, but one error is left
    # a read of a directory is not a missing file, even when a listing and a read follow
    plan = SEEDED + [("call", "read_file", {"path": "docs"}), GLOB, READ_B, ANSWER]
    status, _rows, reasons, extra = _run(tmp_path / "b", plan)
    assert status == "rejected" and "tool_error" in reasons
    assert "recovered_guess" not in extra["verification"]


def test_a_bash_guess_and_a_bash_recovery_are_kept(tmp_path: Path, no_outbound: Any) -> None:
    plan = SEEDED + [("call", "bash", {"command": "cat docs/guessed_name.md"}),
                     ("call", "bash", {"command": "ls docs"}),
                     ("call", "bash", {"command": "head -n 20 docs/b.md"}), ANSWER]
    status, _rows, reasons, extra = _run(tmp_path, plan)
    assert status == "kept", reasons
    assert extra["verification"]["recovered_guess"] == [
        {"call_index": 0, "tool": "bash", "path": "docs/guessed_name.md", "listing_index": 1, "read_index": 2}]


def test_another_area_is_unchanged(tmp_path: Path, no_outbound: Any) -> None:
    t = tg.GenTask(id="ts-plain", kind="agentic", domain="other", family="taskset:general",
                   prompt="Summarize what docs/b.md says.", origin=tg.ORIGIN_TASKSET, setup_files=dict(FILES),
                   meta={"provenance": tg.TASKSET_PROVENANCE, "task_id": "tg-plain"})
    t.validate()
    assert t.area is None
    with FakeTeacher(scripted({"Summarize": [GUESS, GLOB, READ_B, ANSWER]})) as fake:
        status, _rows, reasons, extra = _process(t, settings_for(tmp_path, fake.url))
    assert status == "rejected" and "tool_error" in reasons
    assert "recovered_guess" not in extra["verification"]
    # a round-2 task of another area too
    t2 = _lc(tmp_path)
    t2.r2 = dict(t2.r2, area="daily-agent")
    status, _rows, reasons, extra = _run(tmp_path / "b", SEEDED + [GUESS, GLOB, READ_B, ANSWER], task=t2)
    assert status == "rejected" and "tool_error" in reasons
    assert "recovered_guess" not in extra["verification"]


# ── the rule itself (recovered_guesses) ────────────────────────────────────────

ROOT = PurePosixPath("/sb")


def _resolve(path: str, cwd: PurePosixPath | None) -> PurePosixPath | None:
    if path.startswith("~"):
        path = str(ROOT) + path[1:]
    p = PurePosixPath(posixpath.normpath(posixpath.join(str(cwd or ROOT), path)))
    return p if p == ROOT or ROOT in p.parents else None


def _ev(name: str, args: dict[str, Any], outcome: str = "ok", result: str = "") -> dict[str, Any]:
    return {"name": name, "input": args, "outcome": outcome, "result": result}


def _miss(path: str) -> dict[str, Any]:
    return _ev("read_file", {"path": path}, "error", f"File not found: {ROOT}/{path}")


def _read(path: str) -> dict[str, Any]:
    return _ev("read_file", {"path": path})


def _sh(cmd: str, outcome: str = "ok", result: str = "") -> dict[str, Any]:
    return _ev("bash", {"command": cmd}, outcome, result)


def _forgiven(events: list[dict[str, Any]]) -> list[int]:
    return [g["call_index"] for g in tg.recovered_guesses(events, _resolve)]


GUESSED = "P/tests/test_guess.py"
REAL = "P/tests/test_real.py"


@pytest.mark.parametrize("listing", [
    _ev("glob", {"pattern": "P/tests/test_*.py"}),
    _ev("glob", {"pattern": "**/test_*.py"}),            # any depth from the root
    _ev("glob", {"pattern": "*.py", "root": "P/tests"}),
    _ev("glob", {"pattern": f"{ROOT}/P/**/*.py"}),       # absolute
    _sh("ls P/tests"),
    _sh("ls -la P/tests/ | grep test_"),
    _sh("cd P && ls tests"),
    _sh("ls -R P"),                                      # an ancestor, recursively
    _sh("find P -name 'test_*.py'"),
    _sh("find . -maxdepth 3 -name '*.py'"),              # P/tests is 2 below the root: within depth 3
    _sh("grep -rl frob P"),
    _sh("grep -l frob P/tests/*.py"),
    _sh("ls ~/P/tests"),
])
def test_a_listing_of_the_guessed_files_directory_or_a_covering_ancestor_forgives_it(listing: dict[str, Any]) -> None:
    assert _forgiven([_miss(GUESSED), listing, _read(REAL)]) == [0]


@pytest.mark.parametrize("listing", [
    _sh("ls P"),                                         # an ancestor, not recursive: tests/ entries not shown
    _sh("ls P/src"),
    _ev("glob", {"pattern": "P/*.py"}),
    _ev("glob", {"pattern": "P/src/**/*.py"}),
    _sh("find P/src -name '*.py'"),
    _sh("find . -maxdepth 2 -name '*.py'"),             # stops one level short of P/tests
    _sh("grep -l frob P/*.py"),                           # files in P, not P/tests
    _sh("grep -rn frob P"),                               # a search, not a listing (no -l)
    _ev("grep", {"pattern": "frob", "root": "P"}),        # the grep tool lists matches, not files
    _ev("glob", {"pattern": "P/tests/*.py"}, "error", "boom"),  # a failed listing
    _sh("ls P/tests", "error", "ls: P/tests: No such file or directory"),
])
def test_a_listing_that_does_not_cover_the_guessed_files_directory_does_not(listing: dict[str, Any]) -> None:
    assert _forgiven([_miss(GUESSED), listing, _read(REAL)]) == []


def test_the_window_is_the_next_two_calls_listing_then_read() -> None:
    assert (tg.RECOVERED_GUESS_WINDOW, tg.RECOVERED_GUESS_READ_WINDOW) == (1, 1)
    lst = _sh("ls P/tests")
    assert _forgiven([_miss(GUESSED), lst, _read(REAL)]) == [0]
    assert _forgiven([_miss(GUESSED), _read("P/a.py"), lst, _read(REAL)]) == []              # listing at +2
    assert _forgiven([_miss(GUESSED), _read("P/a.py"), _read("P/b.py"), lst, _read(REAL)]) == []
    assert _forgiven([_miss(GUESSED), lst, _read("P/a.py"), _read(REAL)]) == []              # read at +3
    assert _forgiven([_miss(GUESSED), lst, _read("P/tests/a.py"), _read(REAL)]) == [0]       # read at +2
    # two guesses before one listing: only the one right before it is recovered (the row still fails)
    assert _forgiven([_miss(GUESSED), _miss("P/tests/test_other.py"), lst, _read(REAL)]) == [1]
    # the windows are parameters: the wider reading would forgive these
    wide = [_miss(GUESSED), _read("P/a.py"), lst, _read("P/b.py"), _read(REAL)]
    assert [g["call_index"] for g in tg.recovered_guesses(wide, _resolve, window=2, read_window=None)] == [0]


def test_the_read_must_come_after_the_listing_and_be_where_it_looked() -> None:
    lst = _sh("ls P/tests")
    assert _forgiven([_miss(GUESSED), lst]) == []                                   # no read at all
    assert _forgiven([_read(REAL), _miss(GUESSED), lst]) == []                      # the read was before
    assert _forgiven([_miss(GUESSED), lst, _read("P/src/real.py")]) == []           # elsewhere
    assert _forgiven([_miss(GUESSED), lst, _miss("P/tests/again.py")]) == []        # another guess
    assert _forgiven([_miss(GUESSED), lst, _read("P/src/a.py"), _read(REAL)]) == []  # the next read is elsewhere
    assert _forgiven([_miss(GUESSED), lst, _sh(f"sed -n '1,40p' {REAL}")]) == [0]
    assert _forgiven([_miss(GUESSED), lst, _sh(f"grep -n frob {REAL}")]) == []      # a search is not a read


@pytest.mark.parametrize("guess", [
    _sh(f"cat {GUESSED}", "error", f"cat: {GUESSED}: No such file or directory"),
    _sh(f"head -n 50 {GUESSED}", "error",
        f"head: cannot open '{GUESSED}' for reading: No such file or directory"),
    _sh(f"sed -n '1,80p' {GUESSED}", "error", f"sed: can't read {GUESSED}: No such file or directory"),
    _sh("cd P/tests && cat test_guess.py", "error", "cat: test_guess.py: No such file or directory"),
    _sh(f"cat {GUESSED} | head -20", "error", f"cat: {GUESSED}: No such file or directory"),
])
def test_a_bash_read_of_a_missing_file_is_a_guess(guess: dict[str, Any]) -> None:
    assert _forgiven([guess, _sh("ls P/tests"), _read(REAL)]) == [0]


@pytest.mark.parametrize("call", [
    _sh(f"ls {GUESSED}", "error", f"ls: {GUESSED}: No such file or directory"),           # ls, not a read
    _sh("cd P/nowhere && cat x.py", "error", "sh: cd: can't cd to P/nowhere: no such directory"),
    _sh(f"cat {GUESSED} && echo done", "error", f"cat: {GUESSED}: No such file or directory"),  # not only reads
    _sh(f"sed -i 's/a/b/' {GUESSED}", "error", f"sed: can't read {GUESSED}: No such file or directory"),
    _sh("grep -n zzz P/tests/test_real.py", "error", "(no output)"),
    _ev("read_file", {"path": "P/tests"}, "error", "Cannot read directory: P/tests"),
    _ev("read_file", {"path": GUESSED}, "refused", "Refused: outside"),
    _ev("read_file", {"path": "/etc/passwd"}, "error", "File not found: /etc/passwd"),  # outside: unresolvable
    _ev("edit_file", {"path": REAL, "old_str": "x", "new_str": "y"}, "error", "old_str not found"),
])
def test_other_failures_are_never_forgiven(call: dict[str, Any]) -> None:
    assert _forgiven([call, _sh("ls -R P"), _read(REAL)]) == []


def test_the_record_names_the_call_as_spelled() -> None:
    got = tg.recovered_guesses([_sh("echo hi"), _miss(GUESSED), _ev("glob", {"pattern": "P/tests/*"}),
                                _read(REAL), _ev("todo_write", {})], _resolve)
    assert got == [{"call_index": 1, "tool": "read_file", "path": GUESSED, "listing_index": 2, "read_index": 3}]


def test_the_sandbox_resolver_maps_host_spellings_and_confines(tmp_path: Path) -> None:
    t = _lc(tmp_path)
    root = (tmp_path / "w").resolve()
    root.mkdir()
    env = tg.SandboxEnv(root, t, host=tg.HostPaths())
    resolve = tg.sandbox_resolver(env)
    assert resolve("docs/a.md", None) == PurePosixPath(root / "docs/a.md")
    assert resolve("~/docs", None) == PurePosixPath(root / "docs")
    assert resolve("a.md", PurePosixPath(root / "docs")) == PurePosixPath(root / "docs/a.md")
    assert resolve("/etc/passwd", None) is None and resolve("../x", None) is None
