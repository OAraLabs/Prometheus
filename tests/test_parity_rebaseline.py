"""The rebaseline-from-exchanges mode: re-derive expected files from a replay of the
COMMITTED exchanges, for a change that alters only what the daemon records.

Will's rules (2026-09-26, WP-X.21), each pinned here without a daemon:

* its own explicit harness mode;
* it refuses — exit non-zero, nothing written — when any model request differs
  from its committed exchange, has no committed answer, or a committed request
  is never made;
* it writes only ``*.expected.json``, never a trace;
* it prints the per-column diff (``--dry-run`` prints and writes nothing, so a
  bundle's diff can be attributed one change at a time).

It also refuses a harness error, a daemon step that failed, a scenario whose own
``require`` check now fails, and observables that differ between two replays of
the same tree — each would bake something untrue into a golden. All-or-nothing:
one refusal and no file is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from parity import cli  # noqa: E402
from parity import compare as cmp  # noqa: E402
from parity import traces  # noqa: E402
from parity.model_server import Exchange, Served  # noqa: E402
from parity.runner import RunOutput  # noqa: E402

REQ = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def _run(stores: dict, *, served: list | None = None, unconsumed: list | None = None,
         steps: list | None = None) -> RunOutput:
    return RunOutput(
        scenario="t", mode="replay", steps=steps if steps is not None else [{"op": "chat", "reply": "ok"}],
        stores=stores,
        exchanges=[Exchange(method="POST", path="/v1/chat/completions", status=200,
                            content_type="text/event-stream", body="", request=REQ, upstream="primary")],
        served=served if served is not None else [Served(index=0, recorded_index=0, matched=True, request=REQ)],
        unconsumed=unconsumed or [], turns=[], wall_offset_ns=0, rss=[], boot_seconds=1.0,
        shutdown="clean", daemon_log=Path("/dev/null"),
    )


def _telemetry(session: str | None) -> dict:
    return {"home/.prometheus/telemetry.db": {"sqlite": {"tool_calls": {
        "columns": ["tool_name", "session_id"], "rows": [["_loop_transition", session]]}}}}


EXPECTED = cmp.expected_from(_run(_telemetry(None)), Path("/"))


def _judge(runs, require=None):
    from parity.rebaseline import judge

    return judge("t", runs, EXPECTED, Path("/"), require=require)


# ── the judgement ───────────────────────────────────────────────────────────

def test_a_change_to_what_is_recorded_is_accepted_with_its_per_column_diff():
    v = _judge([_run(_telemetry("desktop:x"))] * 2)
    assert v.refusals == []
    assert v.changed
    assert v.expected == cmp.expected_from(_run(_telemetry("desktop:x")), Path("/"))
    assert "tool_calls.rows[0].session_id" in v.report, "the diff must name the column"


def test_an_unchanged_scenario_is_accepted_and_not_marked_changed():
    v = _judge([_run(_telemetry(None))] * 2)
    assert (v.refusals, v.changed, v.expected) == ([], False, None)


def test_a_request_that_differs_from_its_committed_exchange_is_refused():
    bad = _run(_telemetry("desktop:x"), served=[Served(index=0, recorded_index=0, matched=False,
                                                       request={**REQ, "model": "other"})])
    v = _judge([bad, bad])
    assert any("differ from the committed exchanges" in r for r in v.refusals)
    assert v.expected is None


def test_a_request_with_no_committed_answer_is_refused():
    bad = _run(_telemetry("desktop:x"), served=[
        Served(index=0, recorded_index=0, matched=True, request=REQ),
        Served(index=1, recorded_index=None, matched=False, request=REQ)])
    v = _judge([bad, bad])
    assert any("no committed answer" in r for r in v.refusals)


def test_a_committed_request_the_daemon_never_made_is_refused():
    bad = _run(_telemetry("desktop:x"), unconsumed=[1])
    v = _judge([bad, bad])
    assert any("never made" in r for r in v.refusals)


def test_a_step_failure_is_refused():
    bad = _run(_telemetry("desktop:x"))
    bad.step_failures.append("step chat: daemon returned 500")
    v = _judge([bad, bad])
    assert any("step failed" in r for r in v.refusals)


def test_a_harness_error_is_refused():
    bad = _run(_telemetry("desktop:x"))
    bad.errors.append("daemon did not answer")
    v = _judge([bad, bad])
    assert any("harness error" in r for r in v.refusals)
    assert v.harness_error


def test_a_scenario_whose_own_check_now_fails_is_refused():
    v = _judge([_run(_telemetry("desktop:x"))] * 2, require=lambda ev: ["no executed call"])
    assert any("own check" in r and "no executed call" in r for r in v.refusals)


def test_observables_that_differ_between_two_replays_are_refused():
    v = _judge([_run(_telemetry("desktop:x")), _run(_telemetry("desktop:y"))])
    assert any("differ between" in r for r in v.refusals)


# ── the command: all-or-nothing, expected files only ────────────────────────

def _fixture_root(tmp_path, names=("a", "b")) -> Path:
    src = tmp_path / "src"
    for name in names:
        trace = {"format": traces.FORMAT, "scenario": name, "covers": "c", "recorded": {},
                 "config": "", "files": {}, "git_repos": [], "steps": [], "exchanges": []}
        (traces.trace_dir(src)).mkdir(parents=True, exist_ok=True)
        (traces.trace_dir(src) / f"{name}.trace.json").write_text(json.dumps(trace))
        traces.write_expected(src, name, EXPECTED)
    return src


def _snapshot(src: Path) -> dict:
    return {p.name: p.read_bytes() for p in sorted(traces.trace_dir(src).iterdir())}


def _cli(tmp_path, monkeypatch, runs_by_name: dict, *, dry_run=False):
    """Run the command over fixture files for *runs_by_name*; returns (exit code,
    the fixture files before, after, the fixture root)."""
    src = _fixture_root(tmp_path, tuple(runs_by_name))
    before = _snapshot(src)
    monkeypatch.setattr(cli, "SRC_ROOT", src)
    monkeypatch.setattr(cli, "replay_one", lambda name, root: runs_by_name[name])
    args = argparse.Namespace(scenario=None, root=tmp_path / "root", runs=2, dry_run=dry_run)
    rc = cli.cmd_rebaseline_from_exchanges(args)
    return rc, before, _snapshot(src), src


def test_it_writes_only_the_expected_files_that_changed(tmp_path, monkeypatch):
    rc, before, after, src = _cli(tmp_path, monkeypatch, {
        "a": _run(_telemetry("desktop:x")), "b": _run(_telemetry(None))})
    assert rc == 0
    assert {n for n in after if after[n] != before[n]} == {"a.expected.json"}, (
        "only a's expected file: no trace, and not the unchanged b")
    assert traces.load_expected(src, "a") == cmp.expected_from(_run(_telemetry("desktop:x")), Path("/"))


def test_one_refusal_and_nothing_is_written(tmp_path, monkeypatch):
    rc, before, after, _ = _cli(tmp_path, monkeypatch, {
        "a": _run(_telemetry("desktop:x")), "b": _run(_telemetry("desktop:x"), unconsumed=[1])})
    assert rc == 1
    assert after == before, "a's change must not be written while b is refused"


def test_a_harness_error_exits_2_and_writes_nothing(tmp_path, monkeypatch):
    bad = _run(_telemetry("desktop:x"))
    bad.errors.append("daemon did not answer")
    rc, before, after, _ = _cli(tmp_path, monkeypatch, {"a": bad})
    assert (rc, after) == (2, before)


def test_a_dry_run_writes_nothing(tmp_path, monkeypatch):
    rc, before, after, _ = _cli(tmp_path, monkeypatch, {
        "a": _run(_telemetry("desktop:x")), "b": _run(_telemetry(None))}, dry_run=True)
    assert (rc, after) == (0, before)


def test_the_mode_is_a_subcommand_of_its_own(capsys):
    # --help exits 0; an unknown subcommand would exit 2 — assert which.
    with pytest.raises(SystemExit) as exit_:
        cli.main(["rebaseline-from-exchanges", "--help"])
    assert exit_.value.code == 0
    assert "--dry-run" in capsys.readouterr().out
